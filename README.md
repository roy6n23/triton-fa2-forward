# triton-fa2-forward

[![tests (Triton interpreter)](https://github.com/roy6n23/triton-fa2-forward/actions/workflows/tests.yml/badge.svg)](https://github.com/roy6n23/triton-fa2-forward/actions/workflows/tests.yml)

A FlashAttention-2 style **forward** attention kernel in Triton: causal or full,
FP16/BF16, with grouped-query attention (GQA). About 200 lines of kernel code.
It comes with a pure-PyTorch mirror of the same algorithm, a correctness suite
graded against PyTorch's own error, a benchmark against FlashAttention-2 and -3,
cuDNN, FlashInfer and PyTorch SDPA, and a breakdown of where the kernel's time goes.

> **Status (2026-10-04):** third H100 SXM run, on a Verda VM where Nsight Compute can
> read the GPU's performance counters. 119 tests pass on the GPU. The kernel now has
> GQA head packing and the two fixes its profile pointed to (7% to 10% less time), it is
> 22% to 37% faster than FlashAttention-2, and the gap to FlashAttention-3 is measured
> down to the instruction below ([WORKLOG](WORKLOG.md), 2026-10-03 and 2026-10-04).

## Layout

```
fa2/kernel.py        Triton kernel + attention() wrapper
fa2/reference.py     naive_attention (ground truth) and blocked_attention
                     (the kernel's algorithm, line for line, in plain torch)
tests/               correctness suite
bench/bench.py       benchmark: warmup/IQR protocol, baselines, env pinning
bench/ablation.py    knockout copies of the kernel: time with one kind of work removed
bench/sass_stats.py  instructions per K/V tile and their schedule, from the SASS (no GPU needed)
bench/profile_ncu.sh Nsight Compute on the kernel, two knockouts and FA3 (needs counter access)
bench/gpu_session.sh every GPU measurement in one logged run; bench/results/<date>/ is its output
bench/vm_session.sh  the same run on a VM with sudo, in the vLLM image with CAP_SYS_ADMIN, so ncu can run
bench/ncu_summary.py the Nsight Compute reports side by side, and where the warps stall (CSV only, no GPU)
WORKLOG.md           what broke and how it was found
.github/workflows/   CI: the test suite under TRITON_INTERPRET=1 on a CPU runner
```

## Quick start

```bash
git clone https://github.com/roy6n23/triton-fa2-forward && cd triton-fa2-forward
pip install -e ".[test]"
pytest                                   # GPU box: full sweep
TRITON_INTERPRET=1 pytest                # no GPU: kernel runs in Triton's numpy interpreter
pytest -k blocked                        # no Triton at all (e.g. macOS): algorithm only
python bench/bench.py                    # GPU only
python bench/ablation.py --rounds 5      # GPU: knockout timings, interleaved
python bench/sass_stats.py               # Linux with Triton, no GPU needed: SASS of one K/V tile (sm_90)
```

```python
from fa2.kernel import attention
o = attention(q, k, v, causal=True)      # q [B,Hq,N,D], k/v [B,Hkv,N,D], Hq % Hkv == 0
```

## The algorithm

Attention for one query row is `o = Σ_j softmax(s)_j v_j`, where `s_j = q·k_j / √d`.
The softmax needs the row max `m` for stability, plus the normaliser
`l = Σ_j exp(s_j − m)`. Both seem to need the whole row before any output can
be produced. That would force the N×N score matrix out to HBM.

**Online softmax** removes that dependency. Walk the row in tiles and keep a
running `(m, l, acc)`. When a new tile arrives with scores `s`:

```
m_new = max(m, max(s))
alpha = exp(m − m_new)                    # how much the old state has to shrink
l     = l · alpha + Σ exp(s − m_new)
acc   = acc · alpha + Σ exp(s − m_new) · v
m     = m_new
```

At the end, `o = acc / l`. The rescale by `alpha` makes this exact, not an
approximation: after any prefix of tiles, `(m, l, acc)` equals what you would
get from processing that prefix in one shot. So scores only ever live in
on-chip SRAM, one `BLOCK_M × BLOCK_N` tile at a time.
[`fa2/reference.py:blocked_attention`](fa2/reference.py) is this loop in plain
torch. [`fa2/kernel.py:_attn_fwd_inner`](fa2/kernel.py) is the same loop in Triton.

## Design decisions

| Decision | Why |
|---|---|
| **Q tiles on the grid, K/V in the inner loop** (FA2 order) | Each program owns its output rows outright, so programs never communicate or merge partial results. The grid also spans the sequence dimension, which keeps SMs busy at batch 1. |
| **Normalise once, after the loop** | `acc` stays unnormalised inside the loop, and the divide by `l` happens once per row. Non-matmul FLOPs run on much slower units than tensor-core matmuls, so cutting them matters even though the total FLOP count barely changes. |
| **Causal block skipping in two stages** | Tiles fully left of the diagonal need no causal mask (stage 1). Only the `BLOCK_M`-wide diagonal band pays for it (stage 2). Tiles right of it are never loaded or multiplied. For long sequences that removes about half the work. |
| **`exp2`, with `log2 e` and the softmax scale in one FFMA** | `exp(x) = exp2(x·log2 e)`, so every exponential is a single hardware `exp2`. The row max is taken on the raw scores and scaled once per row, so each score costs one FFMA, `qk·scale − m`, instead of a multiply and a subtract. This needs scale > 0. |
| **No end-of-sequence mask when N_kv is a multiple of BLOCK_N** (`EVEN_N`) | No tile then reaches past the end, and stage 1 drops the 32 selects per tile it otherwise spends on that mask. With the FFMA above, one K/V tile goes from 447 to 371 instructions ([Where the time goes](#where-the-time-goes)). |
| **P cast back to FP16/BF16 before `P @ V`** | Keeps the second matmul on tensor cores. The statistics `m`, `l` and the accumulator stay in FP32. |
| **GQA by index mapping** (`kv_head = q_head // group`) | No K/V replication in memory, and it is a single integer divide per program. |
| **GQA head packing when it fills tiles better** | A group's query heads can share one program: row r of the tile is position r // G of head r % G, so each K/V tile serves the whole group. `attention()` does this when it fills the `BLOCK_M` rows at least 10% better (FA3's rule). It is what makes few queries per head against a long K/V fast ([below](#gqa-head-packing)). |
| **`-inf` masking without NaN guards** | The first tile any row processes always contains column 0, which every row can see (stage 1 starts there, and stage 2 does whenever stage 1 is empty). So `m_new` is finite from the first update on, and `exp(-inf − -inf)` can never happen. The reasoning is written next to the code. |

## Correctness protocol

Ground truth is `naive_attention` computed in **fp64** on the same FP16 inputs.
A kernel passes if its max-abs error is **≤ 2× the error of
`torch.nn.functional.scaled_dot_product_attention` in FP16** on the same inputs
(plus a 1e-4 absolute floor). The sweep covers causal and non-causal; MHA, GQA
32/8 and MQA; head dim 64 and 128; sequence lengths including 1000, which is not
a multiple of any block size; non-contiguous input strides; and N_q ≠ N_kv for
non-causal.

The suite is checked against bugs deliberately planted in the kernel.
Dropping the `alpha` rescale of `acc` fails 26 of the kernel tests. A
causal mask that is off by one (`>` instead of `>=`) fails all 19 causal tests.

Interpreter results (CPU, `D=64`), recorded on 2026-09-22, before the GPU run.
The interpreter's `tl.dot` is more exact than real tensor cores, so these ratios
are optimistic and are not GPU measurements:

| causal | heads q/kv | N | ours max-abs | torch max-abs | ratio |
|---|---|---|---|---|---|
| yes | 4/4 | 128 | 7.22e-4 | 7.22e-4 | 1.00 |
| yes | 4/1 | 128 | 7.94e-4 | 7.94e-4 | 1.00 |
| no | 4/4 | 128 | 2.71e-4 | 2.77e-4 | 0.98 |
| no | 4/1 | 128 | 2.75e-4 | 2.50e-4 | 1.10 |

The GPU runs then applied the same 2× bar on real tensor cores. On the H100,
`pytest` reported 77 passed on 2026-10-02, 115 on 2026-10-03 (the GQA packing tests
added) and 119 on 2026-10-04, BF16 included, which the interpreter has to skip. In the
2026-10-02 num_warps x num_stages sweep (D=128, FP16, causal, N=4096 and 16384),
the max abs error vs SDPA was 4.88e-4 in every config ([WORKLOG](WORKLOG.md)).
That number is a direct difference from SDPA's output, not the fp64-referenced
error in the table above.

## Benchmark protocol

`bench/bench.py`:

- **Warmup 5 + measured 20**, CUDA events per iteration.
- **L2 flushed** (256 MB write) before every measured iteration.
- Report the **median and the IQR (p25–p75)**, never the mean or the best run.
- **Correctness gate:** every provider's output is compared with SDPA before it is timed.
- **Full stack pinned** into the results JSON: GPU SKU and SM count, driver,
  CUDA, cuDNN, PyTorch, Triton, flash-attn, FlashInfer and vLLM versions.
- **Differences of a few percent** are measured interleaved: `bench/ablation.py
  --rounds 5` times every variant once per round and pools the samples, so a slow
  drift in clock or temperature hits all of them alike. At N=16384 two runs of
  identical code differed by 2-4% when measured one after the other.
- **Kernels under about 0.05 ms** (N ≤ 1024 at this shape) are timed one launch per
  event pair, so host-side launch delay inside the window shows up as kernel time.
  In every run with FlashInfer, its first call (at N=512) left the next two providers
  at N=1024 at half speed with large IQRs; the README uses a rerun without FlashInfer
  for those cells, and FlashInfer's own N=512 number measures its launch overhead.

The baselines are FlashAttention-2 and FlashAttention-3, cuDNN (via SDPA), FlashInfer
and SDPA's flash backend. A comparison against torch SDPA alone would be a 2023
comparison. FA2 and FA3 come from vLLM's build (`vllm.vllm_flash_attn`,
`flash_attn_varlen_func` with `fa_version=2` and `3`): the Dao-AILab kernels,
already compiled for this torch and CUDA, where upstream `flash-attn` would have to
be built from source. `bench.py` also tries the upstream packages and skips them
when they are not installed.

## Results

**2026-10-04, H100 80GB HBM3 SXM** (Verda VM, FIN-02), driver 580.178.04, the same
image as before: CUDA 13.0, torch 2.13.0+cu130, Triton 3.7.1, FlashInfer 0.6.18.post1,
vLLM 0.30.0. Default shape: Llama-3-8B attention (H_q=32, H_kv=8, D=128), batch 1,
causal, FP16, 8 warps x 4 stages, kernel at commit 83dbbe8. Median TFLOP/s; the IQRs
and the full environment are in `bench/results/2026-10-04/bench_fp16.json`:

| N | triton-fa2 | FA2 | FA3 | sdpa-flash | sdpa-cudnn | flashinfer |
|---|---|---|---|---|---|---|
| 512 | 112.3 | 86.1 | 107.2 | 86.1 | 126.4 | (19.4) |
| 1024 | 182.1 | 149.4 | 268.2 | 143.3 | 245.8 | 248.3 |
| 2048 | 292.3 | 226.4 | 443.9 | 210.2 | 395.4 | 407.2 |
| 4096 | 389.5 | 296.2 | 589.5 | 274.7 | 527.3 | 543.0 |
| 8192 | 442.0 | 323.7 | 661.1 | 312.5 | 599.4 | 568.3 |
| 16384 | 455.3 | 348.9 | 645.1 | 335.6 | 594.0 | 586.0 |

- **vs FA2:** 22% to 37% faster at every N. FA2's schedule was designed for Ampere.
- **vs FA3, cuDNN, FlashInfer:** the Hopper-specific kernels are 1.3x to 1.5x faster
  from N=2048 up. The next section measures where that gap comes from.
- **BF16** is within 1.5% of FP16 at every N (462.3 TFLOP/s at N=16384, `bench_bf16.json`).
- **The two fixes** (FFMA scale, `EVEN_N`), interleaved on this box against the kernel
  without them: 10.0%, 9.2% and 6.6% less time at N = 4096, 8192 and 16384
  (`ablation_fp16.json`, variant `before_fixes`).
- **vs the 2026-10-03 table** (RunPod, before the fixes; `bench/results/2026-10-03/`):
  +6% at N=16384 and +11% to +16% below it. FA3 measured 640 and 645 TFLOP/s on the two
  boxes, so they are comparable.
- The N=1024 cells for triton-fa2 and sdpa-flash come from a rerun without FlashInfer
  in the process, and FlashInfer's N=512 cell is launch overhead (see the protocol).

**Occupancy (D=128, 8 warps x 4 stages):** 186 registers per thread, no spills, 160
KiB of shared memory. Both cap it at one CTA (two warpgroups, 8 warps) per SM, and
Nsight Compute measures 12.5% achieved occupancy, the same as the theoretical one.

## Where the time goes

At N=16384 the kernel runs at 455 TFLOP/s, 46% of the H100's 989.4. Under this kernel
the GPU sits at its 700 W power cap ("SW power cap" in 74 of 75 `nvidia-smi` samples)
and the SM clock averages 1.61 GHz instead of the 1.83 GHz behind 989.4, so the
ceiling on this box is about 870 TFLOP/s and the kernel reaches 52% of it.

**Nsight Compute** (2025.3.1; `bench/results/2026-10-04/ncu/`, summarised by
`python bench/ncu_summary.py bench/results/2026-10-04/ncu`). Our kernel, the kernel
before its two fixes, the knockout copy with the softmax removed, and FA3, all on the
default shape at N=16384:

| | triton-fa2 | before the fixes | matmuls only | FA3 |
|---|---|---|---|---|
| duration at ncu's locked 1.44 GHz (ms) | 5.53 | 6.27 | 3.42 | 3.53 |
| tensor pipe active (% of cycles) | 51.5 | 45.4 | 83.6 | 81.2 |
| MUFU pipe, which runs `exp2` (%) | 27.4 | 24.2 | 0 | 41.9 |
| instructions executed (millions) | 1,614 | 1,932 | 574 | 1,081 |
| issue slots busy (%) | 38.4 | 40.5 | 22.2 | 40.6 |
| warps per scheduler | 2 | 2 | 2 | 3 |
| L2 hit rate / DRAM throughput (% of peak) | 98.0 / 1.8 | 98.2 / 1.6 | 98.0 / 2.2 | 97.7 / 2.8 |

- **The gap to FA3 is the tensor pipe's idle time.** It is busy 51.5% of the cycles in
  our kernel and 81.2% in FA3: a ratio of 1.58, against 1.57 between the durations.
- **The matmuls themselves are fine.** With the softmax removed, the same schedule keeps
  the tensor pipe 84% busy, as busy as FA3. What costs time is what sits between them.
- **Memory is not the limit.** One KV head's K and V at N=16384 are 8 MB; they stay in
  the 50 MB L2 (98% hit rate) and DRAM runs at 1.8% of its peak.
- **The fixes:** 16% fewer instructions, tensor pipe 45% -> 52% busy.

Where the warps wait, from PC sampling over the whole kernel, grouped by the step of
the K/V-tile loop the waiting instruction belongs to (a sample lands on the instruction
that is waiting, which is not always the one being waited for):

| step | share of samples |
|---|---|
| the softmax: `MUFU.EX2`, FP32 adds, multiplies and max, conversions, shuffles | 41% |
| address arithmetic and loop bookkeeping | 22% |
| `WARPGROUP.DEPBAR.LE gsb0, 0x0`: wait for the Q K^T matmul | 19% |
| issuing the matmuls (`HGMMA`, `WARPGROUP.ARRIVE`) | 11% |
| barriers and K/V copies | 8% |

The one instruction with the most samples is that `WARPGROUP.DEPBAR` (18% of all
samples). The SASS shows why (`bench/sass_stats.py`; GPU build, stage-1 loop, 371
instructions per iteration). Each of the two warpgroups:

```
DEPBAR.LE / BAR.SYNC               wait for this tile's K/V copies; CTA-wide barrier
HGMMA.64x64x16.F32 x8              S = Q K^T on the tensor cores (async)
WARPGROUP.DEPBAR.LE gsb0, 0x0      wait for every matmul in flight
255 instructions                   softmax: 66 FMUL, 36 FMNMX, 34 MUFU.EX2, 32 FFMA, 32 FADD, ...
HGMMA.64x128x16.F32 x4             O += P V (async, waited at the next iteration)
BAR.SYNC / LDGSTS x8               barrier, then cp.async copies of a later K/V tile
```

A warpgroup gives the tensor cores nothing new while it runs its softmax, and both
warpgroups pass the same barriers, so they run their softmaxes at about the same time.
That is the idle half of the tensor pipe. FA3 runs three warpgroups per CTA: one only
loads, and the other two take turns (ping-pong), so one's softmax overlaps the other's
matmuls. That is the next change to try here, with Triton's warp specialization or by
hand.

**Knockouts** (`bench/ablation.py`): copies of the kernel with one kind of work
removed; the copy with nothing removed is bit-identical to the kernel. Five interleaved
rounds on the same box:

| removed (time saved) | N=4096 | N=8192 | N=16384 |
|---|---|---|---|
| `exp2` only (MUFU) | 5.3% | 5.7% | 9.8% |
| the whole softmax (scale, mask, max, exp, sum, rescale) | 29.4% | 32.5% | 26.9% |
| reading new K/V tiles (every iteration rereads the first one) | 3.8% | -2.9% | 6.3% |
| softmax and K/V reads: matmuls only | 34.2% | 34.8% | 33.0% |

**What Nsight Compute corrected.** Before it could run (2026-10-03, RunPod), the
breakdown came from the clock, the SASS and these knockouts. Two of its readings were
wrong:
- The knockout that rereads the first K/V tile was read as the cost of HBM traffic.
  DRAM runs at 1.8% of its peak; K and V already came from L2. What that knockout
  removes is the wait for L2 (the "long scoreboard" stalls).
- From timings alone, the matmuls-only copy looked like it reached 74% of the peak.
  That assumed the full kernel's clock. At ncu's locked clock it keeps the tensor pipe
  84% busy, so under the power cap it probably runs at a lower clock than the full kernel.

The ranking held: the softmax costs more than its exponentials, the tensor pipe waits
while it runs, and memory matters little. The two fixes the SASS suggested went in
before this run and are in the table above.

## GQA head packing

With H_q = 32 and H_kv = 8, four query heads share every K/V tile. Unpacked, each
query head's program loads the K/V tiles on its own, and a head with N_q queries fills
N_q / 128 of its tile. Packed, a group's four heads share one program's 128 rows
([`fa2/kernel.py`](fa2/kernel.py) docstring). `attention()` packs when that fills the
tiles at least 10% better, FA3's rule (`should_pack_gqa`).

Few queries per head against a long K/V, as in decode or speculative decoding:
batch 16, N_kv = 4096, non-causal, FP16, median microseconds
(`bench/results/2026-10-04/pack_short.json`):

| N_q | unpacked | packed | speedup | FA3 | FA2 | K/V read rate, packed |
|---|---|---|---|---|---|---|
| 1 | 296.2 | 106.5 | 2.78x | 111.9 | 118.7 | 2.52 TB/s |
| 16 | 296.2 | 105.5 | 2.81x | 112.7 | 359.8 | 2.54 TB/s |
| 32 | 294.1 | 107.6 | 2.73x | 112.8 | 357.2 | 2.50 TB/s |
| 64 | 294.2 | 151.2 | 1.95x | 135.7 | 359.4 | 1.78 TB/s |
| 128 | 307.9 | not chosen | | 248.4 | 386.3 | |
| 256 | 642.5 | not chosen | | 473.4 | 758.3 | |

- Up to N_q = 32 the packed kernel reads K and V at 2.5 TB/s, 75% of the HBM peak, and
  is slightly faster than FA3. FA2 is fast only at N_q = 1, where it moves the query
  group into the sequence dimension, and 3.3x to 3.4x slower at 16 and 32.
- Unpacked, the 512 programs (16 batches x 32 heads) need four waves on 132 SMs.
  Packed, 128 programs run in one wave, limited by HBM bandwidth.
- At N_q = 64 the packed kernel needs two waves and FA3 is 1.11x faster. From N_q = 128
  a head fills its own tile and the rule leaves packing off.
- Long causal prefill: in interleaved runs on two boxes, forcing packing on changed the
  time by -2.5%, +0.2%, -1.4% (RunPod) and -2.7%, -4.7%, -0.9% (Verda) at N = 4096,
  8192 and 16384. A small gain the rule leaves on the table; why it gains is open.

## Limitations

- Forward only. There is no backward and no dropout.
- Only the D=128 config was tuned (one sweep, two lengths); the rest are starting
  points, and there is no autotuning.
- Causal assumes `N_q == N_kv`, i.e. prefill. Decode-style bottom-right alignment,
  variable-length batches and paged KV are out of scope.
- No split-KV. With few queries and a small batch, the grid can be smaller than the
  GPU. FA3 is still 1.11x faster at N_q = 64 in the packing table; whether split-KV
  or its tile shapes explain that was not measured.
- Plain pointer arithmetic, with no TMA / tensor descriptors. That is a
  deliberate choice for portability and interpreter support, and it is also the
  first thing Hopper wants changed (below).

## What I'd change for Hopper (FA3) and Blackwell (FA4)

**1. The algorithm stays; the schedule has to be redone for every GPU generation.**
FA2 reaches 50–73% of peak on A100, but only about 35% on H100 unmodified,
because its schedule was designed for Ampere. FA3 changed nothing in the maths.
It rescheduled for Hopper's new units and got back to about 75% of peak (up to
740 TFLOP/s FP16) [3]. On the box above, under its power cap: FA2 349 TFLOP/s (35%),
FA3 up to 661 (67%), causal, D=128. FA4 repeats the pattern on Blackwell [4]. The
online-softmax recurrence in this repo is the part that carries over. The loop
structure is the part that does not.

**2. On Hopper [3]:**
- K/V loads through **TMA** instead of pointer arithmetic.
- **Warp specialisation:** producer warps only move data and consumer warps only
  run WGMMA matmuls.
- **Ping-pong scheduling**, so one warpgroup's softmax overlaps another's matmul.
  The reason is the gap between H100's 989 TFLOP/s dense FP16 matmul peak and
  its roughly 3.9 TFLOP/s for special functions like `exp`.

[Where the time goes](#where-the-time-goes) ranks these for this kernel. The tensor
pipe is idle half the time, while the warpgroups run their softmax, and DRAM runs at
1.8% of its peak. So TMA on its own would buy little here, and overlapping softmax and
matmul (warp specialisation, ping-pong) is the lever.

**3. On Blackwell [4]:** the FA4 paper calls the problem *asymmetric hardware
scaling*: tensor-core throughput doubles, but shared-memory bandwidth and the
exponential units scale more slowly or not at all. Matmul gets cheaper while
the non-matmul part of this kernel does not, so a schedule tuned on H100 ends
up limited by `exp` and SMEM. That is why H100 numbers do not transfer to B200.
FA4's answers map directly onto lines in `_attn_fwd_inner`:
- **`tl.math.exp2` → software exponential.** FA4 emulates part of the
  exponentials with a polynomial (Cody-Waite range reduction plus Horner
  evaluation) on the FMA units, which lets them run alongside the saturated
  hardware `exp` unit.
- **`acc * alpha` every tile → conditional rescaling.** FA4 only rescales when
  the running max has grown enough to matter, which skips most rescale
  operations. Here the rescale is unconditional on every tile.
- **Accumulator in registers → tensor memory.** Blackwell's `tcgen05` MMAs are
  asynchronous and accumulate into TMEM (256 KB per SM on B200), which frees
  registers and SMEM bandwidth. Larger tiles and the 2-CTA MMA mode build on that.

For scale, FA4 reports 1613 TFLOP/s BF16 on B200 (71% utilisation), which is
up to 1.3× cuDNN 9.13 and **2.7× Triton** [4]. That gap is the honest ceiling
for a portable Triton kernel like this one on Blackwell.

**Boundary:** everything in points 2–3 is a known next step, not something
implemented here. This kernel targets Ampere/Hopper-class GPUs; it has run on H100
SXM three times (2026-10-02 to 10-04, see [Results](#results) and [WORKLOG](WORKLOG.md)).

## References

1. M. Milakov, N. Gimelshein. *Online normalizer calculation for softmax.* 2018. [arXiv:1805.02867](https://arxiv.org/abs/1805.02867)
2. T. Dao. *FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning.* 2023. [arXiv:2307.08691](https://arxiv.org/abs/2307.08691) (FlashAttention-1: [arXiv:2205.14135](https://arxiv.org/abs/2205.14135))
3. J. Shah et al. *FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-precision.* 2024. [arXiv:2407.08608](https://arxiv.org/abs/2407.08608)
4. T. Zadouri et al. *FlashAttention-4: Algorithm and Kernel Pipelining Co-Design for Asymmetric Hardware Scaling.* 2026. [arXiv:2603.05451](https://arxiv.org/abs/2603.05451)
5. J. Ainslie et al. *GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints.* 2023. [arXiv:2305.13245](https://arxiv.org/abs/2305.13245)
6. Triton tutorial 06, *Fused Attention.* [triton-lang.org](https://triton-lang.org/main/getting-started/tutorials/06-fused-attention.html)

## License

MIT, see [LICENSE](LICENSE).

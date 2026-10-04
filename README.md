# triton-fa2-forward

[![tests (Triton interpreter)](https://github.com/roy6n23/triton-fa2-forward/actions/workflows/tests.yml/badge.svg)](https://github.com/roy6n23/triton-fa2-forward/actions/workflows/tests.yml)

A FlashAttention-2 style **forward** attention kernel in Triton: causal or full,
FP16/BF16, with grouped-query attention (GQA). About 200 lines of kernel code.
It comes with a pure-PyTorch mirror of the same algorithm, a correctness suite
graded against PyTorch's own error, a benchmark against FlashAttention-2 and -3,
cuDNN, FlashInfer and PyTorch SDPA, and a breakdown of where the kernel's time goes.

> **Status (2026-10-03):** second H100 SXM run (RunPod), 115 tests pass on the GPU.
> The default D=128 config is now 8 warps × 4 stages, GQA head packing is in, the
> FA2/FA3 baselines are measured, and the time per K/V tile is broken down below.
> Nsight Compute could not run in the rented container (no access to the GPU's
> performance counters); its script is ready ([WORKLOG](WORKLOG.md), 2026-10-03).

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
| **Causal block skipping in two stages** | Tiles fully left of the diagonal need no causal mask (stage 1). Only the `BLOCK_M`-wide diagonal band pays for it (stage 2). Tiles right of it are never loaded or multiplied. For long sequences that removes about half the work. (Stage 1 still applies the end-of-sequence mask; see [Where the time goes](#where-the-time-goes).) |
| **`exp2` with `log2 e` folded into the scale** | `exp(x) = exp2(x·log2 e)`. One multiply at setup turns every exponential into a single hardware `exp2`. |
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
`pytest` reported 77 passed on 2026-10-02 and 115 passed on 2026-10-03 (the GQA
packing tests added), BF16 included, which the interpreter has to skip. In the
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

The baselines are FlashAttention-2 and FlashAttention-3, cuDNN (via SDPA), FlashInfer
and SDPA's flash backend. A comparison against torch SDPA alone would be a 2023
comparison. FA2 and FA3 come from vLLM's build (`vllm.vllm_flash_attn`,
`flash_attn_varlen_func` with `fa_version=2` and `3`): the Dao-AILab kernels,
already compiled for this torch and CUDA, where upstream `flash-attn` would have to
be built from source. `bench.py` also tries the upstream packages and skips them
when they are not installed.

## Results

**2026-10-03, H100 80GB HBM3 SXM** (RunPod), driver 580.126.09, CUDA 13.0, torch
2.13.0+cu130, Triton 3.7.1, FlashInfer 0.6.18.post1, vLLM 0.30.0. Default shape:
Llama-3-8B attention (H_q=32, H_kv=8, D=128), batch 1, causal, FP16, 8 warps x 4
stages. Median TFLOP/s; the IQRs and the full environment are in
`bench/results/2026-10-03/bench_fp16.json`:

| N | triton-fa2 | FA2 | FA3 | sdpa-flash | sdpa-cudnn | flashinfer |
|---|---|---|---|---|---|---|
| 512 | 96.8 | 79.8 | 99.1 | 80.7 | 112.3 | 126.2 |
| 1024 | 161.5 | 143.9 | 252.4 | 137.8 | 229.8 | 236.0 |
| 2048 | 260.0 | 221.7 | 430.6 | 205.4 | 382.8 | 396.2 |
| 4096 | 342.4 | 289.7 | 579.8 | 272.4 | 516.0 | 539.1 |
| 8192 | 398.8 | 331.3 | 668.0 | 319.6 | 600.6 | 589.6 |
| 16384 | 427.6 | 356.7 | 640.4 | 346.1 | 609.3 | 549.3 |

- **vs FA2:** 12% to 21% faster at every N. FA2's schedule was designed for Ampere.
- **vs FA3, cuDNN, FlashInfer:** the Hopper-specific kernels are 1.3x to 1.7x faster
  from N=2048 up. The next section says where that gap comes from.
- **BF16** is within 1.5% of FP16 at every N (433.4 TFLOP/s at N=16384,
  `bench_bf16.json`).
- **vs the 2026-10-02 run** (8 warps x 2 stages, same box type and stack): +10% to
  +13% at every N, from the 8 x 4 config the sweep picked
  (`bench/results/NVIDIA-H100-80GB-HBM3_bf16_2026-10-02.json` for that run's BF16).

**Occupancy (D=128, 8 warps x 4 stages):** 183 registers per thread, no spills,
160 KiB of shared memory. Both cap it at one CTA (two warpgroups) per SM.

## Where the time goes

At N=16384 the kernel runs at about 430 TFLOP/s, 43% of the H100's 989.4. Nsight
Compute could not run in the rented container, so the breakdown below comes from
three sources that need no hardware counters.

**1. The clock.** Under this kernel the GPU sits at its 700 W power cap
(`nvidia-smi` throttle reason "SW power cap" in 74 of 75 samples over 15 s) and the
SM clock averages 1.68 GHz, not the 1.83 GHz behind the 989.4 figure. The ceiling
on this box is about 911 TFLOP/s.

**2. The SASS of one K/V tile** (`bench/sass_stats.py`; the GPU's own build and an
ahead-of-time sm_90 build on a Mac give the same 447 instructions). Per iteration of
the stage-1 loop, each of the two warpgroups:

```
DEPBAR.LE / BAR.SYNC               wait for this tile's K/V copies; CTA-wide barrier
HGMMA.64x64x16.F32 x8              S = Q K^T on the tensor cores (async)
WARPGROUP.DEPBAR.LE gsb0, 0x0      wait for every matmul in flight
313 instructions                   softmax: 96 FMUL, 63 FADD, 36 FMNMX, 34 MUFU.EX2, 32 FSEL, ...
HGMMA.64x128x16.F32 x4             O += P V (async, waited at the next iteration)
BAR.SYNC / LDGSTS x8               barrier, then cp.async copies of a later K/V tile
```

The tensor cores get nothing new while a warpgroup runs its softmax, and the two
warpgroups pass the same barriers, so they reach the softmax at about the same
time. Per SM and iteration (both warpgroups), each resource alone at its peak would need: tensor cores
1024 clocks, instruction issue 894, MUFU 544, FP32 pipe 332. Measured: about 2260
(at 1.68 GHz, iterations spread evenly over the SMs).

**3. Knockouts** (`bench/ablation.py`): copies of the kernel with one kind of work
removed. The copy with nothing removed is bit-identical to the kernel (tested). The
time a removal saves is what that work costs where it sits in this schedule; the
parts overlap, so the savings do not add exactly. Five interleaved rounds, FP16,
causal, default shape:

| removed (time saved) | N=4096 | N=8192 | N=16384 |
|---|---|---|---|
| `exp2` only (MUFU) | 7.6% | 8.0% | 10.9% |
| the whole softmax (scale, mask, max, exp, sum, rescale) | 35.1% | 36.3% | 32.5% |
| HBM traffic (every iteration reads the first K/V tile, served from L2) | 6.2% | -1.3% | 5.8% |
| softmax and HBM traffic: matmuls and loads from L2 only | 38.7% | 38.6% | 39.2% |

At N=8192 the L2-only copy came out 1.3% slower, inside that run's noise. At N=16384
the kernel and its identical copy differed by 3.8%, so the shares there are good to
a few percent; N=4096 has IQRs of 0.25%.

Put together for N=16384 (5.38 ms, the mean of the kernel and its copy), adding the
work back in this order:

| | ms | share |
|---|---|---|
| tensor-core work at peak, at 1.68 GHz | 2.41 | 45% |
| matmuls below peak even with nothing else to do | 0.86 | 16% |
| HBM traffic not hidden | 0.36 | 7% |
| softmax not hidden | 1.75 | 33% |

So the softmax is the largest share, and the exponentials are only a third of it.
The rest is FP32 and select instructions, shuffles and conversions that the
warpgroups issue while the tensor cores wait. Even with no softmax and K/V served
from L2, the matmuls reach only 74% of the clock-adjusted peak: each iteration waits
for Q K^T before it can compute P, and both warpgroups wait at the same barriers.
HBM traffic matters little at this size.

**Two fixes the SASS points to, measured but not in the kernel yet**
(`bench/ablation.py`, variants `even_n` and `ffma_scale`):
- Stage 1 still applies the end-of-sequence mask to every tile (the 32 `FSEL`).
  When N_kv is a multiple of BLOCK_N, it can be skipped.
- Fold the softmax scale into the exponent as one FFMA, `qk * scale - m`, as
  Triton's tutorial 06 does, instead of scaling every score first.

Together they cut the stage-1 loop from 447 to 371 instructions (ahead-of-time
build) and the time by 10.0%, 8.6% and 9.5% at N = 4096, 8192 and 16384 (452
TFLOP/s at 16384), and they pass the same accuracy bar (`tests/test_ablation.py`). The structural fix is FA3's: overlap one warpgroup's softmax with the
other's matmuls (ping-pong), and tile j's softmax with tile j+1's Q K^T.

**Nsight Compute.** `bench/profile_ncu.sh` runs it with `--set full` on the kernel,
two knockouts and FA3. It needs a machine where the profiler may read the GPU's
counters, such as a VM or bare metal with root; in the RunPod container it stops at
`ERR_NVGPUCTRPERM`. It would measure two things the reading above infers: how busy
the tensor pipe is, and why warps stall, instruction by instruction.

## GQA head packing

With H_q = 32 and H_kv = 8, four query heads share every K/V tile. Unpacked, each
query head's program loads the K/V tiles on its own, and a head with N_q queries fills
N_q / 128 of its tile. Packed, a group's four heads share one program's 128 rows
([`fa2/kernel.py`](fa2/kernel.py) docstring). `attention()` packs when that fills the
tiles at least 10% better, FA3's rule (`should_pack_gqa`).

Few queries per head against a long K/V, as in decode or speculative decoding:
batch 16, N_kv = 4096, non-causal, FP16, median microseconds
(`bench/results/2026-10-03/pack_short.json`):

| N_q | unpacked | packed | speedup | FA3 | FA2 | K/V read rate, packed |
|---|---|---|---|---|---|---|
| 1 | 314.1 | 106.4 | 2.95x | 110.1 | 118.3 | 2.52 TB/s |
| 16 | 314.6 | 106.6 | 2.95x | 111.5 | 361.6 | 2.52 TB/s |
| 32 | 315.3 | 107.9 | 2.92x | 112.3 | 362.3 | 2.49 TB/s |
| 64 | 318.4 | 170.1 | 1.87x | 134.3 | 358.8 | 1.58 TB/s |
| 128 | 328.1 | not chosen | | 242.4 | 372.5 | |
| 256 | 667.2 | not chosen | | 467.6 | 737.8 | |

- Up to N_q = 32 the packed kernel reads K and V at 2.5 TB/s, 75% of the HBM peak,
  and is as fast as FA3. FA2 is fast only at N_q = 1, where it moves the query group
  into the sequence dimension, and 3.4x slower at 16 and 32.
- Unpacked, the 512 programs (16 batches x 32 heads) need four waves on 132 SMs.
  Packed, 128 programs run in one wave, limited by HBM bandwidth.
- At N_q = 64 the packed kernel needs two waves and FA3 is 1.27x faster. From
  N_q = 128 a head fills its own tile, and forcing packing on changes nothing (329.3
  and 666.1 µs).
- Long causal prefill: in the interleaved runs, forcing packing on changed the time
  by -2.5%, +0.2% and -1.4% at N = 4096, 8192 and 16384. At most a small gain, so
  leaving it off there, as the rule does, costs little.

## Limitations

- Forward only. There is no backward and no dropout.
- Only the D=128 config was tuned (one sweep, two lengths); the rest are starting
  points, and there is no autotuning.
- Causal assumes `N_q == N_kv`, i.e. prefill. Decode-style bottom-right alignment,
  variable-length batches and paged KV are out of scope.
- No split-KV. With few queries and a small batch, the grid can be smaller than the
  GPU. FA3 is still 1.27x faster at N_q = 64 in the packing table; whether split-KV
  or its tile shapes explain that was not measured.
- Plain pointer arithmetic, with no TMA / tensor descriptors. That is a
  deliberate choice for portability and interpreter support, and it is also the
  first thing Hopper wants changed (below).
- Nsight Compute has not been run on it yet (see above).

## What I'd change for Hopper (FA3) and Blackwell (FA4)

**1. The algorithm stays; the schedule has to be redone for every GPU generation.**
FA2 reaches 50–73% of peak on A100, but only about 35% on H100 unmodified,
because its schedule was designed for Ampere. FA3 changed nothing in the maths.
It rescheduled for Hopper's new units and got back to about 75% of peak (up to
740 TFLOP/s FP16) [3]. On the box above, under its power cap: FA2 357 TFLOP/s (36%),
FA3 up to 668 (68%), causal, D=128. FA4 repeats the pattern on Blackwell [4]. The
online-softmax recurrence in this repo is the part that carries over. The loop
structure is the part that does not.

**2. On Hopper [3]:**
- K/V loads through **TMA** instead of pointer arithmetic.
- **Warp specialisation:** producer warps only move data and consumer warps only
  run WGMMA matmuls.
- **Ping-pong scheduling**, so one warpgroup's softmax overlaps another's matmul.
  The reason is the gap between H100's 989 TFLOP/s dense FP16 matmul peak and
  its roughly 3.9 TFLOP/s for special functions like `exp`.

The measurements in [Where the time goes](#where-the-time-goes) rank these for this
kernel: the softmax, which runs while the tensor cores wait, is a third of the time
at N=16384, and the exponentials are a third of that. HBM traffic is 7%, so TMA on
its own would buy little here; overlapping softmax and matmul is the bigger lever.

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
implemented here. This kernel targets Ampere/Hopper-class GPUs, and its first
real-hardware validation was an H100 run on 2026-10-02 (see
[Results](#results) and [WORKLOG](WORKLOG.md)).

## References

1. M. Milakov, N. Gimelshein. *Online normalizer calculation for softmax.* 2018. [arXiv:1805.02867](https://arxiv.org/abs/1805.02867)
2. T. Dao. *FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning.* 2023. [arXiv:2307.08691](https://arxiv.org/abs/2307.08691) (FlashAttention-1: [arXiv:2205.14135](https://arxiv.org/abs/2205.14135))
3. J. Shah et al. *FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-precision.* 2024. [arXiv:2407.08608](https://arxiv.org/abs/2407.08608)
4. T. Zadouri et al. *FlashAttention-4: Algorithm and Kernel Pipelining Co-Design for Asymmetric Hardware Scaling.* 2026. [arXiv:2603.05451](https://arxiv.org/abs/2603.05451)
5. J. Ainslie et al. *GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints.* 2023. [arXiv:2305.13245](https://arxiv.org/abs/2305.13245)
6. Triton tutorial 06, *Fused Attention.* [triton-lang.org](https://triton-lang.org/main/getting-started/tutorials/06-fused-attention.html)

## License

MIT, see [LICENSE](LICENSE).

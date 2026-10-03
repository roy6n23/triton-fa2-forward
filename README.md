# triton-fa2-forward

[![tests (Triton interpreter)](https://github.com/roy6n23/triton-fa2-forward/actions/workflows/tests.yml/badge.svg)](https://github.com/roy6n23/triton-fa2-forward/actions/workflows/tests.yml)

A FlashAttention-2 style **forward** attention kernel in Triton: causal or full,
FP16/BF16, with grouped-query attention (GQA). About 200 lines of kernel code.
It comes with a pure-PyTorch mirror of the same algorithm, a correctness suite
graded against PyTorch's own error, and a benchmark against FlashInfer, cuDNN
and flash-attn.

> **Status (2026-10-02):** first GPU run done on an H100 SXM (RunPod). The full
> suite passes on real hardware, BF16 included (77 passed). Results and the stack
> are below; the CPU interpreter suite still runs in CI on every push
> (see [WORKLOG](WORKLOG.md), 2026-10-02).

## Layout

```
fa2/kernel.py        Triton kernel + attention() wrapper
fa2/reference.py     naive_attention (ground truth) and blocked_attention
                     (the kernel's algorithm, line for line, in plain torch)
tests/               correctness suite
bench/bench.py       benchmark: warmup/IQR protocol, baselines, env pinning
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
| **Causal block skipping in two stages** | Tiles fully left of the diagonal need no mask (stage 1). Only the `BLOCK_M`-wide diagonal band pays for the elementwise mask (stage 2). Tiles right of it are never loaded or multiplied. For long sequences that removes about half the work. |
| **`exp2` with `log2 e` folded into the scale** | `exp(x) = exp2(x·log2 e)`. One multiply at setup turns every exponential into a single hardware `exp2`. |
| **P cast back to FP16/BF16 before `P @ V`** | Keeps the second matmul on tensor cores. The statistics `m`, `l` and the accumulator stay in FP32. |
| **GQA by index mapping** (`kv_head = q_head // group`) | No K/V replication in memory, and it is a single integer divide per program. |
| **`-inf` masking without NaN guards** | The first tile any row processes always contains a visible column (column 0 in stage 1, or the diagonal in stage 2). So `m_new` is finite from the first update on, and `exp(-inf − -inf)` can never happen. The reasoning is written next to the code. |

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

The GPU run then applied the same 2× bar on real tensor cores. On the H100
(2026-10-02), `pytest` reported 77 passed, including both BF16 cases, which the
interpreter has to skip. In that run's num_warps x num_stages sweep (D=128,
FP16, causal, N=4096 and 16384), the max abs error vs SDPA was 4.88e-4 in every
config ([WORKLOG](WORKLOG.md), 2026-10-02). That number is a direct difference
from SDPA's output, not the fp64-referenced error in the table above.

## Benchmark protocol

`bench/bench.py`:

- **Warmup 5 + measured 20**, CUDA events per iteration.
- **L2 flushed** (256 MB write) before every measured iteration.
- Report the **median and the IQR (p25–p75)**, never the mean or the best run.
- **Correctness gate:** every provider's output is compared with SDPA before it is timed.
- **Full stack pinned** into the results JSON: GPU SKU and SM count, driver,
  CUDA, cuDNN, PyTorch, Triton, flash-attn and FlashInfer versions.

The baselines are FlashInfer, cuDNN (via SDPA), flash-attn FA2, FA3 on Hopper,
and SDPA's flash backend. A comparison against torch SDPA alone would be a 2023
comparison.

**Results (2026-10-02, H100 80GB HBM3 SXM, driver 580.126.09, CUDA 13.0, torch
2.13.0+cu130, Triton 3.7.1, FlashInfer 0.6.18.post1).** Default shape: Llama-3-8B
attention (H_q=32, H_kv=8, D=128), batch 1, causal. Median TFLOP/s (IQR in ms);
flash-attn and FA3 were not installed on the box, so they are absent. These runs
used the earlier 8 warps x 2 stages config:

| N | triton-fa2 | sdpa-flash | sdpa-cudnn | flashinfer |
|---|---|---|---|---|
| 512 | 87.0 (0.000) | 82.7 (0.000) | 114.9 (0.000) | 24.4 (0.067) |
| 1024 | 146.5 (0.001) | 140.1 (0.001) | 232.2 (0.001) | 237.6 (0.000) |
| 2048 | 230.9 (0.001) | 204.2 (0.009) | 383.7 (0.002) | 397.7 (0.000) |
| 4096 | 303.3 (0.001) | 271.8 (0.012) | 517.8 (0.002) | 541.8 (0.001) |
| 8192 | 360.7 (0.002) | 320.9 (0.030) | 604.1 (0.004) | 632.7 (0.001) |
| 16384 | 389.5 (0.076) | 342.1 (0.080) | 595.1 (0.101) | 590.5 (0.041) |

FP16 above; BF16 is within 1-2% at every N (390.1 TFLOP/s at N=16384,
`bench/results/NVIDIA-H100-80GB-HBM3_bf16_2026-10-02.json`). The kernel is ahead of
SDPA's FA2 backend at every length (+5% to +14%) and 35-45% behind cuDNN and
FlashInfer from N=4096 up, which are Hopper-specific (WGMMA/TMA) kernels; the FA3 section below is about
exactly that gap.

**Occupancy (D=128, 8 warps x 2 stages):** 184 registers per thread, no spills,
96 KB shared. Registers limit it to 1 block per SM (65536 / (184 x 32 x 8) = 1.4).
A num_warps x num_stages sweep at N=4096 and 16384 found 8 x 4 fastest:
349.6 and 415.9 TFLOP/s (+14% and +13%), with the same max error. That is now
the D=128 default; the full N range has not been re-benchmarked with it yet.

## Limitations

- Forward only. There is no backward and no dropout.
- Only the D=128 config was tuned (one sweep, two lengths); the rest are starting
  points, and there is no autotuning.
- Causal assumes `N_q == N_kv`, i.e. prefill. Decode-style bottom-right alignment,
  variable-length batches and paged KV are out of scope.
- GQA maps heads but does not *pack* them. Each query head's program reloads
  the shared K/V tile, and any reuse depends on L2. The obvious next step is to
  put a group's query heads into one program's M dimension so each K/V tile is
  read once per group.
- Plain pointer arithmetic, with no TMA / tensor descriptors. That is a
  deliberate choice for portability and interpreter support, and it is also the
  first thing Hopper wants changed (below).

## What I'd change for Hopper (FA3) and Blackwell (FA4)

**1. The algorithm stays; the schedule has to be redone for every GPU generation.**
FA2 reaches 50–73% of peak on A100, but only about 35% on H100 unmodified,
because its schedule was designed for Ampere. FA3 changed nothing in the maths.
It rescheduled for Hopper's new units and got back to about 75% of peak (up to
740 TFLOP/s FP16) [3]. FA4 repeats the pattern on Blackwell [4]. The
online-softmax recurrence in this repo is the part that carries over. The loop
structure is the part that does not.

**2. On Hopper [3]:**
- K/V loads through **TMA** instead of pointer arithmetic.
- **Warp specialisation:** producer warps only move data and consumer warps only
  run WGMMA matmuls.
- **Ping-pong scheduling**, so one warpgroup's softmax overlaps another's matmul.
  The reason is the gap between H100's 989 TFLOP/s dense FP16 matmul peak and
  its roughly 3.9 TFLOP/s for special functions like `exp`.

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
[Results](#benchmark-protocol) and [WORKLOG](WORKLOG.md)).

## References

1. M. Milakov, N. Gimelshein. *Online normalizer calculation for softmax.* 2018. [arXiv:1805.02867](https://arxiv.org/abs/1805.02867)
2. T. Dao. *FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning.* 2023. [arXiv:2307.08691](https://arxiv.org/abs/2307.08691) (FlashAttention-1: [arXiv:2205.14135](https://arxiv.org/abs/2205.14135))
3. J. Shah et al. *FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-precision.* 2024. [arXiv:2407.08608](https://arxiv.org/abs/2407.08608)
4. T. Zadouri et al. *FlashAttention-4: Algorithm and Kernel Pipelining Co-Design for Asymmetric Hardware Scaling.* 2026. [arXiv:2603.05451](https://arxiv.org/abs/2603.05451)
5. J. Ainslie et al. *GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints.* 2023. [arXiv:2305.13245](https://arxiv.org/abs/2305.13245)
6. Triton tutorial 06, *Fused Attention.* [triton-lang.org](https://triton-lang.org/main/getting-started/tutorials/06-fused-attention.html)

## License

MIT, see [LICENSE](LICENSE).

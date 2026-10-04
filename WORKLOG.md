# WORKLOG

Dated, append-only. Record what broke, how it was found, and what it taught.
Keep entries factual, including the ones that make me look slow.

## 2026-09-22: first full draft, verified without a GPU

**Context.** The design inputs were the online-softmax recurrence, FA1 §3
(tiling and recomputation), FA2 (loop order, split-Q, causal block skipping)
and the forward half of Triton tutorial 06. The goal for this pass was
"correct first, fast later": a naive FA2-order forward kernel, not FA2/FA3-level
scheduling.

**Problem: no GPU on the dev machine (macOS, arm64).** Triton ships no macOS
wheels. Two fallbacks:
1. `fa2/reference.py:blocked_attention` is the kernel's algorithm in plain torch,
   testable on the Mac directly (25 tests, fp64 inputs, fp32 state, agrees with
   naive attention to 1e-5).
2. The real kernel runs under `TRITON_INTERPRET=1` in a `python:3.12-slim`
   arm64 container (Triton 3.8.0, torch 2.14.0+cpu). Triton's interpreter runs
   the kernel with numpy, so no GPU is needed.

**Bug hunt: every BF16 case was wrong by about 8e8, FP16 was fine.**
- The first thing I noticed was that the error was the same constant (7.969e8)
  across every sequence length, causal or not, and every row was bad. That does
  not look like an off-by-one or masking bug, which would hit specific rows. It
  looks like bits being misread.
- To bisect, I wrote three minimal kernels: copy, `load → fp32 → store`, and a
  single 16×16 `tl.dot`. Copy and round-trip were exact in BF16. **The BF16
  `tl.dot` was off by 2.6e10**, while the FP16 `tl.dot` was off by 3.8e-3.
- Conclusion: this is an interpreter limitation (BF16 matmul emulation), not a
  kernel bug. BF16 tests are skipped under the interpreter, with the reason
  given, and BF16 is still unverified until the GPU run.
- Lesson: when every output is wrong by the same absurd constant, suspect
  representation before logic, and bisect with the smallest possible kernel
  rather than staring at the big one.

**Are the tests actually able to fail?** I planted two bugs deliberately:
- Dropped the `alpha` rescale of `acc`: 26 kernel tests fail. The 12 that still
  pass are all N=1 (a single K/V tile, so there is nothing to rescale), which is
  expected.
- Causal mask `>=` changed to `>`: all 19 causal tests fail. The 19 that pass
  are exactly the non-causal ones, which never touch the mask.

**Numbers so far (interpreter, D=64).** Our max-abs error is 0.6–1.1× the error
of PyTorch's FP16 SDPA, well inside the 2× bar. This is optimistic: the
interpreter's matmul is more exact than tensor cores.

**Open for the first GPU run:**
- [x] full `pytest` sweep on real hardware (D=128, N up to 4096, BF16) (2026-10-02)
- [x] `bench/bench.py` default shape; record the env JSON (2026-10-02)
- [x] check the register/occupancy of `_default_config` for D=128 (8 warps, 2 stages is a guess) (2026-10-02)

## 2026-10-02: first GPU run (H100 SXM)

**Box.** RunPod 2x H100 80GB HBM3 SXM, driver 580.126.09, the vLLM 0.30.0 image
(torch 2.13.0+cu130, Triton 3.7.1, FlashInfer 0.6.18.post1). The kernel ran on GPU1
while another job used GPU0; timing runs waited until GPU0 was idle.

**Correctness.** `pytest -v`: 77 passed, including both BF16 cases, which the
interpreter had to skip (the BF16 `tl.dot` emulation bug, 2026-09-22). So the
kernel was right all along and only the interpreter was wrong.

**Problem: `pip install -e ".[test]"` failed.** setuptools' flat-layout discovery
found two top-level packages, `fa2` and `bench`, and refused the build. CI never saw
it because it runs pytest from the repo root (`pythonpath = ["."]`). Fixed with
`[tool.setuptools] packages = ["fa2"]`; an editable install in a clean venv now
imports `fa2` from outside the repo.

**Benchmark** (default shape, fp16, 8 warps x 2 stages): 87 TFLOP/s at N=512 up to
389.5 at N=16384. Ahead of SDPA's flash backend at every N (+5% to +14%); 35-45% behind
cuDNN and FlashInfer from N=4096 up (FlashInfer is slower than us at N=512). BF16 within 1-2%. flash-attn was not installed (it
would have built from source), so that baseline is missing.

**Occupancy.** 184 registers/thread, 0 spills, 96 KB shared, 8 warps: registers
cap it at 1 block per SM (65536 / (184 x 32 x 8) = 1.4; shared would allow 2).

**Sweep** (time-boxed; N=4096 and 16384, fp16, causal):

| warps x stages | N=4096 | N=16384 |
|---|---|---|
| 4 x 1 | 278.6 | 364.6 |
| 4 x 2 | 315.4 | 411.1 |
| 4 x 3 | 246.6 | 314.8 |
| 4 x 4 | 238.9 | 304.6 |
| 8 x 1 | 250.0 | 310.1 |
| 8 x 2 (old default) | 306.9 | 368.4 |
| 8 x 3 | 337.4 | 411.4 |
| 8 x 4 | 349.6 | 415.9 |

Max abs error vs SDPA was 4.88e-4 in every config. 8 x 4 is now the D=128 default.
My guess, not checked with a profiler: with one block per SM, deeper pipelining
hides more K/V load latency. At 4 warps the extra stages hurt instead, so the win
is not simply "more stages".

**Also fixed:** `bench.py` wrote `<gpu>_<date>.json` for both dtypes, so the BF16
run overwrote the FP16 JSON. The name now includes the dtype; the FP16 numbers
survive only in the run log and the README table.

**Open:**
- [x] re-run the full N range with the 8 x 4 default (2026-10-03)
- [x] flash-attn / FA3 baselines (needs a prebuilt wheel for this torch) (2026-10-03, vLLM's build)

## 2026-10-03: second GPU run: FA2/FA3 baselines, GQA head packing, where the time goes

**Box.** RunPod 1x H100 80GB HBM3 SXM (US-NE-1, Secure Cloud, $3.49/h), the same
vLLM 0.30.0 image, driver 580.126.09. About 30 minutes, about $1.7. Everything ran
from `bench/gpu_session.sh`; the outputs are in `bench/results/2026-10-03/`.

**Prepared before the rental, without a GPU.**
- GQA head packing, test first under the interpreter: 29 new tests failed, then
  passed. Two planted bugs were caught: every row reading the group's first head (27
  failures), and stage 1 not rounded down to a whole K/V tile (4 failures).
- `bench/sass_stats.py`. Triton ships ptxas and nvdisasm, so the kernel compiles for
  sm_90 inside a CPU container on the Mac and its SASS can be read with no GPU. The
  schedule in the README was known before the box was up; the GPU's own build later
  had the same 447 instructions in the stage-1 loop.
- `bench/ablation.py`: the kernel copied with switches that remove one kind of work.
  With every switch off it is bit-identical to the kernel (interpreter, then GPU).
- FA2 and FA3 baselines through vLLM's build, which the image already has.

**Problem 1: Nsight Compute cannot read counters in a RunPod container.** `ncu`
2026.3.1 (NVIDIA devtools apt repo) stops at `ERR_NVGPUCTRPERM`. The host driver has
`RmProfilingAdminOnly: 1` (`/proc/driver/nvidia/params`) and the container has
neither `CAP_SYS_ADMIN` nor `CAP_PERFMON` (`capsh --print`), so nothing inside the
pod can change it. Proton's CUPTI PC sampling, the other route to stall reasons,
aborted with `cuptiPCSamplingGetStallReasons` error 1 (`pcsampling.log`). Proton's
in-kernel instrumentation is off by default for Triton-language kernels, because
pipelining moves the timestamps (its own docstring says so), so I did not use it.
The breakdown in the README comes from the clock, the SASS and the knockout timings
instead. `bench/profile_ncu.sh` is ready for a machine that allows counter access (a
VM or bare metal with root), and `gpu_session.sh` now skips it with a message when
the counters are closed.

**Problem 2: the GPU test run failed at collection.** `tests/test_ablation.py`
imported `tests.test_correctness`, and the vLLM image has a `tests` package in
`dist-packages`, which won the import. The CPU container had no such package, so it
passed there. The test now imports `test_correctness` directly (pytest puts `tests/`
on the path). 115 passed after the fix.

**Problem 3: a few percent of noise at N=16384.** In the first knockout run the
copy with nothing removed was 2.4% faster than the kernel it is identical to, and the
two candidate optimizations below looked like slowdowns on their own but a speedup
together. With 2-4% between identical code, variants that close cannot be ranked
from sequential runs. `ablation.py --rounds 5` interleaves the variants and pools
the samples. With it N=4096 has IQRs of 0.25% and every conclusion in the README
holds at all three lengths, though at N=16384 identical code still differed by 3.8%.

**Problem 4: one BF16 cell was an outlier.** The first BF16 table had 100.6
TFLOP/s at N=1024 (FP16: 161.5) with a large IQR. A rerun gave 163.9, so the whole
BF16 table was rerun (`bench_bf16.json`; the first run is kept as
`bench_bf16_first.json`).

**The clock.** In a 15 s loop of the kernel at N=16384 the GPU drew 677 W on
average, `nvidia-smi` reported "SW power cap" in 74 of 75 samples, and the SM clock
averaged 1685 MHz (p10 1635, p90 1755; `clock_n16384.csv`). The realistic ceiling is
about 911 TFLOP/s, not 989.4.

**Results** (details in the README):
- 8 x 4 config: +10% to +13% over the 8 x 2 run at every N. 12-21% ahead of FA2;
  FA3, cuDNN and FlashInfer are 1.3x to 1.7x ahead from N=2048 up.
- Time at N=16384: tensor-core work at peak 45%, matmuls below peak 16%, HBM traffic
  7%, softmax 33%, of which the exponentials are 11%.
- Packing at batch 16, N_kv = 4096: 2.95x up to N_q = 32, as fast as FA3; 1.87x at
  N_q = 64; nothing from N_q = 128, where the rule turns it off. Long prefill: within
  -2.5% to +0.2%.
- Two fixes measured in the knockout copy: skip the end-of-sequence mask when N_kv
  is a multiple of BLOCK_N, and fold the scale into the exponent as one FFMA. 447 ->
  371 instructions per tile, 8.6% to 10.0% faster. Not in the kernel yet.

**Lesson.** The famous number is the exponentials: 989 vs 3.9 TFLOP/s on H100. In
this kernel they cost 11% of the time. The FP32 work around them, issued while the
tensor cores wait, costs twice that, and the matmul schedule itself loses another
16%. Counting instructions per tile told me that before any timing did.

**Open:**
- [ ] Put `even_n` and `ffma_scale` into the kernel, then rerun the table.
- [ ] Nsight Compute on a machine with counter access: tensor-pipe utilization and
  stall reasons per instruction, to check the breakdown.
- [ ] Overlap softmax and matmul: Triton's warp specialization, or a two-warpgroup
  ping-pong.
- [ ] Why FA3 is 1.27x faster at N_q = 64 in the packing table (split-KV? tile shape?).


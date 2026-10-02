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
- [ ] re-run the full N range with the 8 x 4 default
- [ ] flash-attn / FA3 baselines (needs a prebuilt wheel for this torch)

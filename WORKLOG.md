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
- [ ] full `pytest` sweep on real hardware (D=128, N up to 4096, BF16)
- [ ] `bench/bench.py` default shape; record the env JSON
- [ ] check the register/occupancy of `_default_config` for D=128 (8 warps, 2 stages is a guess)

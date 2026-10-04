# Nsight Compute, 2026-10-04 (Verda H100 SXM VM)

`bench/profile_ncu.sh` with `--set full`, N=16384, FP16, causal, default shape. Per target:

- `<target>.details.csv`: the report's details page (all sections)
- `<target>.raw.csv`: every raw metric
- `<target>.sass.csv`: per-SASS-instruction counters and warp stall samples

The `.ncu-rep` files (open them in the Nsight Compute GUI) are assets of the GitHub release
[`ncu-2026-10-04`](https://github.com/roy6n23/triton-fa2-forward/releases/tag/ncu-2026-10-04),
with a SHA256SUMS file.

Targets: `triton-fa2` (the kernel at 83dbbe8), `before_fixes` (the kernel at e4b8e5e, via
bench/ablation.py), `no_exp` and `matmul_only` (knockouts), `fa3-vllm` (vLLM's FA3 build).

Summary: `python bench/ncu_summary.py bench/results/2026-10-04/ncu`.

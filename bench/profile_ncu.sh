#!/usr/bin/env bash
# Nsight Compute on the kernel, two knockouts, the kernel before its 2026-10-03 fixes, and FA3 (GPU box). Needs access to the GPU performance
# counters: in a container that is the host's call (NVreg_RestrictProfilingToAdminUsers=0), and ncu fails
# with ERR_NVGPUCTRPERM without it.
#
#   bash bench/profile_ncu.sh [OUT_DIR]          # default bench/results/ncu, N=16384, fp16, causal
#
# Per target: <name>.ncu-rep (open in the Nsight Compute GUI), and CSV exports of the details page, the raw
# metrics and the per-instruction SASS view. ncu locks the SM clock to base by default, so durations are
# longer than in bench.py; the percent-of-peak metrics are what to compare.
set -uo pipefail
NCU=${NCU:-ncu}
OUT=${1:-bench/results/ncu}
N=${N:-16384}
mkdir -p "$OUT"

prof() {   # prof NAME KERNEL_REGEX [profile_target.py args...]
    local name=$1 regex=$2
    shift 2
    echo "== ncu $name"
    "$NCU" --set full --import-source yes --kernel-name-base demangled --kernel-name "regex:$regex" \
        --launch-skip 1 --launch-count 1 -f -o "$OUT/$name" \
        python bench/profile_target.py --seqlen "$N" "$@" || { echo "ncu failed for $name"; return 1; }
    "$NCU" -i "$OUT/$name.ncu-rep" --page details --csv > "$OUT/$name.details.csv"
    "$NCU" -i "$OUT/$name.ncu-rep" --page raw --csv > "$OUT/$name.raw.csv"
    "$NCU" -i "$OUT/$name.ncu-rep" --page source --print-source sass --csv > "$OUT/$name.sass.csv" 2>/dev/null \
        || echo "(no source page export for $name)"
}

prof triton-fa2 '^_attn_fwd$'
prof matmul_only '_attn_fwd_ablate' --variant matmul_only
prof no_exp '_attn_fwd_ablate' --variant no_exp
prof before_fixes '_attn_fwd_ablate' --variant before_fixes
prof fa3-vllm 'FlashAttnFwdSm90' --provider fa3-vllm

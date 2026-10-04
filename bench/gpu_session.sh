#!/usr/bin/env bash
# One GPU session, every step logged to bench/results/<UTC date>/. A failed step is logged and skipped.
#
#   bash bench/gpu_session.sh            # in tmux; ~1 h on an H100
set -uo pipefail
cd "$(dirname "$0")/.."
OUT=bench/results/$(date -u +%F)
mkdir -p "$OUT"
step() {   # step NAME CMD...
    local name=$1
    shift
    echo "=== $(date -u +%T) $name: $*" | tee -a "$OUT/session.log"
    "$@" > "$OUT/$name.log" 2>&1
    local rc=$?
    echo "=== $(date -u +%T) $name exit $rc" | tee -a "$OUT/session.log"
    tail -n 25 "$OUT/$name.log"
}

step env python -c "import json, sys; sys.path.insert(0, 'bench'); import bench; print(json.dumps(bench.environment(), indent=1))"
step pytest python -m pytest -q -rs
step sass-fp16 python bench/sass_stats.py --from-cache --dump "$OUT/sass_fp16_causal.txt"
step sass-pack python bench/sass_stats.py --from-cache --pack 4 --dump "$OUT/sass_fp16_causal_pack4.txt"
# 1. The default table with the 8x4 config and every baseline, both dtypes.
step bench-fp16 python bench/bench.py --out "$OUT/bench_fp16.json"
step bench-bf16 python bench/bench.py --dtype bf16 --out "$OUT/bench_bf16.json"
# 2. GQA head packing: few queries per head against a long K/V, then long causal prefill.
step pack-short python bench/bench.py --no-causal --batch 16 --kv-len 4096 --seqlens 1 16 32 64 128 256 \
    --providers triton-fa2-nopack triton-fa2-pack triton-fa2 fa2-vllm fa3-vllm sdpa-flash --out "$OUT/pack_short.json"
step pack-prefill python bench/bench.py --seqlens 512 2048 8192 16384 \
    --providers triton-fa2-nopack triton-fa2-pack --out "$OUT/pack_prefill.json"
# 3. Where the time goes: knockouts (interleaved), then Nsight Compute if this machine lets it read the
#    GPU's performance counters (a RunPod container does not: ERR_NVGPUCTRPERM, WORKLOG 2026-10-03).
step ablation python bench/ablation.py --rounds 5 --seqlens 4096 8192 16384 --out "$OUT/ablation_fp16.json"
if ! command -v "${NCU:-ncu}" > /dev/null; then
    echo "=== ncu skipped: not installed" | tee -a "$OUT/session.log"
elif "${NCU:-ncu}" --metrics sm__cycles_elapsed.avg python -c "import torch; torch.ones(1, device='cuda').add_(1)" 2>&1 \
        | grep -q ERR_NVGPUCTRPERM; then
    echo "=== ncu skipped: no access to the GPU performance counters (ERR_NVGPUCTRPERM)" | tee -a "$OUT/session.log"
else
    step ncu bash bench/profile_ncu.sh "$OUT/ncu"
fi
echo "done: $OUT"

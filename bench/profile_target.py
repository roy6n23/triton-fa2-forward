"""One shape, one provider, a few launches: the process Nsight Compute profiles (bench/profile_ncu.sh).

  python bench/profile_target.py --provider triton-fa2 --seqlen 16384
  python bench/profile_target.py --variant no_exp --seqlen 16384      # a knockout from bench/ablation.py
  python bench/profile_target.py --provider fa3-vllm --seqlen 16384

The first launch compiles (Triton) or warms up; profile the last one (ncu --launch-skip).
"""

import argparse
import importlib.util
import os
import sys

import torch

HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(HERE, ".."))


def _load(name, file):
    spec = importlib.util.spec_from_file_location(name, os.path.join(HERE, file))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    bench = _load("fa2_bench", "bench.py")
    ablation = _load("fa2_ablation", "ablation.py")
    ap = argparse.ArgumentParser()
    ap.add_argument("--provider", default="triton-fa2", choices=list(bench.PROVIDERS))
    ap.add_argument("--variant", choices=list(ablation.VARIANTS), help="run a knockout copy instead")
    ap.add_argument("--seqlen", type=int, default=16384)
    ap.add_argument("--hq", type=int, default=32)
    ap.add_argument("--hkv", type=int, default=8)
    ap.add_argument("--head-dim", type=int, default=128)
    ap.add_argument("--dtype", choices=["fp16", "bf16"], default="fp16")
    ap.add_argument("--launches", type=int, default=2)
    args = ap.parse_args()
    dtype = {"fp16": torch.float16, "bf16": torch.bfloat16}[args.dtype]
    torch.manual_seed(0)
    q = torch.randn(1, args.hq, args.seqlen, args.head_dim, device="cuda", dtype=dtype)
    k = torch.randn(1, args.hkv, args.seqlen, args.head_dim, device="cuda", dtype=dtype)
    v = torch.randn_like(k)
    if args.variant:
        on = ablation.VARIANTS[args.variant]
        knobs = {kn: True for kn in on if kn != "pack"}
        fn = lambda: ablation.attention_ablate(q, k, v, causal=True, pack_gqa="pack" in on, **knobs)   # noqa: E731
    else:
        fn = bench.PROVIDERS[args.provider](q, k, v, True)
    for _ in range(args.launches):
        fn()
    torch.cuda.synchronize()


if __name__ == "__main__":
    main()

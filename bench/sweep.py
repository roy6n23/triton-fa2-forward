"""Time attention() over tile configs (BLOCK_M, BLOCK_N, num_warps, num_stages), interleaved. GPU only.

Every config is checked against SDPA first (the same 2e-2 max-abs gate as bench.py), then the configs are timed
once per round for --rounds rounds and the samples pooled, so clock drift hits them alike (README, "Benchmark
protocol"). Registers, spills and shared memory of each compiled kernel are printed at the end.

  python bench/sweep.py                                                # D=128 candidates, N = 2048..16384
  python bench/sweep.py --configs 128,64,8,4 128,128,8,3 --seqlens 16384 --rounds 5
"""

import argparse
import importlib.util
import json
import os
import sys

import torch
import torch.nn.functional as F

HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(HERE, ".."))
from fa2.kernel import _attn_fwd, attention  # noqa: E402

DEFAULT_CONFIGS = ["128,64,8,4", "128,64,8,3", "128,128,8,2", "128,128,8,3"]


def _harness():
    spec = importlib.util.spec_from_file_location("fa2_bench", os.path.join(HERE, "bench.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    h = _harness()
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", nargs="+", default=DEFAULT_CONFIGS, help="BLOCK_M,BLOCK_N,num_warps,num_stages")
    ap.add_argument("--seqlens", type=int, nargs="+", default=[2048, 4096, 8192, 16384])
    ap.add_argument("--hq", type=int, default=32)
    ap.add_argument("--hkv", type=int, default=8)
    ap.add_argument("--head-dim", type=int, default=128)
    ap.add_argument("--dtype", choices=["fp16", "bf16"], default="fp16")
    ap.add_argument("--no-causal", action="store_true")
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    causal = not args.no_causal
    dtype = {"fp16": torch.float16, "bf16": torch.bfloat16}[args.dtype]
    configs = [tuple(int(x) for x in c.split(",")) for c in args.configs]
    env = h.environment()
    rows = []
    for N in args.seqlens:
        torch.manual_seed(0)
        q = torch.randn(1, args.hq, N, args.head_dim, device="cuda", dtype=dtype)
        k = torch.randn(1, args.hkv, N, args.head_dim, device="cuda", dtype=dtype)
        v = torch.randn_like(k)
        ref = F.scaled_dot_product_attention(q, k, v, is_causal=causal, enable_gqa=True)
        flops = h.attn_flops(1, args.hq, N, N, args.head_dim, causal)
        fns, errs = {}, {}
        for bm, bn, nw, ns in configs:
            fn = (lambda bm=bm, bn=bn, nw=nw, ns=ns:
                  attention(q, k, v, causal=causal, block_m=bm, block_n=bn, num_warps=nw, num_stages=ns))
            try:
                err = (fn().float() - ref.float()).abs().max().item()
            except Exception as e:  # noqa: BLE001 - a config that does not compile or launch is reported, not fatal
                print(f"  skip {bm},{bn},{nw},{ns} N={N}: {type(e).__name__}: {str(e)[:160]}")
                continue
            if not err <= 2e-2:
                print(f"  skip {bm},{bn},{nw},{ns} N={N}: max-abs {err:.3e} vs SDPA")
                continue
            fns[(bm, bn, nw, ns)], errs[(bm, bn, nw, ns)] = fn, err
        samples = {c: [] for c in fns}
        for _ in range(args.rounds):
            for c, fn in fns.items():
                samples[c] += h.measure(fn, args.warmup, args.iters)
        base = None
        print(f"\nN={N}  ({args.dtype}, causal={causal}, {args.rounds} interleaved rounds)")
        print("| BLOCK_M, BLOCK_N, warps, stages | ms (IQR) | vs first | TFLOP/s | max-abs vs SDPA |")
        print("|---|---|---|---|---|")
        for c in fns:
            st = h.summarize(samples[c])
            base = base or st["median_ms"]
            tf = flops / (st["median_ms"] * 1e-3) / 1e12
            rows.append(dict(seqlen=N, config=list(c), tflops=tf, max_abs_err_vs_sdpa=errs[c],
                             **{k: v for k, v in st.items() if k != "samples_ms"}))
            print(f"| {', '.join(map(str, c))} | {st['median_ms']:.3f} ({st['iqr_ms']:.3f}) | "
                  f"{st['median_ms'] / base - 1:+.1%} | {tf:.1f} | {errs[c]:.2e} |")
        del q, k, v, ref
        torch.cuda.empty_cache()
    print("\ncompiled kernels:")
    kernels = []
    for caches in _attn_fwd.device_caches.values():
        for ck in caches[0].values():
            md = ck.metadata
            kernels.append(dict(num_warps=md.num_warps, num_stages=md.num_stages, shared=md.shared,
                                n_regs=getattr(ck, "n_regs", None), n_spills=getattr(ck, "n_spills", None)))
            print(f"  warps {md.num_warps}, stages {md.num_stages}, shared {md.shared / 1024:.0f} KiB, "
                  f"registers {getattr(ck, 'n_regs', '?')}, spills {getattr(ck, 'n_spills', '?')}")
    out = args.out or os.path.join(HERE, "results", f"sweep_{env['gpu'].replace(' ', '-')}_{args.dtype}_{env['date'][:10]}.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump({"env": env, "config": vars(args), "results": rows, "kernels": kernels}, f, indent=2)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()

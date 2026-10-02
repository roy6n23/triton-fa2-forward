"""Forward-attention benchmark: this kernel vs 2026 baselines. GPU only.

Protocol:
  * warmup 5 + measured 20 iterations per (provider, shape)
  * L2 flushed before every measured iteration (256 MB write), so no provider
    benefits from K/V still sitting in L2 from the previous call
  * report the median and the IQR (25th-75th percentile), never the mean or the best
  * every provider's output is checked against torch SDPA before it is timed
  * the full software/hardware stack is written next to the numbers

Providers (each one is skipped, with the reason printed, if it cannot run):
  triton-fa2      this repo
  sdpa-flash      torch SDPA, FLASH_ATTENTION backend (FA2 inside PyTorch)
  sdpa-cudnn      torch SDPA, CUDNN_ATTENTION backend
  flash-attn      Tri Dao's flash-attn package (FA2 kernels)
  flash-attn-3    flash_attn_interface (FA3, Hopper only)
  flashinfer      flashinfer.single_prefill_with_kv_cache (batch == 1 only)

Usage:
  python bench/bench.py                                   # Llama-3-8B-like heads, causal
  python bench/bench.py --head-dim 64 --hq 16 --hkv 16    # MHA, d=64
"""

import argparse
import datetime
import json
import math
import os
import platform
import statistics
import subprocess
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from fa2.kernel import attention  # noqa: E402


# ---------------------------------------------------------------------------
# timing
# ---------------------------------------------------------------------------

_L2_FLUSH = None


def flush_l2():
    global _L2_FLUSH
    if _L2_FLUSH is None:
        _L2_FLUSH = torch.empty(256 * 1024 * 1024, dtype=torch.uint8, device="cuda")
    _L2_FLUSH.zero_()


def measure(fn, warmup, iters, flush=True):
    """Per-iteration CUDA-event timings in ms."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    for s, e in zip(starts, ends):
        if flush:
            flush_l2()
        s.record()
        fn()
        e.record()
    torch.cuda.synchronize()
    return [s.elapsed_time(e) for s, e in zip(starts, ends)]


def summarize(ms):
    q1, med, q3 = statistics.quantiles(ms, n=4, method="inclusive")
    return {"median_ms": med, "p25_ms": q1, "p75_ms": q3, "iqr_ms": q3 - q1, "samples_ms": ms}


def attn_flops(B, H_q, N_q, N_kv, D, causal):
    # Two matmuls (QK^T and PV), 2 FLOPs per multiply-add. Causal keeps only the
    # lower triangle including the diagonal.
    pairs = N_q * (N_q + 1) / 2 if causal else N_q * N_kv
    return 4 * B * H_q * pairs * D


# ---------------------------------------------------------------------------
# providers: each returns a zero-arg callable, or raises to be skipped
# ---------------------------------------------------------------------------

def p_triton_fa2(q, k, v, causal):
    return lambda: attention(q, k, v, causal=causal)


def _sdpa(backend):
    def make(q, k, v, causal):
        from torch.nn.attention import SDPBackend, sdpa_kernel
        be = getattr(SDPBackend, backend)

        def run():
            with sdpa_kernel(be):
                return F.scaled_dot_product_attention(q, k, v, is_causal=causal, enable_gqa=True)
        return run
    return make


def p_flash_attn(q, k, v, causal):
    from flash_attn import flash_attn_func
    # flash-attn wants [B, N, H, D]; the layout change is done once, outside the timed region.
    qt, kt, vt = (t.transpose(1, 2).contiguous() for t in (q, k, v))
    return lambda: flash_attn_func(qt, kt, vt, causal=causal)


def p_flash_attn_3(q, k, v, causal):
    if torch.cuda.get_device_capability()[0] != 9:
        raise RuntimeError("FA3 kernels target Hopper (sm_90) only")
    from flash_attn_interface import flash_attn_func
    qt, kt, vt = (t.transpose(1, 2).contiguous() for t in (q, k, v))
    return lambda: flash_attn_func(qt, kt, vt, causal=causal)


def p_flashinfer(q, k, v, causal):
    import flashinfer
    if q.shape[0] != 1:
        raise RuntimeError("single_prefill_with_kv_cache handles batch == 1 only")
    # NHD layout: [N, H, D].
    qt, kt, vt = (t[0].transpose(0, 1).contiguous() for t in (q, k, v))
    return lambda: flashinfer.single_prefill_with_kv_cache(qt, kt, vt, causal=causal)


PROVIDERS = {
    "triton-fa2": p_triton_fa2,
    "sdpa-flash": _sdpa("FLASH_ATTENTION"),
    "sdpa-cudnn": _sdpa("CUDNN_ATTENTION"),
    "flash-attn": p_flash_attn,
    "flash-attn-3": p_flash_attn_3,
    "flashinfer": p_flashinfer,
}


def to_bhnd(out, like):
    """Bring any provider's output back to [B, H, N, D] for the correctness check."""
    if isinstance(out, tuple):          # some FA3 builds return (out, lse)
        out = out[0]
    if out.shape == like.shape:
        return out
    if out.dim() == 3:                  # flashinfer NHD, batch 1
        return out.transpose(0, 1).unsqueeze(0)
    return out.transpose(1, 2)          # flash-attn BNHD


# ---------------------------------------------------------------------------
# environment pinning
# ---------------------------------------------------------------------------

def _version(mod):
    try:
        return __import__(mod).__version__
    except Exception:
        return None


def environment():
    props = torch.cuda.get_device_properties(0)
    try:
        driver = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            capture_output=True, text=True, check=True,
        ).stdout.strip().splitlines()[0]
    except Exception:
        driver = None
    return {
        "gpu": props.name,
        "compute_capability": f"{props.major}.{props.minor}",
        "sm_count": props.multi_processor_count,
        "driver": driver,
        "cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "torch": torch.__version__,
        "triton": _version("triton"),
        "flash_attn": _version("flash_attn"),
        "flashinfer": _version("flashinfer"),
        "python": platform.python_version(),
        "date": datetime.datetime.now().isoformat(timespec="seconds"),
    }


# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seqlens", type=int, nargs="+", default=[512, 1024, 2048, 4096, 8192, 16384])
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--hq", type=int, default=32)
    ap.add_argument("--hkv", type=int, default=8)
    ap.add_argument("--head-dim", type=int, default=128)
    ap.add_argument("--dtype", choices=["fp16", "bf16"], default="fp16")
    ap.add_argument("--no-causal", action="store_true")
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--no-flush", action="store_true")
    ap.add_argument("--providers", nargs="+", default=list(PROVIDERS))
    ap.add_argument("--out", default=None, help="JSON path (default: bench/results/<gpu>_<dtype>_<date>.json)")
    args = ap.parse_args()

    assert torch.cuda.is_available(), "benchmark needs a CUDA GPU"
    causal = not args.no_causal
    dtype = {"fp16": torch.float16, "bf16": torch.bfloat16}[args.dtype]
    env = environment()
    print(json.dumps(env, indent=2))

    rows = []
    for N in args.seqlens:
        torch.manual_seed(0)
        q = torch.randn(args.batch, args.hq, N, args.head_dim, device="cuda", dtype=dtype)
        k = torch.randn(args.batch, args.hkv, N, args.head_dim, device="cuda", dtype=dtype)
        v = torch.randn_like(k)
        ref = F.scaled_dot_product_attention(q, k, v, is_causal=causal, enable_gqa=True)
        flops = attn_flops(args.batch, args.hq, N, N, args.head_dim, causal)

        for name in args.providers:
            try:
                fn = PROVIDERS[name](q, k, v, causal)
                out = to_bhnd(fn(), q)
                err = (out.float() - ref.float()).abs().max().item()
                if not math.isfinite(err) or err > 2e-2:
                    raise RuntimeError(f"output mismatch vs SDPA: max-abs {err:.3e}")
                stats = summarize(measure(fn, args.warmup, args.iters, flush=not args.no_flush))
            except Exception as e:  # noqa: BLE001 - any failure just skips this provider
                print(f"  skip {name:13s} N={N:6d}: {type(e).__name__}: {str(e)[:120]}")
                continue
            stats.update(provider=name, seqlen=N, max_abs_err_vs_sdpa=err,
                         tflops=flops / (stats["median_ms"] * 1e-3) / 1e12)
            rows.append(stats)
            print(f"  {name:13s} N={N:6d}  {stats['median_ms']:8.3f} ms "
                  f"[IQR {stats['p25_ms']:.3f}-{stats['p75_ms']:.3f}]  {stats['tflops']:6.1f} TFLOP/s")
        del q, k, v, ref
        torch.cuda.empty_cache()

    # Markdown table: rows = seqlen, columns = provider, cells = median TFLOP/s (IQR ms).
    names = [p for p in args.providers if any(r["provider"] == p for r in rows)]
    print(f"\nB={args.batch} H_q={args.hq} H_kv={args.hkv} D={args.head_dim} "
          f"{args.dtype} causal={causal} | median TFLOP/s (IQR ms)\n")
    print("| N | " + " | ".join(names) + " |")
    print("|---|" + "---|" * len(names))
    for N in args.seqlens:
        cells = []
        for p in names:
            r = next((r for r in rows if r["provider"] == p and r["seqlen"] == N), None)
            cells.append(f"{r['tflops']:.1f} ({r['iqr_ms']:.3f})" if r else "—")
        print(f"| {N} | " + " | ".join(cells) + " |")

    out = args.out or os.path.join(
        os.path.dirname(__file__), "results",
        f"{env['gpu'].replace(' ', '-')}_{args.dtype}_{env['date'][:10]}.json",
    )
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump({"env": env, "config": vars(args), "results": rows}, f, indent=2)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()

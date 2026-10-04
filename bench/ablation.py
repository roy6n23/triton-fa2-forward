"""Knockout variants of the kernel, for attributing its time. Profiling only; GPU for timing.

A copy of fa2/kernel.py's _attn_fwd with switches that each remove one kind of work, so the time that kind
costs (the part not hidden behind the rest) is the difference to the unmodified copy:

  no_exp       exp2 -> a subtraction: no MUFU work, everything else unchanged
  no_softmax   no max, exp, sum, mask or rescale: acc += (Q K^T) V, the two matmuls and the loads only
  kv_l2        every iteration loads the first K/V tile again: same instructions, but served from L2
               instead of HBM, so the difference is what HBM traffic costs

and two candidate optimizations that the SASS suggests (fewer FP32 instructions per tile):

  ffma_scale   fold the softmax scale into the exp argument, qk * scale - m (one FFMA), as Triton's
               tutorial 06 does, instead of scaling every score and then subtracting
  even_n       skip the end-of-sequence mask outside the last tile when N_kv % BLOCK_N == 0

With every switch off the copy computes exactly what fa2.kernel.attention does (tests/test_ablation.py checks
that bit for bit). The knockouts produce wrong attention outputs by design; only their time means anything.

Usage:
  python bench/ablation.py                              # N = 4096 and 16384, fp16, causal, default shape
  python bench/ablation.py --seqlens 16384 --dtype bf16
  python bench/ablation.py --rounds 5                   # interleave: every variant once per round, 5 rounds,
                                                        # samples pooled (for differences of a few percent)
"""

import argparse
import json
import math
import os
import sys

import torch
import triton
import triton.language as tl

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from fa2.kernel import _default_config  # noqa: E402

KNOBS = ("no_exp", "no_softmax", "kv_l2", "ffma_scale", "even_n")
VARIANTS = {                     # name -> switches on
    "copy": (),
    "no_exp": ("no_exp",),
    "no_softmax": ("no_softmax",),
    "kv_l2": ("kv_l2",),
    "matmul_only": ("no_softmax", "kv_l2"),
    "ffma_scale": ("ffma_scale",),
    "even_n": ("even_n",),
    "ffma_scale+even_n": ("ffma_scale", "even_n"),
    "pack": ("pack",),           # the copy with GQA head packing forced on
}


@triton.jit
def _inner(
    acc, l_i, m_i, q,
    k_base, v_base, stride_kn, stride_kd, stride_vn, stride_vd,
    offs_m, offs_d, lo, hi, N_KV, qk_scale,
    BLOCK_N: tl.constexpr, CAUSAL_MASK: tl.constexpr, PAD_MASK: tl.constexpr,
    NO_EXP: tl.constexpr, NO_SOFTMAX: tl.constexpr, KV_L2: tl.constexpr, FFMA_SCALE: tl.constexpr,
):
    offs_n_base = tl.arange(0, BLOCK_N)
    for start_n in range(lo, hi, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        offs_n = start_n + offs_n_base
        col_ok = offs_n < N_KV
        if KV_L2:
            offs_ld = offs_n_base
        else:
            offs_ld = offs_n

        k = tl.load(
            k_base + offs_d[:, None] * stride_kd + offs_ld[None, :] * stride_kn,
            mask=col_ok[None, :], other=0.0,
        )
        if FFMA_SCALE:
            qk = tl.dot(q, k)
        else:
            qk = tl.dot(q, k) * qk_scale

        if NO_SOFTMAX:
            p = qk
        else:
            if PAD_MASK:
                visible = col_ok[None, :]
                if CAUSAL_MASK:
                    visible = visible & (offs_m[:, None] >= offs_n[None, :])
                qk = tl.where(visible, qk, float("-inf"))
            elif CAUSAL_MASK:
                qk = tl.where(offs_m[:, None] >= offs_n[None, :], qk, float("-inf"))
            if FFMA_SCALE:
                m_new = tl.maximum(m_i, tl.max(qk, 1) * qk_scale)
                s = qk * qk_scale - m_new[:, None]
            else:
                m_new = tl.maximum(m_i, tl.max(qk, 1))
                s = qk - m_new[:, None]
            if NO_EXP:
                p = s
                alpha = m_i - m_new
            else:
                p = tl.math.exp2(s)
                alpha = tl.math.exp2(m_i - m_new)
            l_i = l_i * alpha + tl.sum(p, 1)

        v = tl.load(
            v_base + offs_ld[:, None] * stride_vn + offs_d[None, :] * stride_vd,
            mask=col_ok[:, None], other=0.0,
        )
        if NO_SOFTMAX:
            acc = acc + tl.dot(p.to(v.dtype), v)
        else:
            acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)
            m_i = m_new
    return acc, l_i, m_i


@triton.jit
def _attn_fwd_ablate(
    Q, K, V, Out,
    stride_qb, stride_qh, stride_qn, stride_qd,
    stride_kb, stride_kh, stride_kn, stride_kd,
    stride_vb, stride_vh, stride_vn, stride_vd,
    stride_ob, stride_oh, stride_on, stride_od,
    H_Q, N_Q, N_KV, qk_scale,
    GQA_GROUP: tl.constexpr, PACK: tl.constexpr, HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, CAUSAL: tl.constexpr,
    NO_EXP: tl.constexpr, NO_SOFTMAX: tl.constexpr, KV_L2: tl.constexpr, FFMA_SCALE: tl.constexpr,
    EVEN_N: tl.constexpr,
):
    start_m = tl.program_id(0)
    off_bh = tl.program_id(1)
    off_b = off_bh // (H_Q // PACK)
    off_hq0 = (off_bh % (H_Q // PACK)) * PACK
    off_hkv = off_hq0 // GQA_GROUP

    k_base = K + off_b * stride_kb + off_hkv * stride_kh
    v_base = V + off_b * stride_vb + off_hkv * stride_vh

    offs_r = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_m = offs_r // PACK
    offs_hq = off_hq0 + offs_r % PACK
    offs_d = tl.arange(0, HEAD_DIM)
    row_ok = offs_m < N_Q
    q_rows = Q + off_b * stride_qb + offs_hq * stride_qh + offs_m * stride_qn
    o_rows = Out + off_b * stride_ob + offs_hq * stride_oh + offs_m * stride_on

    q = tl.load(q_rows[:, None] + offs_d[None, :] * stride_qd, mask=row_ok[:, None], other=0.0)

    m_i = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)

    if CAUSAL:
        lo_m = start_m * (BLOCK_M // PACK)
        hi_m = lo_m + BLOCK_M // PACK
        stage1_end = (lo_m // BLOCK_N) * BLOCK_N
        acc, l_i, m_i = _inner(
            acc, l_i, m_i, q, k_base, v_base, stride_kn, stride_kd, stride_vn, stride_vd,
            offs_m, offs_d, 0, stage1_end, N_KV, qk_scale,
            BLOCK_N, False, not EVEN_N, NO_EXP, NO_SOFTMAX, KV_L2, FFMA_SCALE,
        )
        acc, l_i, m_i = _inner(
            acc, l_i, m_i, q, k_base, v_base, stride_kn, stride_kd, stride_vn, stride_vd,
            offs_m, offs_d, stage1_end, tl.minimum(hi_m, N_KV), N_KV, qk_scale,
            BLOCK_N, True, True, NO_EXP, NO_SOFTMAX, KV_L2, FFMA_SCALE,
        )
    else:
        acc, l_i, m_i = _inner(
            acc, l_i, m_i, q, k_base, v_base, stride_kn, stride_kd, stride_vn, stride_vd,
            offs_m, offs_d, 0, N_KV, N_KV, qk_scale,
            BLOCK_N, False, not EVEN_N, NO_EXP, NO_SOFTMAX, KV_L2, FFMA_SCALE,
        )

    if NO_SOFTMAX:
        l_i = l_i + 1.0
    acc = acc / l_i[:, None]
    tl.store(o_rows[:, None] + offs_d[None, :] * stride_od, acc.to(Out.dtype.element_ty), mask=row_ok[:, None])


def attention_ablate(q, k, v, causal=True, pack_gqa=False, **knobs):
    """fa2.kernel.attention with the switches in KNOBS (default config only)."""
    unknown = set(knobs) - set(KNOBS)
    assert not unknown, f"unknown switches {sorted(unknown)}"
    B, H_q, N_q, D = q.shape
    H_kv, N_kv = k.shape[1], k.shape[2]
    bm, bn, nw, ns = _default_config(D, q.is_cuda)
    group = H_q // H_kv
    pack = group if pack_gqa else 1
    even_n = knobs.get("even_n", False)
    assert not even_n or N_kv % bn == 0, "even_n needs N_kv % BLOCK_N == 0"
    out = torch.empty_like(q)
    grid = (triton.cdiv(N_q * pack, bm), B * H_q // pack)
    _attn_fwd_ablate[grid](
        q, k, v, out, *q.stride(), *k.stride(), *v.stride(), *out.stride(),
        H_q, N_q, N_kv, 1.0 / math.sqrt(D) * 1.4426950408889634,
        GQA_GROUP=group, PACK=pack, HEAD_DIM=D, BLOCK_M=bm, BLOCK_N=bn, CAUSAL=causal,
        NO_EXP=knobs.get("no_exp", False), NO_SOFTMAX=knobs.get("no_softmax", False),
        KV_L2=knobs.get("kv_l2", False), FFMA_SCALE=knobs.get("ffma_scale", False), EVEN_N=even_n,
        num_warps=nw, num_stages=ns,
    )
    return out


def _harness():
    """bench/bench.py as a module (timing protocol and environment record), loaded by path."""
    import importlib.util

    spec = importlib.util.spec_from_file_location("fa2_bench", os.path.join(os.path.dirname(__file__), "bench.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    from fa2.kernel import attention

    h = _harness()
    attn_flops, environment, measure, summarize = h.attn_flops, h.environment, h.measure, h.summarize

    ap = argparse.ArgumentParser()
    ap.add_argument("--seqlens", type=int, nargs="+", default=[4096, 16384])
    ap.add_argument("--hq", type=int, default=32)
    ap.add_argument("--hkv", type=int, default=8)
    ap.add_argument("--head-dim", type=int, default=128)
    ap.add_argument("--dtype", choices=["fp16", "bf16"], default="fp16")
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--variants", nargs="+", default=list(VARIANTS), choices=list(VARIANTS))
    ap.add_argument("--rounds", type=int, default=1, help="interleaved rounds; samples are pooled across rounds")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    dtype = {"fp16": torch.float16, "bf16": torch.bfloat16}[args.dtype]
    env = environment()
    rows = []
    for N in args.seqlens:
        torch.manual_seed(0)
        q = torch.randn(1, args.hq, N, args.head_dim, device="cuda", dtype=dtype)
        k = torch.randn(1, args.hkv, N, args.head_dim, device="cuda", dtype=dtype)
        v = torch.randn_like(k)
        flops = attn_flops(1, args.hq, N, N, args.head_dim, True)
        ref = attention(q, k, v, causal=True)
        fns = {"production": lambda: attention(q, k, v, causal=True)}
        info = {}
        for name in args.variants:
            knobs = {kn: True for kn in VARIANTS[name] if kn != "pack"}
            fn = (lambda knobs=knobs, pack="pack" in VARIANTS[name]:
                  attention_ablate(q, k, v, causal=True, pack_gqa=pack, **knobs))
            out = fn()
            info[name] = dict(bit_exact=torch.equal(out, ref), max_abs_diff=(out.float() - ref.float()).abs().max().item())
            fns[name] = fn
        samples = {name: [] for name in fns}
        for _ in range(args.rounds):                    # interleaved, so slow drift hits every variant alike
            for name, fn in fns.items():
                samples[name] += measure(fn, args.warmup, args.iters)
        base = summarize(samples["production"])
        rows.append(dict(seqlen=N, variant="production", rounds=args.rounds, **base))
        for name in args.variants:
            rows.append(dict(seqlen=N, variant=name, rounds=args.rounds, **info[name], **summarize(samples[name])))
        print(f"\nN={N}  (production {base['median_ms']:.3f} ms, {flops / base['median_ms'] / 1e9:.1f} TFLOP/s)")
        print("| variant | ms (IQR) | vs production | TFLOP/s | output |")
        print("|---|---|---|---|---|")
        for r in rows:
            if r["seqlen"] != N:
                continue
            delta = r["median_ms"] - base["median_ms"]
            out = "" if r["variant"] == "production" else ("bit-exact" if r["bit_exact"] else
                                                           f"max diff {r['max_abs_diff']:.2e}")
            print(f"| {r['variant']} | {r['median_ms']:.3f} ({r['iqr_ms']:.3f}) | {delta:+.3f} ({delta / base['median_ms']:+.1%}) | "
                  f"{flops / r['median_ms'] / 1e9:.1f} | {out} |")
        del q, k, v, ref
        torch.cuda.empty_cache()
    out = args.out or os.path.join(os.path.dirname(__file__), "results",
                                   f"ablation_{env['gpu'].replace(' ', '-')}_{args.dtype}_{env['date'][:10]}.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump({"env": env, "config": vars(args), "results": [{k: v for k, v in r.items() if k != "samples_ms"}
                                                                for r in rows]}, f, indent=2)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()

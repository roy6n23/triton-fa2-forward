"""FlashAttention-2 style forward kernel in Triton.

Scope: forward only, FP16/BF16 inputs, optional causal mask, GQA.

Layout matches torch SDPA: q is [B, H_q, N_q, D], k/v are [B, H_kv, N_kv, D],
with H_q % H_kv == 0. Every H_q // H_kv consecutive query heads share one
KV head (GQA; H_kv == H_q is plain MHA, H_kv == 1 is MQA).

Algorithm, per program instance (one Q tile of one (batch, q_head)):

    load Q_i once                                  # stays in SRAM for the whole loop
    m_i = -inf, l_i = 0, acc = 0                   # online-softmax state, fp32
    for each K/V tile j that Q_i can see:
        S    = Q_i @ K_j^T * scale                 # never written to HBM
        m_new = max(m_i, rowmax(S))
        P    = exp(S - m_new)
        alpha = exp(m_i - m_new)                   # correction factor for old state
        l_i  = l_i * alpha + rowsum(P)
        acc  = acc * alpha + P @ V_j               # unnormalised output
        m_i  = m_new
    O_i = acc / l_i                                # normalise once, at the end (FA2)
    store O_i                                      # the only HBM write of the output

Loop order is FA2's: Q tiles on the outer (parallel) grid axis, K/V on the inner
loop, so no two programs ever touch the same output rows -> no inter-block
communication.

Causal handling (FA2 block skipping), for a Q tile covering rows [lo_m, lo_m + BLOCK_M):
    stage 1  K/V columns [0, lo_m)                 fully visible  -> no elementwise mask
    stage 2  K/V columns [lo_m, lo_m + BLOCK_M)    straddle the diagonal -> elementwise mask
    columns >= lo_m + BLOCK_M                      fully invisible -> never loaded (skipped)

We work in base 2: exp(x) == exp2(x * log2(e)), so log2(e) is folded into the
softmax scale once and every exponential becomes a single exp2 instruction.
"""

import math

import torch
import triton
import triton.language as tl


@triton.jit
def _attn_fwd_inner(
    acc, l_i, m_i, q,
    k_base, v_base, stride_kn, stride_kd, stride_vn, stride_vd,
    offs_m, offs_d, lo, hi, N_KV, qk_scale,
    BLOCK_N: tl.constexpr, CAUSAL_MASK: tl.constexpr,
):
    """Run the online-softmax recurrence over K/V columns [lo, hi)."""
    offs_n_base = tl.arange(0, BLOCK_N)
    for start_n in range(lo, hi, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        offs_n = start_n + offs_n_base
        col_ok = offs_n < N_KV

        # K is loaded already transposed: a [D, BLOCK_N] tile, so q @ k is Q K^T.
        k = tl.load(
            k_base + offs_d[:, None] * stride_kd + offs_n[None, :] * stride_kn,
            mask=col_ok[None, :], other=0.0,
        )
        qk = tl.dot(q, k) * qk_scale  # [BLOCK_M, BLOCK_N], fp32

        # Padding columns past the end of the sequence must not get probability mass.
        visible = col_ok[None, :]
        if CAUSAL_MASK:
            visible = visible & (offs_m[:, None] >= offs_n[None, :])
        qk = tl.where(visible, qk, float("-inf"))

        # Online-softmax update. m_new is always finite here: the first tile a row
        # sees always contains at least one visible column (column 0 in stage 1,
        # or column lo_m <= row in stage 2), so -inf - -inf never happens.
        m_new = tl.maximum(m_i, tl.max(qk, 1))
        p = tl.math.exp2(qk - m_new[:, None])
        alpha = tl.math.exp2(m_i - m_new)
        l_i = l_i * alpha + tl.sum(p, 1)

        v = tl.load(
            v_base + offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vd,
            mask=col_ok[:, None], other=0.0,
        )
        # P goes back to the input dtype so the second matmul also runs on tensor cores.
        acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)
        m_i = m_new
    return acc, l_i, m_i


@triton.jit
def _attn_fwd(
    Q, K, V, Out,
    stride_qb, stride_qh, stride_qn, stride_qd,
    stride_kb, stride_kh, stride_kn, stride_kd,
    stride_vb, stride_vh, stride_vn, stride_vd,
    stride_ob, stride_oh, stride_on, stride_od,
    H_Q, N_Q, N_KV, qk_scale,
    GQA_GROUP: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
    CAUSAL: tl.constexpr,
):
    # --- which tile am I? --------------------------------------------------
    start_m = tl.program_id(0)          # Q tile index along the sequence
    off_bh = tl.program_id(1)           # flattened (batch, q_head)
    off_b = off_bh // H_Q
    off_hq = off_bh % H_Q
    off_hkv = off_hq // GQA_GROUP       # GQA: query heads share a KV head

    q_base = Q + off_b * stride_qb + off_hq * stride_qh
    k_base = K + off_b * stride_kb + off_hkv * stride_kh
    v_base = V + off_b * stride_vb + off_hkv * stride_vh
    o_base = Out + off_b * stride_ob + off_hq * stride_oh

    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, HEAD_DIM)
    row_ok = offs_m < N_Q

    # --- load Q_i once; it is reused against every K/V tile ------------------
    q = tl.load(
        q_base + offs_m[:, None] * stride_qn + offs_d[None, :] * stride_qd,
        mask=row_ok[:, None], other=0.0,
    )

    m_i = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)

    if CAUSAL:
        # stage 1: tiles strictly left of the diagonal block, no elementwise mask.
        acc, l_i, m_i = _attn_fwd_inner(
            acc, l_i, m_i, q, k_base, v_base, stride_kn, stride_kd, stride_vn, stride_vd,
            offs_m, offs_d, 0, start_m * BLOCK_M, N_KV, qk_scale,
            BLOCK_N, False,
        )
        # stage 2: the diagonal block, elementwise causal mask.
        # Everything to the right is skipped entirely (never loaded, never multiplied).
        acc, l_i, m_i = _attn_fwd_inner(
            acc, l_i, m_i, q, k_base, v_base, stride_kn, stride_kd, stride_vn, stride_vd,
            offs_m, offs_d, start_m * BLOCK_M, tl.minimum((start_m + 1) * BLOCK_M, N_KV),
            N_KV, qk_scale,
            BLOCK_N, True,
        )
    else:
        acc, l_i, m_i = _attn_fwd_inner(
            acc, l_i, m_i, q, k_base, v_base, stride_kn, stride_kd, stride_vn, stride_vd,
            offs_m, offs_d, 0, N_KV, N_KV, qk_scale,
            BLOCK_N, False,
        )

    # --- normalise once, write the output once --------------------------------
    acc = acc / l_i[:, None]
    tl.store(
        o_base + offs_m[:, None] * stride_on + offs_d[None, :] * stride_od,
        acc.to(Out.dtype.element_ty),
        mask=row_ok[:, None],
    )


def _default_config(head_dim, on_cuda):
    """(BLOCK_M, BLOCK_N, num_warps, num_stages). Untuned starting points."""
    if not on_cuda:
        # TRITON_INTERPRET=1 on CPU: small tiles keep the numpy emulation fast.
        return 32, 16, 1, 1
    if head_dim <= 64:
        return 128, 64, 4, 3
    return 128, 64, 8, 2


def attention(q, k, v, causal=True, sm_scale=None, block_m=None, block_n=None,
              num_warps=None, num_stages=None):
    """softmax(Q K^T * sm_scale [+ causal mask]) V, fused.

    q: [B, H_q, N_q, D]; k, v: [B, H_kv, N_kv, D]; returns [B, H_q, N_q, D].
    """
    assert q.dim() == k.dim() == v.dim() == 4
    B, H_q, N_q, D = q.shape
    Bk, H_kv, N_kv, Dk = k.shape
    assert v.shape == k.shape and Bk == B and Dk == D
    assert q.dtype == k.dtype == v.dtype and q.dtype in (torch.float16, torch.bfloat16)
    assert H_q % H_kv == 0, "GQA needs H_q divisible by H_kv"
    assert D in (16, 32, 64, 128), "head_dim must be a power of two in [16, 128]"
    if causal:
        # Top-left aligned causal mask (prefill). Decode-style bottom-right alignment
        # with N_q < N_kv is out of scope for this kernel.
        assert N_q == N_kv, "causal attention here assumes N_q == N_kv"

    bm, bn, nw, ns = _default_config(D, q.is_cuda)
    bm, bn = block_m or bm, block_n or bn
    nw, ns = num_warps or nw, num_stages or ns
    # Stage 2 covers exactly one BLOCK_M-wide diagonal band in BLOCK_N steps.
    assert bm % bn == 0, "BLOCK_M must be a multiple of BLOCK_N"

    if sm_scale is None:
        sm_scale = 1.0 / math.sqrt(D)
    qk_scale = sm_scale * 1.4426950408889634  # log2(e): exp(x) == exp2(x * log2 e)

    out = torch.empty_like(q)
    grid = (triton.cdiv(N_q, bm), B * H_q)
    _attn_fwd[grid](
        q, k, v, out,
        *q.stride(), *k.stride(), *v.stride(), *out.stride(),
        H_q, N_q, N_kv, qk_scale,
        GQA_GROUP=H_q // H_kv,
        HEAD_DIM=D,
        BLOCK_M=bm, BLOCK_N=bn,
        CAUSAL=causal,
        num_warps=nw, num_stages=ns,
    )
    return out

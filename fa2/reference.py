"""Pure-PyTorch references. No Triton needed, runs anywhere (including a Mac CPU).

naive_attention      the textbook formula, materialises the full N x N score matrix.
                     Used as the ground truth (run it in fp32/fp64).
blocked_attention    the exact same tiled online-softmax recurrence as kernel.py,
                     written with torch ops and Python loops. Slow, but you can step
                     through it in a debugger, and it proves the algorithm is right
                     independently of any Triton bug.
"""

import math

import torch


def _expand_kv(k, v, h_q):
    """GQA -> MHA: repeat each KV head for the query heads that share it."""
    group = h_q // k.shape[1]
    return k.repeat_interleave(group, dim=1), v.repeat_interleave(group, dim=1)


def naive_attention(q, k, v, causal=True, sm_scale=None):
    """softmax(Q K^T * scale) V with the full score matrix. Computes in q's dtype."""
    D = q.shape[-1]
    sm_scale = sm_scale if sm_scale is not None else 1.0 / math.sqrt(D)
    k, v = _expand_kv(k, v, q.shape[1])
    s = (q @ k.transpose(-1, -2)) * sm_scale
    if causal:
        n_q, n_kv = s.shape[-2:]
        mask = torch.ones(n_q, n_kv, dtype=torch.bool, device=q.device).tril()
        s = s.masked_fill(~mask, float("-inf"))
    return torch.softmax(s, dim=-1) @ v


def blocked_attention(q, k, v, causal=True, sm_scale=None, block_m=32, block_n=16):
    """Line-for-line mirror of _attn_fwd / _attn_fwd_inner, in fp32 state.

    Each (b, h, start_m) iteration of the outer loops is one Triton program instance.
    """
    B, H_q, N_q, D = q.shape
    H_kv, N_kv = k.shape[1], k.shape[2]
    group = H_q // H_kv
    sm_scale = sm_scale if sm_scale is not None else 1.0 / math.sqrt(D)
    qk_scale = sm_scale * 1.4426950408889634  # fold log2(e) in, use exp2 below
    out = torch.empty_like(q)

    def inner(acc, l_i, m_i, q_tile, kh, vh, offs_m, lo, hi, causal_mask):
        for start_n in range(lo, hi, block_n):
            offs_n = torch.arange(start_n, min(start_n + block_n, N_kv), device=q.device)
            k_tile = kh[offs_n].float()                     # [bn, D]
            v_tile = vh[offs_n].float()
            qk = (q_tile @ k_tile.T) * qk_scale             # [bm, bn]
            if causal_mask:
                visible = offs_m[:, None] >= offs_n[None, :]
                qk = qk.masked_fill(~visible, float("-inf"))
            m_new = torch.maximum(m_i, qk.max(dim=1).values)
            p = torch.exp2(qk - m_new[:, None])
            alpha = torch.exp2(m_i - m_new)
            l_i = l_i * alpha + p.sum(dim=1)
            acc = acc * alpha[:, None] + p @ v_tile
            m_i = m_new
        return acc, l_i, m_i

    for b in range(B):
        for h in range(H_q):
            kh, vh = k[b, h // group], v[b, h // group]    # GQA head mapping
            for start_m in range(0, N_q, block_m):
                offs_m = torch.arange(start_m, min(start_m + block_m, N_q), device=q.device)
                q_tile = q[b, h, offs_m].float()
                m_i = torch.full((len(offs_m),), float("-inf"), device=q.device)
                l_i = torch.zeros(len(offs_m), device=q.device)
                acc = torch.zeros(len(offs_m), D, device=q.device)
                if causal:
                    acc, l_i, m_i = inner(acc, l_i, m_i, q_tile, kh, vh, offs_m,
                                          0, start_m, False)
                    acc, l_i, m_i = inner(acc, l_i, m_i, q_tile, kh, vh, offs_m,
                                          start_m, min(start_m + block_m, N_kv), True)
                else:
                    acc, l_i, m_i = inner(acc, l_i, m_i, q_tile, kh, vh, offs_m,
                                          0, N_kv, False)
                out[b, h, offs_m] = (acc / l_i[:, None]).to(q.dtype)
    return out

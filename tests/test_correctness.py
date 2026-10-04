"""Correctness tests.

Acceptance criterion: our max-abs error against a high-precision
reference must be <= 2x the error of PyTorch's own fp16 attention on the same
inputs. "Close to fp64" alone is not a useful bar for fp16 outputs; being no
worse than 2x what PyTorch ships is.

Where it runs:
  * GPU box:           pytest tests/                          (full sweep)
  * CPU, no Triton:    pytest tests/ -k blocked               (algorithm only)
  * CPU with Triton:   TRITON_INTERPRET=1 pytest tests/       (small sweep, kernel
                       executed by Triton's numpy interpreter)
"""

import importlib.util
import os

import pytest
import torch
import torch.nn.functional as F

from fa2.reference import blocked_attention, naive_attention

HAS_TRITON = importlib.util.find_spec("triton") is not None
ON_GPU = torch.cuda.is_available()
INTERPRET = os.environ.get("TRITON_INTERPRET") == "1"
DEVICE = "cuda" if ON_GPU else "cpu"

# Absolute slack added to the 2x bound, so a near-zero PyTorch error on a tiny
# shape cannot make the test flaky. Well below one fp16 ulp at 1.0 (~9.8e-4).
ATOL_FLOOR = 1e-4


def make_qkv(B, H_q, H_kv, N_q, N_kv, D, dtype, seed=0):
    g = torch.Generator(device=DEVICE).manual_seed(seed)
    q = torch.randn(B, H_q, N_q, D, device=DEVICE, dtype=dtype, generator=g)
    k = torch.randn(B, H_kv, N_kv, D, device=DEVICE, dtype=dtype, generator=g)
    v = torch.randn(B, H_kv, N_kv, D, device=DEVICE, dtype=dtype, generator=g)
    return q, k, v


def errors(out, ref):
    diff = (out.double() - ref).abs()
    return diff.max().item(), (diff.norm() / ref.norm()).item()


# ---------------------------------------------------------------------------
# 1. The algorithm itself (pure torch, runs anywhere).
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("causal", [True, False])
@pytest.mark.parametrize("H_q,H_kv", [(4, 4), (4, 2), (4, 1)])
@pytest.mark.parametrize("N", [1, 16, 37, 100])
def test_blocked_matches_naive(causal, H_q, H_kv, N):
    q, k, v = make_qkv(2, H_q, H_kv, N, N, 32, torch.float64)
    ref = naive_attention(q, k, v, causal=causal)
    out = blocked_attention(q, k, v, causal=causal, block_m=32, block_n=16)
    # fp32 accumulator state inside blocked_attention -> fp32-level agreement.
    torch.testing.assert_close(out, ref, atol=1e-5, rtol=1e-5)


def test_blocked_noncausal_different_lengths():
    q, k, v = make_qkv(1, 4, 2, 23, 70, 32, torch.float64)
    ref = naive_attention(q, k, v, causal=False)
    out = blocked_attention(q, k, v, causal=False)
    torch.testing.assert_close(out, ref, atol=1e-5, rtol=1e-5)


# ---------------------------------------------------------------------------
# 2. The Triton kernel.
# ---------------------------------------------------------------------------

needs_kernel = pytest.mark.skipif(
    not HAS_TRITON or not (ON_GPU or INTERPRET),
    reason="needs Triton plus a GPU, or TRITON_INTERPRET=1",
)

if ON_GPU:
    SEQ_LENS = [128, 1000, 2048, 4096]      # 1000: not a multiple of any block size
    HEAD_DIMS = [64, 128]
    HEADS = [(32, 32), (32, 8), (32, 1)]    # MHA, GQA (Llama-3-8B ratio), MQA
    BATCH = 2
else:
    SEQ_LENS = [1, 37, 128]                 # interpreter is slow: keep it small
    HEAD_DIMS = [16, 64]
    HEADS = [(4, 4), (4, 2), (4, 1)]
    BATCH = 1


def check_against_torch(q, k, v, causal, _impl=None, **kw):
    from fa2.kernel import attention

    ref = naive_attention(q.double(), k.double(), v.double(), causal=causal)
    ours = (_impl or attention)(q, k, v, causal=causal, **kw)
    theirs = F.scaled_dot_product_attention(q, k, v, is_causal=causal, enable_gqa=True)

    ours_abs, ours_rel = errors(ours, ref)
    theirs_abs, theirs_rel = errors(theirs, ref)
    assert torch.isfinite(ours).all()
    assert ours_abs <= 2 * theirs_abs + ATOL_FLOOR, (
        f"max-abs {ours_abs:.3e} vs torch {theirs_abs:.3e} "
        f"(rel {ours_rel:.3e} vs {theirs_rel:.3e})"
    )


@needs_kernel
@pytest.mark.parametrize("causal", [True, False])
@pytest.mark.parametrize("H_q,H_kv", HEADS)
@pytest.mark.parametrize("D", HEAD_DIMS)
@pytest.mark.parametrize("N", SEQ_LENS)
def test_kernel_fp16(causal, H_q, H_kv, D, N):
    q, k, v = make_qkv(BATCH, H_q, H_kv, N, N, D, torch.float16)
    check_against_torch(q, k, v, causal)


@needs_kernel
@pytest.mark.skipif(
    INTERPRET and not ON_GPU,
    reason="Triton 3.8's CPU interpreter computes bf16 tl.dot wrong (a bare 16x16 bf16 "
           "dot is off by ~1e10); bf16 is only checkable on a real GPU",
)
@pytest.mark.parametrize("causal", [True, False])
def test_kernel_bf16(causal):
    q, k, v = make_qkv(BATCH, 4, 2, 100, 100, 64, torch.bfloat16)
    check_against_torch(q, k, v, causal)


@needs_kernel
def test_kernel_noncausal_different_lengths():
    q, k, v = make_qkv(BATCH, 4, 2, 23, 70, 64, torch.float16)
    check_against_torch(q, k, v, causal=False)


@needs_kernel
def test_kernel_noncontiguous_input():
    """[B, N, H, D] storage viewed as [B, H, N, D], as it comes out of a QKV projection."""
    q, k, v = (t.transpose(1, 2).contiguous().transpose(1, 2)
               for t in make_qkv(BATCH, 4, 2, 64, 64, 64, torch.float16))
    assert not q.is_contiguous()
    check_against_torch(q, k, v, causal=True)


# ---------------------------------------------------------------------------
# 3. GQA head packing: a group's query heads share one program's M dimension.
# ---------------------------------------------------------------------------

if ON_GPU:
    PACK_HEADS = [(32, 8), (32, 1)]         # group 4 (Llama-3-8B), MQA (group 32)
    PACK_LENS = [1, 64, 1000, 2048]
else:
    PACK_HEADS = [(4, 2), (4, 1), (8, 2)]
    PACK_LENS = [1, 5, 37, 128]


@needs_kernel
@pytest.mark.parametrize("causal", [True, False])
@pytest.mark.parametrize("H_q,H_kv", PACK_HEADS)
@pytest.mark.parametrize("N", PACK_LENS)
def test_kernel_pack_gqa(causal, H_q, H_kv, N):
    """Packed and unpacked runs compute every row with the same tiles in the same order, so they agree bit for
    bit; the packed output also meets the 2x bar on its own."""
    from fa2.kernel import attention

    q, k, v = make_qkv(BATCH, H_q, H_kv, N, N, 64, torch.float16)
    packed = attention(q, k, v, causal=causal, pack_gqa=True)
    unpacked = attention(q, k, v, causal=causal, pack_gqa=False)
    assert torch.equal(packed, unpacked)
    check_against_torch(q, k, v, causal, pack_gqa=True)


@needs_kernel
@pytest.mark.parametrize("N_q", [1, 3, 16])
def test_kernel_pack_gqa_short_queries_long_kv(N_q):
    """The case packing is for: a few new tokens per head against a long K/V (decode, chunked prefill)."""
    q, k, v = make_qkv(BATCH, 8, 2, N_q, 200, 64, torch.float16)
    check_against_torch(q, k, v, causal=False, pack_gqa=True)


@needs_kernel
def test_pack_gqa_needs_block_m_divisible_by_group():
    from fa2.kernel import attention

    q, k, v = make_qkv(BATCH, 6, 2, 16, 16, 64, torch.float16)    # group 3 does not divide BLOCK_M
    with pytest.raises(ValueError, match="BLOCK_M"):
        attention(q, k, v, causal=True, pack_gqa=True)
    check_against_torch(q, k, v, causal=True)                       # the default does not pack it


@needs_kernel
def test_should_pack_gqa_follows_tile_efficiency():
    from fa2.kernel import should_pack_gqa

    assert should_pack_gqa(1, 4, 128)                  # 1 of 128 rows used vs 4 of 128
    assert should_pack_gqa(64, 4, 128)                 # half a tile vs exactly two tiles
    assert not should_pack_gqa(4096, 4, 128)           # both use every row
    assert not should_pack_gqa(1000, 4, 128)           # 1000/1024 vs 4000/4096: no gain
    assert not should_pack_gqa(64, 1, 128)             # MHA: nothing to pack
    assert not should_pack_gqa(64, 3, 128)             # group 3 does not divide 128

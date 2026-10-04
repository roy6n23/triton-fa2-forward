"""The knockout copy in bench/ablation.py must be the production kernel when every switch is off."""

import importlib.util
import os

import pytest
import torch

from test_correctness import BATCH, INTERPRET, ON_GPU, check_against_torch, make_qkv

HAS_TRITON = importlib.util.find_spec("triton") is not None
pytestmark = pytest.mark.skipif(not HAS_TRITON or not (ON_GPU or INTERPRET),
                                reason="needs Triton plus a GPU, or TRITON_INTERPRET=1")


@pytest.mark.parametrize("causal", [True, False])
@pytest.mark.parametrize("pack_gqa", [False, True])
@pytest.mark.parametrize("N", [37, 128])
def test_copy_is_the_production_kernel(causal, pack_gqa, N):
    from bench.ablation import attention_ablate
    from fa2.kernel import attention

    q, k, v = make_qkv(BATCH, 8, 2, N, N, 64, torch.float16)
    assert torch.equal(attention_ablate(q, k, v, causal=causal, pack_gqa=pack_gqa),
                       attention(q, k, v, causal=causal, pack_gqa=pack_gqa))


@pytest.mark.parametrize("knob", ["no_exp", "no_softmax", "kv_l2"])
def test_knockouts_change_the_output(knob):
    from bench.ablation import attention_ablate
    from fa2.kernel import attention

    q, k, v = make_qkv(BATCH, 4, 2, 128, 128, 64, torch.float16)
    assert not torch.equal(attention_ablate(q, k, v, **{knob: True}), attention(q, k, v))


@pytest.mark.parametrize("knobs", [{"scale_first": True}, {"mask_every_tile": True},
                                   {"scale_first": True, "mask_every_tile": True}])
@pytest.mark.parametrize("causal", [True, False])
def test_the_kernel_before_its_two_fixes_is_still_attention(knobs, causal):
    from bench import ablation

    q, k, v = make_qkv(BATCH, 4, 2, 128, 128, 64, torch.float16)

    def run(q, k, v, causal):
        return ablation.attention_ablate(q, k, v, causal=causal, **knobs)

    check_against_torch(q, k, v, causal, _impl=run)

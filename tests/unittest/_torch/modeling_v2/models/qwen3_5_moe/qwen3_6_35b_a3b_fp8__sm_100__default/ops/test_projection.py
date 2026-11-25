# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import true_fp32_matmul
from tensorrt_llm._torch._experimental.modeling_v2.models.qwen3_5_moe.qwen3_6_35b_a3b_fp8__sm_100__default.kernels.prefill_gemm import (  # noqa: E501
    PrefillGemm,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.qwen3_5_moe.qwen3_6_35b_a3b_fp8__sm_100__default.ops.epilogue import (  # noqa: E501
    block_quantize,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.qwen3_5_moe.qwen3_6_35b_a3b_fp8__sm_100__default.ops.projection import (  # noqa: E501
    decode_gates,
    fp8_projection,
    routed_experts,
    uses_cutlass,
)

__extra_import_path__ = ["../../.."]
from _validating import assert_within_ulp, certifying, validating  # noqa: E402,F401

assert torch.cuda.is_available()
_DEVICE = torch.device("cuda")
_BF16 = torch.bfloat16
_FP8 = torch.float8_e4m3fn


def _weights(n, k, experts=0):
    leading = (experts,) if experts else ()
    weight = torch.empty((*leading, n, k), dtype=_FP8, device=_DEVICE)
    source = torch.randn(n, k, device=_DEVICE, dtype=_BF16).mul_(40).to(_FP8)
    weight.copy_(source)
    scales = torch.rand((*leading, n // 128, k // 128), device=_DEVICE, dtype=_BF16)
    scales.mul_(0.001).add_(0.0002)
    return weight, scales


@pytest.mark.parametrize("cell", fp8_projection.CELLS, ids=lambda cell: cell.why)
def test_fp8_projection(cell, certifying):  # noqa: F811
    certifying(fp8_projection)
    torch.manual_seed(13)
    spec = cell.spec
    n, k = spec["n"], spec["k"]
    x = torch.randn(spec["rows"], k, device=_DEVICE, dtype=_BF16)
    q = block_quantize.reference(
        x, mn_major=not spec["decode"] and uses_cutlass(spec["rows"], False)
    )
    args = q, _weights(n, k), spec["decode"], PrefillGemm(_DEVICE)
    with true_fp32_matmul():
        got, want = fp8_projection(*args), fp8_projection.reference(*args)
    assert_within_ulp(got, want, element_ulp=2, rms_ulp=0.2)


@pytest.mark.parametrize("cell", decode_gates.CELLS, ids=lambda cell: cell.why)
def test_decode_gates(cell, certifying):  # noqa: F811
    certifying(decode_gates)
    torch.manual_seed(17)
    x = torch.randn(cell.spec["rows"], 2048, device=_DEVICE, dtype=_BF16)
    w = torch.randn(64, 2048, device=_DEVICE, dtype=_BF16) * 0.03
    a = torch.randn(32, device=_DEVICE, dtype=_BF16)
    dt = torch.randn(32, device=_DEVICE, dtype=_BF16)
    with true_fp32_matmul():
        decay, beta = decode_gates(x, w, a, dt)
        want_decay, want_beta = decode_gates.reference(x, w, a, dt)
    middle = slice(x.shape[0] // 3, x.shape[0] // 3 + 1)
    one_decay, one_beta = decode_gates(x[middle], w, a, dt)
    assert torch.equal(decay[middle], one_decay)
    assert torch.equal(beta[middle], one_beta)
    torch.testing.assert_close(decay, want_decay, atol=0.0015, rtol=0.01)
    torch.testing.assert_close(beta, want_beta, atol=0.001, rtol=0.006)


@pytest.mark.parametrize("cell", routed_experts.CELLS, ids=lambda cell: cell.why)
def test_routed_experts(cell, certifying):  # noqa: F811
    certifying(routed_experts)
    torch.manual_seed(101)
    rows = cell.spec["rows"]
    x = torch.randn(rows, 2048, device=_DEVICE, dtype=_BF16)
    x[:, 0] = 2
    x[:, 1] = torch.where(torch.arange(rows, device=_DEVICE) % 2 == 0, 1.0, -1.0)
    router = torch.randn(257, 2048, device=_DEVICE, dtype=_BF16) * 0.001
    anchors = torch.linspace(0.5, 1.5, 8, device=_DEVICE, dtype=_BF16)
    router[:16, 0] = anchors.repeat(2)
    router[:8, 1] = 2
    router[8:16, 1] = -2
    # Both route groups have well-separated BF16 logits. This way differences
    # in expert output rows measure GEMM and packing, without top-k tie swaps.
    q = block_quantize.reference(x)
    gate_up = _weights(1024, 2048, 257)
    down = _weights(2048, 512, 257)
    args = (
        x,
        q,
        router,
        gate_up,
        down,
        cell.spec["decode"],
        torch.zeros(257, device=_DEVICE, dtype=torch.int32),
        PrefillGemm(_DEVICE),
    )
    query_rows = cell.spec.get("query_rows", 1)
    short_context = cell.spec.get("short_context", False)
    with true_fp32_matmul():
        out, probability, logits, permutation = routed_experts(
            *args, query_rows=query_rows, short_context=short_context
        )
        want, want_probability, want_logits, _ = routed_experts.reference(
            *args, query_rows=query_rows, short_context=short_context
        )
    if permutation is not None:
        out = out[permutation.long().flatten()]
    if cell.spec["decode"]:
        middle = slice(rows // 3, rows // 3 + 1)
        quantized = tuple(t[middle] for t in q)
        _, one_prob, one_logits, _ = routed_experts(x[middle], quantized, *args[2:])
        assert torch.equal(logits[middle], one_logits)
        assert torch.equal(probability[middle], one_prob)
    torch.testing.assert_close(logits, want_logits)
    # Compare probability against the actual BF16 logits as well as the
    # FP32-dot reference. One BF16 logit step can move softmax by several ulps.
    actual_scores, actual_ids = logits[:, :256].float().sort(descending=True, stable=True)
    _, want_ids = want_logits[:, :256].float().sort(descending=True, stable=True)
    assert torch.equal(actual_ids[:, :8], want_ids[:, :8])
    torch.testing.assert_close(probability, actual_scores[:, :8].softmax(-1).bfloat16())
    assert_within_ulp(probability, want_probability, element_ulp=8, rms_ulp=0.3)
    # A BF16 dot result on an FP8 decision boundary can move its input to
    # the down projection across one code. Compare the full rows by relative RMS.
    diff = out.float() - want.float()
    relative_rms = diff.square().mean().sqrt() / want.float().square().mean().sqrt()
    assert relative_rms < 0.006


def test_triton_fp8_projection_rejects_wrong_scale_layout():
    torch.manual_seed(44)
    x = torch.randn(7, 2048, device=_DEVICE, dtype=_BF16)
    q = block_quantize.reference(x, mn_major=True)
    args = q, _weights(2048, 2048), False, PrefillGemm(_DEVICE)
    with true_fp32_matmul():
        actual, expected = fp8_projection(*args), fp8_projection.reference(*args)
    assert (actual.float() - expected.float()).square().mean() > 1e-3
    with validating(fp8_projection), pytest.raises(ValueError, match="row-major"):
        fp8_projection(*args)

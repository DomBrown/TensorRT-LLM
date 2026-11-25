# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.qwen3_5_moe.qwen3_6_35b_a3b_fp8__sm_100__default.ops.epilogue import (  # noqa: E501
    block_quantize,
    gated_output_quantize,
    gdn_gates,
    mtp_input,
    residual_norm_quantize,
)

__extra_import_path__ = ["../../.."]
from _validating import assert_within_ulp, certifying  # noqa: E402,F401

assert torch.cuda.is_available()


def _bf16(*shape):
    return torch.randn(*shape, device="cuda", dtype=torch.bfloat16)


def _compare_quantized(out, ref):
    quant, scales = out
    ref_quant, ref_scales = ref
    assert quant.shape == ref_quant.shape
    assert quant.dtype == ref_quant.dtype == torch.float8_e4m3fn
    assert scales.stride() == ref_scales.stride()
    torch.testing.assert_close(scales, ref_scales, atol=1e-8, rtol=2**-7)
    # BF16-rounded gates can move an operand at an FP8 rounding boundary by one
    # representable FP8 code. The dequantized error must stay below one FP8 ulp.
    a = quant.float().view(*scales.shape, 128) * scales[..., None]
    b = ref_quant.float().view(*scales.shape, 128) * ref_scales[..., None]
    assert_within_ulp(a, b, element_ulp=1, rms_ulp=0.15, ulp=1 / 14 + 1e-7)


@pytest.mark.parametrize("cell", block_quantize.CELLS, ids=lambda cell: cell.why)
def test_block_quantize(cell, certifying):  # noqa: F811
    certifying(block_quantize)
    torch.manual_seed(6)
    swiglu = cell.spec.get("swiglu", False)
    x = _bf16(cell.spec["rows"], cell.spec["width"] * (2 if swiglu else 1))
    args = x, swiglu, cell.spec.get("mn_major", False)
    _compare_quantized(block_quantize(*args), block_quantize.reference(*args))


@pytest.mark.parametrize("cell", residual_norm_quantize.CELLS, ids=lambda cell: cell.why)
def test_residual_norm_quantize(cell, certifying):  # noqa: F811
    certifying(residual_norm_quantize)
    torch.manual_seed(13)
    rows = cell.spec["rows"]
    mode = cell.spec["mode"]
    x, weight = _bf16(rows, 2048), _bf16(2048) * 0.04
    delta = probability = router = permutation = None
    if mode == "add":
        delta = _bf16(rows, 2048)
    if mode == "moe":
        delta = _bf16(rows * 9, 2048)
        probability = _bf16(rows, 8).float().softmax(-1).bfloat16()
        router = _bf16(rows, 257)
        if cell.spec.get("packed", False):
            permutation = torch.randperm(rows * 9, device="cuda", dtype=torch.int32)
    args = (
        x,
        weight,
        delta,
        probability,
        router,
        permutation,
        cell.spec.get("quantize", True),
        cell.spec.get("mn_major", False),
    )
    residual, normalized, quantized = residual_norm_quantize(*args)
    ref_residual, ref_normalized, ref_quantized = residual_norm_quantize.reference(*args)
    assert_within_ulp(residual, ref_residual, element_ulp=2, rms_ulp=0.75)
    assert_within_ulp(normalized, ref_normalized, element_ulp=2, rms_ulp=0.75)
    if quantized is not None:
        _compare_quantized(quantized, ref_quantized)
    else:
        assert ref_quantized is None


@pytest.mark.parametrize("cell", gated_output_quantize.CELLS, ids=lambda cell: cell.why)
def test_gated_output_quantize(cell, certifying):  # noqa: F811
    certifying(gated_output_quantize)
    torch.manual_seed(31)
    n, recurrent = cell.spec["rows"], cell.spec["recurrent"]
    projection_width = 12288 if recurrent else 9216
    if cell.spec.get("gate_only"):
        projection_width = 4096
    args = (
        _bf16(n, 4096),
        _bf16(n, projection_width),
        _bf16(128) if recurrent else None,
        cell.spec.get("mn_major", False),
    )
    _compare_quantized(gated_output_quantize(*args), gated_output_quantize.reference(*args))


@pytest.mark.parametrize("cell", gdn_gates.CELLS, ids=lambda cell: cell.why)
def test_gdn_gates(cell, certifying):  # noqa: F811
    certifying(gdn_gates)
    torch.manual_seed(17)
    args = _bf16(cell.spec["rows"], 64) * 4, _bf16(32) * 2, _bf16(32)
    decay, beta = gdn_gates(*args)
    want_decay, want_beta = gdn_gates.reference(*args)
    torch.testing.assert_close(decay, want_decay, rtol=2e-5, atol=3e-6)
    torch.testing.assert_close(beta, want_beta, rtol=0.004, atol=1e-7)


@pytest.mark.parametrize("cell", mtp_input.CELLS, ids=lambda cell: cell.why)
def test_mtp_input(cell, certifying):  # noqa: F811
    certifying(mtp_input)
    torch.manual_seed(21)
    rows = cell.spec["rows"]
    input_ids = torch.randint(0, 41, (rows,), device="cuda", dtype=torch.int32)
    embedding = _bf16(41, 2048)
    hidden = _bf16(rows, 2048)
    embedding_norm, hidden_norm = _bf16(2048) * 0.05, _bf16(2048) * 0.05
    args = input_ids, embedding, hidden, embedding_norm, hidden_norm
    combined = mtp_input(*args)
    expected = mtp_input.reference(*args)
    assert_within_ulp(combined, expected, element_ulp=1, rms_ulp=0.03)
    embeds = residual_norm_quantize(embedding[input_ids], embedding_norm, quantize=False)[1]
    contexts = residual_norm_quantize(hidden, hidden_norm, quantize=False)[1]
    assert torch.equal(combined[:, :2048], embeds)
    assert torch.equal(combined[:, 2048:], contexts)

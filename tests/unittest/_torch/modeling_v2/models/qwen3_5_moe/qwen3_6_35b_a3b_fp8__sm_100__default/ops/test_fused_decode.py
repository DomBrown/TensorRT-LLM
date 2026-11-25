# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import true_fp32_matmul
from tensorrt_llm._torch._experimental.modeling_v2.models.qwen3_5_moe.qwen3_6_35b_a3b_fp8__sm_100__default.ops.draft_logits import (  # noqa: E501
    draft_argmax,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.qwen3_5_moe.qwen3_6_35b_a3b_fp8__sm_100__default.ops.fused_verify_input import (  # noqa: E501
    fused_verify_input,
)

__extra_import_path__ = ["../../.."]
from _validating import assert_within_ulp, certifying  # noqa: E402,F401


@pytest.mark.parametrize("cell", draft_argmax.CELLS, ids=lambda cell: cell.why)
def test_draft_argmax(cell, certifying):  # noqa: F811
    certifying(draft_argmax)
    torch.manual_seed(41)
    rows = cell.spec["rows"]
    columns = 66560
    active = columns - 137
    logits = torch.randn(rows, columns, dtype=torch.bfloat16, device="cuda").clamp_(-3, 3)
    logits[:, 17] = logits[:, 2050] = 4
    logits[:, active:] = torch.inf
    if rows > 1:
        logits[1, :active] = -torch.inf
        logits[2, 50] = logits[2, 3100] = torch.nan
    ids = torch.randperm(248320, dtype=torch.int32, device="cuda")[:columns]
    length = torch.full((1,), active, dtype=torch.int32, device="cuda")
    out = torch.empty(rows, dtype=torch.int32, device="cuda") if cell.spec.get("inplace") else None
    actual = draft_argmax(logits, ids, length, out)
    expected = draft_argmax.reference(logits, ids, length)
    assert torch.equal(actual, expected)
    if out is not None:
        assert actual is out


@pytest.mark.parametrize("cell", fused_verify_input.CELLS, ids=lambda cell: cell.why)
def test_fused_verify_input(cell, certifying):  # noqa: F811
    certifying(fused_verify_input)
    torch.manual_seed(43)
    width = cell.spec["width"]
    a = torch.randint(-7, 8, (width, 2048), device="cuda").bfloat16().to(torch.float8_e4m3fn)
    a_scale = torch.full((width, 16), 1 / 64, dtype=torch.float32, device="cuda")
    w = torch.randint(-7, 8, (12288, 2048), device="cuda").bfloat16().to(torch.float8_e4m3fn)
    w_scale = torch.full((96, 16), 1 / 32, dtype=torch.bfloat16, device="cuda")
    conv = torch.randn(8192, 1, 4, dtype=torch.bfloat16, device="cuda") * 0.1
    history = torch.randn(4, 3, 8192, dtype=torch.bfloat16, device="cuda").transpose(1, 2)
    slots = torch.tensor([2], device="cuda", dtype=torch.int32)
    shape = (1, width, 8192)
    if not cell.spec.get("raw_only"):
        shape = (*shape, 3)
    actual_stage = torch.empty(shape, dtype=torch.bfloat16, device="cuda")
    expected_stage = torch.empty_like(actual_stage)
    args = ((a, a_scale), (w, w_scale), conv, history, slots)
    with true_fp32_matmul():
        gate, mixed = fused_verify_input(*args, actual_stage)
        expected_gate, expected_mixed = fused_verify_input.reference(*args, expected_stage)
    assert torch.equal(gate, expected_gate)
    assert torch.equal(actual_stage, expected_stage)
    assert_within_ulp(mixed, expected_mixed, element_ulp=2, rms_ulp=0.1)

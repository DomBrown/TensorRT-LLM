# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import true_fp32_matmul
from tensorrt_llm._torch._experimental.modeling_v2.models.qwen3_5_moe.qwen3_6_35b_a3b_fp8__sm_100__default.ops.attention import (  # noqa: E501
    paged_decode_attention,
    qkv_projection,
)

__extra_import_path__ = ["../../.."]
from _validating import assert_within_ulp, certifying  # noqa: E402,F401

assert torch.cuda.is_available()


@pytest.mark.parametrize("cell", qkv_projection.CELLS, ids=lambda cell: cell.why)
def test_qkv_projection(cell, certifying):  # noqa: F811
    certifying(qkv_projection)
    torch.manual_seed(47)
    rows = cell.spec["rows"]
    phase = torch.arange(4096, device="cuda")[:, None] * (
        1e7 ** (-torch.arange(0, 64, 2, device="cuda", dtype=torch.float32) / 64)
    )
    args = (
        torch.randn(rows, 9216, device="cuda", dtype=torch.bfloat16),
        torch.randn(256, device="cuda", dtype=torch.bfloat16),
        torch.randn(256, device="cuda", dtype=torch.bfloat16),
        phase.cos(),
        phase.sin(),
        torch.randperm(4096, device="cuda", dtype=torch.int32)[:rows],
        cell.spec.get("packed", True),
    )
    actual = qkv_projection(*args)
    expected = qkv_projection.reference(*args)
    if isinstance(actual, tuple):
        for got, want in zip(actual, expected, strict=True):
            assert_within_ulp(got, want, element_ulp=1, rms_ulp=0.1)
    else:
        assert_within_ulp(actual, expected, element_ulp=1, rms_ulp=0.1)


@pytest.mark.parametrize("cell", paged_decode_attention.CELLS, ids=lambda cell: cell.why)
def test_paged_decode_attention(cell, certifying):  # noqa: F811
    certifying(paged_decode_attention)
    torch.manual_seed(41)
    positions = cell.spec["positions"]
    layers = cell.spec.get("layers", 1)
    page_size, width = 32, 16
    rows = len(positions)
    storage = torch.randn(64, layers, 2, 2, page_size, 256, device="cuda", dtype=torch.bfloat16)
    expected_storage = storage.clone()
    pages = torch.randperm(64, device="cuda", dtype=torch.int32)[: rows * width].view(rows, width)
    key_pages = pages * (2 * layers)
    positions = torch.tensor(positions, device="cuda", dtype=torch.int32)
    qkv = torch.randn(rows, 5120, device="cuda", dtype=torch.bfloat16)
    actual = paged_decode_attention(qkv, storage[:, -1], positions, key_pages, width * page_size)
    with true_fp32_matmul():
        expected = paged_decode_attention.reference(
            qkv, expected_storage[:, -1], positions, key_pages, width * page_size
        )
    assert_within_ulp(actual, expected, element_ulp=2, rms_ulp=0.85)
    assert torch.equal(storage, expected_storage)

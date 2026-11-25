# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import true_fp32_matmul
from tensorrt_llm._torch._experimental.modeling_v2.models.qwen3_5_moe.qwen3_6_35b_a3b_fp8__sm_100__default.ops.attention import (  # noqa: E501
    paged_decode_attention,
    qkv_projection,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.qwen3_5_moe.qwen3_6_35b_a3b_fp8__sm_100__default.ops.epilogue import (  # noqa: E501
    gated_output_quantize,
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
    query_rows = cell.spec.get("query_rows", 1)
    batch = rows // query_rows
    storage = torch.randn(64, layers, 2, 2, page_size, 256, device="cuda", dtype=torch.bfloat16)
    expected_storage = storage.clone()
    pages = torch.randperm(64, device="cuda", dtype=torch.int32)[: batch * width].view(batch, width)
    key_pages = pages * (2 * layers)
    positions = torch.tensor(positions, device="cuda", dtype=torch.int32)
    qkv = torch.randn(rows, 5120, device="cuda", dtype=torch.bfloat16)
    args = qkv, storage[:, -1], positions, key_pages, width * page_size, query_rows
    projection = None
    if cell.spec.get("quantize", False):
        projection = torch.randn(rows, 9216, device="cuda", dtype=torch.bfloat16)
    mn_major = cell.spec.get("mn_major", False)
    raw_picks = cell.spec.get("pick_rows")
    pick_rows = None if raw_picks is None else torch.tensor(raw_picks, device="cuda")
    actual = paged_decode_attention(
        *args,
        projection=projection,
        mn_major=mn_major,
        pick_rows=pick_rows,
    )
    with true_fp32_matmul():
        expected = paged_decode_attention.reference(
            qkv,
            expected_storage[:, -1],
            positions,
            key_pages,
            width * page_size,
            query_rows,
            projection=projection,
            mn_major=mn_major,
            pick_rows=pick_rows,
        )
    assert torch.equal(storage, expected_storage)
    if projection is None:
        assert_within_ulp(actual, expected, element_ulp=2, rms_ulp=0.85)
    else:
        quant, scales = actual
        ref_quant, ref_scales = expected
        out = quant.float().reshape(-1, 32, 128) * scales[..., None]
        ref = ref_quant.float().reshape(-1, 32, 128) * ref_scales[..., None]
        assert_within_ulp(out, ref, element_ulp=2, rms_ulp=0.3, ulp=1 / 14 + 1e-7)
        old_core = paged_decode_attention(*args)
        if cell.spec.get("check_independent_chains", False):
            separate = []
            for request in range(batch):
                chain = slice(request * query_rows, (request + 1) * query_rows)
                separate.append(
                    paged_decode_attention(
                        qkv[chain],
                        storage[:, -1],
                        positions[chain],
                        key_pages[request : request + 1],
                        width * page_size,
                        query_rows,
                    )
                )
            assert torch.equal(old_core, torch.cat(separate))
        if pick_rows is not None:
            old_core = old_core.index_select(0, pick_rows)
            projection = projection.index_select(0, pick_rows)
        old_quant, old_scales = gated_output_quantize(old_core, projection, None, mn_major=mn_major)
        assert scales.stride() == old_scales.stride()
        assert torch.equal(quant.view(torch.uint8), old_quant.view(torch.uint8))
        assert torch.equal(scales, old_scales)

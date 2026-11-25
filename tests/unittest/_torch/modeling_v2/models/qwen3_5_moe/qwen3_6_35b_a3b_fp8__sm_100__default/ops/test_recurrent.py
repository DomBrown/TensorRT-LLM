# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Cover carried history and recurrent state at non-dense manager slot indices."""

import itertools

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import true_fp32_matmul
from tensorrt_llm._torch._experimental.modeling_v2.models.qwen3_5_moe.qwen3_6_35b_a3b_fp8__sm_100__default.modeling import (  # noqa: E501
    _ContextGroup,
    _Decoder,
    _partition_context,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.qwen3_5_moe.qwen3_6_35b_a3b_fp8__sm_100__default.ops.recurrent import (  # noqa: E501
    SHORT_CONTEXT,
    causal_convolution,
    gated_delta_rule,
    replay_gdn,
    verify_conv,
    verify_gdn,
)

__extra_import_path__ = ["../../.."]
from _validating import assert_within_ulp, certifying  # noqa: E402,F401

assert torch.cuda.is_available()


def _layout(spec):
    lengths = spec["lengths"]
    slots = torch.tensor(spec["slots"], device="cuda", dtype=torch.int32)
    rows = sum(lengths) if lengths else slots.numel()
    offsets = initial = row_map = None
    if lengths:
        offsets = torch.tensor(
            list(itertools.accumulate(lengths, initial=0)), device="cuda", dtype=torch.int32
        )
        row_map = torch.arange(len(lengths), device="cuda", dtype=torch.int32).repeat_interleave(
            torch.tensor(lengths, device="cuda")
        )
        initial = torch.tensor(spec["initial"], device="cuda", dtype=torch.bool)
    return lengths, rows, slots, offsets, row_map, initial


@pytest.mark.parametrize("cell", causal_convolution.CELLS, ids=lambda cell: cell.why)
def test_causal_convolution(cell, certifying):  # noqa: F811
    certifying(causal_convolution)
    torch.manual_seed(7)
    _, rows, slots, offsets, row_map, initial = _layout(cell.spec)
    split = cell.spec.get("split", False)
    raw = torch.randn(rows, 12288, device="cuda", dtype=torch.bfloat16)
    weights = torch.randn(8192, 1, 4, device="cuda", dtype=torch.bfloat16)
    storage = torch.randn(7, 3, 8192, 3, device="cuda", dtype=torch.bfloat16)
    reference_storage = storage.clone()
    actual = causal_convolution(
        raw, weights, storage[:, 1], slots, offsets, row_map, initial, split
    )
    with true_fp32_matmul():
        expected = causal_convolution.reference(
            raw, weights, reference_storage[:, 1], slots, offsets, row_map, initial, split
        )
    if split:
        for got, want in zip(actual, expected, strict=True):
            assert got.is_contiguous()
            assert_within_ulp(got, want, element_ulp=2, rms_ulp=0.1)
    else:
        assert_within_ulp(actual, expected, element_ulp=2, rms_ulp=0.1)
    assert torch.equal(storage, reference_storage)


@pytest.mark.parametrize("cell", gated_delta_rule.CELLS, ids=lambda cell: cell.why)
def test_gated_delta_rule(cell, certifying):  # noqa: F811
    certifying(gated_delta_rule)
    torch.manual_seed(19)
    lengths, rows, slots, offsets, _, initial = _layout(cell.spec)
    mixed = torch.randn(rows, 8192, device="cuda", dtype=torch.bfloat16)
    if max(lengths, default=0) > SHORT_CONTEXT:
        q, k, v = (x.contiguous() for x in mixed.split((2048, 2048, 4096), dim=1))
        mixed = q.view(rows, 16, 128), k.view(rows, 16, 128), v.view(rows, 32, 128)
    alpha = torch.rand(rows, 32, device="cuda") * 0.3 + 0.7
    beta = torch.randn(rows, 32, device="cuda").sigmoid().bfloat16().float()
    num_slots = max(cell.spec["slots"]) + 2
    storage = torch.randn(num_slots, 3, 32, 128, 128, device="cuda") * 0.04
    reference_storage = storage.clone()
    read_slots = torch.where(initial, slots, -1) if initial is not None else None
    args = slots, offsets, initial, read_slots, lengths
    actual = gated_delta_rule(mixed, alpha, beta, storage[:, 1], *args)
    expected = gated_delta_rule.reference(mixed, alpha, beta, reference_storage[:, 1], *args)
    # Chunking forms BF16 block-factor products before applying them to FP32 carry.
    assert_within_ulp(actual, expected, element_ulp=3, rms_ulp=1)
    torch.testing.assert_close(storage[:, 1], reference_storage[:, 1], atol=0.006, rtol=0.02)
    untouched = sorted(set(range(num_slots)) - set(cell.spec["slots"]))
    assert torch.equal(storage[untouched], reference_storage[untouched])
    assert torch.equal(storage[:, ::2], reference_storage[:, ::2])


def test_short_recurrence_is_independent_of_context_packing():
    torch.manual_seed(23)
    lengths = (9, 14, 30)
    rows = sum(lengths)
    mixed = torch.randn(rows, 8192, device="cuda", dtype=torch.bfloat16)
    alpha = torch.rand(rows, 32, device="cuda") * 0.3 + 0.7
    beta = torch.rand(rows, 32, device="cuda")
    slots = torch.tensor([4, 1, 3], device="cuda", dtype=torch.int32)
    offsets = torch.tensor([0, 9, 23, 53], device="cuda", dtype=torch.int32)
    initial = torch.tensor([True, False, True], device="cuda")
    state = torch.randn(5, 32, 128, 128, device="cuda") * 0.04
    packed_state = state.clone()
    packed = gated_delta_rule(
        mixed, alpha, beta, packed_state, slots, offsets, initial, None, lengths
    )
    start = 0
    for i, length in enumerate(lengths):
        end = start + length
        single = gated_delta_rule(
            mixed[start:end],
            alpha[start:end],
            beta[start:end],
            state,
            slots[i : i + 1],
            offsets[i : i + 2] - start,
            initial[i : i + 1],
            None,
            (length,),
        )
        assert torch.equal(packed[start:end], single)
        start = end
    assert torch.equal(state, packed_state)


def test_mixed_short_and_long_prefill_preserves_slot_state():
    torch.manual_seed(83)
    lengths, rows, slots, offsets, row_map, initial = _layout(
        {"lengths": (9, 103, 13), "slots": (4, 1, 3), "initial": (True, True, False)}
    )
    read_slots = torch.where(initial, slots, -1)
    context = _ContextGroup(None, slots, offsets, row_map, initial, read_slots, lengths)
    options = {"device": "cuda", "dtype": torch.bfloat16}
    projection = torch.randn(rows, 12288, **options)
    hidden = torch.randn(rows, 2048, **options)
    weights = (
        torch.randn(64, 2048, **options) * 0.02,
        torch.randn(8192, 1, 4, **options) * 0.25,
        torch.randn(32, **options),
        torch.randn(32, **options),
        torch.ones(128, **options),
    )
    conv = torch.randn(5, 8192, 3, **options)
    state = torch.randn(5, 32, 128, 128, device="cuda") * 0.04
    packed_conv, packed_state = conv.clone(), state.clone()
    packed_output = torch.empty(rows, 4096, **options)
    for group in _partition_context(context):
        indices = group.rows
        assert indices is not None
        outputs = _Decoder._prefill_gdn(
            projection[indices], hidden[indices], weights, packed_conv, packed_state, group
        )
        packed_output.index_copy_(0, indices, outputs)
    start = 0
    for idx, length in enumerate(lengths):
        end = start + length
        item = slice(idx, idx + 1)
        solo_group = _ContextGroup(
            None,
            slots[item],
            offsets[idx : idx + 2] - start,
            torch.zeros_like(row_map[start:end]),
            initial[item],
            read_slots[item],
            (length,),
        )
        solo = _Decoder._prefill_gdn(
            projection[start:end], hidden[start:end], weights, conv, state, solo_group
        )
        assert torch.equal(packed_output[start:end], solo)
        start = end
    assert torch.equal(conv, packed_conv)
    assert torch.equal(state, packed_state)


@pytest.mark.parametrize("cell", verify_conv.CELLS, ids=lambda cell: cell.why)
def test_verify_convolution(cell, certifying):  # noqa: F811
    certifying(verify_conv)
    torch.manual_seed(17)
    slots = torch.tensor(cell.spec["slots"], device="cuda", dtype=torch.int32)
    width = cell.spec["width"]
    raw = torch.randn(slots.numel() * width, 12288, device="cuda", dtype=torch.bfloat16)
    weights = torch.randn(8192, 1, 4, device="cuda", dtype=torch.bfloat16)
    history = torch.randn(8, 3, 8192, 3, device="cuda", dtype=torch.bfloat16)[:, 1]
    saved = history.clone()
    staged = torch.zeros(4, 4, 8192, 3, device="cuda", dtype=torch.bfloat16)
    expected_staged = staged.clone()
    actual = verify_conv(raw, weights, slots, history, staged, width)
    with true_fp32_matmul():
        expected = verify_conv.reference(raw, weights, slots, history, expected_staged, width)
    assert_within_ulp(actual, expected, element_ulp=2, rms_ulp=0.1)
    assert torch.equal(history, saved)
    assert torch.equal(staged, expected_staged)


@pytest.mark.parametrize("cell", verify_gdn.CELLS, ids=lambda cell: cell.why)
def test_verify_gated_delta_rule(cell, certifying):  # noqa: F811
    certifying(verify_gdn)
    torch.manual_seed(27)
    slots = torch.tensor(cell.spec["slots"], device="cuda", dtype=torch.int32)
    width = cell.spec["width"]
    rows = width * slots.numel()
    state = torch.randn(8, 3, 32, 128, 128, device="cuda")[:, 1] * 0.02
    saved = state.clone()
    mixed = torch.randn(rows, 8192, device="cuda", dtype=torch.bfloat16)
    alpha = torch.rand(rows, 32, device="cuda") * 0.3 + 0.7
    beta = torch.rand(rows, 32, device="cuda")
    staged = torch.zeros(4, 4, 32, 128, 128, device="cuda")
    expected_staged = staged.clone()
    actual = verify_gdn(mixed, alpha, beta, slots, state, staged, width)
    expected = verify_gdn.reference(mixed, alpha, beta, slots, state, expected_staged, width)
    assert_within_ulp(actual, expected, element_ulp=1, rms_ulp=0.1)
    assert torch.equal(state, saved)
    torch.testing.assert_close(staged, expected_staged, atol=2e-7, rtol=1e-5)


@pytest.mark.parametrize("cell", replay_gdn.CELLS, ids=lambda cell: cell.why)
def test_replay_gated_delta_rule(cell, certifying):  # noqa: F811
    certifying(replay_gdn)
    torch.manual_seed(51)
    slots = torch.tensor(cell.spec["slots"], device="cuda", dtype=torch.int32)
    count = slots.numel()
    replay_slots = 6 - slots
    history = 8
    spec_width = 4
    state = torch.randn(8, 2, 32, 128, 128, device="cuda") * 0.03
    keys = torch.zeros(8, 2, history, 16, 128, device="cuda")
    innovations = torch.zeros(8, 2, history, 32, 128, device="cuda")
    decays = torch.zeros(8, 2, 32, history, device="cuda")
    meta = torch.zeros(8, 4, device="cuda", dtype=torch.int32)
    accepted, bank = meta[:, 0], meta[:, 1]
    actual_buffers = (state[:, 1], keys, innovations, decays)
    expected_buffers = tuple(t.clone() for t in actual_buffers)
    for step, width in enumerate(cell.spec["widths"]):
        rows = width * count
        mixed = torch.randn(rows, 8192, device="cuda", dtype=torch.bfloat16)
        alpha = torch.rand(rows, 32, device="cuda") * 0.3 + 0.7
        beta = torch.rand(rows, 32, device="cuda")
        new_count = 1 + (torch.arange(count, device="cuda", dtype=torch.int32) + step) % width
        args = (accepted, bank, width, spec_width)
        actual = replay_gdn(
            mixed,
            alpha,
            beta,
            actual_buffers[0],
            slots,
            replay_slots,
            *actual_buffers[1:],
            *args,
        )
        expected = replay_gdn.reference(
            mixed,
            alpha,
            beta,
            expected_buffers[0],
            slots,
            replay_slots,
            *expected_buffers[1:],
            *args,
        )
        assert_within_ulp(actual, expected, element_ulp=1, rms_ulp=0.15)
        for got, want in zip(actual_buffers, expected_buffers, strict=True):
            torch.testing.assert_close(got, want, atol=3e-7, rtol=2e-5)
        before = accepted[replay_slots]
        flushed = before + spec_width > history
        accepted[replay_slots] = torch.where(flushed, new_count, before + new_count)
        bank[replay_slots] = torch.where(flushed, 1 - bank[replay_slots], bank[replay_slots])

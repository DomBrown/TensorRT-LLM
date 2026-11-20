# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Independent references for the Qwen3.6 SM100 text operations."""
# ruff: noqa: F811

from itertools import accumulate

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.qwen3_5_moe.qwen3_6_35b_a3b_fp8__sm_100__default.kernels import (  # noqa: E501
    dg_prefill,
    fp8,
    mx_decode,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.qwen3_5_moe.qwen3_6_35b_a3b_fp8__sm_100__default.ops.attention import (  # noqa: E501
    KvPages,
    paged_decode_attention,
    project_and_store,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.qwen3_5_moe.qwen3_6_35b_a3b_fp8__sm_100__default.ops.linear_attention import (  # noqa: E501
    ConvAcceptance,
    GdnWeights,
    ReplayState,
    StatePool,
    accept_convolution,
    context_gdn,
    gated_delta_step,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.qwen3_5_moe.qwen3_6_35b_a3b_fp8__sm_100__default.ops.multi_token import (  # noqa: E501
    draft_argmax,
    draft_input,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.qwen3_5_moe.qwen3_6_35b_a3b_fp8__sm_100__default.ops.projections import (  # noqa: E501
    RoutedWeights,
    context_projection,
    dense_projection,
    routed_experts,
    rowwise_projection,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.qwen3_5_moe.qwen3_6_35b_a3b_fp8__sm_100__default.weights import (  # noqa: E501
    DenseWeight,
)

__extra_import_path__ = ["../../.."]
from _validating import assert_within_ulp, certifying  # noqa: E402,F401

BF16 = torch.bfloat16
FP8 = torch.float8_e4m3fn


def _rand(shape: tuple[int, ...], scale: float = 1.0) -> torch.Tensor:
    return (torch.randn(shape, device="cuda") * scale).to(BF16)


def _begin() -> None:
    assert torch.cuda.is_available()
    if torch.cuda.get_device_capability() != (10, 0):
        pytest.skip("Qwen3.6 FP8 target requires SM100")
    torch.manual_seed(314)


@pytest.mark.parametrize("cell", paged_decode_attention.CELLS, ids=lambda cell: cell.why)
def test_paged_decode_attention(cell, certifying) -> None:
    _begin()
    certifying(paged_decode_attention)
    lengths = cell.spec["lengths"]
    page_size = cell.spec["page_size"]
    batch = len(lengths)
    width = (max(lengths) + page_size - 1) // page_size
    slots = batch * width + 7
    cache = _rand((slots, 2, page_size, 2, 256), 0.2)
    blocks = torch.randperm(slots, device="cuda")[: batch * width].view(batch, width) * 2
    pages = KvPages(cache, torch.stack((blocks, blocks + 1), dim=1).int())
    positions = torch.tensor(lengths, device="cuda", dtype=torch.int32) - 1
    requests = torch.arange(batch, device="cuda", dtype=torch.int32).flip(0)
    q = _rand((batch, 16, 256), 0.5)
    gate = _rand((batch, 4096), 0.5)
    actual, scales, _ = paged_decode_attention(q, gate, positions, requests, pages, True, True)
    expected = paged_decode_attention.reference(q, gate, positions, requests, pages, True, True)
    dequantized = actual.float() * scales.repeat_interleave(128, dim=-1)
    # E4M3 rounds individual values by up to 1/16 of their magnitude.
    assert_within_ulp(dequantized.to(BF16), expected, element_ulp=16, rms_ulp=8)
    scrambled = expected.roll(128, dims=-1)
    assert (dequantized - scrambled).norm() > 10 * (dequantized - expected).norm()


@pytest.mark.parametrize("cell", gated_delta_step.CELLS, ids=lambda cell: cell.why)
def test_gated_delta_step(cell, certifying) -> None:
    _begin()
    certifying(gated_delta_step)
    slots = cell.spec["pool_slots"]
    steps = cell.spec["steps"]
    batch = len(slots)
    projected = _rand((batch * steps, 12288), 0.25)
    gates = _rand((batch * steps, 64), 0.5)
    weights = GdnWeights(_rand((8192, 4), 0.25), _rand((32,), 0.5), _rand((32,), 0.5))
    history = _rand((6, 8192, 3), 0.25)
    state = _rand((6, 32, 128, 128), 0.01).float()
    indices = torch.tensor(slots, device="cuda", dtype=torch.int32)
    initial = torch.tensor([i % 2 == 0 for i in range(batch)], device="cuda")
    pool = StatePool(history, state, indices, initial)
    expected_pool = StatePool(history.clone(), state.clone(), indices, initial)
    replay = None
    expected_replay = None
    if steps > 1:
        replay_indices = torch.tensor([1, 3], device="cuda", dtype=torch.int32)
        scratch = torch.zeros((batch, steps, 8192), device="cuda", dtype=BF16)
        factors = torch.zeros((6, steps - 1, 32, 257), device="cuda")
        pending = torch.zeros(6, device="cuda", dtype=torch.int32)
        replay = ReplayState(factors, pending, replay_indices, scratch)
        expected_replay = ReplayState(
            factors.clone(), pending.clone(), replay_indices, scratch.clone()
        )

    actual = gated_delta_step(projected, gates, weights, pool, steps, True, True, replay)
    expected = gated_delta_step.reference(
        projected, gates, weights, expected_pool, steps, True, True, expected_replay
    )
    assert_within_ulp(actual, expected, element_ulp=2, rms_ulp=0.5)
    torch.testing.assert_close(pool.conv, expected_pool.conv, rtol=0, atol=0)
    torch.testing.assert_close(pool.temporal, expected_pool.temporal, rtol=0.006, atol=2e-5)

    if replay is not None:
        accepted_positions = torch.tensor([2, 1], device="cuda", dtype=torch.int32)
        source = torch.arange(batch, device="cuda", dtype=torch.int32)
        accepted = ConvAcceptance(source, indices, accepted_positions + 1, replay.slots, None)
        accept_convolution(
            pool.conv.unsqueeze(0), replay.raw_conv.unsqueeze(0), accepted, replay.pending
        )
        accept_convolution.reference(
            expected_pool.conv.unsqueeze(0),
            expected_replay.raw_conv.unsqueeze(0),
            accepted,
            expected_replay.pending,
        )
        projected = _rand((batch, 12288), 0.25)
        gates = _rand((batch, 64), 0.5)
        actual = gated_delta_step(projected, gates, weights, pool, 1, False, True, replay)
        expected = gated_delta_step.reference(
            projected, gates, weights, expected_pool, 1, False, True, expected_replay
        )
        assert_within_ulp(actual, expected, element_ulp=2, rms_ulp=0.5)
        torch.testing.assert_close(pool.conv, expected_pool.conv, rtol=0, atol=0)
        torch.testing.assert_close(pool.temporal, expected_pool.temporal, rtol=0.006, atol=2e-5)


@pytest.mark.parametrize("cell", dense_projection.CELLS, ids=lambda cell: cell.why)
def test_dense_projection(cell, certifying) -> None:
    _begin()
    certifying(dense_projection)
    rows, columns, width = (cell.spec[name] for name in ("rows", "columns", "width"))
    x = fp8.quant(_rand((rows, width), 0.1), True)
    w = _rand((columns, width), 32).to(FP8)
    scales = torch.full((columns // 128, width // 128), 0.001953125, device="cuda")
    mx = mx_decode.swizzle(scales, columns, width, row_group=128)
    packed = dg_prefill.encode(scales, columns, width, row_group=128)
    weight = DenseWeight(w, scales, mx, packed)
    decode_rows = cell.spec.get("decode_rows", 0)
    actual = dense_projection(x, weight, decode_rows)
    expected = dense_projection.reference(x, weight, decode_rows)
    assert_within_ulp(actual, expected, element_ulp=2, rms_ulp=0.5)
    if decode_rows:
        solo = dense_projection(tuple(t[-decode_rows:] for t in x), weight, decode_rows)
        torch.testing.assert_close(actual[-decode_rows:], solo, rtol=0, atol=0)


@pytest.mark.parametrize("cell", routed_experts.CELLS, ids=lambda cell: cell.why)
def test_routed_experts(cell, certifying) -> None:
    _begin()
    certifying(routed_experts)
    rows = cell.spec["rows"]
    incoming = fp8.quant(_rand((rows, 2048), 0.1))
    route = torch.stack([torch.randperm(256, device="cuda")[:8] for _ in range(rows)]).int()
    up = torch.empty((256, 1024, 2048), device="cuda", dtype=FP8)
    down = torch.empty((256, 2048, 512), device="cuda", dtype=FP8)
    for expert in route.unique().tolist():
        up[expert].copy_(_rand((1024, 2048), 32))
        down[expert].copy_(_rand((2048, 512), 32))
    up_scales = torch.rand((256, 8, 16), device="cuda") / 512 + 0.001
    down_scales = torch.rand((256, 16, 4), device="cuda") / 512 + 0.001
    weights = RoutedWeights((up, up_scales), (down, down_scales))
    actual, inverse = routed_experts(incoming, route, weights, rows == 1, None, True)
    if inverse is not None:
        actual = actual[inverse.flatten().long()]
    expected = routed_experts.reference(incoming, route, weights, rows == 1, None, True)
    assert_within_ulp(actual, expected, element_ulp=5, rms_ulp=1.5)


@pytest.mark.parametrize("cell", project_and_store.CELLS, ids=lambda cell: cell.why)
def test_qkv_rope_paged_store(cell, certifying) -> None:
    _begin()
    certifying(project_and_store)
    context_rows = cell.spec["context_rows"]
    projected = _rand((3, 9216), 0.2)
    q_norm, k_norm = _rand((256,), 0.05), _rand((256,), 0.05)
    angle = torch.rand((64, 32), device="cuda")
    rope = torch.cat((angle.cos(), angle.sin()), dim=-1)
    positions = torch.tensor([33, 9, 63], device="cuda", dtype=torch.int32)
    requests = torch.tensor([2, 1, 0], device="cuda", dtype=torch.int32)
    blocks = torch.tensor([[6, 2], [4, 16], [12, 10]], device="cuda", dtype=torch.int32)
    table = torch.stack((blocks, blocks + 1), dim=1)
    cache = torch.zeros((10, 2, 32, 2, 256), device="cuda", dtype=BF16)
    expected_cache = cache.clone()
    actual = project_and_store(
        projected,
        q_norm,
        k_norm,
        rope,
        positions,
        requests,
        KvPages(cache, table),
        context_rows,
        not context_rows,
    )
    expected = project_and_store.reference(
        projected,
        q_norm,
        k_norm,
        rope,
        positions,
        requests,
        KvPages(expected_cache, table),
        context_rows,
        not context_rows,
    )
    for output, reference in zip(actual, expected, strict=True):
        if reference.numel():
            assert_within_ulp(output, reference, element_ulp=2, rms_ulp=0.5)
    assert_within_ulp(cache.flatten(0, 3), expected_cache.flatten(0, 3), element_ulp=2, rms_ulp=0.5)


@pytest.mark.parametrize("cell", context_gdn.CELLS, ids=lambda cell: cell.why)
def test_packed_context_gdn(cell, certifying) -> None:
    _begin()
    certifying(context_gdn)
    lengths = cell.spec["lengths"]
    initial = cell.spec["initial"]
    rows = sum(lengths)
    slots = torch.tensor((4, 1)[: len(lengths)], device="cuda", dtype=torch.int32)
    projected = _rand((rows, 12288), 0.25)
    ba = _rand((rows, 64), 0.5)
    weights = GdnWeights(_rand((8192, 4), 0.25), _rand((32,), 0.5), _rand((32,), 0.5))
    history = _rand((6, 8192, 3), 0.25)
    state = _rand((6, 32, 128, 128), 0.01).float()
    has_initial = torch.tensor(initial, device="cuda")
    pool = StatePool(history, state, slots, has_initial)
    ref_pool = StatePool(history.clone(), state.clone(), slots, has_initial)
    cu_cpu = tuple(accumulate(lengths, initial=0))
    cu = torch.tensor(cu_cpu, device="cuda", dtype=torch.int64)
    requests = torch.tensor(
        [req for req, length in enumerate(lengths) for _ in range(length)],
        device="cuda",
        dtype=torch.int32,
    )
    actual = context_gdn(projected, ba, weights, pool, requests, cu, cu_cpu, any(initial))
    expected = context_gdn.reference(
        projected, ba, weights, ref_pool, requests, cu, cu_cpu, any(initial)
    )
    # The chunk scan uses BF16 dot products while the reference accumulates sequentially in FP32.
    assert_within_ulp(actual, expected, element_ulp=5, rms_ulp=1.5)
    torch.testing.assert_close(pool.conv, ref_pool.conv, rtol=0, atol=0)
    # A wrong slot or missed state reset would leave order-one error in the final state.
    error = (pool.temporal[slots] - ref_pool.temporal[slots]).norm()
    assert error / ref_pool.temporal[slots].norm() < 0.008
    untouched = torch.ones(6, dtype=torch.bool, device="cuda")
    untouched[slots] = False
    torch.testing.assert_close(
        pool.temporal[untouched], ref_pool.temporal[untouched], rtol=0, atol=0
    )


@pytest.mark.parametrize("cell", draft_input.CELLS, ids=lambda cell: cell.why)
def test_mtp_draft_input(cell, certifying) -> None:
    _begin()
    certifying(draft_input)
    rows = cell.spec["rows"]
    ids = torch.tensor([3, 2, 7, 3][:rows], device="cuda", dtype=torch.int32)
    embed = _rand((11, 2048))
    hidden = _rand((rows, 2048))
    embedding_norm, hidden_norm = _rand((2048,), 0.05), _rand((2048,), 0.05)
    actual = draft_input(ids, embed, hidden, embedding_norm, hidden_norm)
    expected = draft_input.reference(ids, embed, hidden, embedding_norm, hidden_norm)
    assert_within_ulp(actual, expected, element_ulp=2, rms_ulp=0.5)


@pytest.mark.parametrize("cell", draft_argmax.CELLS, ids=lambda cell: cell.why)
def test_mtp_draft_argmax(cell, certifying) -> None:
    _begin()
    certifying(draft_argmax)
    rows, width = (cell.spec[name] for name in ("rows", "width"))
    logits = -_rand((rows, width), 0.25).abs() - 1
    logits[:, 1025] = 0
    logits[:, -1] = -0.0
    if rows > 1:
        logits[1].fill_(float("-inf"))
        logits[2] -= 1
    out = torch.empty(rows, device="cuda", dtype=torch.int32)
    assert draft_argmax(logits, out) is out
    torch.testing.assert_close(out, draft_argmax.reference(logits).int())


@pytest.mark.parametrize("cell", rowwise_projection.CELLS, ids=lambda cell: cell.why)
def test_rowwise_projection(cell, certifying) -> None:
    _begin()
    certifying(rowwise_projection)
    rows, columns = (cell.spec[name] for name in ("rows", "columns"))
    hidden = _rand((rows, 2048), 0.4)
    weight = _rand((columns, 2048), 0.1)
    packed = rowwise_projection(hidden, weight)
    solo = rowwise_projection(hidden[-1:], weight, True)
    torch.testing.assert_close(packed[-1:], solo, rtol=0, atol=0)
    expected = rowwise_projection.reference(hidden, weight)
    assert_within_ulp(packed, expected, element_ulp=2, rms_ulp=0.5)


@pytest.mark.parametrize("cell", context_projection.CELLS, ids=lambda cell: cell.why)
def test_context_projection(cell, certifying) -> None:
    _begin()
    certifying(context_projection)
    length, batch, columns = (cell.spec[name] for name in ("length", "batch", "columns"))
    hidden = _rand((batch * length, 2048), 0.4)
    weight = _rand((columns, 2048), 0.1)
    packed = context_projection(hidden, weight, length)
    expected = context_projection.reference(hidden, weight, length)
    torch.testing.assert_close(packed, expected, rtol=0, atol=0)


@pytest.mark.parametrize("cell", accept_convolution.CELLS, ids=lambda cell: cell.why)
def test_accepted_convolution(cell, certifying) -> None:
    _begin()
    certifying(accept_convolution)
    counts = cell.spec["accepted"]
    n = len(counts)
    history = _rand((2, 7, 8192, 3))
    raw = _rand((2, n, 4, 8192))
    pending = torch.zeros(11, device="cuda", dtype=torch.int32)[1::2]
    expected_history = history.clone()
    expected_pending = pending.clone()
    source = torch.arange(n, device="cuda", dtype=torch.int32).flip(0)
    destinations = torch.tensor([5, 3, 1][:n], device="cuda", dtype=torch.int32)
    slots = torch.tensor([4, 1, 3][:n], device="cuda", dtype=torch.int32)
    dummy = torch.tensor([False, False, True], device="cuda") if n == 3 else None
    count = torch.tensor(counts, device="cuda", dtype=torch.int32)
    batch = ConvAcceptance(source, destinations, count, slots, dummy)
    accept_convolution(history, raw, batch, pending)
    accept_convolution.reference(expected_history, raw, batch, expected_pending)
    torch.testing.assert_close(history, expected_history, rtol=0, atol=0)
    torch.testing.assert_close(pending, expected_pending, rtol=0, atol=0)

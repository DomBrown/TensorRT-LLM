# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Split-KV BF16 decode from the reference engine, with runtime page addresses."""

import torch
import triton
import triton.language as tl
from triton.experimental import gluon
from triton.experimental.gluon import language as gl

from .kernels import FP8, _scales

_HEAD_DIM = 256
_Q_HEADS = 16
_KV_HEADS = 2
# Two narrow decode splits preserve their reduction order at wide batch sizes.
_SHORT_HISTORY = 256


@triton.jit
def _append_decode(
    QKV,
    POOL,
    POS,
    BLOCKS,
    BLOCK_ROW_STRIDE: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    PAGE_STRIDE: tl.constexpr,
    WIDTH: tl.constexpr,
    QUERY_ROWS: tl.constexpr,
):
    row = tl.program_id(0)
    head = tl.program_id(1)
    d = tl.arange(0, 256)
    position = tl.load(POS + row)
    page = tl.load(BLOCKS + (row // QUERY_ROWS) * BLOCK_ROW_STRIDE + position // PAGE_SIZE).to(
        tl.int64
    )
    page_offset = page * PAGE_STRIDE
    offset = (head * PAGE_SIZE + position % PAGE_SIZE) * 256 + d
    k = tl.load(QKV + row * WIDTH + 4096 + head * 256 + d)
    v = tl.load(QKV + row * WIDTH + 4608 + head * 256 + d)
    tl.store(POOL + page_offset + offset, k, page >= 0)
    tl.store(POOL + (page_offset + PAGE_STRIDE) + offset, v, page >= 0)


@triton.jit
def _decode_attention(
    QKV,
    POOL,
    POS,
    BLOCKS,
    PART,
    LSE,
    PICKS,
    BLOCK_ROW_STRIDE: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    PAGE_STRIDE: tl.constexpr,
    SPLITS: tl.constexpr,
    SPLIT_LEN: tl.constexpr,
    WIDTH: tl.constexpr,
    QUERY_ROWS: tl.constexpr,
    GROUP_ROWS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    SHORT_LIMIT: tl.constexpr = 0,
    SELECT_ROWS: tl.constexpr = False,
):
    first_output = tl.program_id(0) * GROUP_ROWS
    first_row = tl.load(PICKS + first_output) if SELECT_ROWS else first_output
    sequence = first_row // QUERY_ROWS
    kv_head = tl.program_id(1)
    split = tl.program_id(2)
    start = split * SPLIT_LEN
    last_end = tl.load(POS + first_row + GROUP_ROWS - 1) + 1
    if start >= last_end:
        return
    if SHORT_LIMIT:
        if last_end > SHORT_LIMIT:
            return
    channel = tl.arange(0, 256)
    lane = tl.arange(0, BLOCK_M)
    rank = lane // 8
    head = kv_head * 8 + lane % 8
    row = first_row + rank
    end = tl.load(POS + row, rank < GROUP_ROWS, 0) + 1
    valid = (rank < GROUP_ROWS) & (start < end)
    q = tl.load(
        QKV + row[:, None] * WIDTH + head[:, None] * 256 + channel[None, :],
        valid[:, None],
        0,
    )
    highest = tl.full((BLOCK_M,), -float("inf"), tl.float32)
    denom = tl.full((BLOCK_M,), 0.0, tl.float32)
    acc = tl.full((BLOCK_M, 256), 0.0, tl.float32)
    for key_start in range(start, tl.minimum(last_end, start + SPLIT_LEN), 32):
        position = key_start + tl.arange(0, 32)
        page = tl.load(
            BLOCKS + sequence * BLOCK_ROW_STRIDE + position // PAGE_SIZE,
            position < last_end,
            -1,
        ).to(tl.int64)
        address = page * PAGE_STRIDE + (kv_head * PAGE_SIZE + position % PAGE_SIZE) * 256
        live = (position < last_end) & (page >= 0)
        key = tl.load(POOL + address[None, :] + channel[:, None], live[None, :], 0)
        value = tl.load(POOL + address[:, None] + PAGE_STRIDE + channel[None, :], live[:, None], 0)
        scores = tl.dot(q, key) * 0.09016844005556021  # log2(e) / sqrt(256)
        scores = tl.where(
            valid[:, None] & (position[None, :] < end[:, None]), scores, -float("inf")
        )
        next_high = tl.where(valid, tl.maximum(highest, tl.max(scores, 1)), 0.0)
        old = tl.exp2(highest - next_high)
        probability = tl.exp2(scores - next_high[:, None])
        denom = denom * old + tl.sum(probability, 1)
        acc = tl.dot(probability.to(value.dtype), value, acc * old[:, None])
        highest = next_high
    index = ((first_output + rank) * SPLITS + split) * 16 + head
    tl.store(PART + index[:, None] * 256 + channel[None, :], acc / denom[:, None], valid[:, None])
    tl.store(LSE + index, highest + tl.log2(denom), valid)


@triton.jit
def _merge(PART, LSE, POS, OUTPUT, SPLITS: tl.constexpr, NS: tl.constexpr, SPLIT_LEN: tl.constexpr):
    row = tl.program_id(0)
    head = tl.program_id(1)
    end = tl.load(POS + row) + 1
    s = tl.arange(0, NS)
    d = tl.program_id(2) * 64 + tl.arange(0, 64)
    live = (s < SPLITS) & (s * SPLIT_LEN < end)
    base = (row * SPLITS + s) * 16 + head
    lse = tl.load(LSE + base, live, -float("inf"))
    weight = tl.exp2(lse - tl.max(lse, 0))
    values = tl.load(PART + base[:, None] * 256 + d[None, :], live[:, None], 0)
    attn = tl.sum(values * weight[:, None], 0) / tl.sum(weight, 0)
    tl.store(OUTPUT + (row * 16 + head) * 256 + d, attn)


@gluon.jit
def _merged_core(
    PART,
    LSE,
    row,
    head,
    group,
    end,
    SPLITS: gl.constexpr,
    NS: gl.constexpr,
    SPLIT_LEN: gl.constexpr,
    vector: gl.constexpr,
):
    # Match the split-axis reduction of the BF16 merge before quantization.
    tile: gl.constexpr = gl.BlockedLayout([1, 4], [2, 16], [4, 1], [1, 0])
    split = gl.arange(0, NS, layout=gl.SliceLayout(1, tile))
    channel = group * 128 + gl.arange(0, 128, layout=gl.SliceLayout(0, tile))
    live = (split < SPLITS) & (split * SPLIT_LEN < end)
    base = (row * SPLITS + split) * 16 + head
    lse = gl.load(LSE + base, live, -float("inf"))
    lse_vector = gl.convert_layout(lse, vector)
    highest = gl.max(lse_vector, 0)
    weights = gl.exp2(lse - highest)
    denom = gl.sum(gl.exp2(lse_vector - highest), 0)
    values = gl.load(PART + base[:, None] * 256 + channel[None, :], live[:, None], 0)
    core = (gl.sum(values * weights[:, None], 0) / denom).to(gl.bfloat16).to(gl.float32)
    return gl.convert_layout(core, vector)


@gluon.jit
def _merge_quantize(
    PART,
    LSE,
    POS,
    PROJECTION,
    NATIVE,
    PICKS,
    OUTPUT,
    SCALES,
    SPLITS: gl.constexpr,
    NS: gl.constexpr,
    SPLIT_LEN: gl.constexpr,
    MN_MAJOR: gl.constexpr,
    SHORT_LIMIT: gl.constexpr,
    QUERY_ROWS: gl.constexpr,
    SELECT_ROWS: gl.constexpr,
):
    row = gl.program_id(0)
    source = gl.load(PICKS + row) if SELECT_ROWS else row
    head = gl.program_id(1)
    group = gl.program_id(2)
    end = gl.load(POS + source) + 1
    vector: gl.constexpr = gl.BlockedLayout([1], [32], [4], [0])
    output_channel = group * 128 + gl.arange(0, 128, layout=vector)
    if SHORT_LIMIT:
        last_end = gl.load(POS + (row // QUERY_ROWS + 1) * QUERY_ROWS - 1) + 1
        if last_end > SHORT_LIMIT:
            core = gl.load(NATIVE + (row * 16 + head) * 256 + output_channel).to(gl.float32)
        else:
            core = _merged_core(PART, LSE, row, head, group, end, SPLITS, NS, SPLIT_LEN, vector)
    else:
        core = _merged_core(PART, LSE, row, head, group, end, SPLITS, NS, SPLIT_LEN, vector)
    z = gl.load(PROJECTION + source * 9216 + head * 512 + 256 + output_channel).to(gl.float32)
    gate = 1.0 / (1.0 + gl.exp(-z))
    x = (core * gate).to(gl.bfloat16).to(gl.float32)
    maximum = gl.maximum(gl.max(gl.abs(x), 0), 1e-10)
    scale = maximum * (1.0 / 448.0)
    quant = gl.clamp(x * (448.0 / maximum), -448.0, 448.0).to(gl.float8e4nv)
    scale_column = head * 2 + group
    scale_offset = scale_column * gl.num_programs(0) + row if MN_MAJOR else row * 32 + scale_column
    gl.store(OUTPUT + (row * 16 + head) * 256 + output_channel, quant)
    gl.store(SCALES + scale_offset, scale)


def paged_decode(
    qkv: torch.Tensor,
    pool: torch.Tensor,
    positions: torch.Tensor,
    key_pages: torch.Tensor,
    max_seq_len: int,
    query_rows: int = 1,
    *,
    projection: torch.Tensor | None = None,
    mn_major: bool = False,
    native_output: torch.Tensor | None = None,
    pick_rows: torch.Tensor | None = None,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Attend over HND pages, reusing the native kernel's writes if supplied."""
    rows, width = qkv.shape
    page_size = pool.shape[-2]
    stride = pool.stride(1)
    short_limit = _SHORT_HISTORY if native_output is not None else 0
    split_length = 128 if rows == query_rows or short_limit else 256
    max_keys = min(max_seq_len, short_limit) if short_limit else max_seq_len
    splits = triton.cdiv(max_keys, split_length)
    output_rows = rows if pick_rows is None else pick_rows.numel()
    partial = torch.empty(
        (output_rows, splits, _Q_HEADS, _HEAD_DIM), device=qkv.device, dtype=torch.float32
    )
    lse = torch.empty((output_rows, splits, _Q_HEADS), device=qkv.device, dtype=torch.float32)
    if native_output is None:
        _append_decode[(rows, _KV_HEADS)](
            qkv,
            pool,
            positions,
            key_pages,
            key_pages.stride(0),
            page_size,
            stride,
            width,
            query_rows,
            num_warps=4,
        )
    # A singleton uses the narrower query tiles to maximize parallelism.
    group_rows = query_rows if rows > query_rows and pick_rows is None else 1
    _decode_attention[(output_rows // group_rows, _KV_HEADS, splits)](
        qkv,
        pool,
        positions,
        key_pages,
        partial,
        lse,
        pick_rows,
        key_pages.stride(0),
        page_size,
        stride,
        splits,
        split_length,
        width,
        query_rows,
        group_rows,
        max(16, triton.next_power_of_2(group_rows * 8)),
        short_limit,
        pick_rows is not None,
        num_warps=4,
    )
    output_width = _Q_HEADS * _HEAD_DIM
    if projection is not None:
        out = qkv.new_empty((output_rows, output_width), dtype=FP8)
        scales = _scales(output_rows, output_width // 128, qkv.device, mn_major)
        _merge_quantize[(output_rows, _Q_HEADS, _HEAD_DIM // 128)](
            partial,
            lse,
            positions,
            projection,
            native_output,
            pick_rows,
            out,
            scales,
            splits,
            triton.next_power_of_2(splits),
            split_length,
            mn_major,
            short_limit,
            query_rows,
            pick_rows is not None,
            num_warps=4,
        )
        return out, scales
    out = qkv.new_empty((output_rows, output_width))
    _merge[(rows, _Q_HEADS, _HEAD_DIM // 64)](
        partial,
        lse,
        positions,
        out,
        splits,
        triton.next_power_of_2(splits),
        split_length,
        num_warps=4,
    )
    return out

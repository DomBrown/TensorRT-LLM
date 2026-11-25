# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Gated DeltaNet's depthwise SiLU convolution over manager-owned history."""

import torch
import triton
import triton.language as tl


@triton.jit
def _history_input(
    raw,
    history,
    row,
    channel,
    local_pos,
    step,
    has_prefix,
    width: tl.constexpr,
    channel_stride: tl.constexpr,
    time_stride: tl.constexpr,
):
    x = tl.load(raw + (row - step) * width + channel, local_pos >= step, 0).to(tl.float32)
    prev = tl.load(
        history + channel * channel_stride + (local_pos + 3 - step) * time_stride,
        (local_pos < step) & has_prefix,
        0,
    ).to(tl.float32)
    return tl.where(local_pos >= step, x, prev)


@triton.jit
def _conv(
    raw,
    weights,
    output,
    history,
    slots,
    offsets,
    row_map,
    initial,
    WIDTH: tl.constexpr,
    SLOT_STRIDE: tl.constexpr,
    CHANNEL_STRIDE: tl.constexpr,
    TIME_STRIDE: tl.constexpr,
    PREFILL: tl.constexpr,
    SPLIT: tl.constexpr,
    TILE: tl.constexpr,
):
    row = tl.program_id(0)
    channel = tl.program_id(1) * TILE + tl.arange(0, TILE)
    request = tl.load(row_map + row) if PREFILL else row
    slot = tl.load(slots + request).to(tl.int64)
    history += slot * SLOT_STRIDE
    pos = row - tl.load(offsets + request) if PREFILL else 0
    has_prefix = tl.load(initial + request) if PREFILL else True

    x0 = _history_input(
        raw, history, row, channel, pos, 3, has_prefix, WIDTH, CHANNEL_STRIDE, TIME_STRIDE
    )
    x1 = _history_input(
        raw, history, row, channel, pos, 2, has_prefix, WIDTH, CHANNEL_STRIDE, TIME_STRIDE
    )
    x2 = _history_input(
        raw, history, row, channel, pos, 1, has_prefix, WIDTH, CHANNEL_STRIDE, TIME_STRIDE
    )
    x3 = tl.load(raw + row * WIDTH + channel).to(tl.float32)
    w = weights + channel * 4
    value = x0 * tl.load(w).to(tl.float32)
    value = tl.fma(x1, tl.load(w + 1).to(tl.float32), value)
    value = tl.fma(x2, tl.load(w + 2).to(tl.float32), value)
    value = tl.fma(x3, tl.load(w + 3).to(tl.float32), value)
    value *= tl.sigmoid(value)
    if not PREFILL:
        ptr = history + channel * CHANNEL_STRIDE
        tl.store(ptr, x1)
        tl.store(ptr + TIME_STRIDE, x2)
        tl.store(ptr + 2 * TIME_STRIDE, x3)
    if SPLIT:
        section = tl.minimum(tl.program_id(1) * TILE // 2048, 2)
        section_width = tl.where(section == 2, 4096, 2048)
        output_index = section * tl.num_programs(0) * 2048 + row * section_width
        output_index += channel - section * 2048
    else:
        output_index = row * 8192 + channel
    tl.store(output + output_index, value)


@triton.jit
def _save_history(
    raw,
    history,
    slots,
    offsets,
    initial,
    WIDTH: tl.constexpr,
    SLOT_STRIDE: tl.constexpr,
    CHANNEL_STRIDE: tl.constexpr,
    TIME_STRIDE: tl.constexpr,
    TILE: tl.constexpr,
):
    request = tl.program_id(0)
    channel = tl.program_id(1) * TILE + tl.arange(0, TILE)
    slot = tl.load(slots + request).to(tl.int64)
    start = tl.load(offsets + request)
    end = tl.load(offsets + request + 1)
    has_prefix = tl.load(initial + request)
    if end > start:
        row = end - 1
        history += slot * SLOT_STRIDE
        x1 = _history_input(
            raw,
            history,
            row,
            channel,
            row - start,
            2,
            has_prefix,
            WIDTH,
            CHANNEL_STRIDE,
            TIME_STRIDE,
        )
        x2 = _history_input(
            raw,
            history,
            row,
            channel,
            row - start,
            1,
            has_prefix,
            WIDTH,
            CHANNEL_STRIDE,
            TIME_STRIDE,
        )
        x3 = tl.load(raw + row * WIDTH + channel)
        ptr = history + channel * CHANNEL_STRIDE
        tl.store(ptr, x1)
        tl.store(ptr + TIME_STRIDE, x2)
        tl.store(ptr + 2 * TIME_STRIDE, x3)


def convolution(
    raw: torch.Tensor,
    weights: torch.Tensor,
    history: torch.Tensor,
    slots: torch.Tensor,
    offsets: torch.Tensor | None,
    row_map: torch.Tensor | None,
    initial: torch.Tensor | None,
    split: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Mix [rows, 12288] projections against [slots, 8192, 3] conv history."""
    rows = raw.shape[0]
    mixed = torch.empty((rows, 8192), device=raw.device, dtype=raw.dtype)
    tile = 256
    _conv[(rows, 8192 // tile)](
        raw,
        weights,
        mixed,
        history,
        slots,
        offsets,
        row_map,
        initial,
        raw.shape[1],
        *history.stride(),
        offsets is not None,
        split,
        tile,
        num_warps=4,
    )
    if offsets is not None:
        # Every context row can read the previous window. Publish the new window
        # only after those reads have finished.
        _save_history[(slots.numel(), 8192 // tile)](
            raw,
            history,
            slots,
            offsets,
            initial,
            raw.shape[1],
            *history.stride(),
            tile,
            num_warps=4,
        )
    if split:
        q, k, v = mixed.flatten().split((rows * 2048, rows * 2048, rows * 4096))
        return q.view(rows, 16, 128), k.view(rows, 16, 128), v.view(rows, 32, 128)
    return mixed

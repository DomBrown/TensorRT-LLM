# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""QK normalization and compensated GQA adapted to runtime KV page tables."""

import triton as tr
import triton.language as tl


@tr.jit
def qkv_rope_paged(
    X,
    QW,
    KW,
    Rope,
    Positions,
    Requests,
    Pages,
    Cache,
    Q,
    K,
    V,
    Gate,
    TABLE_STRIDE: tl.constexpr,
    PAGE_STRIDE: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    CONTEXT_ROWS: tl.constexpr,
    PDL: tl.constexpr,
):
    """Normalize and rotate Q/K, write paged K/V and keep dense prefill K/V."""
    row, head = tl.program_id(0), tl.program_id(1)
    position = tl.load(Positions + row)
    offsets = tl.arange(0, 256)
    if head < 16:
        weights = tl.load(QW + offsets).to(tl.float32)
    else:
        weights = tl.load(KW + offsets).to(tl.float32)
    if PDL:
        tl.extra.cuda.gdc_wait()
        tl.extra.cuda.gdc_launch_dependents()
    base = row * 9216
    start = tl.where(head < 16, head * 512, 8192 + (head - 16) * 256)
    x = tl.load(X + base + start + offsets).to(tl.float32)
    norm = x * tl.rsqrt(tl.sum(x * x, 0) * (1.0 / 256.0) + 1.0e-6)
    norm = (norm * (1.0 + weights)).to(tl.bfloat16).to(tl.float32)
    other = tl.gather(norm, offsets ^ 32, 0)
    cos = tl.load(Rope + position * 64 + offsets % 32)
    sin = tl.load(Rope + position * 64 + 32 + offsets % 32)
    rotated = norm * cos + tl.where(offsets < 32, -other, other) * sin
    y = tl.where(offsets < 64, rotated, norm)
    if head < 16:
        gate = tl.load(X + base + head * 512 + 256 + offsets)
        destination = (row * 16 + head) * 256 + offsets
        tl.store(Q + destination, y)
        tl.store(Gate + destination, gate)
    else:
        value = tl.load(X + base + 8704 + (head - 16) * 256 + offsets)
        request = tl.load(Requests + row)
        page_offset = request * 2 * TABLE_STRIDE + position // PAGE_SIZE
        key_page = tl.load(Pages + page_offset).to(tl.int64)
        value_page = tl.load(Pages + page_offset + TABLE_STRIDE).to(tl.int64)
        within = (position % PAGE_SIZE * 2 + head - 16) * 256 + offsets
        tl.store(Cache + key_page * PAGE_STRIDE + within, y, key_page >= 0)
        tl.store(Cache + value_page * PAGE_STRIDE + within, value, value_page >= 0)
        if row < CONTEXT_ROWS:
            packed = (row * 2 + head - 16) * 256 + offsets
            tl.store(K + packed, y)
            tl.store(V + packed, value)


@tr.jit
def _attention_slice(
    Q,
    Pages,
    Cache,
    Part,
    Lse,
    end,
    request,
    TABLE_STRIDE: tl.constexpr,
    PAGE_STRIDE: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    SPLITS: tl.constexpr,
    ACTIVE_SPLITS: tl.constexpr,
    TILE: tl.constexpr,
):
    kv, split, row = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    range_size = tl.cdiv(end, ACTIVE_SPLITS * TILE) * TILE
    start = split * range_size
    stop = tl.minimum(end, start + range_size)
    heads = tl.arange(0, 16)
    channels = tl.arange(0, 256)
    head = kv * 8 + heads
    local = heads < 8
    if split < ACTIVE_SPLITS and start < stop:
        q = tl.load(Q + (row * 16 + head[:, None]) * 256 + channels[None, :], local[:, None], 0)
        peak = tl.full((16,), -float("inf"), tl.float32)
        integral = tl.full((16,), 0.0, tl.float32)
        acc = tl.full((16, 256), 0.0, tl.float32)
        for pos in range(start, stop, TILE):
            token = pos + tl.arange(0, TILE)
            table = request * 2 * TABLE_STRIDE + token // PAGE_SIZE
            key_page = tl.load(Pages + table, token < stop, 0).to(tl.int64)
            value_page = tl.load(Pages + table + TABLE_STRIDE, token < stop, 0).to(tl.int64)
            within = (token % PAGE_SIZE * 2 + kv) * 256
            key = tl.load(
                Cache + (key_page * PAGE_STRIDE + within)[:, None] + channels[None, :],
                (token[:, None] < stop) & (key_page[:, None] >= 0),
                0,
            )
            value = tl.load(
                Cache + (value_page * PAGE_STRIDE + within)[:, None] + channels[None, :],
                (token[:, None] < stop) & (value_page[:, None] >= 0),
                0,
            )
            scores = tl.dot(q, tl.trans(key)) * (1.0 / 16.0)
            scores = tl.where(token[None, :] < stop, scores, -float("inf"))
            top = tl.maximum(peak, tl.max(scores, 1))
            probs = tl.exp(scores - top[:, None])
            old = tl.exp(peak - top)
            high = probs.to(tl.bfloat16)
            low = (probs - high.to(tl.float32)).to(tl.bfloat16)
            pv = tl.dot(low, value, tl.dot(high, value))
            acc = acc * old[:, None] + pv
            integral = integral * old + tl.sum(probs, 1)
            peak = top
        tl.store(
            Part + ((row * 16 + head[:, None]) * SPLITS + split) * 256 + channels[None, :],
            acc / integral[:, None],
            local[:, None],
        )
        tl.store(Lse + (row * 16 + head) * SPLITS + split, peak + tl.log(integral), local)
    else:
        tl.store(Lse + (row * 16 + head) * SPLITS + split, -float("inf"), local)


@tr.jit
def attention_paged(
    Q,
    Pages,
    Cache,
    Positions,
    Requests,
    Part,
    Lse,
    TABLE_STRIDE: tl.constexpr,
    PAGE_STRIDE: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    SPLITS: tl.constexpr,
    PDL: tl.constexpr,
):
    row = tl.program_id(2)
    end = tl.load(Positions + row) + 1
    request = tl.load(Requests + row)
    if PDL:
        tl.extra.cuda.gdc_wait()
        tl.extra.cuda.gdc_launch_dependents()
    if end > 2048:
        _attention_slice(
            Q,
            Pages,
            Cache,
            Part,
            Lse,
            end,
            request,
            TABLE_STRIDE,
            PAGE_STRIDE,
            PAGE_SIZE,
            SPLITS,
            min(SPLITS, 16),
            64,
        )
    else:
        _attention_slice(
            Q,
            Pages,
            Cache,
            Part,
            Lse,
            end,
            request,
            TABLE_STRIDE,
            PAGE_STRIDE,
            PAGE_SIZE,
            SPLITS,
            SPLITS,
            32,
        )


@tr.jit
def gather_paged(
    Pages,
    Cache,
    K,
    V,
    request,
    TABLE_STRIDE: tl.constexpr,
    PAGE_STRIDE: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    LENGTH: tl.constexpr,
    TILE: tl.constexpr,
):
    offset = tl.program_id(0) * TILE + tl.arange(0, TILE)
    token = offset // 512
    column = offset % 512
    table = request * 2 * TABLE_STRIDE + token // PAGE_SIZE
    key_page = tl.load(Pages + table, token < LENGTH, 0).to(tl.int64)
    value_page = tl.load(Pages + table + TABLE_STRIDE, token < LENGTH, 0).to(tl.int64)
    within = token % PAGE_SIZE * 512 + column
    k = tl.load(Cache + key_page * PAGE_STRIDE + within, token < LENGTH, 0)
    v = tl.load(Cache + value_page * PAGE_STRIDE + within, token < LENGTH, 0)
    tl.store(K + offset, k, token < LENGTH)
    tl.store(V + offset, v, token < LENGTH)

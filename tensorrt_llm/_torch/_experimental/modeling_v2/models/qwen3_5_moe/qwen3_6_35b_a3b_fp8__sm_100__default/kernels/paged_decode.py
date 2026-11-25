# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Split-KV BF16 decode from the reference engine, with runtime page addresses."""

import torch
import triton
import triton.language as tl

_HEAD_DIM = 256
_Q_HEADS = 16
_KV_HEADS = 2


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
    BLOCK_ROW_STRIDE: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    PAGE_STRIDE: tl.constexpr,
    SPLITS: tl.constexpr,
    BN: tl.constexpr,
    SPLIT_LEN: tl.constexpr,
    WIDTH: tl.constexpr,
    QUERY_ROWS: tl.constexpr,
):
    row = tl.program_id(0)
    kv_head = tl.program_id(1)
    split = tl.program_id(2)
    end = tl.load(POS + row) + 1
    start = split * SPLIT_LEN
    d = tl.arange(0, 256)
    m = tl.arange(0, 16)
    n = tl.arange(0, BN)
    if start < end:
        q = tl.load(
            QKV + row * WIDTH + (kv_head * 8 + m[:, None]) * 256 + d[None, :],
            m[:, None] < 8,
            0,
        )
        high = tl.full((16,), -float("inf"), tl.float32)
        den = tl.full((16,), 0.0, tl.float32)
        acc = tl.full((16, 256), 0.0, tl.float32)
        for idx in range(start, tl.minimum(end, start + SPLIT_LEN), BN):
            p = idx + n
            page = tl.load(
                BLOCKS + (row // QUERY_ROWS) * BLOCK_ROW_STRIDE + p // PAGE_SIZE, p < end, -1
            ).to(tl.int64)
            address = page * PAGE_STRIDE + (kv_head * PAGE_SIZE + p % PAGE_SIZE) * 256
            live = (p < end) & (page >= 0)
            k = tl.load(POOL + address[None, :] + d[:, None], live[None, :], 0)
            v = tl.load(POOL + address[:, None] + PAGE_STRIDE + d[None, :], live[:, None], 0)
            score = tl.dot(q, k) * 0.09016844005556021  # log2(e) / sqrt(256)
            score = tl.where(p[None, :] < end, score, -float("inf"))
            hi = tl.maximum(high, tl.max(score, 1))
            prior = tl.exp2(high - hi)
            probability = tl.exp2(score - hi[:, None])
            den = den * prior + tl.sum(probability, 1)
            acc = tl.dot(probability.to(v.dtype), v, acc * prior[:, None])
            high = hi
        head = kv_head * 8 + m
        base = (row * SPLITS + split) * 16 + head
        tl.store(PART + base[:, None] * 256 + d[None, :], acc / den[:, None], m[:, None] < 8)
        tl.store(LSE + base, high + tl.log2(den), m < 8)


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


def paged_decode(
    qkv: torch.Tensor,
    pool: torch.Tensor,
    positions: torch.Tensor,
    key_pages: torch.Tensor,
    max_seq_len: int,
    query_rows: int = 1,
) -> torch.Tensor:
    """Append and read packed QKV against paired HND K/V pages."""
    rows, width = qkv.shape
    page_size = pool.shape[-2]
    stride = pool.stride(1)
    split_length = 128 if rows == query_rows else 256
    splits = triton.cdiv(max_seq_len, split_length)
    partial = torch.empty(
        (rows, splits, _Q_HEADS, _HEAD_DIM), device=qkv.device, dtype=torch.float32
    )
    lse = torch.empty((rows, splits, _Q_HEADS), device=qkv.device, dtype=torch.float32)
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
    _decode_attention[(rows, _KV_HEADS, splits)](
        qkv,
        pool,
        positions,
        key_pages,
        partial,
        lse,
        key_pages.stride(0),
        page_size,
        stride,
        splits,
        32,
        split_length,
        width,
        query_rows,
        num_warps=4,
    )
    out = qkv.new_empty((rows, _Q_HEADS * _HEAD_DIM))
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

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Stable BF16 vocabulary reduction with a device-resident shortlist length."""

import torch
import triton
import triton.language as tl

_INDEX_MASK = tl.constexpr((1 << 18) - 1)
_BLOCK = 1024


@triton.jit
def _partial(LOGITS, LENGTH, PARTIALS, STRIDE: tl.constexpr, TILES: tl.constexpr):
    tile = tl.program_id(0)
    row = tl.program_id(1)
    local = tl.arange(0, 1024)
    index = tile * 1024 + local
    valid = index < tl.load(LENGTH)
    value = tl.load(LOGITS + row * STRIDE + index, valid, 0)
    bits = value.to(tl.uint16, bitcast=True).to(tl.uint32)
    bits = tl.where((bits & 32767) == 0, 0, bits)
    order = bits ^ tl.where((bits & 32768) != 0, 65535, 32768).to(tl.uint32)
    is_nan = ((bits & 32640) == 32640) & ((bits & 127) != 0)
    order = tl.where(is_nan, 65535, order)
    key = order * 1024 + (1023 - local).to(tl.uint32)
    best = tl.max(tl.where(valid, key, 0), 0)
    winner = tile * 1024 + 1023 - (best & 1023)
    packed = ((best >> 10).to(tl.uint64) << 18) | (_INDEX_MASK - winner).to(tl.uint64)
    tl.store(PARTIALS + row * TILES + tile, packed)


@triton.jit
def _tokens(PARTIALS, IDS, OUTPUT, TILES: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    parts = tl.arange(0, BLOCK)
    key = tl.max(tl.load(PARTIALS + row * TILES + parts, parts < TILES, 0), 0)
    index = _INDEX_MASK - (key & _INDEX_MASK)
    tl.store(OUTPUT + row, tl.load(IDS + index))


def shortlist_argmax(logits, ids, length, out):
    rows, columns = logits.shape
    tiles = triton.cdiv(columns, _BLOCK)
    partials = torch.empty((rows, tiles), dtype=torch.uint64, device=logits.device)
    if out is None:
        out = torch.empty(rows, dtype=torch.int32, device=logits.device)
    _partial[(tiles, rows)](logits, length, partials, logits.stride(0), tiles, num_warps=4)
    _tokens[(rows,)](partials, ids, out, tiles, triton.next_power_of_2(tiles), num_warps=1)
    return out

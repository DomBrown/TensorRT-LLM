# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Fused MTP embedding and trunk-state normalization."""

import triton as tr
import triton.language as tl


@tr.jit
def draft_input(Ids, Embed, Hidden, WE, WH, X):
    row = tl.program_id(0)
    offsets = tl.arange(0, 2048)
    token = tl.load(Ids + row)
    e = tl.load(Embed + token * 2048 + offsets).to(tl.float32)
    h = tl.load(Hidden + row * 2048 + offsets).to(tl.float32)
    ew = 1.0 + tl.load(WE + offsets).to(tl.float32)
    hw = 1.0 + tl.load(WH + offsets).to(tl.float32)
    es = tl.rsqrt(tl.sum(e * e, 0) * (1.0 / 2048.0) + 1.0e-6)
    hs = tl.rsqrt(tl.sum(h * h, 0) * (1.0 / 2048.0) + 1.0e-6)
    tl.store(X + row * 4096 + offsets, (e * es) * ew)
    tl.store(X + row * 4096 + 2048 + offsets, (h * hs) * hw)


@tr.jit
def draft_max_tiles(Logits, Partial, WIDTH: tl.constexpr, TILES: tl.constexpr, B: tl.constexpr):
    tile, row = tl.program_id(0), tl.program_id(1)
    index = tile * B + tl.arange(0, B)
    value = tl.load(Logits + row * WIDTH + index, index < WIDTH, -float("inf"))
    bits = value.to(tl.uint16, bitcast=True).to(tl.uint32)
    bits = tl.where(value == 0.0, 0, bits)
    rank = tl.where(bits >= 32768, 65535 - bits, bits + 32768)
    key = (rank << 16) | (65535 - index.to(tl.uint32))
    tl.store(Partial + row * TILES + tile, tl.max(tl.where(index < WIDTH, key, 0), 0))


@tr.jit
def draft_max_finish(Partial, Tokens, TILES: tl.constexpr, B: tl.constexpr):
    row = tl.program_id(0)
    tiles = tl.arange(0, B)
    key = tl.max(tl.load(Partial + row * TILES + tiles, tiles < TILES, 0), 0)
    tl.store(Tokens + row, 65535 - (key & 65535))

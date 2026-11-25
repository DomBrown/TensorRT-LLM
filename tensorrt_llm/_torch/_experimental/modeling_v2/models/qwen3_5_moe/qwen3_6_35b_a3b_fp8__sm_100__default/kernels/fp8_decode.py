# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Small-row SM100 E4M3 GEMMs with pipelined TMA weight loads."""

from functools import lru_cache

import torch
import triton
import triton.language as tl
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.tools.tensor_descriptor import TensorDescriptor

ROUTE_BLOCK = 8


@lru_cache(maxsize=640)
def weight_tiles(weight, rows):
    matrix = weight.view(-1, weight.shape[-1])
    return TensorDescriptor.from_tensor(matrix, [rows, 128])


@triton.jit(do_not_specialize=["M"])
def _transposed_dot(
    A,
    AS,
    W,
    WS,
    Y,
    ROUTE,
    SORTED,
    START,
    END,
    TILE_EXPERT,
    M,
    N: tl.constexpr,
    K: tl.constexpr,
    WT: tl.constexpr,
    AT: tl.constexpr,
    MODE: tl.constexpr,
    UP: tl.constexpr,
):
    row_tile = tl.program_id(0)
    weight_tile = tl.program_id(1)
    n = weight_tile * WT + tl.arange(0, WT)
    m = tl.arange(0, AT)
    if MODE == "group":
        expert = tl.load(TILE_EXPERT + row_tile)
        if expert >= 257:
            return
        lo = tl.load(START + expert)
        hi = tl.load(END + expert)
        index = lo + row_tile * AT - tl.load(START + 257 + expert) + m
        live = index < hi
        row = tl.load(SORTED + index, live, 0).to(tl.int32)
    elif MODE == "slot":
        expert = tl.load(ROUTE + row_tile).to(tl.int32)
        row = row_tile + m
        live = m == 0
    else:
        expert = 0
        row = row_tile * AT + m
        live = row < M
    src = row // 9 if UP else row

    kk = tl.arange(0, 128)
    acc = tl.full((WT, AT), 0.0, tl.float32)
    for g in range(K // 128):
        w = W.load([expert * N + weight_tile * WT, 128 * g])
        a = tl.load(A + src[None, :] * K + 128 * g + kk[:, None], live[None, :], 0.0)
        d = tl.dot(w, a, max_num_imprecise_acc=0)
        sa = tl.load(AS + src * (K // 128) + g, live, 0.0)
        sw = tl.load(WS + (expert * (N // 128) + n // 128) * (K // 128) + g).to(tl.float32)
        acc = tl.fma(d, sw[:, None] * sa[None, :], acc)
    tl.store(Y + row[None, :] * N + n[:, None], acc, live[None, :])


@triton.jit
def _grouped_expert_tile(
    A,
    AS,
    W,
    WS,
    Y,
    SORTED,
    START,
    END,
    EXP,
    row_tile,
    weight_tile,
    N: tl.constexpr,
    K: tl.constexpr,
    WT: tl.constexpr,
    AT: tl.constexpr,
    UP: tl.constexpr,
    STAGES: tl.constexpr,
):
    expert = tl.load(EXP + row_tile)
    n = weight_tile * WT + tl.arange(0, WT)
    m = tl.arange(0, AT)
    lo = tl.load(START + expert)
    hi = tl.load(END + expert)
    index = lo + row_tile * AT - tl.load(START + 257 + expert) + m
    live = index < hi
    row = tl.load(SORTED + index, live, 0).to(tl.int32)
    src = row // 9 if UP else row

    kk = tl.arange(0, 128)
    acc = tl.full((WT, AT), 0.0, tl.float32)
    for g in tl.range(K // 128, num_stages=STAGES):
        w = W.load([expert * N + weight_tile * WT, 128 * g])
        a = tl.load(A + src[None, :] * K + 128 * g + kk[:, None], live[None, :], 0.0)
        d = tl.dot(w, a, max_num_imprecise_acc=0)
        sa = tl.load(AS + src * (K // 128) + g, live, 0.0)
        sw = tl.load(WS + (expert * (N // 128) + n // 128) * (K // 128) + g).to(tl.float32)
        acc = tl.fma(d, sw[:, None] * sa[None, :], acc)
    tl.store(Y + row[None, :] * N + n[:, None], acc, live[None, :])


@triton.jit
def _persistent_experts(
    A,
    AS,
    W,
    WS,
    Y,
    ORDER,
    START,
    END,
    EXPERTS,
    COUNT,
    N: tl.constexpr,
    K: tl.constexpr,
    WT: tl.constexpr,
    AT: tl.constexpr,
    UP: tl.constexpr,
    STAGES: tl.constexpr,
):
    active = tl.load(COUNT)
    tile_cols: tl.constexpr = N // WT
    for block in range(tl.program_id(0), active * tile_cols, tl.num_programs(0)):
        tile = block // tile_cols
        column = block % tile_cols
        _grouped_expert_tile(
            A, AS, W, WS, Y, ORDER, START, END, EXPERTS, tile, column, N, K, WT, AT, UP, STAGES
        )


@triton.jit
def _partial_dots(
    A, W, RAW, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr, WT: tl.constexpr, AT: tl.constexpr
):
    nt = tl.program_id(0)
    g = tl.program_id(1)
    n = nt * WT + tl.arange(0, WT)
    m = tl.arange(0, AT)
    z = tl.arange(0, 128)
    a = tl.load(A + m[None, :] * K + g * 128 + z[:, None], m[None, :] < M, 0.0)
    w = W.load([nt * WT, g * 128])
    d = tl.dot(w, a, max_num_imprecise_acc=0)
    tl.store(RAW + (g * M + m[None, :]) * N + n[:, None], d, m[None, :] < M)


@triton.jit
def _ordered_group_sum(
    RAW, AS, WS, Y, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr, TILE: tl.constexpr
):
    o = tl.program_id(0) * TILE + tl.arange(0, TILE)
    m = tl.program_id(1)
    acc = tl.full((TILE,), 0.0, tl.float32)
    for g in tl.static_range(K // 128):
        v = tl.load(RAW + (g * M + m) * N + o)
        sa = tl.load(AS + m * (K // 128) + g)
        sw = tl.load(WS + (o // 128) * (K // 128) + g).to(tl.float32)
        acc = tl.fma(v, sa * sw, acc)
    tl.store(Y + m * N + o, acc)


@gluon.jit
def _group_row(tile, j: gl.constexpr, layout: gl.constexpr):
    row = gl.gather(tile, gl.full((1, 32), j, gl.int32, layout), axis=0)
    return gl.sum(row, 0)


@gluon.jit
def _prefetched_group_sum(RAW, AS, WS, Y, M: gl.constexpr, N: gl.constexpr):
    row = gl.program_id(1)
    tile = gl.program_id(0)
    layout: gl.constexpr = gl.BlockedLayout([1, 1], [1, 32], [1, 1], [1, 0])
    col = tile * 32 + gl.arange(0, 32, layout=gl.SliceLayout(0, layout))
    group = gl.arange(0, 16, layout=gl.SliceLayout(1, layout))
    total = gl.full((32,), 0.0, gl.float32, gl.SliceLayout(0, layout))
    for half in gl.static_range(2):
        k = half * 16 + group
        dots = gl.load(RAW + ((k[:, None] * M + row) * N + col[None, :]))
        act = gl.load(AS + row * 32 + k)
        weight = gl.load(WS + (tile // 4) * 32 + k).to(gl.float32)
        scale, _ = gl.broadcast((act * weight)[:, None], dots)
        for j in gl.static_range(16):
            total = gl.fma(_group_row(dots, j, layout), _group_row(scale, j, layout), total)
    gl.store(Y + row * N + col, total)


def dense(activations, scales, weight, weight_scales, split=True, prefetch_groups=False):
    m, k = activations.shape
    n = weight.shape[0]
    out = torch.empty((m, n), dtype=torch.bfloat16, device=activations.device)
    if split and m <= 32 and n == 2048:
        wt = 256
        at = max(8, triton.next_power_of_2(m))
        raw = torch.empty((k // 128, m, n), dtype=torch.float32, device=activations.device)
        _partial_dots[(n // wt, k // 128)](
            activations, weight_tiles(weight, wt), raw, m, n, k, wt, at, num_warps=4
        )
        if prefetch_groups and k == 4096:
            _prefetched_group_sum[(n // 32, m)](raw, scales, weight_scales, out, m, n, num_warps=1)
        else:
            tile = 64 if m <= 8 else 128
            _ordered_group_sum[(n // tile, m)](
                raw, scales, weight_scales, out, m, n, k, tile, num_warps=2 if tile == 64 else 4
            )
    else:
        at = min(32, max(8, triton.next_power_of_2(m)))
        wt = 64 if m <= 8 else 128
        _transposed_dot[(triton.cdiv(m, at), n // wt)](
            activations,
            scales,
            weight_tiles(weight, wt),
            weight_scales,
            out,
            out,
            out,
            out,
            out,
            out,
            m,
            n,
            k,
            wt,
            at,
            "dense",
            False,
            num_warps=4,
            num_stages=3,
        )
    return out


def routed(
    activations,
    scales,
    weight,
    weight_scales,
    routes,
    schedule,
    up,
    tile_rows=ROUTE_BLOCK,
    tuned=True,
    multiprocessors=0,
):
    m, k = activations.shape
    n = weight.shape[-2]
    b = routes.shape[0]
    out = torch.empty((b * 9, n), dtype=torch.bfloat16, device=activations.device)
    if schedule is None:
        sorted_rows = starts = ends = experts = routes
        mode = "slot"
        count = routes.numel()
    else:
        sorted_rows, starts, ends, experts = schedule
        mode = "group"
        count = experts.numel()
    if multiprocessors:
        columns, stages = 128, 3
        _persistent_experts[(4 * multiprocessors,)](
            activations,
            scales,
            weight_tiles(weight, columns),
            weight_scales,
            out,
            sorted_rows,
            starts,
            ends,
            experts,
            experts[-1:],
            n,
            k,
            columns,
            tile_rows,
            up,
            stages,
            num_warps=4,
            num_stages=stages,
        )
        return out
    if up:
        wt = 128 if tuned and (1 < b <= 8 or b >= 128) else 64
        stages = 3
    elif b == 1:
        wt, stages = 128, 3
    elif tuned and b <= 8:
        wt, stages = 256, 3
    elif tuned and 128 <= b < 256:
        wt, stages = 256, 1
    elif tuned and b >= 256:
        wt, stages = 128, 3
    elif b <= 32:
        wt, stages = 256, 2
    else:
        wt, stages = 128, 2
    _transposed_dot[(count, n // wt)](
        activations,
        scales,
        weight_tiles(weight, wt),
        weight_scales,
        out,
        routes,
        sorted_rows,
        starts,
        ends,
        experts,
        m,
        n,
        k,
        wt,
        tile_rows,
        mode,
        up,
        num_warps=4,
        num_stages=stages,
    )
    return out

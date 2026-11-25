# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Routed W8A8 projections using double-buffered multi-group TMA weight panels."""

from functools import lru_cache

import torch
import triton
import triton.language as tl
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language.nvidia.blackwell import (
    TensorMemoryLayout,
    allocate_tensor_memory,
    tcgen05_commit,
    tcgen05_mma,
)
from triton.experimental.gluon.language.nvidia.hopper import fence_async_shared, mbarrier, tma
from triton.experimental.gluon.nvidia.hopper import TensorDescriptor


@gluon.jit
def _routed_panel(
    A,
    AS,
    WD,
    WS,
    Y,
    SORTED,
    START,
    END,
    TILE_EXPERT,
    N: gl.constexpr,
    K: gl.constexpr,
    WT: gl.constexpr,
    AT: gl.constexpr,
    GROUPS: gl.constexpr,
):
    route_tile = gl.program_id(0)
    weight_tile = gl.program_id(1)
    a_layout: gl.constexpr = gl.BlockedLayout([1, 16], [4, 8], [1, 4], [1, 0])
    r = gl.arange(0, AT, layout=gl.SliceLayout(1, a_layout))
    ch = gl.arange(0, GROUPS * 128, layout=gl.SliceLayout(0, a_layout))
    expert = gl.load(TILE_EXPERT + route_tile).to(gl.int32)
    if expert >= 257:
        return
    lo = gl.load(START + expert)
    end = gl.load(END + expert)
    index = lo + route_tile * AT - gl.load(START + 257 + expert) + r
    live = index < end
    row = gl.load(SORTED + index, live, 0)
    act_row = row // 9
    panel_w: gl.constexpr = GROUPS * 128
    num_panels: gl.constexpr = K // panel_w
    bar = gl.allocate_shared_memory(gl.int64, [2, 1], mbarrier.MBarrierLayout())
    done = gl.allocate_shared_memory(gl.int64, [1], mbarrier.MBarrierLayout())
    for stage_id in gl.static_range(2):
        mbarrier.init(bar.index(stage_id), count=1)
    mbarrier.init(done, count=1)
    w = gl.allocate_shared_memory(gl.float8e4nv, [2, WT, panel_w], WD.layout)
    a_shared_layout: gl.constexpr = gl.NVMMASharedLayout.get_default_for(
        [AT, panel_w], gl.float8e4nv
    )
    a = gl.allocate_shared_memory(gl.float8e4nv, [AT, panel_w], a_shared_layout)
    partial = allocate_tensor_memory(
        gl.float32, [GROUPS, WT, AT], TensorMemoryLayout([WT, AT], col_stride=1)
    )
    y_layout: gl.constexpr = partial.index(0).get_reg_layout()
    result = gl.full((WT, AT), 0.0, gl.float32, y_layout)
    y_row = gl.convert_layout(row, gl.SliceLayout(0, y_layout))
    y_act = gl.convert_layout(act_row, gl.SliceLayout(0, y_layout))
    y_live = gl.convert_layout(live, gl.SliceLayout(0, y_layout))
    column = weight_tile * WT + gl.arange(0, WT, layout=gl.SliceLayout(1, y_layout))
    wrow = expert * N + weight_tile * WT

    # Keep the next weight panel in flight while the current one uses tensor cores.
    mbarrier.expect(bar.index(0), WD.block_type.nbytes)
    tma.async_load(WD, [wrow, 0], bar.index(0), w.index(0))
    for panel in range(num_panels):
        stage = panel % 2
        if panel + 1 < num_panels:
            mbarrier.expect(bar.index(1 - stage), WD.block_type.nbytes)
            tma.async_load(
                WD, [wrow, (panel + 1) * panel_w], bar.index(1 - stage), w.index(1 - stage)
            )
        x = gl.load(A + act_row[:, None] * K + panel * panel_w + ch[None, :], live[:, None], 0.0)
        a.store(x)
        fence_async_shared()
        mbarrier.wait(bar.index(stage), phase=(panel // 2) % 2)
        for group in gl.static_range(GROUPS):
            weight = w.index(stage).slice(group * 128, 128, dim=1)
            act = a.slice(group * 128, 128, dim=1).permute((1, 0))
            tcgen05_mma(weight, act, partial.index(group), use_acc=False)
        tcgen05_commit(done)
        mbarrier.wait(done, phase=panel % 2)
        for group in gl.static_range(GROUPS):
            scale_group = panel * GROUPS + group
            aw = gl.load(AS + y_act * (K // 128) + scale_group, y_live, 0.0)
            ww = gl.load(WS + (expert * (N // 128) + column // 128) * (K // 128) + scale_group).to(
                gl.float32
            )
            result = gl.fma(partial.index(group).load(), ww[:, None] * aw[None, :], result)
    for stage_id in gl.static_range(2):
        mbarrier.invalidate(bar.index(stage_id))
    mbarrier.invalidate(done)
    gl.store(Y + y_row[None, :] * N + column[:, None], result, y_live[None, :])


@lru_cache(maxsize=640)
def _descriptor(weight, columns, groups):
    shape = [columns, groups * 128]
    layout = gl.NVMMASharedLayout.get_default_for(shape, gl.float8e4nv)
    return TensorDescriptor.from_tensor(weight.view(-1, weight.shape[-1]), shape, layout)


def gate_up(a, scale, weight, weight_scale, schedule, tile_rows):
    n, k = weight.shape[-2:]
    result = torch.empty((a.shape[0] * 9, n), dtype=torch.bfloat16, device=a.device)
    order, starts, ends, experts = schedule
    columns, groups = 64, 4
    _routed_panel[(experts.numel(), n // columns)](
        a,
        scale,
        _descriptor(weight, columns, groups),
        weight_scale,
        result,
        order,
        starts,
        ends,
        experts,
        n,
        k,
        columns,
        tile_rows,
        groups,
        num_warps=4,
    )
    return result


@gluon.jit
def _routed_part(A, WD, RAW, ROUTES, WT: gl.constexpr, AT: gl.constexpr, K_GROUPS: gl.constexpr):
    tile = gl.program_id(0)
    wt = gl.program_id(1)
    p = gl.program_id(2)
    expert = gl.load(ROUTES + tile).to(gl.int32)
    if expert >= 257:
        return
    act_layout: gl.constexpr = gl.BlockedLayout([1, 16], [4, 8], [1, 4], [1, 0])
    r = gl.arange(0, AT, layout=gl.SliceLayout(1, act_layout))
    k = gl.arange(0, K_GROUPS * 128, layout=gl.SliceLayout(0, act_layout))
    live = r == 0
    row = tile + r
    source = row // 9
    block: gl.constexpr = K_GROUPS * 128
    weight = gl.allocate_shared_memory(gl.float8e4nv, [WT, block], WD.layout)
    act_layout_shared: gl.constexpr = gl.NVMMASharedLayout.get_default_for(
        [AT, block], gl.float8e4nv
    )
    act = gl.allocate_shared_memory(gl.float8e4nv, [AT, block], act_layout_shared)
    ready = gl.allocate_shared_memory(gl.int64, [1], mbarrier.MBarrierLayout())
    done = gl.allocate_shared_memory(gl.int64, [1], mbarrier.MBarrierLayout())
    mbarrier.init(ready, count=1)
    mbarrier.init(done, count=1)
    mbarrier.expect(ready, WD.block_type.nbytes)
    tma.async_load(WD, [expert * 1024 + wt * WT, p * block], ready, weight)
    a = gl.load(A + source[:, None] * 2048 + p * block + k[None, :], live[:, None], 0.0)
    act.store(a)
    fence_async_shared()
    result = allocate_tensor_memory(
        gl.float32, [K_GROUPS, WT, AT], TensorMemoryLayout([min(WT, 128), AT], col_stride=1)
    )
    mbarrier.wait(ready, phase=0)
    for i in gl.static_range(K_GROUPS):
        tcgen05_mma(
            weight.slice(i * 128, 128, dim=1),
            act.slice(i * 128, 128, dim=1).permute((1, 0)),
            result.index(i),
            use_acc=False,
        )
    tcgen05_commit(done)
    mbarrier.wait(done, phase=0)
    raw_layout: gl.constexpr = result.index(0).get_reg_layout()
    y_row = gl.convert_layout(row, gl.SliceLayout(0, raw_layout))
    y_live = gl.convert_layout(live, gl.SliceLayout(0, raw_layout))
    column = wt * WT + gl.arange(0, WT, layout=gl.SliceLayout(1, raw_layout))
    for i in gl.static_range(K_GROUPS):
        offset = ((p * K_GROUPS + i) * 9 + y_row[None, :]) * 1024 + column[:, None]
        gl.store(RAW + offset, result.index(i).load(), y_live[None, :])
    mbarrier.invalidate(ready)
    mbarrier.invalidate(done)


@triton.jit
def _swiglu_finish(P, AS, WS, ROUTE, Q, S, G: tl.constexpr):
    row = tl.program_id(0)
    expert = tl.load(ROUTE + row).to(tl.int32)
    g = tl.program_id(1)
    n = g * 128 + tl.arange(0, 128)
    gate = tl.full((128,), 0.0, tl.float32)
    up = tl.full((128,), 0.0, tl.float32)
    # Apply each group scale in K order before the checkpoint's BF16 rounding.
    for k in tl.static_range(G):
        sc = tl.load(AS + k)
        wg = tl.load(WS + (expert * 8 + g) * G + k).to(tl.float32)
        wu = tl.load(WS + (expert * 8 + g + 4) * G + k).to(tl.float32)
        partial = P + (k * 9 + row) * 1024 + n
        gate = tl.fma(tl.load(partial), sc * wg, gate)
        up = tl.fma(tl.load(partial + 512), sc * wu, up)
    gate = gate.to(tl.bfloat16).to(tl.float32)
    up = up.to(tl.bfloat16).to(tl.float32)
    values = (gate * tl.sigmoid(gate) * up).to(tl.bfloat16).to(tl.float32)
    maximum = tl.maximum(tl.max(tl.abs(values), 0), 1.0e-10)
    scale = maximum * (1.0 / 448.0)
    quant = tl.clamp(values * (448.0 / maximum), -448.0, 448.0).to(tl.float8e4nv)
    tl.store(Q + row * 512 + n, quant)
    tl.store(S + row * 4 + g, scale)


def gate_swiglu(a, scales, weight, ws, route):
    raw = torch.empty((16, 9, 1024), device=a.device, dtype=torch.float32)
    out = torch.empty((9, 512), device=a.device, dtype=a.dtype)
    quant_scale = torch.empty((9, 4), device=a.device, dtype=torch.float32)
    _routed_part[(route.numel(), 16, 2)](
        a,
        _descriptor(weight, 64, 8),
        raw,
        route,
        64,
        8,
        8,
        num_warps=4,
    )
    _swiglu_finish[(route.numel(), 4)](
        raw,
        scales,
        ws,
        route,
        out,
        quant_scale,
        16,
        num_warps=1,
    )
    return out, quant_scale

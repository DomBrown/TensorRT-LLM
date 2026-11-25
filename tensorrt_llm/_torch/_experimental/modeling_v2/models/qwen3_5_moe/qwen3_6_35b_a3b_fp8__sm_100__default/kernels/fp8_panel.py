# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Small-row E4M3 projections with TMA activations and weight panels."""

from functools import lru_cache

import torch
import triton
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language.nvidia.blackwell import (
    TensorMemoryLayout,
    allocate_tensor_memory,
    tcgen05_commit,
    tcgen05_mma,
)
from triton.experimental.gluon.language.nvidia.hopper import mbarrier, tma
from triton.experimental.gluon.nvidia.hopper import TensorDescriptor

from .fp8_decode import _prefetched_group_sum


@gluon.jit
def _tma_input_projection(
    AD, S, WD, WS, OUTPUT, M: gl.constexpr, N: gl.constexpr, K: gl.constexpr, GROUPS: gl.constexpr
):
    row_tile = gl.program_id(0)
    weight_tile = gl.program_id(1)
    prog_rows: gl.constexpr = 8
    kg: gl.constexpr = K // 128
    panel: gl.constexpr = GROUPS * 128
    panels: gl.constexpr = K // panel
    weights = gl.allocate_shared_memory(gl.float8e4nv, [2, 64, panel], WD.layout)
    activations = gl.allocate_shared_memory(gl.float8e4nv, [8, K], AD.layout)
    ready = mbarrier.allocate_mbarrier(2)
    done = mbarrier.allocate_mbarrier()
    for stage in gl.static_range(2):
        mbarrier.init(ready.index(stage), count=1)
    mbarrier.init(done, count=1)
    raw = allocate_tensor_memory(
        gl.float32, [GROUPS, 64, prog_rows], TensorMemoryLayout([64, prog_rows], col_stride=1)
    )
    out_layout: gl.constexpr = raw.index(0).get_reg_layout()
    row = row_tile * prog_rows + gl.arange(0, prog_rows, layout=gl.SliceLayout(0, out_layout))
    col = weight_tile * 64 + gl.arange(0, 64, layout=gl.SliceLayout(1, out_layout))
    live = row < M
    acc = gl.full((64, prog_rows), 0.0, gl.float32, out_layout)
    first = ready.index(0)
    # Load the complete activation tile once and overlap subsequent weight panels.
    mbarrier.expect(first, WD.nbytes_per_cta + AD.nbytes_per_cta)
    tma.async_load(WD, [weight_tile * 64, 0], first, weights.index(0))
    tma.async_load(AD, [row_tile * 8, 0], first, activations)
    for pi in gl.static_range(panels):
        buf = pi % 2
        if pi + 1 < panels:
            next_ready = ready.index(1 - buf)
            mbarrier.expect(next_ready, WD.nbytes_per_cta)
            tma.async_load(
                WD, [weight_tile * 64, (pi + 1) * panel], next_ready, weights.index(1 - buf)
            )
        mbarrier.wait(ready.index(buf), phase=(pi // 2) % 2)
        for j in gl.static_range(GROUPS):
            x = activations.slice(pi * panel + j * 128, 128, dim=1)
            w = weights.index(buf).slice(j * 128, 128, dim=1)
            tcgen05_mma(w, x.permute((1, 0)), raw.index(j), use_acc=False)
        tcgen05_commit(done)
        mbarrier.wait(done, phase=pi % 2)
        for j in gl.static_range(GROUPS):
            group = pi * GROUPS + j
            sa = gl.load(S + row * kg + group, live, 0.0)
            sw = gl.load(WS + (col // 128) * kg + group).to(gl.float32)
            acc = gl.fma(raw.index(j).load(), sw[:, None] * sa[None, :], acc)
    for stage in gl.static_range(2):
        mbarrier.invalidate(ready.index(stage))
    mbarrier.invalidate(done)
    gl.store(OUTPUT + row[None, :] * N + col[:, None], acc, live[None, :])


@lru_cache(maxsize=128)
def _descriptor(weight, width, columns=64):
    shape = [columns, width]
    layout = gl.NVMMASharedLayout.get_default_for(shape, gl.float8e4nv)
    return TensorDescriptor.from_tensor(weight, shape, layout)


def dense(a, scale, weight, weight_scale):
    rows, k = a.shape
    n = weight.shape[0]
    out = torch.empty((rows, n), dtype=torch.bfloat16, device=a.device)
    groups = 8 if n == 9216 and rows <= 4 else 4
    shape = [8, k]
    layout = gl.NVMMASharedLayout.get_default_for(shape, gl.float8e4nv)
    activation = TensorDescriptor.from_tensor(a, shape, layout)
    _tma_input_projection[(triton.cdiv(rows, 8), n // 64)](
        activation,
        scale,
        _descriptor(weight, groups * 128),
        weight_scale,
        out,
        rows,
        n,
        k,
        groups,
        num_warps=4,
    )
    return out


@gluon.jit
def _partial_panel(
    AD, WD, RAW, M: gl.constexpr, N: gl.constexpr, COLS: gl.constexpr, GROUPS: gl.constexpr
):
    column_tile = gl.program_id(0)
    panel = gl.program_id(1)
    channels: gl.constexpr = GROUPS * 128
    a = gl.allocate_shared_memory(gl.float8e4nv, [8, channels], AD.layout)
    w = gl.allocate_shared_memory(gl.float8e4nv, [COLS, channels], WD.layout)
    ready = mbarrier.allocate_mbarrier()
    done = mbarrier.allocate_mbarrier()
    mbarrier.init(ready, count=1)
    mbarrier.init(done, count=1)
    mbarrier.expect(ready, AD.nbytes_per_cta + WD.nbytes_per_cta)
    tma.async_load(AD, [0, panel * channels], ready, a)
    tma.async_load(WD, [column_tile * COLS, panel * channels], ready, w)
    partial = allocate_tensor_memory(
        gl.float32, [GROUPS, COLS, 8], TensorMemoryLayout([COLS, 8], col_stride=1)
    )
    mbarrier.wait(ready, phase=0)
    for group in gl.static_range(GROUPS):
        activation = a.slice(group * 128, 128, dim=1)
        weight = w.slice(group * 128, 128, dim=1)
        tcgen05_mma(weight, activation.permute((1, 0)), partial.index(group), use_acc=False)
    tcgen05_commit(done)
    mbarrier.wait(done, phase=0)
    layout: gl.constexpr = partial.index(0).get_reg_layout()
    row = gl.arange(0, 8, layout=gl.SliceLayout(0, layout))
    column = column_tile * COLS + gl.arange(0, COLS, layout=gl.SliceLayout(1, layout))
    for group in gl.static_range(GROUPS):
        offset = ((panel * GROUPS + group) * M + row[None, :]) * N + column[:, None]
        gl.store(RAW + offset, partial.index(group).load(), row[None, :] < M)
    mbarrier.invalidate(ready)
    mbarrier.invalidate(done)


def split_output(a, scale, weight, weight_scale):
    """Load four K groups per FP8 tile and finish their scale FMAs in order."""
    rows, k = a.shape
    columns = weight.shape[0]
    raw = torch.empty((k // 128, rows, columns), device=a.device, dtype=torch.float32)
    result = torch.empty((rows, columns), device=a.device, dtype=torch.bfloat16)
    groups, tile = 4, 128
    shape = [8, groups * 128]
    layout = gl.NVMMASharedLayout.get_default_for(shape, gl.float8e4nv)
    activation = TensorDescriptor.from_tensor(a, shape, layout)
    _partial_panel[(columns // tile, k // (groups * 128))](
        activation,
        _descriptor(weight, groups * 128, tile),
        raw,
        rows,
        columns,
        tile,
        groups,
        num_warps=4,
    )
    _prefetched_group_sum[(columns // 32, rows)](
        raw, scale, weight_scale, result, rows, columns, num_warps=1
    )
    return result

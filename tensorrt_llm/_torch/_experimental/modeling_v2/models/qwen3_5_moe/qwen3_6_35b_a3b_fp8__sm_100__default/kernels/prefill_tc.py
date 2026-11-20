# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Native FP8 expert tiles for short prefill batches."""

import torch
from triton.experimental import gluon as gj
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language.nvidia.blackwell import (
    TensorMemoryLayout,
    allocate_tensor_memory,
    tcgen05_commit,
    tcgen05_mma,
)
from triton.experimental.gluon.language.nvidia.hopper import fence_async_shared, mbarrier, tma
from triton.experimental.gluon.nvidia.hopper import TensorDescriptor

from . import fp8 as kernels


@gj.jit
def expert_tiles(
    X,
    XS,
    WD,
    WS,
    Pairs,
    Experts,
    Counts,
    Y,
    N: gl.constexpr,
    K: gl.constexpr,
    BM: gl.constexpr,
    BN: gl.constexpr,
    G: gl.constexpr,
    DOWN: gl.constexpr,
):
    col, block = gl.program_id(0), gl.program_id(1)
    if block >= gl.load(Counts + 256):
        return
    expert = gl.load(Experts + block)
    w_row = expert * N + col * BN
    groups: gl.constexpr = K // 128

    w_smem = gl.allocate_shared_memory(gl.float8e4nv, (G, BN, 128), WD.layout)
    a_smem = gl.allocate_shared_memory(
        gl.float8e4nv, (G, BM, 128), gl.NVMMASharedLayout.get_default_for([BM, 128], gl.float8e4nv)
    )
    tm = allocate_tensor_memory(gl.float32, (G, BN, BM), TensorMemoryLayout([BN, BM], col_stride=1))
    transfer = gl.allocate_shared_memory(gl.int64, (1,), mbarrier.MBarrierLayout())
    finished = gl.allocate_shared_memory(gl.int64, (1,), mbarrier.MBarrierLayout())
    mbarrier.init(transfer, count=1)
    mbarrier.init(finished, count=1)

    act_layout: gl.constexpr = gl.BlockedLayout([1, 4], [1, 32], [4, 1], [1, 0])
    a_lane = gl.arange(0, BM, layout=gl.SliceLayout(1, act_layout))
    channel = gl.arange(0, 128, layout=gl.SliceLayout(0, act_layout))
    mapping = gl.load(Pairs + block * BM + a_lane)
    idx = gl.maximum(mapping, 0)
    a_row = idx if DOWN else idx // 8
    a_ptr = X + a_row[:, None] * K + channel[None, :]

    out_layout: gl.constexpr = gl.BlockedLayout([1, 1], [8, 4], [4, 1], [0, 1])
    o_lane = gl.arange(0, BM, layout=gl.SliceLayout(0, out_layout))
    columns = gl.arange(0, BN, layout=gl.SliceLayout(1, out_layout))
    pair = gl.load(Pairs + block * BM + o_lane)
    target = gl.maximum(pair, 0)
    scale_row = target if DOWN else target // 8
    acc = gl.full((BN, BM), 0.0, gl.float32, out_layout)

    for phase in range(groups // G):
        first = phase * G
        mbarrier.expect(transfer, G * BN * 128)
        for i in gl.static_range(G):
            tma.async_load(WD, [w_row, (first + i) * 128], transfer, w_smem.index(i))
        for i in gl.static_range(G):
            tile = gl.load(a_ptr + (first + i) * 128)
            a_smem.index(i).store(tile)
        fence_async_shared()
        mbarrier.wait(transfer, phase=phase & 1)
        for i in gl.static_range(G):
            tcgen05_mma(
                w_smem.index(i), a_smem.index(i).permute((1, 0)), tm.index(i), use_acc=False
            )
        tcgen05_commit(finished)
        mbarrier.wait(finished, phase=phase & 1)
        for i in gl.static_range(G):
            w_scale = gl.load(WS + (w_row // 128) * groups + first + i)
            a_scale = gl.load(XS + scale_row * groups + first + i)
            raw = gl.convert_layout(tm.index(i).load(), out_layout)
            acc = gl.fma(raw, (a_scale * w_scale)[None, :], acc)

    dest = Y + target[None, :] * N + col * BN + columns[:, None]
    gl.store(dest, acc, pair[None, :] >= 0)
    mbarrier.invalidate(transfer)
    mbarrier.invalidate(finished)


def project(
    activation,
    parameter,
    grouping,
    tokens,
    *,
    down=False,
    col_tile=128,
    group=4,
    dtype=torch.bfloat16,
):
    x, scale = activation[:2]
    w, ws = parameter
    per_tile = kernels.grouping_tile(tokens)
    n, width = w.shape[-2:]
    pairs, experts, counts = grouping
    out = torch.empty((tokens * 8, n), device=x.device, dtype=dtype)
    layout = gl.NVMMASharedLayout.get_default_for([col_tile, 128], gl.float8e4nv)
    desc = TensorDescriptor.from_tensor(w.view(-1, width), [col_tile, 128], layout)
    expert_tiles[(n // col_tile, experts.numel())](
        x,
        scale,
        desc,
        ws,
        pairs,
        experts,
        counts,
        out,
        n,
        width,
        per_tile,
        col_tile,
        group,
        down,
        num_warps=4,
    )
    return out

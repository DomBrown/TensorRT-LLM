# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Small-row FP8 GEMMs with native Blackwell tensor cores and TMA weight loads."""

import torch
import triton as tr
import triton.language as tl
from triton.experimental import gluon as gl_jit
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


@gl_jit.jit
def small_fp8(
    X,
    XS,
    SX,
    SS,
    WD,
    WS,
    Routes,
    Pairs,
    Y,
    AD,
    N: gl.constexpr,
    K: gl.constexpr,
    M: gl.constexpr,
    MODE: gl.constexpr,
    BM: gl.constexpr,
    BN: gl.constexpr,
    SP: gl.constexpr,
    G: gl.constexpr,
    SLOTS: gl.constexpr,
    EXACT_PARTIALS: gl.constexpr,
    PDL: gl.constexpr,
    A_TMA: gl.constexpr,
    INTERLEAVE: gl.constexpr,
    EARLY: gl.constexpr,
):
    tile = gl.program_id(0)
    if INTERLEAVE:
        part, slot = gl.program_id(1), gl.program_id(2)
    else:
        slot, part = gl.program_id(1), gl.program_id(2)
    groups: gl.constexpr = K // 128
    if MODE:
        if M == 1 and SLOTS == 9:
            expert = gl.load(Routes + slot, slot < 8, 256)
        else:
            expert = gl.load(Routes + slot)
        rows: gl.constexpr = M * SLOTS
        if M > 1 and expert < 0:
            return
    else:
        expert = 0
        rows: gl.constexpr = M
    w_start = expert * N + tile * BN
    w_smem = gl.allocate_shared_memory(gl.float8e4nv, (G, BN, 128), WD.layout)
    act_smem_layout: gl.constexpr = gl.NVMMASharedLayout.get_default_for([BM, 128], gl.float8e4nv)
    a_smem = gl.allocate_shared_memory(gl.float8e4nv, (G, BM, 128), act_smem_layout)
    mem_layout: gl.constexpr = TensorMemoryLayout([BN, BM], col_stride=1)
    tm = allocate_tensor_memory(gl.float32, (G, BN, BM), mem_layout)

    transfer = gl.allocate_shared_memory(gl.int64, [1], mbarrier.MBarrierLayout())
    finished = gl.allocate_shared_memory(gl.int64, [1], mbarrier.MBarrierLayout())
    mbarrier.init(transfer, count=1)
    mbarrier.init(finished, count=1)
    act_layout: gl.constexpr = gl.BlockedLayout([1, 4], [1, 32], [4, 1], [1, 0])
    m = gl.arange(0, BM, layout=gl.SliceLayout(1, act_layout))
    quad = gl.arange(0, 128, layout=gl.SliceLayout(0, act_layout))
    if MODE == 1:
        x_row = gl.minimum(m, M - 1)
    elif MODE == 2 and M > 1:
        a_pair = gl.load(Pairs + slot * M + m, m < M, -1)
        x_row = gl.maximum(a_pair, 0)
    elif MODE == 2:
        x_row = gl.full((BM,), slot, gl.int32, gl.SliceLayout(1, act_layout))
    else:
        x_row = gl.minimum(m, M - 1)
    if MODE == 1 and SLOTS == 9:
        base = gl.where(expert == 256, SX, X)
        scale_base = gl.where(expert == 256, SS, XS)
    else:
        base = X
        scale_base = XS
    xp = base + x_row[:, None] * K + quad[None, :]

    out_layout: gl.constexpr = gl.BlockedLayout([1, 1], [8, 4], [4, 1], [0, 1])
    outputs = gl.arange(0, BN, layout=gl.SliceLayout(1, out_layout))
    row = gl.arange(0, BM, layout=gl.SliceLayout(0, out_layout))
    if MODE and M > 1:
        out_row = gl.load(Pairs + slot * M + row, row < M, -1)
        valid = out_row >= 0
        scale_row = gl.maximum(out_row, 0)
        if MODE == 1:
            scale_row = gl.minimum(row, M - 1)
    elif MODE:
        out_row = gl.full((BM,), slot, gl.int32, gl.SliceLayout(0, out_layout))
        scale_row = (
            out_row if MODE == 2 else gl.full((BM,), 0, gl.int32, gl.SliceLayout(0, out_layout))
        )
        valid = row == 0
    else:
        out_row = row
        scale_row = gl.minimum(row, M - 1)
        valid = row < M

    offs = out_row[None, :] * N + tile * BN + outputs[:, None]
    acc = gl.full((BN, BM), 0.0, gl.float32, out_layout)
    for batch in range(groups // (SP * G)):
        first = (part * (groups // (SP * G)) + batch) * G
        mbarrier.expect(transfer, G * (BN + (BM if A_TMA else 0)) * 128)
        if PDL:
            preload = tile % 4 == 0
            if preload:
                tma.async_load(WD, [w_start, first * 128], transfer, w_smem.index(0))
            gl.inline_asm_elementwise(
                "griddepcontrol.wait; // dependency $0",
                "=r",
                [],
                dtype=gl.int32,
                is_pure=False,
                pack=1,
            )
            if EARLY:
                gl.inline_asm_elementwise(
                    "griddepcontrol.launch_dependents; // $0",
                    "=r",
                    [],
                    dtype=gl.int32,
                    is_pure=False,
                    pack=1,
                )
            if not preload:
                tma.async_load(WD, [w_start, first * 128], transfer, w_smem.index(0))
        for i in gl.static_range(1 if PDL else 0, G):
            tma.async_load(WD, [w_start, (first + i) * 128], transfer, w_smem.index(i))
        if A_TMA:
            for i in gl.static_range(G):
                tma.async_load(AD, [0, (first + i) * 128], transfer, a_smem.index(i))
        else:
            for i in gl.static_range(G):
                a = gl.load(xp + (first + i) * 128)
                a_smem.index(i).store(a)
            fence_async_shared()
        mbarrier.wait(transfer, phase=batch & 1)

        for i in gl.static_range(G):
            tcgen05_mma(
                w_smem.index(i), a_smem.index(i).permute((1, 0)), tm.index(i), use_acc=False
            )
        tcgen05_commit(finished)
        mbarrier.wait(finished, phase=batch & 1)

        if EXACT_PARTIALS and part > 0:
            for i in gl.static_range(G):
                raw = gl.convert_layout(tm.index(i).load(), out_layout)
                addr = (1 + first + i - groups // SP) * rows * N + offs
                gl.store(Y + addr, raw, valid[None, :])
        else:
            for i in gl.static_range(G):
                xs = gl.load(scale_base + scale_row * groups + first + i)
                ws = gl.load(WS + (w_start // 128) * groups + first + i)
                raw = gl.convert_layout(tm.index(i).load(), out_layout)
                acc = gl.fma(raw, (xs * ws)[None, :], acc)

    if EXACT_PARTIALS:
        if part == 0:
            gl.store(Y + offs, acc, valid[None, :])
    else:
        gl.store(Y + part * rows * N + offs, acc, valid[None, :])
    mbarrier.invalidate(transfer)
    mbarrier.invalidate(finished)


@tr.jit
def route_plan(Routes, Experts, Pairs, M: tl.constexpr, B: tl.constexpr, SLOTS: tl.constexpr):
    i = tl.arange(0, B)
    active = i < M * SLOTS
    if SLOTS == 9:
        lane = i % SLOTS
        e = tl.load(Routes + (i // SLOTS) * 8 + lane, active & (lane < 8), -1)
        e = tl.where(active & (lane == 8), 256, e)
    else:
        e = tl.load(Routes + i, active, -1)
    equal = (e[:, None] == e[None, :]) & active[:, None]
    first = tl.min(tl.where(equal, i[None, :], B), 1)
    unique = first == i
    tl.store(Experts + i, tl.where(unique, e, -1), active)
    for row in tl.static_range(M):
        pair = tl.min(tl.where(equal & (i[None, :] // SLOTS == row), i[None, :], B), 1)
        tl.store(Pairs + i * M + row, tl.where(pair < B, pair, -1), active & unique)


def plan(routes, slots=8):
    m = routes.shape[0]
    if m == 1:
        return routes, routes
    experts = torch.empty((m * slots,), dtype=torch.int32, device=routes.device)
    pairs = torch.empty((m * slots, m), dtype=torch.int32, device=routes.device)
    route_plan[(1,)](routes, experts, pairs, m, tr.next_power_of_2(m * slots), slots, num_warps=4)
    return experts, pairs


def matmul(
    a,
    weight,
    routes=None,
    down=False,
    *,
    tile=128,
    group=4,
    split=1,
    dtype=torch.bfloat16,
    audit=False,
    routing=None,
    slots=8,
    shared=None,
    exact_partials=False,
    pdl=False,
    a_tma=False,
    interleave=False,
    early=False,
):
    x, xs = a[:2]
    w, ws = weight
    n, width = w.shape[-2:]
    mode = (2 if down else 1) if routes is not None else 0
    m = routes.shape[0] if mode else x.shape[0]
    rows = m * slots if mode else m
    ax, ascale = shared[:2] if shared is not None else (x, xs)
    experts, pairs = routing or (routes, routes) if mode else (x, x)
    bm = 8 if m <= 8 else 16
    if m > bm or (width // 128) % (group * split):
        raise ValueError("unsupported small-row FP8 tile")
    dl = gl.NVMMASharedLayout.get_default_for([tile, 128], gl.float8e4nv)
    desc = TensorDescriptor.from_tensor(w.view(-1, width), [tile, 128], dl)
    activation_desc = None
    if a_tma:
        activation_layout = gl.NVMMASharedLayout.get_default_for([bm, 128], gl.float8e4nv)
        activation_desc = TensorDescriptor.from_tensor(x, [bm, 128], activation_layout)
    if exact_partials:
        partials = 1 + width // 128 - width // (128 * split)
        y = out = torch.empty((partials, rows, n), device=x.device, dtype=torch.float32)
    else:
        y = torch.empty((rows, n), device=x.device, dtype=dtype)
        out = (
            y if split == 1 else torch.empty((split, rows, n), device=x.device, dtype=torch.float32)
        )
    blocks = (split, rows if mode else 1) if interleave else (rows if mode else 1, split)
    compiled = small_fp8[(n // tile, *blocks)](
        x,
        xs,
        ax,
        ascale,
        desc,
        ws,
        experts,
        pairs,
        out,
        activation_desc,
        n,
        width,
        m,
        mode,
        bm,
        tile,
        split,
        group,
        slots,
        exact_partials,
        pdl,
        a_tma,
        interleave,
        early,
        num_warps=4,
        launch_pdl=pdl,
    )
    if audit and "tcgen05.mma.cta_group::1.kind::f8f6f4" not in compiled.asm["ptx"]:
        raise RuntimeError("small-row projection did not compile to native FP8")
    if split > 1 and not exact_partials:
        kernels.add_partials[(tr.cdiv(rows * n, 256),)](out, y, rows * n, split, 256)
    return y

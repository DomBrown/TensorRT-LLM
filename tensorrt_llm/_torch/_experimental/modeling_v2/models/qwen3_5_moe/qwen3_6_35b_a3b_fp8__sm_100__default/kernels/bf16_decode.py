# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Small BF16 tensor-core projections with explicit split-K reductions."""

from functools import lru_cache

import torch
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice
from triton.tools.tensor_descriptor import TensorDescriptor

from . import kernels as ops
from .kernels import _gdn_gate_values


@triton.jit
def _bf16_part(
    X,
    W,
    P,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    MT: tl.constexpr,
    NT: tl.constexpr,
    KT: tl.constexpr,
    ST: tl.constexpr,
    K_PART: tl.constexpr = 0,
):
    mi = tl.program_id(0) * MT + tl.arange(0, MT)
    ni = tl.program_id(1) * NT + tl.arange(0, NT)
    slice = tl.program_id(2)
    acc = tl.full((NT, MT), 0.0, tl.float32)
    if K_PART:
        for i in range(tl.cdiv(K_PART, KT)):
            index = i * KT + tl.arange(0, KT)
            ki = slice * K_PART + index
            live = (ki < K) & (index < K_PART)
            w = tl.load(W + ni[:, None] * K + ki[None, :], (ni[:, None] < N) & live[None, :], 0)
            x = tl.load(X + mi[None, :] * K + ki[:, None], (mi[None, :] < M) & live[:, None], 0)
            acc = tl.dot(w, x, acc)
    else:
        for i in range(K // (ST * KT)):
            ki = (slice * (K // (ST * KT)) + i) * KT + tl.arange(0, KT)
            w = tl.load(W + ni[:, None] * K + ki[None, :], ni[:, None] < N, 0)
            x = tl.load(X + mi[None, :] * K + ki[:, None], mi[None, :] < M, 0)
            acc = tl.dot(w, x, acc)
    tl.store(
        P + (slice * M + mi[None, :]) * N + ni[:, None], acc, (mi[None, :] < M) & (ni[:, None] < N)
    )


@triton.jit
def _bf16_finish(P, Y, M: tl.constexpr, N: tl.constexpr, ST: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    i = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    slice = tl.arange(0, ST)
    part = tl.load(P + (slice[:, None] * M + row) * N + i[None, :], i[None, :] < N, 0)
    tl.store(Y + row * N + i, tl.sum(part, 0), i < N)


def router(x, w, batch):
    m, n = x.shape[0], w.shape[0]
    p, st = _router_parts(x, w, batch)
    y = torch.empty((m, n), dtype=x.dtype, device=x.device)
    _bf16_finish[(m, triton.cdiv(n, 64))](p, y, m, n, st, 64, num_warps=4)
    return y


@triton.jit
def _single_router_finish(
    P, Y, ROUTES, PROB, M: tl.constexpr, ST: tl.constexpr, TOPK: tl.constexpr
):
    row = tl.program_id(0)
    if TOPK:
        e = tl.arange(0, 256)
        acc = tl.full((256,), 0.0, tl.float32)
        shared = tl.full((), 0.0, tl.float32)
        for part in tl.static_range(ST):
            base = (part * M + row) * 257
            acc += tl.load(P + base + e)
            shared += tl.load(P + base + 256)
        logits = acc.to(tl.bfloat16)
        tl.store(Y + row * 257 + e, logits)
        tl.store(Y + row * 257 + 256, shared)
        bits = logits.to(tl.uint16, bitcast=True).to(tl.uint32)
        bits = tl.where((bits & 32767) == 0, 0, bits)
        ordered = bits ^ tl.where((bits & 32768) != 0, 65535, 32768).to(tl.uint32)
        key = (ordered << 8) | (255 - e).to(tl.uint32)
        ids = (255 - (tl.topk(key, 8) & 255)).to(tl.int32)
        score = tl.gather(logits, ids, 0).to(tl.float32)
        ex = libdevice.exp(score - tl.max(score, 0))
        probability = libdevice.div_rn(ex, tl.sum(ex, 0))
        j = tl.arange(0, 8)
        tl.store(ROUTES + row * 9 + j, ids)
        tl.store(ROUTES + row * 9 + 8, 256)
        tl.store(PROB + row * 8 + j, probability)
    else:
        e = tl.program_id(1) * 64 + tl.arange(0, 64)
        acc = tl.full((64,), 0.0, tl.float32)
        for part in tl.static_range(ST):
            acc += tl.load(P + (part * M + row) * 257 + e, e < 257, 0.0)
        tl.store(Y + row * 257 + e, acc, e < 257)


def single_router(x, w, fused, round_router):
    m, k = x.shape
    mt, nt, kt, st = 16, 32, 64, 11
    partials = torch.empty((st, m, 257), dtype=torch.float32, device=x.device)
    logits = torch.empty((m, 257), dtype=x.dtype, device=x.device)
    _bf16_part[(1, 9, st)](x, w, partials, m, 257, k, mt, nt, kt, st, 192, num_warps=4)
    if fused:
        ids = torch.empty((m, 9), dtype=torch.int64, device=x.device)
        prob = torch.empty(
            (m, 8), dtype=x.dtype if round_router else torch.float32, device=x.device
        )
    else:
        ids = prob = logits
    _single_router_finish[(m, 1 if fused else 5)](
        partials, logits, ids, prob, m, st, fused, num_warps=4
    )
    return logits, ids, prob


@triton.jit
def _gate_finish(
    P, PARAM, DT, ALPHA, BETA, M: tl.constexpr, ST: tl.constexpr, ROUND_BETA: tl.constexpr
):
    row = tl.program_id(0)
    h = tl.arange(0, 32)
    a = tl.full((32,), 0.0, tl.float32)
    b = tl.full((32,), 0.0, tl.float32)
    # Match the original projection's FP32 partition reduction order.
    for part in tl.static_range(ST):
        offset = (part * M + row) * 64 + h
        a += tl.load(P + offset)
        b += tl.load(P + offset + 32)
    a = a.to(tl.bfloat16).to(tl.float32)
    b = b.to(tl.bfloat16).to(tl.float32)
    alpha, beta = _gdn_gate_values(a, b, PARAM, DT, h, ROUND_BETA)
    tl.store(ALPHA + row * 32 + h, alpha)
    tl.store(BETA + row * 32 + h, beta)


def gates(x, w, a_log, dt, round_beta, batch=1):
    m, k = x.shape
    if batch == 1:
        mt, kt, st = 16, 64, 16
    elif batch <= 32:
        mt, kt, st = 8, 128, 2
    else:
        mt, kt, st = 16, 256, 2
    nt = 32
    partials = torch.empty((st, m, 64), dtype=torch.float32, device=x.device)
    alpha = torch.empty((m, 32), dtype=torch.float32, device=x.device)
    beta = torch.empty_like(alpha)
    _bf16_part[(triton.cdiv(m, mt), 2, st)](
        x,
        w,
        partials,
        m,
        64,
        k,
        mt,
        nt,
        kt,
        st,
        num_warps=4,
    )
    _gate_finish[(m,)](partials, a_log, dt, alpha, beta, m, st, round_beta, num_warps=1)
    return alpha, beta


@triton.jit
def _even_part(base, stride, ST: tl.constexpr):
    # The ordinary split-K epilogue first combines the even and odd planes.
    a = tl.load(base)
    b = tl.load(base + 2 * stride)
    c = tl.load(base + 4 * stride)
    d = tl.load(base + 6 * stride)
    ac = a + c
    bd = b + d
    if ST == 16:
        # Match the packed fp32x2 tree of the split-K projection epilogue.
        ac += tl.load(base + 8 * stride) + tl.load(base + 12 * stride)
        bd += tl.load(base + 10 * stride) + tl.load(base + 14 * stride)
    return ac + bd


@triton.jit
def _router_topk(
    P,
    Y,
    IDS,
    WEIGHTS,
    ROWS,
    COUNTS,
    ACTIVE,
    M: tl.constexpr,
    ST: tl.constexpr,
    WIDTH: tl.constexpr,
    RETIRE: tl.constexpr,
):
    row = tl.program_id(0)
    j = tl.arange(0, 8)
    if RETIRE:
        if tl.load(ACTIVE + row // WIDTH) == 0:
            tl.store(IDS + row * 9 + j, 257)
            tl.store(IDS + row * 9 + 8, 257)
            tl.store(WEIGHTS + row * 8 + j, 0.0)
            tl.store(Y + row * 257 + 256, 0.0)
            return
    expert = tl.arange(0, 256)
    source = P + row * 257
    stride = M * 257
    addresses = source + expert
    logits = (_even_part(addresses, stride, ST) + _even_part(addresses + stride, stride, ST)).to(
        tl.bfloat16
    )
    shared = (
        _even_part(source + 256, stride, ST) + _even_part(source + stride + 256, stride, ST)
    ).to(tl.bfloat16)
    tl.store(Y + row * 257 + expert, logits)
    tl.store(Y + row * 257 + 256, shared)

    bits = logits.to(tl.uint16, bitcast=True).to(tl.uint32)
    bits = tl.where((bits & 32767) == 0, 0, bits)
    ordered = bits ^ tl.where((bits & 32768) != 0, 65535, 32768).to(tl.uint32)
    key = (ordered << 8) | (255 - expert).to(tl.uint32)
    selected = tl.topk(key, 8)
    ids = (255 - (selected & 255)).to(tl.int32)
    scores = tl.gather(logits, ids, 0).to(tl.float32)
    ex = libdevice.exp(scores - tl.max(scores, 0))
    probability = libdevice.div_rn(ex, tl.sum(ex, 0))
    tl.store(IDS + row * 9 + j, ids)
    tl.store(IDS + row * 9 + 8, 256)
    tl.store(WEIGHTS + row * 8 + j, probability)
    local = tl.atomic_add(COUNTS + ids, 1, sem="relaxed")
    tl.store(ROWS + ids * M + local, row * 9 + j)
    if not RETIRE:
        tl.store(ROWS + 256 * M + row, row * 9 + 8)


@triton.jit
def _router_tma_parts(X, W, P, M: tl.constexpr, ST: tl.constexpr):
    start_x = tl.program_id(0) * 32
    start_w = tl.program_id(1) * 32
    part = tl.program_id(2)
    row = start_x + tl.arange(0, 32)
    expert = start_w + tl.arange(0, 32)
    width: tl.constexpr = 2048 // ST
    acc = tl.full((32, 32), 0.0, tl.float32)
    for offset in range(0, width, 64):
        k = part * width + offset
        w = W.load([start_w, k])
        x = X.load([start_x, k]).T
        acc = tl.dot(w, x, acc)
    address = (part * M + row[None, :]) * 257 + expert[:, None]
    tl.store(P + address, acc, (row[None, :] < M) & (expert[:, None] < 257))


@lru_cache(maxsize=64)
def _decode_router_weight(weight):
    return TensorDescriptor.from_tensor(weight, [32, 64])


def _router_parts(x, w, batch):
    # Verification and ordinary decode keep the same split-K layout for a row.
    mt, st = (16, 16) if batch <= 32 else (32, 8)
    nt, kt = 32, 64
    m, k = x.shape
    n = w.shape[0]
    p = torch.empty((st, m, n), dtype=torch.float32, device=x.device)
    if batch >= 128 and m >= 256:
        activation = TensorDescriptor.from_tensor(x, [32, 64])
        _router_tma_parts[(triton.cdiv(m, 32), 9, st)](
            activation,
            _decode_router_weight(w),
            p,
            m,
            st,
            num_warps=4,
            num_stages=3,
        )
    else:
        _bf16_part[(triton.cdiv(m, mt), triton.cdiv(n, nt), st)](
            x,
            w,
            p,
            m,
            n,
            k,
            mt,
            nt,
            kt,
            st,
            num_warps=4,
        )
    return p, st


def router_topk(
    x, w, batch, round_router, counts, tile_rows, active=None, width=1, emit_count=False
):
    m, n = x.shape[0], w.shape[0]
    p, st = _router_parts(x, w, batch)
    y = torch.empty((m, n), dtype=x.dtype, device=x.device)
    ids = torch.empty((m, 9), dtype=torch.int64, device=x.device)
    probability = torch.empty(
        (m, 8), dtype=x.dtype if round_router else torch.float32, device=x.device
    )
    rows = torch.empty((257, m), device=x.device, dtype=torch.int32)
    _router_topk[(m,)](
        p,
        y,
        ids,
        probability,
        rows,
        counts,
        active,
        m,
        st,
        width,
        active is not None,
        num_warps=4,
    )
    schedule = ops.schedule_expert_lists(
        rows, counts, m, tile_rows, True, active, width, emit_count
    )
    return y, ids, probability, schedule


@triton.jit(do_not_specialize=["M"])
def _prefill_router(X, W, Y, M, MT: tl.constexpr, NT: tl.constexpr, KT: tl.constexpr):
    start_m = tl.program_id(0) * MT
    start_n = tl.program_id(1) * NT
    row = start_m + tl.arange(0, MT)
    expert = start_n + tl.arange(0, NT)
    result = tl.full((NT, MT), 0.0, tl.float32)
    # Accumulate K in order so packed prompts retain the same rounded logits.
    for k in range(0, 2048, KT):
        w = W.load([start_n, k])
        x = X.load([start_m, k]).T
        result = tl.dot(w, x, result)
    tl.store(
        Y + row[None, :] * 257 + expert[:, None],
        result,
        (row[None, :] < M) & (expert[:, None] < 257),
    )


@lru_cache(maxsize=128)
def _prefill_router_weight(weight, columns, group):
    return TensorDescriptor.from_tensor(weight, [columns, group])


def prefill_router(x, weight, narrow=False):
    rows = x.shape[0]
    logits = torch.empty((rows, 257), dtype=x.dtype, device=x.device)
    if narrow and rows <= 1024:
        mt = 16 if rows <= 256 else 32 if rows <= 768 else 64
        nt, kt = 32, 256
    else:
        mt = nt = kt = 64
    activation = TensorDescriptor.from_tensor(x, [mt, kt])
    _prefill_router[(triton.cdiv(rows, mt), triton.cdiv(257, nt))](
        activation,
        _prefill_router_weight(weight, nt, kt),
        logits,
        rows,
        mt,
        nt,
        kt,
        num_warps=4,
        num_stages=3,
    )
    return logits

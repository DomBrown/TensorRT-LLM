# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""FP8, normalization and MoE kernels adapted from the standalone engine's kernels.py."""

import torch
import triton as tr
import triton.language as tl
from triton.language.extra.cuda import libdevice

MX_PAD = tl.constexpr(False)
GLU_WARPS = 1


@tr.jit
def scale_ceil(x):
    bits = x.to(tl.int32, bitcast=True)
    exponent = (bits + 0x7FFFFF) & 0x7F800000
    return exponent.to(tl.float32, bitcast=True)


@tr.jit
def requant_weights(W, S, Output, OS, K: tl.constexpr):
    n, k = tl.program_id(0), tl.program_id(1)
    i = tl.arange(0, 128)
    index = (n * 128 + i[:, None]) * K + k * 128 + i[None, :]
    s = tl.load(S + n * (K // 128) + k)
    x = (tl.load(W + index).to(tl.float32) * s).to(tl.bfloat16).to(tl.float32)
    amax = tl.max(tl.max(tl.abs(x), 1), 0)
    ns = scale_ceil(tl.maximum(amax, 1.0e-4) * (1.0 / 448.0))
    tl.store(Output + index, x / ns)
    tl.store(OS + n * (K // 128) + k, ns)


@tr.jit
def quantize(X, Q, S, K: tl.constexpr, SX: tl.constexpr, POW2: tl.constexpr, GROUPS: tl.constexpr):
    p, b = tl.program_id(0), tl.program_id(1)
    i = b * GROUPS * 128 + tl.arange(0, GROUPS * 128)
    x = tl.load(X + p * SX + i).to(tl.float32)
    tile = tl.reshape(x, (GROUPS, 128))
    amax = tl.maximum(tl.max(tl.abs(tile), 1), 1.0e-10)
    scale = amax * (1.0 / 448.0)
    if POW2:
        scale = scale_ceil(scale)
        inv = 1.0 / scale
    else:
        inv = 448.0 / amax
    q = tl.clamp(tile * inv[:, None], -448.0, 448.0)
    tl.store(Q + p * K + i, tl.reshape(q, (GROUPS * 128,)))
    tl.store(S + p * (K // 128) + b * GROUPS + tl.arange(0, GROUPS), scale)


def quant(x, pow2=False):
    m, k = x.shape
    q = torch.empty((m, k), device=x.device, dtype=torch.float8_e4m3fn)
    s = torch.empty((m, k // 128), device=x.device, dtype=torch.float32)
    groups = 1 if m == 1 else 4
    quantize[(m, k // (128 * groups))](
        x, q, s, k, x.stride(0), pow2, groups, num_warps=1 if groups == 1 else 4
    )
    return q, s


@tr.jit
def add_partials(P, Y, SIZE: tl.constexpr, PARTS: tl.constexpr, B: tl.constexpr):
    i = tl.program_id(0) * B + tl.arange(0, B)
    p = tl.arange(0, PARTS)
    x = tl.load(P + p[:, None] * SIZE + i[None, :])
    tl.store(Y + i, tl.sum(x, 0))


def grouping_tile(m):
    return 16 if m <= 256 else 32


@tr.jit(do_not_specialize=["P"])
def count_routes(R, Counts, Offsets, Start, P, B: tl.constexpr, WINDOW: tl.constexpr):
    i = tl.program_id(0) * B + tl.arange(0, B)
    valid = i < P
    if WINDOW:
        valid &= i >= 8 * tl.load(Start)
    e = tl.load(R + i, valid, 0)
    rank = tl.atomic_add(Counts + e, 1, valid, sem="relaxed")
    tl.store(Offsets + i, rank, valid)


@tr.jit
def make_blocks(Counts, Blocks, Experts, Pairs, BM: tl.constexpr):
    e = tl.arange(0, 256)
    counts = tl.load(Counts + e)
    sizes = tl.cdiv(counts, BM)
    end = tl.cumsum(sizes)
    tl.store(Blocks + e, end - sizes)
    tl.store(Blocks + 256, tl.sum(sizes, 0))
    for j in range(tl.max(sizes, 0)):
        tl.store(Experts + end - sizes + j, e, j < sizes)
    # Only padding entries are initialized. Valid rows are filled by scatter.
    i = tl.arange(0, BM)
    first_pad = (end * BM)[:, None] - (BM - i)[None, :]
    tl.store(
        Pairs + first_pad,
        -1,
        (i[None, :] >= (counts - (sizes - 1) * BM)[:, None]) & (sizes[:, None] > 0),
    )


@tr.jit
def scatter_pairs(
    R, Offsets, Blocks, Pairs, Start, NP, BM: tl.constexpr, B: tl.constexpr, WINDOW: tl.constexpr
):
    i = tl.program_id(0) * B + tl.arange(0, B)
    valid = i < NP
    if WINDOW:
        valid &= i >= 8 * tl.load(Start)
    e = tl.load(R + i, valid, 0)
    rank = tl.load(Offsets + i, valid, 0)
    b = tl.load(Blocks + e)
    tl.store(Pairs + b * BM + rank, i, valid)


def group_routes(routes, start=None):
    m = routes.shape[0]
    bm = grouping_tile(m)
    cap = tr.cdiv(m * 8, bm) + 256
    counts = torch.zeros((256,), dtype=torch.int32, device=routes.device)
    blocks = torch.empty((257,), dtype=torch.int32, device=routes.device)
    offsets = torch.empty_like(routes)
    experts = torch.empty((cap,), dtype=torch.int32, device=routes.device)
    pairs = torch.empty((cap * bm,), dtype=torch.int32, device=routes.device)
    first = routes if start is None else start
    count_routes[(tr.cdiv(m * 8, 128),)](
        routes, counts, offsets, first, m * 8, 128, start is not None
    )
    make_blocks[(1,)](counts, blocks, experts, pairs, bm)
    scatter_pairs[(tr.cdiv(m * 8, 128),)](
        routes, offsets, blocks, pairs, first, m * 8, bm, 128, start is not None
    )
    return pairs, experts, blocks


@tr.jit
def rms_kernel(X, W, Y, K: tl.constexpr, GEMMA: tl.constexpr):
    p = tl.program_id(0)
    i = tl.arange(0, K)
    x = tl.load(X + p * K + i).to(tl.float32)
    w = tl.load(W + i).to(tl.float32)
    if GEMMA:
        w += 1.0
    v = tl.sum(x * x, 0) * (1.0 / K)
    tl.store(Y + p * K + i, (x * tl.rsqrt(v + 1.0e-6)) * w)


def rms(x, w):
    y = torch.empty_like(x)
    k = x.shape[-1]
    rms_kernel[(x.numel() // k,)](x, w, y, k, True, num_warps=4 if k >= 1024 else 1)
    return y


@tr.jit
def routing(
    L, Id, Weight, X, W, Sig, LS: tl.constexpr, FUSED: tl.constexpr, ROUTES: tl.constexpr = 8
):
    r = tl.program_id(0)
    i = tl.arange(0, 256)
    x = tl.load(L + r * LS + i).to(tl.float32)
    bits = x.to(tl.bfloat16).to(tl.int16, bitcast=True).to(tl.int32) & 65535
    ordered = tl.where(bits >= 32768, 65535 - bits, bits + 32768)
    best = tl.topk((ordered << 8) | (255 - i), ROUTES)
    indices = 255 - (best & 255)
    values = tl.load(L + r * LS + indices).to(tl.float32)
    prob = tl.exp(values - tl.max(values, 0))
    prob /= tl.sum(prob, 0)
    slot = tl.arange(0, ROUTES)
    tl.store(Id + r * 8 + slot, indices)
    tl.store(Weight + r * 8 + slot, prob)
    if FUSED:
        j = tl.arange(0, 2048)
        h = tl.load(X + r * 2048 + j).to(tl.float32)
        w = tl.load(W + j).to(tl.float32)
        gate = tl.sum(h * w, 0)
        if FUSED == 2:
            gate = gate.to(tl.bfloat16).to(tl.float32)
            tl.store(Sig + r, tl.sigmoid(gate).to(tl.bfloat16))
        else:
            tl.store(Sig + r, tl.sigmoid(gate))
    else:
        gate = tl.load(L + r * LS + 256).to(tl.float32)
        tl.store(Sig + r, tl.sigmoid(gate).to(tl.bfloat16).to(tl.float32))


@tr.jit
def routing_union(
    L,
    Id,
    Weight,
    Sig,
    Experts,
    Pairs,
    Live,
    LS: tl.constexpr,
    M: tl.constexpr,
    BM: tl.constexpr,
    B: tl.constexpr,
    SLOTS: tl.constexpr,
    ADAPTIVE: tl.constexpr,
    COMPACT: tl.constexpr = False,
    PDL: tl.constexpr = False,
):
    if PDL:
        tl.extra.cuda.gdc_wait()
    live = tl.load(Live) if ADAPTIVE else M
    r = tl.arange(0, BM)
    i = tl.arange(0, 256)
    logits = tl.load(L + r[:, None] * LS + i[None, :], r[:, None] < live, 0)
    bits = logits.to(tl.bfloat16).to(tl.int16, bitcast=True).to(tl.int32) & 65535
    ordered = tl.where(bits >= 32768, 65535 - bits, bits + 32768)
    best = tl.topk((ordered << 8) | (255 - i[None, :]), 8)
    indices = 255 - (best & 255)
    v = tl.load(L + r[:, None] * LS + indices, r[:, None] < live, 0).to(tl.float32)
    p = tl.exp(v - tl.max(v, 1)[:, None])
    p /= tl.sum(p, 1)[:, None]
    to = r[:, None] * 8 + tl.arange(0, 8)[None, :]
    tl.store(Id + to, indices, r[:, None] < live)
    tl.store(Weight + to, p, r[:, None] < live)
    gate = tl.load(L + r * LS + 256, r < live, 0).to(tl.float32)
    tl.store(Sig + r, tl.sigmoid(gate).to(tl.bfloat16), r < live)

    u = tl.arange(0, B)
    active = u < live * SLOTS
    if SLOTS == 9:
        slot = u % SLOTS
        index = tl.minimum(u // SLOTS * 8 + slot, BM * 8 - 1)
        expert = tl.where(slot == 8, 256, tl.gather(tl.reshape(indices, (BM * 8,)), index, 0))
    else:
        expert = tl.reshape(indices, (B,))
    expert = tl.where(active, expert, -1)
    equal = (expert[:, None] == expert[None, :]) & active[:, None]
    first = tl.min(tl.where(equal, u[None, :], B), 1)
    unique = first == u
    if COMPACT:
        count = tl.cumsum(unique.to(tl.int32), 0)
        dest = count - 1
        final = tl.max(count, 0)
        tl.store(Experts + u, -1, (u >= final) & (u < M * SLOTS))
        tl.store(Experts + dest, expert, unique)
    else:
        dest = u
        tl.store(Experts + u, tl.where(active & unique, expert, -1), u < M * SLOTS)
    for row in tl.static_range(M):
        pair = tl.min(tl.where(equal & (u[None, :] // SLOTS == row), u[None, :], B), 1)
        tl.store(Pairs + dest * M + row, tl.where(pair < B, pair, -1), active & unique)


@tr.jit
def encode_mx(scale):
    exp = (scale.to(tl.int32, bitcast=True) >> 23) & 255
    return exp * 0x01010101


@tr.jit
def store_mx_group(Out, scale, group, row, M: tl.constexpr):
    # One int32 word repeats the 128-channel scale in the four hardware
    # 32-channel fields. Each group occupies a 128-row scale tile.
    word = (row % 32) * 4 + row // 32
    tl.store(Out + group * 128 + word, encode_mx(scale))
    if MX_PAD:
        if row == 0:
            pad = tl.arange(0, 128)
            pad_row = (pad % 4) * 32 + pad // 4
            tl.store(Out + group * 128 + pad, 0x7F7F7F7F, pad_row >= M)


def mx_buffer(x, k):
    return torch.empty((128 * (k // 128),), device=x.device, dtype=torch.int32)


@tr.jit
def glu_quant_values(
    gate,
    up,
    is_shared,
    POW2: tl.constexpr,
    ROUND: tl.constexpr,
    FP8_UP: tl.constexpr,
    MIXED_SHARED: tl.constexpr,
    SHARED_POW2: tl.constexpr,
):
    if MIXED_SHARED:
        gate = tl.where(is_shared, gate.to(tl.bfloat16).to(tl.float32), gate)
        up = tl.where(is_shared, up.to(tl.bfloat16).to(tl.float32), up)
    if FP8_UP:
        sg = tl.maximum(tl.max(tl.abs(gate), 0) * (1.0 / 448.0), 1.1754943508222875e-38)
        su = tl.maximum(tl.max(tl.abs(up), 0) * (1.0 / 448.0), 1.1754943508222875e-38)
        g8 = tl.clamp(gate / sg, -448.0, 448.0).to(tl.float8e4nv).to(tl.float32) * sg
        u8 = tl.clamp(up / su, -448.0, 448.0).to(tl.float8e4nv).to(tl.float32) * su
        gate = tl.where(is_shared, gate, g8)
        up = tl.where(is_shared, up, u8)
    val = gate / (1.0 + tl.exp(-gate)) * up
    if ROUND >= 1:
        val = val.to(tl.bfloat16).to(tl.float32)
    elif MIXED_SHARED:
        val = tl.where(is_shared, val.to(tl.bfloat16).to(tl.float32), val)
    amax = tl.maximum(tl.max(tl.abs(val), 0), 1.0e-10)
    scale = amax * (1.0 / 448.0)
    if POW2:
        scale = scale_ceil(scale)
        inv = 1.0 / scale
    else:
        inv = 448.0 / amax
        if MIXED_SHARED and SHARED_POW2:
            scale_p2 = scale_ceil(scale)
            inv = tl.where(is_shared, 1.0 / scale_p2, inv)
            scale = tl.where(is_shared, scale_p2, scale)
    return tl.clamp(val * inv, -448.0, 448.0), scale


@tr.jit
def glu_quant(
    X,
    Q,
    S,
    MX,
    Limit,
    Start,
    POW2: tl.constexpr,
    ROUND: tl.constexpr,
    FP8_UP: tl.constexpr,
    ENCODE_MX: tl.constexpr,
    M: tl.constexpr,
    MIXED_SHARED: tl.constexpr,
    SHARED_POW2: tl.constexpr,
    CAP: tl.constexpr,
    PDL: tl.constexpr,
    WINDOW: tl.constexpr,
):
    if PDL:
        tl.extra.cuda.gdc_wait()
        tl.extra.cuda.gdc_launch_dependents()
    p = tl.program_id(0)
    if CAP and p >= tl.load(Limit):
        return
    if WINDOW and p < 8 * tl.load(Start):
        return
    g = tl.program_id(1)
    i = tl.arange(0, 128)
    gate = tl.load(X + p * 1024 + g * 128 + i).to(tl.float32)
    up = tl.load(X + p * 1024 + 512 + g * 128 + i).to(tl.float32)
    is_shared = p % 9 == 8 if MIXED_SHARED else False
    value, scale = glu_quant_values(
        gate, up, is_shared, POW2, ROUND, FP8_UP, MIXED_SHARED, SHARED_POW2
    )
    tl.store(Q + p * 512 + g * 128 + i, value)
    tl.store(S + p * 4 + g, scale)
    if ENCODE_MX:
        store_mx_group(MX, scale, g, p, M)


def activation(
    x,
    pow2,
    rounded,
    fp8_up=False,
    mx=False,
    mixed_shared=False,
    shared_pow2=False,
    limit=None,
    pdl=False,
    start=None,
):
    m = x.shape[0]
    q = torch.empty((m, 512), device=x.device, dtype=torch.float8_e4m3fn)
    s = torch.empty((m, 4), device=x.device, dtype=torch.float32)
    encoded = mx_buffer(x, 512) if mx else s
    glu_quant[(m, 4)](
        x,
        q,
        s,
        encoded,
        s if limit is None else limit,
        s if start is None else start,
        pow2,
        int(rounded),
        fp8_up,
        mx,
        m if mx else 0,
        mixed_shared,
        shared_pow2,
        limit is not None,
        pdl,
        start is not None,
        num_warps=1,
        launch_pdl=pdl,
    )
    return (q, s, encoded.view(torch.uint8)) if mx else (q, s)


@tr.jit
def prefill_glu_quant(X, Q, S, Limit, ROUND: tl.constexpr, WIDE: tl.constexpr):
    row = tl.program_id(0)
    if row < tl.load(Limit):
        source = row.to(tl.int64) if WIDE else row
        group = tl.arange(0, 4)
        i = group[:, None] * 128 + tl.arange(0, 128)[None, :]
        gate = tl.load(X + source * 1024 + i).to(tl.float32)
        up = tl.load(X + source * 1024 + 512 + i).to(tl.float32)
        val = gate / (1.0 + tl.exp(-gate)) * up
        if ROUND:
            val = val.to(tl.bfloat16).to(tl.float32)
        q, scale = quant_values(val, False)
        tl.store(Q + row * 512 + i, q)
        tl.store(S + row * 4 + group, scale)


def grouped_activation(x, limit, rounded):
    m = x.shape[0]
    q = torch.empty((m, 512), device=x.device, dtype=torch.float8_e4m3fn)
    s = torch.empty((m, 4), device=x.device, dtype=torch.float32)
    prefill_glu_quant[(m,)](x, q, s, limit, rounded, x.numel() >= 2**31, num_warps=4)
    return q, s


@tr.jit
def glu_quant_partials(
    P,
    AS,
    SS,
    WS,
    R,
    Q,
    S,
    Live,
    ROWS: tl.constexpr,
    SLOTS: tl.constexpr,
    FIRST: tl.constexpr,
    BF16_UP: tl.constexpr,
    ROUND: tl.constexpr,
    FP8_UP: tl.constexpr,
    SHARED_POW2: tl.constexpr,
    PDL: tl.constexpr,
    ADAPTIVE: tl.constexpr,
):
    if PDL:
        tl.extra.cuda.gdc_wait()
        tl.extra.cuda.gdc_launch_dependents()
    p, g = tl.program_id(0), tl.program_id(1)
    i = tl.arange(0, 128)
    row, slot = p // SLOTS, p % SLOTS
    if ADAPTIVE and row >= tl.load(Live):
        return
    is_shared = slot == 8 if SLOTS == 9 else False
    expert = tl.load(R + row * 8 + slot, slot < 8, 256)
    xs = tl.where(is_shared, SS, AS) + row * 16
    ws = WS + (expert * 8 + g) * 16
    addr = p * 1024 + g * 128 + i
    gate = tl.load(P + addr)
    up = tl.load(P + addr + 512)
    for k in tl.static_range(FIRST, 16):
        offset = (k - FIRST + 1) * ROWS * 1024
        a = tl.load(xs + k)
        sg = a * tl.load(ws + k)
        su = a * tl.load(ws + k + 64)
        gate = tl.fma(tl.load(P + addr + offset), sg, gate)
        up = tl.fma(tl.load(P + addr + offset + 512), su, up)
    if BF16_UP:
        gate = gate.to(tl.bfloat16).to(tl.float32)
        up = up.to(tl.bfloat16).to(tl.float32)
    value, scale = glu_quant_values(
        gate, up, is_shared, False, ROUND, FP8_UP, SLOTS == 9, SHARED_POW2
    )
    tl.store(Q + p * 512 + g * 128 + i, value)
    tl.store(S + p * 4 + g, scale)


def activation_partials(
    partials,
    incoming,
    shared,
    weight,
    routes,
    slots,
    split,
    bf16_up,
    rounded,
    fp8_up,
    shared_pow2,
    pdl=False,
    live=None,
):
    rows = partials.shape[1]
    q = torch.empty((rows, 512), device=partials.device, dtype=torch.float8_e4m3fn)
    s = torch.empty((rows, 4), device=partials.device, dtype=torch.float32)
    glu_quant_partials[(rows, 4)](
        partials,
        incoming[1],
        shared[1],
        weight[1],
        routes,
        q,
        s,
        routes if live is None else live,
        rows,
        slots,
        16 // split,
        bf16_up,
        rounded,
        fp8_up,
        shared_pow2,
        pdl,
        live is not None,
        num_warps=GLU_WARPS,
        launch_pdl=pdl,
    )
    return q, s


@tr.jit
def moe_combined_values(
    E,
    Weight,
    Shared,
    Sig,
    Order,
    p,
    i,
    RAW_GATE: tl.constexpr,
    SLOTS: tl.constexpr,
    SORTED: tl.constexpr,
    WIDE: tl.constexpr,
):
    slot = tl.arange(0, min(SLOTS, 8))
    w = tl.load(Weight + p * 8 + slot).to(tl.float32)
    row = tl.load(Order + p * 8 + slot) if SORTED else p * SLOTS + slot
    if WIDE:
        row = row.to(tl.int64)
    x = tl.load(E + row[:, None] * 2048 + i[None, :]).to(tl.float32)
    routed = tl.sum(x * w[:, None], 0).to(tl.bfloat16).to(tl.float32)
    sig = tl.load(Sig + p)
    if SLOTS == 9:
        shared = tl.load(E + (p * SLOTS + 8) * 2048 + i).to(tl.float32)
    else:
        shared = tl.load(Shared + p * 2048 + i).to(tl.float32)
    if RAW_GATE:
        result = tl.fma(shared, sig, routed)
    else:
        result = routed + (shared * sig).to(tl.bfloat16).to(tl.float32)
    return result


@tr.jit
def moe_combine(
    E,
    Weight,
    Shared,
    Sig,
    Y,
    Order,
    RAW_GATE: tl.constexpr,
    B: tl.constexpr,
    SLOTS: tl.constexpr = 8,
    SORTED: tl.constexpr = False,
    WIDE: tl.constexpr = False,
):
    p = tl.program_id(0)
    i = tl.program_id(1) * B + tl.arange(0, B)
    result = moe_combined_values(E, Weight, Shared, Sig, Order, p, i, RAW_GATE, SLOTS, SORTED, WIDE)
    tl.store(Y + p * 2048 + i, result)


@tr.jit(do_not_specialize=["M"])
def prefill_gdn_gates(BA, ALOG, DT, Gate, Beta, M, B: tl.constexpr):
    index = tl.program_id(0) * B + tl.arange(0, B)
    row, head = index // 32, index % 32
    b = tl.load(BA + row * 64 + head, row < M, 0).to(tl.float32)
    a = tl.load(BA + row * 64 + 32 + head, row < M, 0).to(tl.float32)
    x = a + tl.load(DT + head).to(tl.float32)
    decay = -libdevice.exp(tl.load(ALOG + head).to(tl.float32))
    softplus = tl.where(x > 20.0, x, libdevice.log1p(libdevice.exp(x)))
    tl.store(Gate + index, libdevice.exp(decay * softplus), row < M)
    beta = libdevice.div_rn(1.0, 1.0 + libdevice.exp(-b))
    tl.store(Beta + index, beta.to(tl.bfloat16).to(tl.float32), row < M)


@tr.jit
def gdn_gated(Y, Z, W, Output, ZS: tl.constexpr):
    r, h = tl.program_id(0), tl.program_id(1)
    i = tl.arange(0, 128)
    y = tl.load(Y + r * 4096 + h * 128 + i).to(tl.float32)
    z = tl.load(Z + r * ZS + h * 128 + i).to(tl.float32)
    w = tl.load(W + i).to(tl.float32)
    rms = tl.rsqrt(tl.sum(y * y, 0) * (1.0 / 128.0) + 1.0e-6)
    tl.store(Output + r * 4096 + h * 128 + i, (y * rms * w) * (z / (1.0 + tl.exp(-z))))


@tr.jit
def sigmoid_mul(X, Gate, Y, N, B: tl.constexpr):
    i = tl.program_id(0) * B + tl.arange(0, B)
    x = tl.load(X + i, i < N, 0).to(tl.float32)
    g = tl.load(Gate + i, i < N, 0).to(tl.float32)
    tl.store(Y + i, x / (1.0 + tl.exp(-g)), i < N)


@tr.jit
def add_rms_kernel(R, X, W, Y, R_OUT, ROUND: tl.constexpr):
    row = tl.program_id(0)
    i = tl.arange(0, 2048)
    off = row * 2048 + i
    x = tl.load(R + off).to(tl.float32) + tl.load(X + off).to(tl.float32)
    tl.store(R_OUT + off, x)
    if ROUND:
        x = x.to(tl.bfloat16).to(tl.float32)
    weight = 1.0 + tl.load(W + i).to(tl.float32)
    inv = tl.rsqrt(tl.sum(x * x, 0) * (1.0 / 2048.0) + 1.0e-6)
    tl.store(Y + off, x * inv * weight)


def add_rms(residual, x, w, rounded=False):
    output = torch.empty_like(residual)
    added = torch.empty_like(residual)
    add_rms_kernel[(x.shape[0],)](residual, x, w, output, added, rounded, num_warps=4)
    return added, output


@tr.jit
def small_bf16_mv(X, W, Output, N: tl.constexpr, ROWS: tl.constexpr, PDL: tl.constexpr):
    token = tl.program_id(1)
    rows = tl.program_id(0) * ROWS + tl.arange(0, ROWS)
    d = tl.arange(0, 2048)
    w = tl.load(W + rows[:, None] * 2048 + d[None, :], rows[:, None] < N, 0).to(tl.float32)
    if PDL:
        tl.extra.cuda.gdc_wait()
        tl.extra.cuda.gdc_launch_dependents()
    x = tl.load(X + token * 2048 + d).to(tl.float32)
    y = tl.sum(x[None, :] * w, 1)
    tl.store(Output + token * N + rows, y, rows < N)


def bf16_linear(x, w, pdl=False):
    n = w.shape[0]
    m = x.shape[0]
    out = torch.empty((m, n), device=x.device, dtype=x.dtype)
    rows = 1 if n <= 64 else 2
    small_bf16_mv[(tr.cdiv(n, rows), m)](x, w, out, n, rows, pdl, num_warps=4, launch_pdl=pdl)
    return out


@tr.jit
def quant_values(x, POW2: tl.constexpr):
    amax = tl.maximum(tl.max(tl.abs(x), 1), 1.0e-10)
    scale = amax * (1.0 / 448.0)
    if POW2:
        scale = scale_ceil(scale)
        inv = 1.0 / scale
    else:
        inv = 448.0 / amax
    return tl.clamp(x * inv[:, None], -448.0, 448.0), scale


@tr.jit
def store_norm_fp8(
    norm,
    row,
    QA,
    SA,
    QB,
    SB,
    MX,
    POW2: tl.constexpr,
    DUAL: tl.constexpr,
    ENCODE_MX: tl.constexpr,
    M: tl.constexpr,
):
    off = row * 2048 + tl.arange(0, 2048)
    groups = tl.reshape(norm.to(tl.float32), (16, 128))
    q, s = quant_values(groups, POW2)
    g = row * 16 + tl.arange(0, 16)
    tl.store(QA + off, tl.reshape(q, (2048,)))
    tl.store(SA + g, s)
    if DUAL:
        q2, s2 = quant_values(groups, True)
        tl.store(QB + off, tl.reshape(q2, (2048,)))
        tl.store(SB + g, s2)
    else:
        s2 = s
    if ENCODE_MX:
        if MX_PAD:
            pad = tl.arange(0, 128)
            pad_row = (pad % 4) * 32 + pad // 4
            val = tl.where(pad_row[None, :] == row, encode_mx(s2)[:, None], 0x7F7F7F7F)
            dst = MX + tl.arange(0, 16)[:, None] * 128 + pad[None, :]
            tl.store(dst, val, ((pad_row == row) | ((row == 0) & (pad_row >= M)))[None, :])
        else:
            word = (row % 32) * 4 + row // 32
            tl.store(MX + tl.arange(0, 16) * 128 + word, encode_mx(s2))


@tr.jit
def add_rms_fp8(
    R,
    X,
    W,
    Y,
    R_OUT,
    QA,
    SA,
    QB,
    SB,
    MX,
    ROUND: tl.constexpr,
    POW2: tl.constexpr,
    DUAL: tl.constexpr,
    ENCODE_MX: tl.constexpr,
    M: tl.constexpr,
    PDL: tl.constexpr,
):
    row = tl.program_id(0)
    i = tl.arange(0, 2048)
    off = row * 2048 + i
    residual = tl.load(R + off).to(tl.float32)
    weight = 1.0 + tl.load(W + i).to(tl.float32)
    if PDL:
        tl.extra.cuda.gdc_wait()
        tl.extra.cuda.gdc_launch_dependents()
    x = residual + tl.load(X + off).to(tl.float32)
    tl.store(R_OUT + off, x)
    if ROUND:
        x = x.to(tl.bfloat16).to(tl.float32)
    inv = tl.rsqrt(tl.sum(x * x, 0) * (1.0 / 2048.0) + 1.0e-6)
    norm = (x * inv * weight).to(tl.bfloat16)
    tl.store(Y + off, norm)
    store_norm_fp8(norm, row, QA, SA, QB, SB, MX, POW2, DUAL, ENCODE_MX, M)


@tr.jit
def moe_next_norm(
    E,
    Prob,
    Shared,
    Sig,
    R,
    W,
    Y,
    R_OUT,
    QA,
    SA,
    MX,
    Order,
    Live,
    Start,
    RAW_GATE: tl.constexpr,
    SLOTS: tl.constexpr,
    ROUND: tl.constexpr,
    QUANT: tl.constexpr,
    POW2: tl.constexpr,
    ENCODE_MX: tl.constexpr,
    M: tl.constexpr,
    SORTED: tl.constexpr,
    ADAPTIVE: tl.constexpr,
    WIDE: tl.constexpr,
    WINDOW: tl.constexpr,
    PDL: tl.constexpr,
):
    row = tl.program_id(0)
    i = tl.arange(0, 2048)
    off = row * 2048 + i
    leftover = tl.load(R + off).to(tl.float32)
    weight = 1.0 + tl.load(W + i).to(tl.float32)
    if PDL:
        tl.extra.cuda.gdc_wait()
        tl.extra.cuda.gdc_launch_dependents()
    valid = True
    if ADAPTIVE:
        valid = row < tl.load(Live)
    if WINDOW:
        valid = valid & (row >= tl.load(Start))
    if valid:
        mix = moe_combined_values(
            E, Prob, Shared, Sig, Order, row, i, RAW_GATE, SLOTS, SORTED, WIDE
        )
    else:
        mix = tl.full((2048,), 0.0, tl.float32)
    x = leftover + mix.to(tl.bfloat16).to(tl.float32)
    tl.store(R_OUT + off, x)
    if ROUND:
        x = x.to(tl.bfloat16).to(tl.float32)
    inv = tl.rsqrt(tl.sum(x * x, 0) * (1.0 / 2048.0) + 1.0e-6)
    norm = (x * inv * weight).to(tl.bfloat16)
    tl.store(Y + off, norm)
    if QUANT:
        store_norm_fp8(norm, row, QA, SA, QA, SA, MX, POW2, False, ENCODE_MX, M)


def combine_next_norm(
    experts,
    probs,
    shared,
    sigmoid,
    residual,
    w,
    raw_gate,
    slots,
    rounded,
    quant,
    pow2,
    mx,
    order=None,
    live=None,
    start=None,
    warps=4,
    pdl=False,
):
    m, width = residual.shape
    output = torch.empty_like(residual)
    added = torch.empty_like(residual)
    q = (
        torch.empty((m, width), device=residual.device, dtype=torch.float8_e4m3fn)
        if quant
        else output
    )
    s = (
        torch.empty((m, width // 128), device=residual.device, dtype=torch.float32)
        if quant
        else output
    )
    mx = mx and quant
    encoded = mx_buffer(residual, width) if mx else s
    moe_next_norm[(m,)](
        experts,
        probs,
        shared,
        sigmoid,
        residual,
        w,
        output,
        added,
        q,
        s,
        encoded,
        probs if order is None else order,
        probs if live is None else live,
        probs if start is None else start,
        raw_gate,
        slots,
        rounded,
        quant,
        pow2,
        mx,
        m if mx else 0,
        order is not None,
        live is not None,
        experts.numel() >= 2**31,
        start is not None,
        pdl,
        num_warps=warps,
        launch_pdl=pdl,
    )
    activation = (q, s, encoded.view(torch.uint8)) if mx else (q, s)
    return added, output, activation


def add_rms_quant(residual, x, w, rounded, pow2, dual=False, mx=False, exact=False, pdl=False):
    m, width = x.shape
    y = torch.empty_like(x)
    added = torch.empty_like(x)
    qa = torch.empty((m, width), device=x.device, dtype=torch.float8_e4m3fn)
    sa = torch.empty((m, width // 128), device=x.device, dtype=torch.float32)
    qb, sb = (torch.empty_like(qa), torch.empty_like(sa)) if dual else (qa, sa)
    encoded = mx_buffer(x, width) if mx else sa
    if exact and not mx:
        from . import norm_exact

        norm_exact.add_rms_quant[(m,)](
            residual, x, w, y, added, qa, sa, qb, sb, rounded, pow2, dual, num_warps=4
        )
    else:
        add_rms_fp8[(m,)](
            residual,
            x,
            w,
            y,
            added,
            qa,
            sa,
            qb,
            sb,
            encoded,
            rounded,
            pow2,
            dual,
            mx,
            m if mx else 0,
            pdl,
            num_warps=4,
            launch_pdl=pdl,
        )
    first, second = (qa, sa), (qb, sb)
    if mx:
        sf = encoded.view(torch.uint8)
        if dual:
            second += (sf,)
        else:
            first += (sf,)
            second = first
    return added, y, first, second


@tr.jit
def gdn_gated_fp8(
    Y,
    Z,
    W,
    Q,
    S,
    MX,
    ZS: tl.constexpr,
    POW2: tl.constexpr,
    ENCODE_MX: tl.constexpr,
    M: tl.constexpr,
):
    r, h = tl.program_id(0), tl.program_id(1)
    i = tl.arange(0, 128)
    y = tl.load(Y + r * 4096 + h * 128 + i).to(tl.float32)
    z = tl.load(Z + r * ZS + h * 128 + i).to(tl.float32)
    w = tl.load(W + i).to(tl.float32)
    rms = tl.rsqrt(tl.sum(y * y, 0) * (1.0 / 128.0) + 1.0e-6)
    u = ((y * rms * w) * (z / (1.0 + tl.exp(-z)))).to(tl.bfloat16).to(tl.float32)
    q, s = quant_values(tl.reshape(u, (1, 128)), POW2)
    tl.store(Q + r * 4096 + h * 128 + i, tl.reshape(q, (128,)))
    scale = tl.sum(s, 0)
    tl.store(S + r * 32 + h, scale)
    if ENCODE_MX:
        store_mx_group(MX, scale, h, r, M)


@tr.jit
def gdn_prefill_output(
    Y, Z, W, Q, S, ZS: tl.constexpr, POW2: tl.constexpr, G: tl.constexpr, WIDE: tl.constexpr
):
    row, block = tl.program_id(0), tl.program_id(1)
    group = block * G + tl.arange(0, G)
    channel = tl.arange(0, 128)
    i = group[:, None] * 128 + channel[None, :]
    y = tl.load(Y + row * 4096 + i).to(tl.float32)
    z_row = row.to(tl.int64) if WIDE else row
    z = tl.load(Z + z_row * ZS + i).to(tl.float32)
    w = tl.load(W + channel).to(tl.float32)
    inv = tl.rsqrt(tl.sum(y * y, 1) * (1.0 / 128.0) + 1.0e-6)
    x = ((y * inv[:, None]) * w[None, :]) * (z / (1.0 + tl.exp(-z)))
    q, s = quant_values(x.to(tl.bfloat16).to(tl.float32), POW2)
    tl.store(Q + row * 4096 + i, q)
    tl.store(S + row * 32 + group, s)


@tr.jit
def softmax_prefill_output(
    Y, Gate, Q, S, MX, POW2: tl.constexpr, G: tl.constexpr, ENCODE_MX: tl.constexpr, M: tl.constexpr
):
    row, block = tl.program_id(0), tl.program_id(1)
    group = block * G + tl.arange(0, G)
    i = row * 4096 + group[:, None] * 128 + tl.arange(0, 128)[None, :]
    y = tl.load(Y + i).to(tl.float32)
    g = tl.load(Gate + i).to(tl.float32)
    x = (y * tl.sigmoid(g)).to(tl.bfloat16).to(tl.float32)
    q, s = quant_values(x, POW2)
    tl.store(Q + i, q)
    tl.store(S + row * 32 + group, s)
    if ENCODE_MX:
        word = (row % 32) * 4 + row // 32
        tl.store(MX + group * 128 + word, encode_mx(s))
        if MX_PAD and row == 0:
            pad = tl.arange(0, 128)
            pad_row = (pad % 4) * 32 + pad // 4
            tl.store(MX + group[:, None] * 128 + pad[None, :], 0x7F7F7F7F, pad_row[None, :] >= M)


@tr.jit
def attn_merge_fp8(
    P,
    L,
    Gate,
    Q,
    QS,
    MX,
    S: tl.constexpr,
    POW2: tl.constexpr,
    ENCODE_MX: tl.constexpr,
    M: tl.constexpr = 1,
    PDL: tl.constexpr = False,
):
    h, block = tl.program_id(0), tl.program_id(1)
    row = tl.program_id(2) if M > 1 else 0
    h += row * 16
    d = block * 128 + tl.arange(0, 128)
    i = tl.arange(0, S)
    gate = tl.load(Gate + h * 256 + d).to(tl.float32)
    if PDL:
        tl.extra.cuda.gdc_wait()
        tl.extra.cuda.gdc_launch_dependents()
    logsumexp = tl.load(L + h * S + i)
    w = tl.exp(logsumexp - tl.max(logsumexp, 0))
    x = tl.load(P + (h * S + i[:, None]) * 256 + d[None, :], logsumexp[:, None] > -float("inf"), 0)
    y = (tl.sum(x * w[:, None], 0) / tl.sum(w, 0)).to(tl.bfloat16).to(tl.float32)
    value = (y / (1.0 + tl.exp(-gate))).to(tl.bfloat16).to(tl.float32)
    q, scale = quant_values(tl.reshape(value, (1, 128)), POW2)
    tl.store(Q + h * 256 + d, tl.reshape(q, (128,)))
    s_val = tl.sum(scale, 0)
    tl.store(QS + h * 2 + block, s_val)
    if ENCODE_MX:
        store_mx_group(MX, s_val, (h % 16) * 2 + block, row, M)

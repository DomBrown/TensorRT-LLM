# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Device kernels for Qwen's block-FP8 projections and attention."""

import torch
import triton
import triton.language as tl
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.language.extra.cuda import libdevice

FP8 = torch.float8_e4m3fn
BF16 = torch.bfloat16


@triton.jit(do_not_specialize=["M"])
def _quantize(
    X,
    Q,
    S,
    M,
    K: tl.constexpr,
    SX: tl.constexpr,
    STYLE: tl.constexpr,
    POW2: tl.constexpr,
    R: tl.constexpr,
    MN_MAJOR: tl.constexpr,
    ACTIVE,
    WIDTH: tl.constexpr,
    RETIRE: tl.constexpr,
):
    r = tl.program_id(0) * R + tl.arange(0, R)
    g = tl.program_id(1)
    c = g * 128 + tl.arange(0, 128)
    live = r < M
    if RETIRE:
        live &= tl.load(ACTIVE + r // (9 * WIDTH), live, 0) != 0
    a = tl.load(X + r[:, None] * SX + c[None, :], live[:, None], 0).to(tl.float32)
    if STYLE == "swiglu":
        b = tl.load(X + r[:, None] * SX + K + c[None, :], live[:, None], 0).to(tl.float32)
        a = (a * tl.sigmoid(a) * b).to(tl.bfloat16).to(tl.float32)
    amax = tl.maximum(tl.max(tl.abs(a), 1), 1.0e-10)
    step = amax * (1.0 / 448.0)
    if POW2:
        step = tl.exp2(tl.ceil(tl.log2(step)))
        recip = 1.0 / step
    else:
        recip = 448.0 / amax
    q = tl.clamp(a * recip[:, None], -448.0, 448.0).to(tl.float8e4nv)
    tl.store(Q + r[:, None] * K + c[None, :], q, live[:, None])
    offset = g * M + r if MN_MAJOR else r * (K // 128) + g
    tl.store(S + offset, step, live)


def _scales(rows, groups, device, mn_major):
    if mn_major:
        return torch.empty((groups, rows), device=device, dtype=torch.float32).T
    return torch.empty((rows, groups), device=device, dtype=torch.float32)


def quantize(x, style="", pow2=False, mn_major=False, active=None, width=1):
    m, stride = x.shape
    k = stride // 2 if style else stride
    q = torch.empty((m, k), device=x.device, dtype=FP8)
    s = _scales(m, k // 128, x.device, mn_major)
    r = 4 if m >= 4 else 1
    _quantize[(triton.cdiv(m, r), k // 128)](
        x,
        q,
        s,
        m,
        k,
        stride,
        style,
        pow2,
        r,
        mn_major,
        x if active is None else active,
        width,
        active is not None,
        num_warps=4 if r == 4 else 1,
    )
    return q, s


@triton.jit(do_not_specialize=["M"])
def _fp8_dot(
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
    BM: tl.constexpr,
    BN: tl.constexpr,
    MODE: tl.constexpr,
    UP: tl.constexpr,
):
    tile = tl.program_id(0)
    col = tl.program_id(1) * BN + tl.arange(0, BN)
    i = tl.arange(0, BM)
    if MODE == "group":
        expert = tl.load(TILE_EXPERT + tile)
        if expert >= 257:
            return
        lo = tl.load(START + expert)
        hi = tl.load(END + expert)
        pair_index = lo + tile * BM - tl.load(START + 257 + expert) + i
        live = pair_index < hi
        row = tl.load(SORTED + pair_index, live, 0).to(tl.int32)
    elif MODE == "slot":
        expert = tl.load(ROUTE + tile).to(tl.int32)
        row = tile + i
        live = i == 0
    else:
        expert = 0
        row = tile * BM + i
        live = row < M
    if UP:
        src = row // 9
    else:
        src = row
    kk = tl.arange(0, 128)
    acc = tl.full((BM, BN), 0, tl.float32)
    for g in range(K // 128):
        a = tl.load(A + src[:, None] * K + 128 * g + kk[None, :], live[:, None], 0.0)
        w = tl.load(W + (expert * N + col[None, :]) * K + 128 * g + kk[:, None])
        dot = tl.dot(a, w, max_num_imprecise_acc=0)
        sa = tl.load(AS + src * (K // 128) + g, live, 0.0)
        sw = tl.load(WS + (expert * (N // 128) + col // 128) * (K // 128) + g).to(tl.float32)
        acc = tl.fma(dot, sa[:, None] * sw[None, :], acc)
    tl.store(Y + row[:, None] * N + col[None, :], acc, live[:, None])


def fp8_linear(a, scale, w, ws, *, route=None, schedule=None, up=False):
    m, k = a.shape
    n = w.shape[-2]
    if route is None:
        mode = ""
        rows = m
        blocks = triton.cdiv(m, 64)
    else:
        rows = route.numel()
        mode = "group" if schedule is not None else "slot"
        blocks = schedule[-1].numel() if schedule is not None else rows
    y = torch.empty((rows, n), dtype=torch.bfloat16, device=a.device)
    sorted_rows, starts, ends, experts = schedule if schedule is not None else (a, a, a, a)
    bn = 64 if m <= 128 else 128
    _fp8_dot[(blocks, n // bn)](
        a,
        scale,
        w,
        ws,
        y,
        route if route is not None else a,
        sorted_rows,
        starts,
        ends,
        experts,
        m,
        n,
        k,
        64,
        bn,
        mode,
        up,
        num_warps=4,
        num_stages=2,
    )
    return y


def _expert_tile_capacity(rows, tile_rows):
    # Each routed expert needs one assignment for its first tile and a full
    # tile of extra assignments for every additional tile it can activate.
    routed = rows * 8
    first = min(routed, 256)
    return first + (routed - first) // tile_rows + triton.cdiv(rows, tile_rows)


def expert_schedule(routes, tile_rows=64):
    """Shared expert is ID 256. Return packed route indices and padded tile metadata."""
    ids = routes.view(-1)
    indices = torch.argsort(ids, stable=False)
    counts = torch.zeros(257, device=ids.device, dtype=torch.int32)
    counts.scatter_add_(0, ids, torch.ones_like(ids, dtype=torch.int32))
    ends = counts.cumsum(0, dtype=torch.int32)
    padded = triton.cdiv(counts, tile_rows) * tile_rows
    padded_ends = padded.cumsum(0, dtype=torch.int32)
    starts = torch.cat((ends - counts, padded_ends - padded))
    upper_bound = _expert_tile_capacity(routes.shape[0], tile_rows)
    tiles = torch.arange(upper_bound, device=ids.device, dtype=torch.int32) * tile_rows
    experts = torch.searchsorted(padded_ends, tiles, right=True, out_int32=True)
    return indices, starts, ends, experts


@triton.jit
def _rms(X, W, Y, K: tl.constexpr, EPS: tl.constexpr):
    r = tl.program_id(0)
    c = tl.arange(0, K)
    x = tl.load(X + r * K + c).to(tl.float32)
    w = tl.load(W + c).to(tl.float32)
    y = x * tl.rsqrt(tl.sum(x * x, 0) * (1.0 / K) + EPS) * (1.0 + w)
    tl.store(Y + r * K + c, y)


def rms(x, w, eps=1e-6):
    y = torch.empty_like(x)
    _rms[(x.numel() // x.shape[-1],)](x, w, y, x.shape[-1], eps, num_warps=4)
    return y


@triton.jit
def _gdn_gate_values(a, b, PARAM, DT, h, ROUND_BETA: tl.constexpr):
    x = a + tl.load(DT + h).to(tl.float32)
    g = -tl.exp(tl.load(PARAM + h).to(tl.float32)) * tl.where(x <= 20.0, tl.log(1.0 + tl.exp(x)), x)
    beta = tl.sigmoid(b)
    if ROUND_BETA:
        beta = beta.to(tl.bfloat16).to(tl.float32)
    return tl.exp(g), beta


@triton.jit
def _gdn_gates(A, PARAM, DT, ALPHA, BETA, ROUND_BETA: tl.constexpr):
    t = tl.program_id(0)
    h = tl.arange(0, 32)
    a = tl.load(A + t * 64 + h).to(tl.float32)
    b = tl.load(A + t * 64 + 32 + h).to(tl.float32)
    alpha, beta = _gdn_gate_values(a, b, PARAM, DT, h, ROUND_BETA)
    tl.store(ALPHA + t * 32 + h, alpha)
    tl.store(BETA + t * 32 + h, beta)


def gdn_gates(ab, a_log, dt, round_beta):
    shape = (ab.shape[0], 32)
    alpha = torch.empty(shape, device=ab.device, dtype=torch.float32)
    beta = torch.empty_like(alpha)
    _gdn_gates[(shape[0],)](ab, a_log, dt, alpha, beta, round_beta, num_warps=1)
    return alpha, beta


@triton.jit
def _gdn(
    C,
    ALPHA,
    BETA,
    Y,
    STATE,
    CU,
    SLOTS,
    INITIAL,
    BV: tl.constexpr,
    PREFILL: tl.constexpr,
    SLOT_STRIDE: tl.constexpr,
    ROUND_QK: tl.constexpr,
):
    b = tl.program_id(0)
    hv = tl.program_id(1)
    v_idx = tl.program_id(2) * BV + tl.arange(0, BV)
    k_idx = tl.arange(0, 128)
    slot = tl.load(SLOTS + b).to(tl.int64)
    ptr = STATE + slot * SLOT_STRIDE + hv * 16384 + v_idx[:, None] * 128 + k_idx[None, :]
    if PREFILL:
        lo = tl.load(CU + b)
        hi = tl.load(CU + b + 1)
        state = tl.load(ptr, tl.load(INITIAL + b), 0).to(tl.float32)
    else:
        lo = b
        hi = b + 1
        state = tl.load(ptr).to(tl.float32)
    for t in range(lo, hi):
        q = tl.load(C + t * 8192 + hv // 2 * 128 + k_idx).to(tl.float32)
        k = tl.load(C + t * 8192 + 2048 + hv // 2 * 128 + k_idx).to(tl.float32)
        v = tl.load(C + t * 8192 + 4096 + hv * 128 + v_idx).to(tl.float32)
        q = q / tl.sqrt(tl.sum(q * q, 0) + 1e-6)
        k = k / tl.sqrt(tl.sum(k * k, 0) + 1e-6)
        if ROUND_QK:
            q = q.to(tl.bfloat16).to(tl.float32)
            k = k.to(tl.bfloat16).to(tl.float32)
        q *= 0.08838834764831845
        a = tl.load(ALPHA + t * 32 + hv)
        beta = tl.load(BETA + t * 32 + hv)
        state *= a
        delta = (v - tl.sum(state * k[None, :], 1)) * beta
        state = tl.fma(delta[:, None], k[None, :], state)
        o = tl.sum(state * q[None, :], 1)
        tl.store(Y + t * 4096 + hv * 128 + v_idx, o)
    tl.store(ptr, state)


def gdn_recurrent(mixed, alpha, beta, state, slots, offsets=None, initial=None):
    decode = offsets is None
    output = torch.empty((mixed.shape[0], 4096), device=mixed.device, dtype=mixed.dtype)
    bv = 8 if decode else 4
    _gdn[(slots.numel(), 32, 128 // bv)](
        mixed,
        alpha,
        beta,
        output,
        state,
        offsets,
        slots,
        initial,
        bv,
        not decode,
        state.stride(0),
        not decode,
        num_warps=4 if decode else 1,
    )
    return output


@gluon.jit
def _gate_quant_group(
    OUTPUT,
    Z,
    W,
    Q,
    S,
    STRIDE: gl.constexpr,
    HEADS: gl.constexpr,
    GDN: gl.constexpr,
    MN: gl.constexpr,
    POW2: gl.constexpr,
    ROUND_SIGMOID: gl.constexpr,
):
    row = gl.program_id(0)
    group = gl.program_id(1)
    layout: gl.constexpr = gl.BlockedLayout([1, 4], [1, 32], [HEADS, 1], [1, 0])
    h = group * HEADS + gl.arange(0, HEADS, layout=gl.SliceLayout(1, layout))
    c = gl.arange(0, 128, layout=gl.SliceLayout(0, layout))
    x = gl.load(OUTPUT + row * 4096 + h[:, None] * 128 + c[None, :]).to(gl.float32)
    if GDN:
        add = STRIDE - 4096 + h * 128
    else:
        add = (h // 2) * 512 + 256 + (h % 2) * 128
    z = gl.load(Z + row * STRIDE + add[:, None] + c[None, :]).to(gl.float32)
    gate = 1.0 / (1.0 + gl.exp(-z))
    if GDN:
        weight = gl.load(W + c).to(gl.float32)
        inverse = gl.rsqrt(gl.sum(x * x, 1) * (1.0 / 128.0) + 1e-6)
        y = x * inverse[:, None] * weight[None, :] * (z * gate)
    else:
        if ROUND_SIGMOID:
            gate = gate.to(gl.bfloat16).to(gl.float32)
        y = x * gate
    y = y.to(gl.bfloat16).to(gl.float32)
    maximum = gl.maximum(gl.max(gl.abs(y), 1), 1.0e-10)
    scale = maximum * (1.0 / 448.0)
    if POW2:
        scale = gl.exp2(gl.ceil(gl.log2(scale)))
        reciprocal = 1.0 / scale
    else:
        reciprocal = 448.0 / maximum
    q = gl.clamp(y * reciprocal[:, None], -448.0, 448.0).to(gl.float8e4nv)
    gl.store(Q + row * 4096 + h[:, None] * 128 + c[None, :], q)
    loc = h * gl.num_programs(0) + row if MN else row * 32 + h
    gl.store(S + loc, scale)


@triton.jit
def _gated_norm_quant(
    OUTPUT, Z, W, Q, S, SZ: tl.constexpr, POW2: tl.constexpr, MN_MAJOR: tl.constexpr
):
    r = tl.program_id(0)
    h = tl.program_id(1)
    c = tl.arange(0, 128)
    x = tl.load(OUTPUT + r * 4096 + h * 128 + c).to(tl.float32)
    z = tl.load(Z + r * SZ + SZ - 4096 + h * 128 + c).to(tl.float32)
    w = tl.load(W + c).to(tl.float32)
    y = x * tl.rsqrt(tl.sum(x * x, 0) / 128.0 + 1e-6) * w
    y = (y * (z * tl.sigmoid(z))).to(tl.bfloat16).to(tl.float32)
    amax = tl.maximum(tl.max(tl.abs(y), 0), 1.0e-10)
    step = amax * (1.0 / 448.0)
    if POW2:
        step = tl.exp2(tl.ceil(tl.log2(step)))
        recip = 1.0 / step
    else:
        recip = 448.0 / amax
    q = tl.clamp(y * recip, -448.0, 448.0).to(tl.float8e4nv)
    tl.store(Q + r * 4096 + h * 128 + c, q)
    tl.store(S + (h * tl.num_programs(0) + r if MN_MAJOR else r * 32 + h), step)


def gdn_output(core, proj, weight, pow2, mn_major=False, grouped=False):
    quant = torch.empty_like(core, dtype=FP8)
    scale = _scales(core.shape[0], 32, core.device, mn_major)
    if grouped and core.shape[0] >= 128:
        _gate_quant_group[(core.shape[0], 8)](
            core,
            proj,
            weight,
            quant,
            scale,
            proj.shape[1],
            4,
            True,
            mn_major,
            pow2,
            False,
            num_warps=4,
        )
    else:
        _gated_norm_quant[(core.shape[0], 32)](
            core, proj, weight, quant, scale, proj.shape[1], pow2, mn_major, num_warps=1
        )
    return quant, scale


@triton.jit
def _qkv(X, QW, KW, COS, SIN, POS, OUT, PACKED: tl.constexpr, K_OUT, V_OUT):
    row = tl.program_id(0)
    head = tl.program_id(1)
    channel = tl.arange(0, 256)
    pos = tl.load(POS + row)
    if head < 18:
        if head < 16:
            offset = head * 512
            weight = QW
        else:
            offset = 8192 + (head - 16) * 256
            weight = KW
        source = X + row * 9216 + offset
        x = tl.load(source + channel).to(tl.float32)
        partner = channel ^ 32
        other = tl.load(source + partner).to(tl.float32)
        norm = tl.rsqrt(tl.sum(x * x, 0) / 256.0 + 1e-6)
        x = x * norm * (1.0 + tl.load(weight + channel).to(tl.float32))
        other = other * norm * (1.0 + tl.load(weight + partner).to(tl.float32))
        c = tl.load(COS + pos * 32 + channel % 32)
        s = tl.load(SIN + pos * 32 + channel % 32)
        rotated = tl.fma(tl.where(channel < 32, -other, other), s, x * c)
        value = tl.where(channel < 64, rotated, x)
    else:
        value = tl.load(X + row * 9216 + 8704 + (head - 18) * 256 + channel).to(tl.float32)
    if PACKED:
        tl.store(OUT + row * 5120 + head * 256 + channel, value)
    elif head < 16:
        tl.store(OUT + row * 4096 + head * 256 + channel, value)
    elif head < 18:
        tl.store(K_OUT + row * 512 + (head - 16) * 256 + channel, value)
    else:
        tl.store(V_OUT + row * 512 + (head - 18) * 256 + channel, value)


def prepare_qkv(proj, qw, kw, cosine, sine, pos, packed=True):
    rows = proj.shape[0]
    if packed:
        out = proj.new_empty((rows, 5120))
        _qkv[(rows, 20)](proj, qw, kw, cosine, sine, pos, out, True, None, None, num_warps=4)
        return out
    q = proj.new_empty((rows, 16, 256))
    k = proj.new_empty((rows, 2, 256))
    v = torch.empty_like(k)
    _qkv[(rows, 20)](proj, qw, kw, cosine, sine, pos, q, False, k, v, num_warps=4)
    return q, k, v


@triton.jit
def _attention_gate_quant(
    OUTPUT, Z, Q, S, POW2: tl.constexpr, ROUND_SIGMOID: tl.constexpr, MN_MAJOR: tl.constexpr
):
    row = tl.program_id(0)
    g = tl.program_id(1)
    c = tl.arange(0, 128)
    x = tl.load(OUTPUT + row * 4096 + g * 128 + c).to(tl.float32)
    z = tl.load(Z + row * 9216 + (g // 2) * 512 + 256 + (g % 2) * 128 + c).to(tl.float32)
    sig = tl.sigmoid(z)
    if ROUND_SIGMOID:
        sig = sig.to(tl.bfloat16).to(tl.float32)
    y = (x * sig).to(tl.bfloat16).to(tl.float32)
    amax = tl.maximum(tl.max(tl.abs(y), 0), 1.0e-10)
    step = amax * (1.0 / 448.0)
    if POW2:
        step = tl.exp2(tl.ceil(tl.log2(step)))
        recip = 1.0 / step
    else:
        recip = 448.0 / amax
    q = tl.clamp(y * recip, -448.0, 448.0).to(tl.float8e4nv)
    tl.store(Q + row * 4096 + g * 128 + c, q)
    tl.store(S + (g * tl.num_programs(0) + row if MN_MAJOR else row * 32 + g), step)


def attention_output(core, proj, pow2, round_sigmoid, mn_major=False, grouped=False):
    q = torch.empty((proj.shape[0], 4096), device=proj.device, dtype=FP8)
    s = _scales(proj.shape[0], 32, proj.device, mn_major)
    if grouped and q.shape[0] >= 128:
        _gate_quant_group[(q.shape[0], 8)](
            core,
            proj,
            None,
            q,
            s,
            9216,
            4,
            False,
            mn_major,
            pow2,
            round_sigmoid,
            num_warps=4,
        )
    else:
        _attention_gate_quant[(q.shape[0], 32)](
            core, proj, q, s, pow2, round_sigmoid, mn_major, num_warps=1
        )
    return q, s


@triton.jit
def _route_logits(LOGITS, ROUTES, WEIGHTS, SLOTS, COUNTS, SCATTER: tl.constexpr):
    row = tl.program_id(0)
    expert = tl.arange(0, 256)
    logits = tl.load(LOGITS + row * 257 + expert)
    bits = logits.to(tl.uint16, bitcast=True).to(tl.uint32)
    bits = tl.where((bits & 32767) == 0, 0, bits)
    ordered = bits ^ tl.where((bits & 32768) != 0, 65535, 32768).to(tl.uint32)
    key = (ordered << 8) | (255 - expert).to(tl.uint32)
    selected = tl.topk(key, 8)
    selected_id = (255 - (selected & 255)).to(tl.int32)
    sorted_logits = tl.load(LOGITS + row * 257 + selected_id).to(tl.float32)
    ex = libdevice.exp(sorted_logits - tl.max(sorted_logits, 0))
    probability = libdevice.div_rn(ex, tl.sum(ex, 0))
    j = tl.arange(0, 8)
    tl.store(ROUTES + row * 9 + j, selected_id)
    tl.store(ROUTES + row * 9 + 8, 256)
    tl.store(WEIGHTS + row * 8 + j, probability)
    if SCATTER:
        local = tl.atomic_add(COUNTS + selected_id, 1, sem="relaxed")
        tl.store(SLOTS + row * 9 + j, local)
        tl.store(SLOTS + row * 9 + 8, row)


def route_logits(logits, round_router, counts=None):
    m = logits.shape[0]
    routes = torch.empty((m, 9), device=logits.device, dtype=torch.int64)
    probs = torch.empty((m, 8), device=logits.device, dtype=BF16 if round_router else torch.float32)
    slots = torch.empty_like(routes, dtype=torch.int32) if counts is not None else routes
    _route_logits[(m,)](
        logits,
        routes,
        probs,
        slots,
        counts if counts is not None else slots,
        counts is not None,
        num_warps=4,
    )
    return (routes, probs, slots) if counts is not None else (routes, probs)


@triton.jit(do_not_specialize=["B"])
def _expert_lists(ROUTES, ROWS, COUNTS, B, RB: tl.constexpr):
    expert = tl.program_id(0)
    i = tl.arange(0, RB)
    rid = tl.load(ROUTES + i, i < B * 9, 257)
    keep = rid == expert
    count = tl.cumsum(keep.to(tl.int32), 0)
    tl.store(ROWS + expert * B + count - 1, i, keep)
    tl.store(COUNTS + expert, tl.sum(keep.to(tl.int32), 0))


@triton.jit(do_not_specialize=["B", "TILES"])
def _expert_offsets(
    COUNTS,
    START,
    END,
    EXPERTS,
    ROWS,
    ACTIVE,
    B,
    TILES,
    AT: tl.constexpr,
    PAT: tl.constexpr,
    PTILES: tl.constexpr,
    RESET: tl.constexpr,
    WIDTH: tl.constexpr,
    RETIRE: tl.constexpr,
    PAD: tl.constexpr,
    EMIT_COUNT: tl.constexpr,
):
    i = tl.arange(0, 512)
    count = tl.load(COUNTS + i, i < 257, 0)
    if RESET:
        # Pack shared rows from the same sequence mask as the routed rows.
        if RETIRE:
            row = tl.arange(0, PAD)
            live = (tl.load(ACTIVE + row // WIDTH, row < B, 0) != 0) & (row < B)
            slot = tl.cumsum(live.to(tl.int32), 0) - 1
            tl.store(ROWS + 256 * B + slot, row * 9 + 8, live)
            shared = tl.sum(live.to(tl.int32), 0)
        else:
            shared = B
        count = tl.where(i == 256, shared, count)
        tl.store(COUNTS + i, 0, i < 256)
    ntiles = tl.cdiv(count, AT)
    exclusive = tl.cumsum(ntiles, 0) - ntiles
    first = i * B
    tl.store(START + i, first, i < 257)
    tl.store(START + 257 + i, exclusive * AT, i < 257)
    tl.store(END + i, first + count, i < 257)

    lane = tl.arange(0, PAT)
    tl.store(
        EXPERTS + exclusive[:, None] + lane[None, :],
        i[:, None],
        (i[:, None] < 257) & (lane[None, :] < ntiles[:, None]),
    )
    last = tl.sum(ntiles, 0)
    if EMIT_COUNT:
        tl.store(EXPERTS + TILES, last)
    t = tl.arange(0, PTILES)
    tl.store(EXPERTS + t, 257, (t >= last) & (t < TILES))


@triton.jit(do_not_specialize=["PAIRS", "TILES"])
def _small_expert_lists(
    ROUTES, ROWS, START, END, TILE_EXPERT, PAIRS, TILES, AT: tl.constexpr, BLOCK: tl.constexpr
):
    i = tl.arange(0, BLOCK)
    ids = tl.load(ROUTES + i, i < PAIRS, 257).to(tl.int32)
    tag = tl.sort(ids * BLOCK + i, descending=False)
    expert = tag // BLOCK
    source = tag % BLOCK
    prev = tl.gather(expert, tl.maximum(0, i - 1), 0)
    following = tl.gather(expert, tl.minimum(BLOCK - 1, i + 1), 0)
    live = i < PAIRS
    first = live & ((i == 0) | (expert != prev))
    last = live & ((i == PAIRS - 1) | (expert != following))
    tile = tl.cumsum(first.to(tl.int32), 0) - 1
    tl.store(ROWS + i, source, live)
    tl.store(START + expert, i, first)
    tl.store(END + expert, i + 1, last)
    tl.store(START + 257 + expert, tile * AT, first)
    tl.store(TILE_EXPERT + tile, expert, first)
    valid_tiles = tl.sum(first.to(tl.int32), 0)
    tl.store(TILE_EXPERT + i, 257, (i >= valid_tiles) & (i < TILES))


def schedule_expert_lists(
    rows, counts, m, tile_rows, reset_counts=False, active=None, width=1, emit_count=False
):
    device = rows.device
    start = torch.empty(514, device=device, dtype=torch.int32)
    end = torch.empty(257, device=device, dtype=torch.int32)
    ntiles = _expert_tile_capacity(m, tile_rows)
    tiles = torch.empty(ntiles + int(emit_count), device=device, dtype=torch.int32)
    _expert_offsets[(1,)](
        counts,
        start,
        end,
        tiles,
        rows,
        active,
        m,
        ntiles,
        tile_rows,
        triton.next_power_of_2(triton.cdiv(m, tile_rows)),
        triton.next_power_of_2(ntiles),
        reset_counts,
        width,
        active is not None,
        triton.next_power_of_2(m),
        emit_count,
        num_warps=4,
    )
    return rows, start, end, tiles


def make_expert_lists(routes, tile_rows=64):
    b = routes.shape[0]
    device = routes.device
    pairs = routes.numel()
    if b <= min(tile_rows, 8):
        start = torch.empty(514, device=device, dtype=torch.int32)
        end = torch.empty(257, device=device, dtype=torch.int32)
        ntiles = _expert_tile_capacity(b, tile_rows)
        tiles = torch.empty(ntiles, device=device, dtype=torch.int32)
        rows = torch.empty(pairs, device=device, dtype=torch.int32)
        _small_expert_lists[(1,)](
            routes,
            rows,
            start,
            end,
            tiles,
            pairs,
            ntiles,
            tile_rows,
            triton.next_power_of_2(pairs),
            num_warps=4,
        )
        return rows, start, end, tiles
    rows = torch.empty((257, b), device=device, dtype=torch.int32)
    counts = torch.empty(257, device=device, dtype=torch.int32)
    _expert_lists[(257,)](routes, rows, counts, b, triton.next_power_of_2(pairs), num_warps=4)
    return schedule_expert_lists(rows, counts, b, tile_rows)

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Gated DeltaNet convolution and recurrent kernels for runtime state slots."""

import triton as tr
import triton.language as tl


@tr.jit
def context_conv(
    X,
    W,
    History,
    Slots,
    Requests,
    Cu,
    Initial,
    Q,
    K,
    V,
    ROWS: tl.constexpr,
    X_STRIDE: tl.constexpr,
    H_SLOT: tl.constexpr,
    H_CHANNEL: tl.constexpr,
    H_TAP: tl.constexpr,
    BR: tl.constexpr,
    BC: tl.constexpr,
):
    """Convolve packed context rows while reading the old history."""
    row = tl.program_id(0) * BR + tl.arange(0, BR)
    channel_block = tl.program_id(1)
    col = channel_block * BC + tl.arange(0, BC)
    request = tl.load(Requests + row, row < ROWS, 0)
    begin = tl.load(Cu + request)
    slot = tl.load(Slots + request).to(tl.int64)
    initial = tl.load(Initial + request, row < ROWS, False)
    acc = tl.full((BR, BC), 0.0, tl.float32)
    for tap in tl.static_range(4):
        source = row + tap - 3
        x = tl.load(
            X + source[:, None] * X_STRIDE + col[None, :],
            (source[:, None] >= begin[:, None]) & (row[:, None] < ROWS),
            0,
        ).to(tl.float32)
        if tap < 3:
            hist = tl.load(
                History
                + slot[:, None] * H_SLOT
                + col[None, :] * H_CHANNEL
                + (source - begin + 3)[:, None] * H_TAP,
                ((source < begin) & initial & (row < ROWS))[:, None],
                0,
            ).to(tl.float32)
            x += hist
        coeff = tl.load(W + col * 4 + tap).to(tl.float32)
        acc += x * coeff[None, :]
    y = acc / (1.0 + tl.exp(-acc))
    if channel_block < 2048 // BC:
        tl.store(Q + row[:, None] * 2048 + col[None, :], y, row[:, None] < ROWS)
    elif channel_block < 4096 // BC:
        tl.store(K + row[:, None] * 2048 + col[None, :] - 2048, y, row[:, None] < ROWS)
    else:
        tl.store(V + row[:, None] * 4096 + col[None, :] - 4096, y, row[:, None] < ROWS)


@tr.jit
def context_history(
    X,
    History,
    Slots,
    Cu,
    Initial,
    X_STRIDE: tl.constexpr,
    H_SLOT: tl.constexpr,
    H_CHANNEL: tl.constexpr,
    H_TAP: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Retain the last three raw rows across a prompt chunk boundary."""
    req = tl.program_id(0)
    col = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    start = tl.load(Cu + req)
    end = tl.load(Cu + req + 1)
    slot = tl.load(Slots + req).to(tl.int64)
    initial = tl.load(Initial + req)
    history = History + slot * H_SLOT + col * H_CHANNEL
    for tap in tl.static_range(3):
        source = end - 3 + tap
        from_history = source < start
        old = tl.load(history + (source - start + 3) * H_TAP, from_history & initial, 0)
        new = tl.load(X + source * X_STRIDE + col, not from_history, 0)
        tl.store(history + tap * H_TAP, tl.where(from_history, old, new))


@tr.jit
def decode_conv(
    X,
    W,
    History,
    Slots,
    Initial,
    Y,
    Raw,
    X_STRIDE: tl.constexpr,
    H_SLOT: tl.constexpr,
    H_CHANNEL: tl.constexpr,
    H_TAP: tl.constexpr,
    STEPS: tl.constexpr,
    CHECKPOINT: tl.constexpr,
    RAW_REQUEST: tl.constexpr,
    RAW_STEP: tl.constexpr,
    RESET_NEW: tl.constexpr,
    PDL: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Convolve a generation block and keep only its raw input rows for acceptance."""
    request = tl.program_id(0)
    col = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    slot = tl.load(Slots + request).to(tl.int64)
    initial = tl.load(Initial + request) if RESET_NEW else True
    history = History + slot * H_SLOT + col * H_CHANNEL
    a = tl.load(history, initial, 0).to(tl.float32)
    b = tl.load(history + H_TAP, initial, 0).to(tl.float32)
    c = tl.load(history + H_TAP * 2, initial, 0).to(tl.float32)
    wa = tl.load(W + col * 4).to(tl.float32)
    wb = tl.load(W + col * 4 + 1).to(tl.float32)
    wc = tl.load(W + col * 4 + 2).to(tl.float32)
    wd = tl.load(W + col * 4 + 3).to(tl.float32)
    if PDL:
        tl.extra.cuda.gdc_wait()
        tl.extra.cuda.gdc_launch_dependents()
    for row in tl.static_range(STEPS):
        idx = request * STEPS + row
        d = tl.load(X + idx * X_STRIDE + col).to(tl.float32)
        value = tl.fma(d, wd, tl.fma(c, wc, tl.fma(b, wb, a * wa)))
        y = value / (1.0 + tl.exp(-value))
        tl.store(Y + idx * 8192 + col, y)
        if CHECKPOINT:
            tl.store(Raw + request * RAW_REQUEST + row * RAW_STEP + col, d)
        a, b, c = b, c, d
    if not CHECKPOINT:
        tl.store(history, a)
        tl.store(history + H_TAP, b)
        tl.store(history + H_TAP * 2, c)


@tr.jit
def recurrent(
    X,
    BA,
    Decay,
    Bias,
    State,
    Slots,
    Initial,
    Y,
    Factors,
    Pending,
    ReplaySlots,
    STATE_STRIDE: tl.constexpr,
    STEPS: tl.constexpr,
    REPLAY: tl.constexpr,
    CHECKPOINT: tl.constexpr,
    FACTOR_SLOT: tl.constexpr,
    FACTOR_STEP: tl.constexpr,
    PENDING_STRIDE: tl.constexpr,
    RESET_NEW: tl.constexpr,
    TILE: tl.constexpr,
    PDL: tl.constexpr,
):
    """Apply the GDN rank-one updates in FP32 with BF16 beta rounding."""
    request, head, tile = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    v = tile * TILE + tl.arange(0, TILE)
    k = tl.arange(0, 128)
    slot = tl.load(Slots + request).to(tl.int64)
    initial = tl.load(Initial + request) if RESET_NEW else True
    ptr = State + slot * STATE_STRIDE + (head * 128 + v[:, None]) * 128 + k[None, :]
    state = tl.load(ptr, initial, 0)
    if REPLAY:
        replay_slot = tl.load(ReplaySlots + request).to(tl.int64)
        factor = Factors + replay_slot * FACTOR_SLOT + head * 257
        length = tl.load(Pending + replay_slot * PENDING_STRIDE)
        for row in range(length):
            update = factor + row * FACTOR_STEP
            previous_key = tl.load(update + k)
            previous_delta = tl.load(update + 128 + v)
            previous_decay = tl.load(update + 256)
            state = tl.fma(previous_delta[:, None], previous_key[None, :], state * previous_decay)
    if PDL:
        tl.extra.cuda.gdc_wait()
    decay = -tl.exp(tl.load(Decay + head).to(tl.float32))
    bias = tl.load(Bias + head).to(tl.float32)
    for row in tl.static_range(STEPS):
        token = request * STEPS + row
        q = tl.load(X + token * 8192 + head // 2 * 128 + k).to(tl.float32)
        key = tl.load(X + token * 8192 + 2048 + head // 2 * 128 + k).to(tl.float32)
        value = tl.load(X + token * 8192 + 4096 + head * 128 + v).to(tl.float32)
        b = tl.load(BA + token * 64 + head).to(tl.float32)
        a = tl.load(BA + token * 64 + 32 + head).to(tl.float32) + bias
        softplus = tl.where(a > 20.0, a, tl.log(1.0 + tl.exp(a)))
        g = tl.exp(decay * softplus)
        beta = (1.0 / (1.0 + tl.exp(-b))).to(tl.bfloat16).to(tl.float32)
        q = (q / tl.sqrt(tl.sum(q * q, 0) + 1.0e-6)) * 0.08838834764831845
        key = key / tl.sqrt(tl.sum(key * key, 0) + 1.0e-6)
        state *= g
        delta = (value - tl.sum(state * key[None, :], 1)) * beta
        state = tl.fma(delta[:, None], key[None, :], state)
        output = tl.sum(state * q[None, :], 1)
        tl.store(Y + token * 4096 + head * 128 + v, output)
        if CHECKPOINT:
            if row == 0:
                tl.store(ptr, state)
            else:
                update = factor + (row - 1) * FACTOR_STEP
                tl.store(update + 128 + v, delta)
                if tile == 0:
                    tl.store(update + k, key)
                    tl.store(update + 256, g)
    if not CHECKPOINT:
        tl.store(ptr, state)


@tr.jit
def zero_fresh_states(State, Slots, Initial, STATE_STRIDE: tl.constexpr, TILE: tl.constexpr):
    request = tl.program_id(0)
    if not tl.load(Initial + request):
        slot = tl.load(Slots + request).to(tl.int64)
        index = tl.program_id(1) * TILE + tl.arange(0, TILE)
        tl.store(State + slot * STATE_STRIDE + index, 0)


@tr.jit
def accept_conv(
    History,
    Raw,
    Source,
    Destination,
    Accepted,
    Dummy,
    ReplaySlots,
    Pending,
    NUM_REQUESTS: tl.constexpr,
    H_LAYER: tl.constexpr,
    H_SLOT: tl.constexpr,
    H_CHANNEL: tl.constexpr,
    H_TAP: tl.constexpr,
    R_LAYER: tl.constexpr,
    R_REQUEST: tl.constexpr,
    R_STEP: tl.constexpr,
    P_STRIDE: tl.constexpr,
    CHANNELS: tl.constexpr,
    B: tl.constexpr,
    HAS_DUMMY: tl.constexpr,
    WRITE_PENDING: tl.constexpr,
):
    """Advance each retained convolution window by its accepted raw inputs."""
    layer_and_request, tile = tl.program_id(0), tl.program_id(1)
    request = layer_and_request % NUM_REQUESTS
    layer = (layer_and_request // NUM_REQUESTS).to(tl.int64)
    if HAS_DUMMY:
        if tl.load(Dummy + request):
            return
    source = tl.load(Source + request).to(tl.int64)
    destination = tl.load(Destination + request).to(tl.int64)
    count = tl.load(Accepted + request)
    col = tile * B + tl.arange(0, B)
    valid = col < CHANNELS
    hist = History + layer * H_LAYER + destination * H_SLOT + col * H_CHANNEL
    raw = Raw + layer * R_LAYER + source * R_REQUEST + col

    # The old tap is always to the right of the tap being overwritten.
    for tap in tl.static_range(3):
        index = count + tap - 3
        if index < 0:
            value = tl.load(hist + (index + 3) * H_TAP, valid, 0)
        else:
            value = tl.load(raw + index * R_STEP, valid, 0)
        tl.store(hist + tap * H_TAP, value, valid)
    if WRITE_PENDING:
        if layer == 0 and tile == 0:
            slot = tl.load(ReplaySlots + request).to(tl.int64)
            tl.store(Pending + slot * P_STRIDE, count - 1)

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Time-tiled Gated DeltaNet recurrence for short prompts."""

import torch
from triton.experimental import gluon
from triton.experimental.gluon import language as gl


@gluon.jit
def _normalize(MIXED, QK):
    row = gl.program_id(0)
    head = gl.program_id(1)
    layout: gl.constexpr = gl.BlockedLayout([4], [32], [1], [0])
    channel = gl.arange(0, 128, layout=layout)
    offset = row * 8192 + head * 128 + channel
    q = gl.load(MIXED + offset).to(gl.float32)
    k = gl.load(MIXED + offset + 2048).to(gl.float32)
    q = q / gl.sqrt(gl.sum(q * q, 0) + 1e-6)
    k = k / gl.sqrt(gl.sum(k * k, 0) + 1e-6)
    output = QK + row * 4096 + head * 128 + channel
    gl.store(output, q)
    gl.store(output + 2048, k)


@gluon.jit
def _time_row(x, index: gl.constexpr, columns: gl.constexpr, layout: gl.constexpr):
    row = gl.gather(x, gl.full((1, columns), index, gl.int32, layout), axis=0)
    return gl.convert_layout(gl.reshape(row, (columns,)), gl.SliceLayout(0, layout))


@gluon.jit
def _update(state, query, key, value, alpha, beta):
    query = query.to(gl.float32) * 0.08838834764831845
    key = key.to(gl.float32)
    state *= alpha
    delta = (value.to(gl.float32) - gl.sum(state * key[None, :], 1)) * beta
    state = gl.fma(delta[:, None], key[None, :], state)
    return state, gl.sum(state * query[None, :], 1)


@gluon.jit
def _time_tiled(MIXED, QK, ALPHA, BETA, Y, STATE, CU, SLOTS, INITIAL, SLOT_STRIDE: gl.constexpr):
    sequence = gl.program_id(0)
    head = gl.program_id(1)
    value_tile = gl.program_id(2)
    # Keep four consecutive keys per lane for the prefill reduction order.
    state_layout: gl.constexpr = gl.BlockedLayout([1, 4], [1, 32], [1, 1], [1, 0])
    preload_layout: gl.constexpr = gl.BlockedLayout([1, 4], [1, 32], [1, 1], [1, 0])
    key_layout: gl.constexpr = gl.SliceLayout(0, state_layout)
    value_layout: gl.constexpr = gl.SliceLayout(1, state_layout)
    value = value_tile * 4 + gl.arange(0, 4, layout=value_layout)
    key = gl.arange(0, 128, layout=key_layout)
    lo = gl.load(CU + sequence)
    hi = gl.load(CU + sequence + 1)
    slot = gl.load(SLOTS + sequence).to(gl.int64)
    offset = slot * SLOT_STRIDE + (head * 128 + value[:, None]) * 128 + key[None, :]
    state = gl.load(STATE + offset, gl.load(INITIAL + sequence), 0).to(gl.float32)
    full_tiles = lo + ((hi - lo) // 8) * 8
    time = gl.arange(0, 8, layout=gl.SliceLayout(1, preload_layout))
    read_key = head // 2 * 128 + gl.arange(0, 128, layout=gl.SliceLayout(0, preload_layout))
    read_value = (
        head * 128 + value_tile * 4 + gl.arange(0, 4, layout=gl.SliceLayout(0, preload_layout))
    )

    for start in range(lo, full_tiles, 8):
        token = start + time
        query = gl.load(QK + token[:, None] * 4096 + read_key[None, :])
        keys = gl.load(QK + token[:, None] * 4096 + 2048 + read_key[None, :])
        values = gl.load(MIXED + token[:, None] * 8192 + 4096 + read_value[None, :])
        alphas = gl.load(ALPHA + token * 32 + head)
        betas = gl.load(BETA + token * 32 + head)
        for index in gl.static_range(8):
            q = gl.convert_layout(_time_row(query, index, 128, preload_layout), key_layout)
            k = gl.convert_layout(_time_row(keys, index, 128, preload_layout), key_layout)
            v = gl.convert_layout(_time_row(values, index, 4, preload_layout), value_layout)
            alpha = gl.sum(gl.where(time == index, alphas, 0.0), 0)
            beta = gl.sum(gl.where(time == index, betas, 0.0), 0)
            state, out = _update(state, q, k, v, alpha, beta)
            gl.store(Y + (start + index) * 4096 + head * 128 + value, out)

    for token in range(full_tiles, hi):
        q = gl.load(QK + token * 4096 + head // 2 * 128 + key)
        k = gl.load(QK + token * 4096 + 2048 + head // 2 * 128 + key)
        v = gl.load(MIXED + token * 8192 + 4096 + head * 128 + value)
        alpha = gl.load(ALPHA + token * 32 + head)
        beta = gl.load(BETA + token * 32 + head)
        state, out = _update(state, q, k, v, alpha, beta)
        gl.store(Y + token * 4096 + head * 128 + value, out)

    gl.store(STATE + offset, state)


def recurrent(mixed, alpha, beta, state, offsets, slots, initial):
    rows = mixed.shape[0]
    qk = torch.empty((rows, 4096), dtype=mixed.dtype, device=mixed.device)
    result = torch.empty_like(qk)
    _normalize[(rows, 16)](mixed, qk, num_warps=1)
    _time_tiled[(slots.numel(), 32, 32)](
        mixed,
        qk,
        alpha,
        beta,
        result,
        state,
        offsets,
        slots,
        initial,
        state.stride(0),
        num_warps=1,
    )
    return result

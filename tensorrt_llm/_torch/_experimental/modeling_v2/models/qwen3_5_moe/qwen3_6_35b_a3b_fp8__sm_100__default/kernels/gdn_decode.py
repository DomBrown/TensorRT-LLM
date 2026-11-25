# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Batch Gated DeltaNet decode with FP32 state and independent warp tiles."""

import torch
import triton
import triton.language as tl
from triton.experimental import gluon
from triton.experimental.gluon import language as gl


@triton.jit
def _qk(MIXED, QK):
    row = tl.program_id(0)
    head = tl.program_id(1)
    index = tl.arange(0, 128)
    q = tl.load(MIXED + row * 8192 + head * 128 + index).to(tl.float32)
    k = tl.load(MIXED + row * 8192 + 2048 + head * 128 + index).to(tl.float32)
    q = q / tl.sqrt(tl.sum(q * q, 0) + 1e-6)
    k = k / tl.sqrt(tl.sum(k * k, 0) + 1e-6)
    tl.store(QK + row * 4096 + head * 128 + index, q * 0.08838834764831845)
    tl.store(QK + row * 4096 + 2048 + head * 128 + index, k)


@gluon.jit
def _gdn_qk(MIXED, QK, ALPHA, BETA, Y, STATE, SLOTS, SLOT_STRIDE: gl.constexpr):
    # Four consecutive keys per lane keep the FP32 state reduction in the
    # same order as the four-warp decoder, without cross-warp synchronization.
    b = gl.program_id(0)
    h = gl.program_id(1)
    v_tile = gl.program_id(2)
    layout: gl.constexpr = gl.BlockedLayout([1, 4], [1, 32], [1, 1], [1, 0])
    v = v_tile * 4 + gl.arange(0, 4, layout=gl.SliceLayout(1, layout))
    k = gl.arange(0, 128, layout=gl.SliceLayout(0, layout))
    slot = gl.load(SLOTS + b).to(gl.int64)
    ptr = STATE + slot * SLOT_STRIDE + (h * 128 + v[:, None]) * 128 + k[None, :]
    state = gl.load(ptr, cache_modifier=".cg").to(gl.float32)
    q = gl.load(QK + b * 4096 + h // 2 * 128 + k)
    key = gl.load(QK + b * 4096 + 2048 + h // 2 * 128 + k)
    val = gl.load(MIXED + b * 8192 + 4096 + h * 128 + v).to(gl.float32)
    alpha = gl.load(ALPHA + b * 32 + h)
    beta = gl.load(BETA + b * 32 + h)
    state *= alpha
    d = (val - gl.sum(state * key[None, :], 1)) * beta
    state = gl.fma(d[:, None], key[None, :], state)
    o = gl.sum(state * q[None, :], 1)
    gl.store(ptr, state, cache_modifier=".wb")
    gl.store(Y + b * 4096 + h * 128 + v, o)


def recurrent(mixed, alpha, beta, state, slots):
    batch = mixed.shape[0]
    qk = torch.empty((batch, 4096), device=mixed.device, dtype=torch.float32)
    output = torch.empty((batch, 4096), device=mixed.device, dtype=mixed.dtype)
    _qk[(batch, 16)](mixed, qk, num_warps=4)
    _gdn_qk[(batch, 32, 32)](
        mixed, qk, alpha, beta, output, state, slots, state.stride(0), num_warps=1
    )
    return output

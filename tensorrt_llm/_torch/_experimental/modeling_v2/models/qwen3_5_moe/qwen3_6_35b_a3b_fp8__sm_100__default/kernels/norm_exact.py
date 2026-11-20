# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Fused prefill norm with the standalone RMS reduction order."""

import torch
import triton as tr
from triton.experimental import gluon as gj
from triton.experimental.gluon import language as gl


@gj.jit
def quant_group(x, peak, pow2: gl.constexpr):
    scale = peak * (1.0 / 448.0)
    if pow2:
        bits = scale.to(gl.int32, bitcast=True)
        scale = ((bits + 0x7FFFFF) & 0x7F800000).to(gl.float32, bitcast=True)
        inv = 1.0 / scale
    else:
        inv = 448.0 / peak
    return gl.clamp(x * inv[:, None], -448.0, 448.0), scale


@gj.jit
def add_rms_quant(
    R, X, W, Y, Added, QA, SA, QB, SB, ROUND: gl.constexpr, POW2: gl.constexpr, DUAL: gl.constexpr
):
    row = gl.program_id(0)
    rms_layout: gl.constexpr = gl.BlockedLayout([8], [32], [4], [0])
    col = gl.arange(0, 2048, layout=rms_layout)
    off = row * 2048 + col
    x = gl.load(R + off).to(gl.float32) + gl.load(X + off).to(gl.float32)
    gl.store(Added + off, x)
    if ROUND:
        x = x.to(gl.bfloat16).to(gl.float32)
    w = 1.0 + gl.load(W + col).to(gl.float32)
    inv = gl.rsqrt(gl.sum(x * x, 0) * (1.0 / 2048.0) + 1.0e-6)
    norm = (x * inv * w).to(gl.bfloat16)
    gl.store(Y + off, norm)

    qlayout: gl.constexpr = gl.BlockedLayout([1, 4], [1, 32], [4, 1], [1, 0])
    group = gl.arange(0, 16, layout=gl.SliceLayout(1, qlayout))
    channel = gl.arange(0, 128, layout=gl.SliceLayout(0, qlayout))
    idx = row * 2048 + group[:, None] * 128 + channel[None, :]
    tile = gl.convert_layout(gl.reshape(norm.to(gl.float32), (16, 128)), qlayout)
    peak = gl.maximum(gl.max(gl.abs(tile), 1), 1.0e-10)
    q, scale = quant_group(tile, peak, POW2)
    gl.store(QA + idx, q)
    gl.store(SA + row * 16 + group, scale)
    if DUAL:
        q, scale = quant_group(tile, peak, True)
        gl.store(QB + idx, q)
        gl.store(SB + row * 16 + group, scale)


@gj.jit
def normalized_router(R, X, Gamma, W, Out, N: gl.constexpr, ROUND: gl.constexpr):
    tile, token = gl.program_id(0), gl.program_id(1)
    rms_layout: gl.constexpr = gl.BlockedLayout([16], [32], [4], [0])
    col = gl.arange(0, 2048, layout=rms_layout)
    off = token * 2048 + col
    x = gl.load(R + off).to(gl.float32) + gl.load(X + off).to(gl.float32)
    if ROUND:
        x = x.to(gl.bfloat16).to(gl.float32)
    weight = 1.0 + gl.load(Gamma + col).to(gl.float32)
    inv = gl.rsqrt(gl.sum(x * x, 0) * (1.0 / 2048.0) + 1.0e-6)
    norm = (x * inv * weight).to(gl.bfloat16).to(gl.float32)

    matrix: gl.constexpr = gl.BlockedLayout([1, 8], [1, 32], [1, 4], [1, 0])
    d_layout: gl.constexpr = gl.SliceLayout(0, matrix)
    row = tile * 2 + gl.arange(0, 2, layout=gl.SliceLayout(1, matrix))
    d = gl.arange(0, 2048, layout=d_layout)
    coeff = gl.load(W + row[:, None] * 2048 + d[None, :], row[:, None] < N, 0).to(gl.float32)
    logits = gl.sum(gl.convert_layout(norm, d_layout)[None, :] * coeff, 1)
    gl.store(Out + token * N + row, logits, row < N)


def router_linear(residual, update, norm, weight, rounded):
    m, n = residual.shape[0], weight.shape[0]
    result = torch.empty((m, n), dtype=residual.dtype, device=residual.device)
    normalized_router[(tr.cdiv(n, 2), m)](
        residual, update, norm, weight, result, n, rounded, num_warps=4
    )
    return result

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Layout-preserving RMSNorm, residual and FP8 producer fusion."""

import torch
from triton.experimental import gluon
from triton.experimental.gluon import language as gl

from . import kernels as ops


@gluon.jit
def _norm_quant(
    X,
    D,
    PROBS,
    ROUTER,
    PERM,
    W,
    R,
    Y,
    Q,
    S,
    ACTIVE,
    K: gl.constexpr,
    POW2: gl.constexpr,
    MODE: gl.constexpr,
    ROUND: gl.constexpr,
    ROUND_MOE: gl.constexpr,
    ROUND_SHARED: gl.constexpr,
    QUANTIZE: gl.constexpr,
    GATHER: gl.constexpr,
    MN_MAJOR: gl.constexpr,
    WIDTH: gl.constexpr,
    RETIRE: gl.constexpr,
):
    row = gl.program_id(0)
    norm_layout: gl.constexpr = gl.BlockedLayout([8], [32], [4], [0])
    c = gl.arange(0, K, layout=norm_layout)
    offset = row * K + c
    if RETIRE:
        if gl.load(ACTIVE + row // WIDTH) == 0:
            gl.store(R + offset, 0.0)
            gl.store(Y + offset, 0.0)
            if QUANTIZE:
                gl.store(Q + offset, 0.0)
                group = gl.arange(0, K // 128, layout=norm_layout)
                scale_offset = (
                    group * gl.num_programs(0) + row if MN_MAJOR else row * (K // 128) + group
                )
                gl.store(S + scale_offset, 1.0)
            return
    x = gl.load(X + offset).to(gl.float32)
    if MODE != "rms":
        if MODE == "moe":
            routed = gl.full((K,), 0.0, gl.float32, norm_layout)
            for i in gl.static_range(8):
                expert_row = gl.load(PERM + row * 9 + i) if GATHER else row * 9 + i
                act = gl.load(D + expert_row * K + c).to(gl.float32)
                weight = gl.load(PROBS + row * 8 + i).to(gl.float32)
                value = act * weight
                if ROUND_MOE:
                    value = value.to(gl.bfloat16).to(gl.float32)
                routed += value
            gate = gl.load(ROUTER + row * 257 + 256).to(gl.float32)
            gate = 1.0 / (1.0 + gl.exp(-gate))
            if ROUND_SHARED:
                gate = gate.to(gl.bfloat16).to(gl.float32)
            expert_row = gl.load(PERM + row * 9 + 8) if GATHER else row * 9 + 8
            shared = gl.load(D + expert_row * K + c).to(gl.float32)
            expert_sum = routed.to(gl.bfloat16).to(gl.float32)
            shared = (shared * gate).to(gl.bfloat16).to(gl.float32)
            delta = (expert_sum + shared).to(gl.bfloat16).to(gl.float32)
        else:
            delta = gl.load(D + offset).to(gl.float32)
        x += delta
        gl.store(R + offset, x)
        if ROUND:
            x = x.to(gl.bfloat16).to(gl.float32)

    w = gl.load(W + c).to(gl.float32)
    y = (x * gl.rsqrt(gl.sum(x * x, 0) * (1.0 / K) + 1e-6) * (1.0 + w)).to(gl.bfloat16)
    gl.store(Y + offset, y)

    if QUANTIZE:
        q_layout: gl.constexpr = gl.BlockedLayout([1, 8], [2, 16], [4, 1], [1, 0])
        groups = gl.convert_layout(gl.reshape(y.to(gl.float32), (K // 128, 128)), q_layout)
        maximum = gl.maximum(gl.max(gl.abs(groups), 1), 1e-10)
        scale = maximum * (1.0 / 448.0)
        if POW2:
            scale = gl.exp2(gl.ceil(gl.log2(scale)))
            reciprocal = 1.0 / scale
        else:
            reciprocal = 448.0 / maximum
        quant = gl.clamp(groups * reciprocal[:, None], -448.0, 448.0).to(gl.float8e4nv)
        g = gl.arange(0, K // 128, layout=gl.SliceLayout(1, q_layout))
        v = gl.arange(0, 128, layout=gl.SliceLayout(0, q_layout))
        gl.store(Q + row * K + g[:, None] * 128 + v[None, :], quant)
        offset = g * gl.num_programs(0) + row if MN_MAJOR else row * (K // 128) + g
        gl.store(S + offset, scale)


def norm_quant(
    x,
    weight,
    pow2=False,
    delta=None,
    round_residual=False,
    probability=None,
    router=None,
    round_experts=True,
    round_shared=True,
    quantize=True,
    permutation=None,
    mn_major=False,
    active=None,
    width=1,
):
    m, k = x.shape
    residual = torch.empty_like(x) if delta is not None else x
    normalized = torch.empty_like(x)
    q = torch.empty(x.shape, device=x.device, dtype=ops.FP8) if quantize else normalized
    s = ops._scales(m, k // 128, x.device, mn_major) if quantize else normalized
    mode = "moe" if probability is not None else "rms" if delta is None else "add"
    _norm_quant[(m,)](
        x,
        delta if delta is not None else x,
        probability if probability is not None else x,
        router if router is not None else x,
        permutation if permutation is not None else x,
        weight,
        residual,
        normalized,
        q,
        s,
        active,
        k,
        pow2,
        mode,
        round_residual,
        round_experts,
        round_shared,
        quantize,
        permutation is not None,
        mn_major,
        width,
        active is not None,
        num_warps=4,
    )
    return residual, normalized, (q, s)

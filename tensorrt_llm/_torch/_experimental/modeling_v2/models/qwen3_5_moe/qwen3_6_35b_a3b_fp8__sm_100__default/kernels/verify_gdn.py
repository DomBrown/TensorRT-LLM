# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Gated DeltaNet verification into the hybrid cache manager's candidate slots."""

import torch
import triton
import triton.language as tl


@triton.jit
def _verify_conv(
    PROJ,
    WEIGHT,
    MIXED,
    SLOTS,
    CONV,
    STAGED,
    WIDTH: tl.constexpr,
    PROJ_STRIDE: tl.constexpr,
    CONV_SLOT: tl.constexpr,
    CONV_CHANNEL: tl.constexpr,
    CONV_TIME: tl.constexpr,
    STAGE_SLOT: tl.constexpr,
    STAGE_TOKEN: tl.constexpr,
    STAGE_CHANNEL: tl.constexpr,
    STAGE_TIME: tl.constexpr,
    TILE: tl.constexpr,
):
    request = tl.program_id(0)
    channel = tl.program_id(1) * TILE + tl.arange(0, TILE)
    slot = tl.load(SLOTS + request).to(tl.int64)
    history = CONV + slot * CONV_SLOT + channel * CONV_CHANNEL
    staged = STAGED + request.to(tl.int64) * STAGE_SLOT + channel * STAGE_CHANNEL
    x0 = tl.load(history).to(tl.float32)
    x1 = tl.load(history + CONV_TIME).to(tl.float32)
    x2 = tl.load(history + 2 * CONV_TIME).to(tl.float32)
    w0 = tl.load(WEIGHT + channel * 4).to(tl.float32)
    w1 = tl.load(WEIGHT + channel * 4 + 1).to(tl.float32)
    w2 = tl.load(WEIGHT + channel * 4 + 2).to(tl.float32)
    w3 = tl.load(WEIGHT + channel * 4 + 3).to(tl.float32)
    for rank in tl.static_range(WIDTH):
        row = request * WIDTH + rank
        x3 = tl.load(PROJ + row * PROJ_STRIDE + channel).to(tl.float32)
        value = x0 * w0
        value = tl.fma(x1, w1, value)
        value = tl.fma(x2, w2, value)
        value = tl.fma(x3, w3, value)
        tl.store(MIXED + row * 8192 + channel, value * tl.sigmoid(value))
        target = staged + rank * STAGE_TOKEN
        tl.store(target, x1)
        tl.store(target + STAGE_TIME, x2)
        tl.store(target + 2 * STAGE_TIME, x3)
        x0, x1, x2 = x1, x2, x3


@triton.jit
def _verify_recurrent(
    MIXED,
    ALPHA,
    BETA,
    OUTPUT,
    SLOTS,
    STATE,
    STAGED,
    WIDTH: tl.constexpr,
    SLOT_STRIDE: tl.constexpr,
    STAGE_REQUEST: tl.constexpr,
    STAGE_TOKEN: tl.constexpr,
    VALUE_TILE: tl.constexpr,
):
    request = tl.program_id(0)
    head = tl.program_id(1)
    v = tl.program_id(2) * VALUE_TILE + tl.arange(0, VALUE_TILE)
    k = tl.arange(0, 128)
    offset = (head * 128 + v[:, None]) * 128 + k[None, :]
    slot = tl.load(SLOTS + request).to(tl.int64)
    state = tl.load(STATE + slot * SLOT_STRIDE + offset).to(tl.float32)
    staged = STAGED + request.to(tl.int64) * STAGE_REQUEST + offset
    for rank in tl.static_range(WIDTH):
        row = request * WIDTH + rank
        q = tl.load(MIXED + row * 8192 + head // 2 * 128 + k).to(tl.float32)
        key = tl.load(MIXED + row * 8192 + 2048 + head // 2 * 128 + k).to(tl.float32)
        value = tl.load(MIXED + row * 8192 + 4096 + head * 128 + v).to(tl.float32)
        q = q / tl.sqrt(tl.sum(q * q, 0) + 1e-6) * 0.08838834764831845
        key = key / tl.sqrt(tl.sum(key * key, 0) + 1e-6)
        decay = tl.load(ALPHA + row * 32 + head)
        beta = tl.load(BETA + row * 32 + head)
        state *= decay
        innovation = (value - tl.sum(state * key[None, :], 1)) * beta
        state = tl.fma(innovation[:, None], key[None, :], state)
        tl.store(OUTPUT + row * 4096 + head * 128 + v, tl.sum(state * q[None, :], 1))
        tl.store(staged + rank * STAGE_TOKEN, state)


def verify_convolution(
    projection: torch.Tensor,
    weight: torch.Tensor,
    slots: torch.Tensor,
    conv: torch.Tensor,
    staged_conv: torch.Tensor,
    width: int,
) -> torch.Tensor:
    """Write each width-token chain's rolling BF16 windows to candidate rows."""
    mixed = projection.new_empty((projection.shape[0], 8192))
    _verify_conv[(slots.numel(), 32)](
        projection,
        weight,
        mixed,
        slots,
        conv,
        staged_conv,
        width,
        projection.stride(0),
        *conv.stride(),
        *staged_conv.stride(),
        256,
        num_warps=4,
    )
    return mixed


def verify_recurrent(
    mixed: torch.Tensor,
    alpha: torch.Tensor,
    beta: torch.Tensor,
    slots: torch.Tensor,
    state: torch.Tensor,
    staged_ssm: torch.Tensor,
    width: int,
) -> torch.Tensor:
    """Record FP32 temporal checkpoints for possible promotion after verification."""
    output = mixed.new_empty((mixed.shape[0], 4096))
    _verify_recurrent[(slots.numel(), 32, 16)](
        mixed,
        alpha,
        beta,
        output,
        slots,
        state,
        staged_ssm,
        width,
        state.stride(0),
        staged_ssm.stride(0),
        staged_ssm.stride(1),
        8,
        num_warps=4,
    )
    return output

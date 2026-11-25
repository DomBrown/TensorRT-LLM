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


@gluon.jit
def _pair_product(left, right):
    left, right = gl.broadcast(left, right)
    return gl.inline_asm_elementwise(
        asm="""{
            .reg .b64 a, b, c;
            mov.b64 a, {$2,$3};
            mov.b64 b, {$4,$5};
            mul.f32x2 c, a, b;
            mov.b64 {$0,$1}, c;
        }""",
        constraints="=f,=f,f,f,f,f",
        args=[left, right],
        dtype=gl.float32,
        is_pure=True,
        pack=2,
    )


@gluon.jit
def _pair_fma(delta, key, old):
    delta, _ = gl.broadcast(delta, old)
    key, _ = gl.broadcast(key, old)
    return gl.inline_asm_elementwise(
        asm="""{
            .reg .b64 d, k, s, x;
            mov.b64 d, {$2,$3};
            mov.b64 k, {$4,$5};
            mov.b64 s, {$6,$7};
            fma.rn.f32x2 x, d, k, s;
            mov.b64 {$0,$1}, x;
        }""",
        constraints="=f,=f,f,f,f,f,f,f",
        args=[delta, key, old],
        dtype=gl.float32,
        is_pure=True,
        pack=2,
    )


@gluon.jit
def _replay_sum(product, TILE: gl.constexpr, layout: gl.constexpr):
    # Preserve the single-warp GDN reduction tree with four keys per thread.
    grouped = gl.reshape(product, (TILE, 4, 8, 4))
    lane_sum = gl.sum(grouped, 3)
    high_bits = gl.sum(lane_sum, 1)
    return gl.convert_layout(gl.sum(high_bits, 1), gl.SliceLayout(1, layout))


@gluon.jit
def _gdn_replay(
    MIXED,
    QK,
    ALPHA,
    BETA,
    OUT,
    SLOTS,
    REPLAY_SLOTS,
    STATE,
    KEYS,
    INNOVATIONS,
    DECAYS,
    ACCEPTED,
    BANK,
    WIDTH: gl.constexpr,
    SPEC_WIDTH: gl.constexpr,
    HISTORY: gl.constexpr,
    STATE_SLOT_STRIDE: gl.constexpr,
    KEYS_SLOT_STRIDE: gl.constexpr,
    INNOVATIONS_SLOT_STRIDE: gl.constexpr,
    DECAYS_SLOT_STRIDE: gl.constexpr,
    META_STRIDE: gl.constexpr,
    TILE: gl.constexpr,
    DIRECT_COMMIT: gl.constexpr,
):
    req = gl.program_id(0)
    h = gl.program_id(1)
    t = gl.program_id(2)
    layout: gl.constexpr = gl.BlockedLayout([1, 4], [4, 8], [1, 1], [1, 0])
    v = t * TILE + gl.arange(0, TILE, layout=gl.SliceLayout(1, layout))
    k = gl.arange(0, 128, layout=gl.SliceLayout(0, layout))
    state_slot = gl.load(SLOTS + req).to(gl.int64)
    replay_slot = gl.load(REPLAY_SLOTS + req).to(gl.int64)
    accepted = gl.load(ACCEPTED + replay_slot * META_STRIDE)
    bank = gl.load(BANK + replay_slot * META_STRIDE)

    state_ptr = STATE + state_slot * STATE_SLOT_STRIDE + (h * 128 + v[:, None]) * 128 + k[None, :]
    key_ptr = KEYS + replay_slot * KEYS_SLOT_STRIDE + h // 2 * 128 + k
    innovation_ptr = INNOVATIONS + replay_slot * INNOVATIONS_SLOT_STRIDE + h * 128 + v
    decay_ptr = DECAYS + replay_slot * DECAYS_SLOT_STRIDE + h * HISTORY
    state = gl.load(state_ptr, cache_modifier=".cg")
    for rank in range(accepted):
        rec = bank * HISTORY + rank
        decay = gl.load(decay_ptr + bank * 32 * HISTORY + rank)
        key = gl.load(key_ptr + rec * 2048)
        innovation = gl.load(innovation_ptr + rec * 4096)
        state = _pair_fma(innovation[:, None], key[None, :], _pair_product(state, decay))

    flush = (accepted + SPEC_WIDTH) > HISTORY
    write_bank = gl.where(flush, 1 - bank, bank)
    write_pos = gl.where(flush, 0, accepted)
    for rank in gl.static_range(WIDTH):
        row = req * WIDTH + rank
        q = gl.load(QK + row * 4096 + h // 2 * 128 + k)
        key = gl.load(QK + row * 4096 + 2048 + h // 2 * 128 + k)
        val = gl.load(MIXED + row * 8192 + 4096 + h * 128 + v).to(gl.float32)
        decay = gl.load(ALPHA + row * 32 + h)
        beta = gl.load(BETA + row * 32 + h)
        state = _pair_product(state, decay)
        innovation = (val - _replay_sum(state * key[None, :], TILE, layout)) * beta
        state = _pair_fma(innovation[:, None], key[None, :], state)
        gl.store(OUT + row * 4096 + h * 128 + v, _replay_sum(state * q[None, :], TILE, layout))
        if not DIRECT_COMMIT:
            # The verifier always consumes the seed, even if it rejects every draft.
            if rank == 0 and flush:
                gl.store(state_ptr, state, cache_modifier=".cs")
            else:
                write_rank = write_pos + rank - flush.to(gl.int32)
                record = write_bank * HISTORY + write_rank
                gl.store(innovation_ptr + record * 4096, innovation)
                if t == 0:
                    gl.store(decay_ptr + write_bank * 32 * HISTORY + write_rank, decay)
                    if h % 2 == 0:
                        gl.store(key_ptr + record * 2048, key)
    if DIRECT_COMMIT:
        gl.store(state_ptr, state, cache_modifier=".cs")


def replay(
    mixed: torch.Tensor,
    alpha: torch.Tensor,
    beta: torch.Tensor,
    state: torch.Tensor,
    slots: torch.Tensor,
    replay_slots: torch.Tensor,
    keys: torch.Tensor,
    innovations: torch.Tensor,
    decays: torch.Tensor,
    accepted: torch.Tensor,
    bank: torch.Tensor,
    width: int,
    spec_width: int,
) -> torch.Tensor:
    """Fold accepted FP32 records and verify the next linear draft chain."""
    rows = mixed.shape[0]
    qk = torch.empty((rows, 4096), device=mixed.device, dtype=torch.float32)
    output = torch.empty((rows, 4096), device=mixed.device, dtype=mixed.dtype)
    _qk[(rows, 16)](mixed, qk, num_warps=4)
    tile = 16 if slots.numel() >= 64 else 8
    _gdn_replay[(slots.numel(), 32, 128 // tile)](
        mixed,
        qk,
        alpha,
        beta,
        output,
        slots,
        replay_slots,
        state,
        keys,
        innovations,
        decays,
        accepted,
        bank,
        width,
        spec_width,
        keys.shape[2],
        state.stride(0),
        keys.stride(0),
        innovations.stride(0),
        decays.stride(0),
        accepted.stride(0),
        tile,
        width == 1,
        num_warps=1,
    )
    return output

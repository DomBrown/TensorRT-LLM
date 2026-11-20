# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Packed Gated DeltaNet operations on runtime convolution and SSM state."""

from typing import NamedTuple

import torch
import torch.nn.functional as F
import triton as tr
from flashinfer.gdn_prefill import chunk_gated_delta_rule

from .....catalog._op import Arch, Cell, OpWrapper
from ..kernels import fp8, gdn


class GdnWeights(NamedTuple):
    conv: torch.Tensor
    decay: torch.Tensor
    bias: torch.Tensor


class StatePool(NamedTuple):
    conv: torch.Tensor
    temporal: torch.Tensor
    slots: torch.Tensor
    has_initial: torch.Tensor


class ReplayState(NamedTuple):
    factors: torch.Tensor
    pending: torch.Tensor
    slots: torch.Tensor
    raw_conv: torch.Tensor


class ConvAcceptance(NamedTuple):
    source: torch.Tensor
    destination: torch.Tensor
    count: torch.Tensor
    replay_slots: torch.Tensor
    dummy: torch.Tensor | None


class AcceptConvolution(OpWrapper):
    """Advance the recurrent convolution window by the accepted verifier rows."""

    ARCHS = frozenset({Arch.SM_100})
    CELLS = (
        Cell("one accepted row through a recycled state slot", {"accepted": (1,)}),
        Cell("mixed acceptance lengths with one graph dummy", {"accepted": (4, 1, 2)}),
    )

    def __call__(
        self,
        history: torch.Tensor,
        raw: torch.Tensor,
        batch: ConvAcceptance,
        pending: torch.Tensor | None,
    ) -> None:
        requests = batch.count.shape[0]
        channels = history.shape[2]
        tile = 256
        gdn.accept_conv[(history.shape[0] * requests, tr.cdiv(channels, tile))](
            history,
            raw,
            batch.source,
            batch.destination,
            batch.count,
            batch.count if batch.dummy is None else batch.dummy,
            batch.replay_slots,
            batch.replay_slots if pending is None else pending,
            requests,
            *history.stride(),
            *raw.stride()[:3],
            0 if pending is None else pending.stride(0),
            channels,
            tile,
            batch.dummy is not None,
            pending is not None,
            num_warps=4,
        )

    def reference(
        self,
        history: torch.Tensor,
        raw: torch.Tensor,
        batch: ConvAcceptance,
        pending: torch.Tensor | None,
    ) -> None:
        for i, (source, destination, count, replay) in enumerate(zip(*batch[:4], strict=True)):
            if batch.dummy is not None and batch.dummy[i]:
                continue
            old = history[:, destination].transpose(1, 2)
            update = torch.cat((old, raw[:, source, :count]), dim=1)[:, -3:]
            history[:, destination].copy_(update.transpose(1, 2))
            if pending is not None:
                pending[replay] = count - 1


accept_convolution = AcceptConvolution()


class GatedDeltaStep(OpWrapper):
    """Causal depthwise convolution and FP32 GDN recurrence on active slots."""

    ARCHS = frozenset({Arch.SM_100})
    CELLS = (
        Cell("single continuation in a nonzero pool slot", {"pool_slots": (2,), "steps": 1}),
        Cell("batched rows with fresh and continuing slots", {"pool_slots": (4, 1), "steps": 1}),
        Cell(
            "batched four-row verification then accepted-prefix replay",
            {"pool_slots": (2, 5), "steps": 4},
        ),
    )
    note = "The convolution and recurrence retain the standalone engine's BF16 round points."

    def __call__(
        self,
        projected: torch.Tensor,
        ba: torch.Tensor,
        weights: GdnWeights,
        pool: StatePool,
        steps: int,
        reset_new: bool,
        pdl: bool,
        replay: ReplayState | None = None,
    ) -> torch.Tensor:
        rows = projected.shape[0]
        requests = rows // steps
        checkpoint = replay is not None and steps > 1
        raw_conv = replay.raw_conv if checkpoint else pool.conv
        factors = replay.factors if replay is not None else pool.temporal
        pending = replay.pending if replay is not None else pool.slots
        replay_slots = replay.slots if replay is not None else pool.slots
        convolved = projected.new_empty((rows, 8192))
        gdn.decode_conv[(requests, 32)](
            projected,
            weights.conv,
            pool.conv,
            pool.slots,
            pool.has_initial,
            convolved,
            raw_conv,
            projected.stride(0),
            *pool.conv.stride(),
            steps,
            checkpoint,
            raw_conv.stride(0),
            raw_conv.stride(1),
            reset_new,
            pdl,
            256,
            num_warps=4,
            launch_pdl=pdl,
        )
        output = projected.new_empty((rows, 4096))
        tile = 8 if checkpoint and requests == 1 else 16
        gdn.recurrent[(requests, 32, 128 // tile)](
            convolved,
            ba,
            weights.decay,
            weights.bias,
            pool.temporal,
            pool.slots,
            pool.has_initial,
            output,
            factors,
            pending,
            replay_slots,
            pool.temporal.stride(0),
            steps,
            replay is not None,
            checkpoint,
            factors.stride(0),
            factors.stride(1),
            pending.stride(0),
            reset_new,
            tile,
            pdl,
            num_warps=4,
            launch_pdl=pdl,
        )
        return output

    def reference(
        self,
        projected: torch.Tensor,
        ba: torch.Tensor,
        weights: GdnWeights,
        pool: StatePool,
        steps: int,
        reset_new: bool,
        pdl: bool,
        replay: ReplayState | None = None,
    ) -> torch.Tensor:
        """Calculate the causal convolution and state updates in PyTorch."""
        del pdl
        return _reference_gdn(
            projected,
            ba,
            weights,
            pool,
            (steps,) * len(pool.slots),
            reset_new,
            replay,
        )


def _reference_gdn(
    projected: torch.Tensor,
    ba: torch.Tensor,
    weights: GdnWeights,
    pool: StatePool,
    lengths: tuple[int, ...],
    reset_new: bool,
    replay: ReplayState | None = None,
) -> torch.Tensor:
    outputs = []
    start = 0
    for request, (slot, steps) in enumerate(zip(pool.slots.tolist(), lengths, strict=True)):
        previous = not reset_new or pool.has_initial[request].item()
        history = pool.conv[slot].T
        state = pool.temporal[slot].float()
        if not previous:
            history = torch.zeros_like(history)
            state = torch.zeros_like(state)
        checkpoint = replay is not None and steps > 1
        if replay is not None:
            replay_slot = int(replay.slots[request])
            length = int(replay.pending[replay_slot])
            for factor in replay.factors[replay_slot, :length]:
                key, delta, decay = factor.split((128, 128, 1), dim=-1)
                state = torch.addcmul(state * decay[:, :, None], delta[:, :, None], key[:, None, :])
        raw = projected[start : start + steps, :8192]
        window = torch.cat((history, raw))
        convolved = F.silu((window.unfold(0, 4, 1).float() * weights.conv.float()).sum(dim=-1)).to(
            raw.dtype
        )
        for offset, mixed in enumerate(convolved):
            q, k, v = mixed.float().split((2048, 2048, 4096))
            q = q.view(16, 128)
            k = k.view(16, 128)
            q = (q / (q.square().sum(-1, keepdim=True) + 1e-6).sqrt()).repeat_interleave(2, 0)
            k = (k / (k.square().sum(-1, keepdim=True) + 1e-6).sqrt()).repeat_interleave(2, 0)
            b, a = ba[start + offset].float().split(32)
            beta = b.sigmoid().to(raw.dtype).float()
            decay = (-weights.decay.float().exp() * F.softplus(a + weights.bias.float())).exp()
            state = state * decay[:, None, None]
            delta = (v.view(32, 128) - (state * k[:, None, :]).sum(-1)) * beta[:, None]
            state = torch.addcmul(state, delta[:, :, None], k[:, None, :])
            outputs.append((state * q[:, None, :]).sum(-1).flatten() / (128**0.5))
            if checkpoint:
                replay.raw_conv[request, offset].copy_(raw[offset])
                if offset == 0:
                    pool.temporal[slot].copy_(state)
                else:
                    replay.factors[replay_slot, offset - 1].copy_(
                        torch.cat((k, delta, decay[:, None]), dim=-1)
                    )
        if not checkpoint:
            pool.conv[slot].copy_(window[-3:].T)
            pool.temporal[slot].copy_(state)
        start += steps
    return torch.stack(outputs).to(projected.dtype)


gated_delta_step = GatedDeltaStep()


class ContextGatedDelta(OpWrapper):
    """Causal convolution and packed FlashInfer GDN scan."""

    ARCHS = frozenset({Arch.SM_100})
    CELLS = (
        Cell("one fresh prompt in a recycled state slot", {"lengths": (7,), "initial": (False,)}),
        Cell("chunk continuation and fresh prompt", {"lengths": (2, 9), "initial": (True, False)}),
    )

    def __call__(
        self,
        projected: torch.Tensor,
        ba: torch.Tensor,
        weights: GdnWeights,
        pool: StatePool,
        requests: torch.Tensor,
        cu: torch.Tensor,
        cu_cpu: tuple[int, ...],
        has_continuation: bool,
    ) -> torch.Tensor:
        """Convolve packed prompts and scan them with FlashInfer's SM100 GDN."""
        rows = projected.shape[0]
        batch = len(cu_cpu) - 1
        if rows == 1:
            return gated_delta_step(projected, ba, weights, pool, 1, True, False)
        q = projected.new_empty((rows, 16, 128))
        k = torch.empty_like(q)
        v = projected.new_empty((rows, 32, 128))
        gdn.context_conv[(tr.cdiv(rows, 8), 32)](
            projected,
            weights.conv,
            pool.conv,
            pool.slots,
            requests,
            cu,
            pool.has_initial,
            q,
            k,
            v,
            rows,
            projected.stride(0),
            *pool.conv.stride(),
            8,
            256,
            num_warps=4,
        )
        gdn.context_history[(batch, 32)](
            projected,
            pool.conv,
            pool.slots,
            cu,
            pool.has_initial,
            projected.stride(0),
            *pool.conv.stride(),
            256,
            num_warps=4,
        )
        gate = projected.new_empty((rows, 32), dtype=torch.float32)
        beta = torch.empty_like(gate)
        fp8.prefill_gdn_gates[(tr.cdiv(rows * 32, 256),)](
            ba, weights.decay, weights.bias, gate, beta, rows, 256
        )
        if has_continuation:
            gdn.zero_fresh_states[(batch, 128)](
                pool.temporal,
                pool.slots,
                pool.has_initial,
                pool.temporal.stride(0),
                4096,
            )
        result, _ = chunk_gated_delta_rule(
            q,
            k,
            v,
            g=gate,
            beta=beta,
            cu_seqlens=cu,
            initial_state=pool.temporal if has_continuation else None,
            output_final_state=True,
            output_state=pool.temporal,
            state_indices=pool.slots,
            use_qk_l2norm_in_kernel=True,
            use_cp=False,
            backend="flashinfer",
            cu_seqlens_cpu=cu_cpu,
            max_seqlen=max(b - a for a, b in zip(cu_cpu, cu_cpu[1:])),
        )
        return result.view(rows, 4096)

    def reference(
        self,
        projected: torch.Tensor,
        ba: torch.Tensor,
        weights: GdnWeights,
        pool: StatePool,
        requests: torch.Tensor,
        cu: torch.Tensor,
        cu_cpu: tuple[int, ...],
        has_continuation: bool,
    ) -> torch.Tensor:
        """Evaluate the packed chunks with FP32 recurrence and state."""
        del requests, cu, has_continuation
        return _reference_gdn(
            projected,
            ba,
            weights,
            pool,
            tuple(b - a for a, b in zip(cu_cpu, cu_cpu[1:])),
            True,
        )


context_gdn = ContextGatedDelta()


def gated_delta_output(
    values: torch.Tensor, projected: torch.Tensor, weight: torch.Tensor, encode_mx: bool
) -> tuple[torch.Tensor, ...]:
    """Apply the GDN output norm and gate, and quantize for the FP8 projection."""
    rows = values.shape[0]
    q = projected.new_empty((rows, 4096), dtype=torch.float8_e4m3fn)
    scales = projected.new_empty((rows, 32), dtype=torch.float32)
    if rows > 8:
        fp8.gdn_prefill_output[(rows, 8)](
            values,
            projected[:, 8192:],
            weight,
            q,
            scales,
            projected.stride(0),
            True,
            4,
            False,
            num_warps=4,
        )
        return q, scales
    mx = fp8.mx_buffer(q, 4096) if encode_mx else scales
    fp8.gdn_gated_fp8[(rows, 32)](
        values,
        projected[:, 8192:],
        weight,
        q,
        scales,
        mx,
        projected.stride(0),
        True,
        encode_mx,
        rows,
        num_warps=1,
    )
    return (q, scales, mx.view(torch.uint8)) if encode_mx else (q, scales)

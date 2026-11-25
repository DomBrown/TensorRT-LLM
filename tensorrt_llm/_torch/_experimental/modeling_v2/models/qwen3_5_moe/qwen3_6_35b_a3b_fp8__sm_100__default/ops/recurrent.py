# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Slot-indexed convolution and recurrence for the Qwen Gated DeltaNet block."""

from __future__ import annotations

import itertools

import torch
from flashinfer.gdn_prefill import chunk_gated_delta_rule
from torch.nn import functional as F

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, Cell, OpWrapper

from ..kernels import gdn_decode, prefill_gdn
from ..kernels.convolution import convolution
from ..kernels.kernels import gdn_recurrent
from ..kernels.verify_gdn import promote_replay_convolution, verify_convolution, verify_recurrent

_CONV_WIDTH = 8192
_QK_WIDTH = 2048
_VALUE_WIDTH = 4096
SHORT_CONTEXT = 96
ProjectionRows = torch.Tensor | tuple[torch.Tensor, torch.Tensor, torch.Tensor]


def _context_ranges(
    rows: int, offsets: torch.Tensor | None, initial: torch.Tensor | None
) -> tuple[list[int], list[bool]]:
    if offsets is None:
        return list(range(rows + 1)), [True] * rows
    return offsets.tolist(), initial.tolist()


def _split_qkv(mixed: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    rows = mixed.shape[0]
    q, k, v = mixed.split((_QK_WIDTH, _QK_WIDTH, _VALUE_WIDTH), dim=-1)
    return q.reshape(rows, 16, 128), k.reshape(rows, 16, 128), v.reshape(rows, 32, 128)


class CausalConvolution(OpWrapper):
    """Apply the four-tap depthwise SiLU convolution and update the last three inputs."""

    ARCHS = frozenset({Arch.SM_100})
    CELLS = (
        Cell("one-token decode, nonzero state slot", {"lengths": (), "slots": (3,)}),
        Cell("batched decode with noncontiguous slots", {"lengths": (), "slots": (4, 1, 5)}),
        Cell(
            "packed fresh context and a one-token continuation",
            {"lengths": (5, 1), "slots": (3, 1), "initial": (False, True)},
        ),
        Cell(
            "chunked recurrence uses contiguous Q, K and V",
            {"lengths": (101, 6), "slots": (0, 2), "initial": (True, False), "split": True},
        ),
    )

    def __call__(
        self,
        raw: torch.Tensor,
        weights: torch.Tensor,
        history: torch.Tensor,
        slots: torch.Tensor,
        offsets: torch.Tensor | None,
        row_map: torch.Tensor | None,
        initial: torch.Tensor | None,
        split: bool = False,
    ) -> ProjectionRows:
        return convolution(raw, weights, history, slots, offsets, row_map, initial, split)

    def reference(
        self,
        raw: torch.Tensor,
        weights: torch.Tensor,
        history: torch.Tensor,
        slots: torch.Tensor,
        offsets: torch.Tensor | None,
        row_map: torch.Tensor | None,
        initial: torch.Tensor | None,
        split: bool = False,
    ) -> ProjectionRows:
        starts, read_history = _context_ranges(raw.shape[0], offsets, initial)
        outputs = []
        for index, slot in enumerate(slots.tolist()):
            prefix = history[slot] if read_history[index] else torch.zeros_like(history[slot])
            src = raw[starts[index] : starts[index + 1], :_CONV_WIDTH].T
            source = torch.cat((prefix, src), dim=-1)
            out = F.conv1d(source[None].float(), weights.float(), groups=_CONV_WIDTH)
            outputs.append(F.silu(out[0].T).to(raw.dtype))
            history[slot].copy_(source[:, -3:])
        mixed = torch.cat(outputs)
        return _split_qkv(mixed) if split else mixed


class GatedDeltaRule(OpWrapper):
    """GDN over indexed [slots, 32, 128, 128] FP32 states in V-major order."""

    ARCHS = frozenset({Arch.SM_100})
    CELLS = (
        Cell("decode single row", {"lengths": (), "slots": (3,)}),
        Cell("decode at batch 32", {"lengths": (), "slots": tuple(range(32, 0, -1))}),
        Cell(
            "short prefill across fresh and carried states",
            {"lengths": (7, 3), "slots": (1, 4), "initial": (False, True)},
        ),
        Cell(
            "time-tiled prefill",
            {"lengths": (23, 12), "slots": (3, 1), "initial": (False, True)},
        ),
        Cell(
            "chunked FlashInfer prefill with indexed input and output",
            {"lengths": (129, 31), "slots": (4, 1), "initial": (False, True)},
        ),
    )
    note = "Uses the reference engine's sequential and time-tiled recurrence and FlashInfer SM100 prefill."

    def __call__(
        self,
        mixed: ProjectionRows,
        alpha: torch.Tensor,
        beta: torch.Tensor,
        state: torch.Tensor,
        slots: torch.Tensor,
        offsets: torch.Tensor | None,
        initial: torch.Tensor | None,
        read_slots: torch.Tensor | None,
        lengths: tuple[int, ...],
    ) -> torch.Tensor:
        if not lengths:
            if mixed.shape[0] > 1:
                return gdn_decode.recurrent(mixed, alpha, beta, state, slots)
            return gdn_recurrent(mixed, alpha, beta, state, slots)
        max_length = max(lengths)
        if max_length > SHORT_CONTEXT:
            q, k, v = mixed
            result, _ = chunk_gated_delta_rule(
                q,
                k,
                v,
                alpha,
                beta,
                initial_state=state if read_slots is not None else None,
                output_final_state=True,
                cu_seqlens=offsets,
                use_qk_l2norm_in_kernel=True,
                output_state=state,
                state_indices=slots if read_slots is None else read_slots,
                output_state_indices=None if read_slots is None else slots,
                max_seqlen=max_length,
                cu_seqlens_cpu=list(itertools.accumulate(lengths, initial=0)),
                backend="flashinfer",
                use_cp=False,
            )
            return result.view(-1, _VALUE_WIDTH)
        return prefill_gdn.recurrent(mixed, alpha, beta, state, offsets, slots, initial)

    def reference(
        self,
        mixed: ProjectionRows,
        alpha: torch.Tensor,
        beta: torch.Tensor,
        state: torch.Tensor,
        slots: torch.Tensor,
        offsets: torch.Tensor | None,
        initial: torch.Tensor | None,
        read_slots: torch.Tensor | None,
        lengths: tuple[int, ...],
    ) -> torch.Tensor:
        q, k, v = _split_qkv(mixed) if isinstance(mixed, torch.Tensor) else mixed
        q_norm = q.float() * (q.float().square().sum(dim=-1, keepdim=True) + 1e-6).rsqrt()
        k_norm = k.float() * (k.float().square().sum(dim=-1, keepdim=True) + 1e-6).rsqrt()
        if lengths:
            q_norm = q_norm.to(q.dtype).float()
            k_norm = k_norm.to(k.dtype).float()
        query = q_norm.repeat_interleave(2, dim=1) * (128**-0.5)
        key = k_norm.repeat_interleave(2, dim=1)
        starts, read_history = _context_ranges(v.shape[0], offsets, initial)
        out = torch.empty_like(v)
        for index, slot in enumerate(slots.tolist()):
            carry = state[slot].float().clone()
            if not read_history[index]:
                carry.zero_()
            for row in range(starts[index], starts[index + 1]):
                carry *= alpha[row, :, None, None]
                innovation = v[row].float() - (carry * key[row, :, None, :]).sum(-1)
                innovation *= beta[row, :, None]
                carry.addcmul_(innovation[:, :, None], key[row, :, None, :])
                out[row].copy_((carry * query[row, :, None, :]).sum(-1))
            state[slot].copy_(carry)
        return out.view(-1, _VALUE_WIDTH)


causal_convolution = CausalConvolution()
gated_delta_rule = GatedDeltaRule()

_VERIFY_CELLS = (
    Cell("two candidates at one nonzero slot", {"slots": (3,), "width": 2}),
    Cell("five-token verification in a remapped slot", {"slots": (4,), "width": 5}),
    Cell("four candidates with noncontiguous state slots", {"slots": (5, 1, 4), "width": 4}),
)


class VerifyConvolution(OpWrapper):
    """Convolve draft rows and stage the inputs needed for acceptance."""

    ARCHS = frozenset({Arch.SM_100})
    CELLS = (
        *_VERIFY_CELLS,
        Cell(
            "compact replay inputs for a four-token draft chain",
            {"slots": (5, 1, 4), "width": 4, "raw_only": True},
        ),
    )

    def __call__(
        self,
        raw: torch.Tensor,
        weights: torch.Tensor,
        slots: torch.Tensor,
        history: torch.Tensor,
        staged: torch.Tensor,
        width: int,
    ) -> torch.Tensor:
        return verify_convolution(raw, weights, slots, history, staged, width)

    def reference(
        self,
        raw: torch.Tensor,
        weights: torch.Tensor,
        slots: torch.Tensor,
        history: torch.Tensor,
        staged: torch.Tensor,
        width: int,
    ) -> torch.Tensor:
        outputs = []
        for request, slot in enumerate(slots.tolist()):
            projected = raw[request * width : (request + 1) * width, :_CONV_WIDTH].T
            carry = torch.cat((history[slot], projected), dim=-1)
            out = F.conv1d(carry[None].float(), weights.float(), groups=_CONV_WIDTH)
            outputs.append(F.silu(out[0].T).to(raw.dtype))
            saved = (
                projected.T if staged.ndim == 3 else carry[:, 1:].unfold(-1, 3, 1).transpose(0, 1)
            )
            staged[request, :width].copy_(saved)
        return torch.cat(outputs)


class VerifyGatedDeltaRule(OpWrapper):
    """Evaluate linear draft chains and save a temporal checkpoint per prefix."""

    ARCHS = frozenset({Arch.SM_100})
    CELLS = _VERIFY_CELLS

    def __call__(
        self,
        mixed: torch.Tensor,
        alpha: torch.Tensor,
        beta: torch.Tensor,
        slots: torch.Tensor,
        state: torch.Tensor,
        staged: torch.Tensor,
        width: int,
    ) -> torch.Tensor:
        return verify_recurrent(mixed, alpha, beta, slots, state, staged, width)

    def reference(
        self,
        mixed: torch.Tensor,
        alpha: torch.Tensor,
        beta: torch.Tensor,
        slots: torch.Tensor,
        state: torch.Tensor,
        staged: torch.Tensor,
        width: int,
    ) -> torch.Tensor:
        carry = state[slots].clone()
        dense_slots = torch.arange(slots.numel(), device=slots.device)
        result = torch.empty((mixed.shape[0], _VALUE_WIDTH), dtype=mixed.dtype, device=mixed.device)
        for rank in range(width):
            result[rank::width] = gated_delta_rule.reference(
                mixed[rank::width],
                alpha[rank::width],
                beta[rank::width],
                carry,
                dense_slots,
                None,
                None,
                None,
                (),
            )
            staged[: slots.numel(), rank].copy_(carry)
        return result


verify_conv = VerifyConvolution()
verify_gdn = VerifyGatedDeltaRule()


class PromoteReplayConvolution(OpWrapper):
    """Commit the accepted convolution history and replay count for each live request."""

    ARCHS = frozenset({Arch.SM_100})
    CELLS = (
        Cell(
            "multiple layers, remapped slots, skipped dummy and zero acceptance",
            {"layers": 3, "accepted": (1, 4, 2, 0, 3), "advance": True},
        ),
        Cell(
            "independent layer promotion without a second replay advance",
            {"layers": 1, "accepted": (3, 1, 4, 2, 1), "advance": False},
        ),
    )

    def __call__(
        self,
        conv: torch.Tensor,
        staged: torch.Tensor,
        source: torch.Tensor,
        destination: torch.Tensor,
        accepted: torch.Tensor,
        dummy: torch.Tensor | None,
        replay_slots: torch.Tensor,
        pending: torch.Tensor,
        bank: torch.Tensor,
        width: int,
        history: int,
        advance: bool = True,
    ) -> None:
        promote_replay_convolution(
            conv,
            staged,
            source,
            destination,
            accepted,
            dummy,
            replay_slots,
            pending,
            bank,
            width,
            history,
            advance,
        )

    def reference(
        self,
        conv: torch.Tensor,
        staged: torch.Tensor,
        source: torch.Tensor,
        destination: torch.Tensor,
        accepted: torch.Tensor,
        dummy: torch.Tensor | None,
        replay_slots: torch.Tensor,
        pending: torch.Tensor,
        bank: torch.Tensor,
        width: int,
        history: int,
        advance: bool = True,
    ) -> None:
        dummy_requests = [False] * source.numel() if dummy is None else dummy.tolist()
        for src, dst, count, slot, skip in zip(
            source.tolist(),
            destination.tolist(),
            accepted.tolist(),
            replay_slots.tolist(),
            dummy_requests,
            strict=True,
        ):
            if skip or count == 0:
                continue
            window = conv[:, dst]
            prefix = staged[:, src, :count].transpose(-1, -2)
            window.copy_(torch.cat((window, prefix), dim=-1)[..., -3:])
            if advance:
                prev = int(pending[slot])
                if prev + width > history:
                    pending[slot] = count - 1
                    bank[slot] = 1 - bank[slot]
                else:
                    pending[slot] = prev + count


promote_replay_conv = PromoteReplayConvolution()


class ReplayGatedDeltaRule(OpWrapper):
    """Verify a linear draft and store request-private FP32 updates for replay."""

    ARCHS = frozenset({Arch.SM_100})
    CELLS = (
        Cell(
            "full and seed-only steps with an accepted prefix", {"slots": (3,), "widths": (4, 4, 1)}
        ),
        Cell(
            "prefix accumulation and checkpoint rollover on independent slots",
            {"slots": (5, 1, 4), "widths": (4, 2, 4, 3, 4, 1)},
        ),
    )

    def __call__(
        self,
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
        return gdn_decode.replay(
            mixed,
            alpha,
            beta,
            state,
            slots,
            replay_slots,
            keys,
            innovations,
            decays,
            accepted,
            bank,
            width,
            spec_width,
        )

    def reference(
        self,
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
        query, key, values = (t.float() for t in _split_qkv(mixed))
        query = query / (query.square().sum(-1, keepdim=True) + 1e-6).sqrt() * (128**-0.5)
        key = key / (key.square().sum(-1, keepdim=True) + 1e-6).sqrt()
        query = query.repeat_interleave(2, dim=1)
        output = torch.empty_like(values, dtype=mixed.dtype)
        for request, (state_slot, replay_slot) in enumerate(
            zip(slots.tolist(), replay_slots.tolist(), strict=True)
        ):
            carry = state[state_slot].clone()
            count = int(accepted[replay_slot])
            old_bank = int(bank[replay_slot])
            for rank in range(count):
                decay = decays[replay_slot, old_bank, :, rank]
                cached_key = keys[replay_slot, old_bank, rank].repeat_interleave(2, dim=0)
                innovation = innovations[replay_slot, old_bank, rank]
                carry.mul_(decay[:, None, None])
                carry.addcmul_(innovation[:, :, None], cached_key[:, None, :])
            flush = count + spec_width > keys.shape[2]
            write_start = 0 if flush else count
            write_bank = 1 - old_bank if flush else old_bank
            for rank in range(width):
                row = request * width + rank
                repeated_key = key[row].repeat_interleave(2, dim=0)
                carry.mul_(alpha[row, :, None, None])
                innovation = values[row] - (carry * repeated_key[:, None, :]).sum(-1)
                innovation *= beta[row, :, None]
                carry.addcmul_(innovation[:, :, None], repeated_key[:, None, :])
                output[row].copy_((carry * query[row, :, None, :]).sum(-1))
                if width > 1:
                    if rank == 0 and flush:
                        state[state_slot].copy_(carry)
                    else:
                        position = write_start + rank - int(flush)
                        keys[replay_slot, write_bank, position].copy_(key[row])
                        innovations[replay_slot, write_bank, position].copy_(innovation)
                        decays[replay_slot, write_bank, :, position].copy_(alpha[row])
            if width == 1:
                state[state_slot].copy_(carry)
        return output.view(-1, _VALUE_WIDTH)


replay_gdn = ReplayGatedDeltaRule()

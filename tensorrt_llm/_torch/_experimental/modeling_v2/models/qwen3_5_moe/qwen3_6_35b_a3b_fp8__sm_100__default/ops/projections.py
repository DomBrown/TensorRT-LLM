# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Block-scaled FP8 and routed MoE operations for the GB200 target."""

from typing import NamedTuple

import torch
import torch.nn.functional as F
import triton as tr

from .....catalog._op import Arch, Cell, OpWrapper, true_fp32_matmul
from ..kernels import bf16, dg_prefill, fp8, mx_decode, prefill_moe, prefill_tc, tc_decode
from ..weights import DenseWeight, LayerWeights

Activation = tuple[torch.Tensor, ...]


class RowwiseProjection(OpWrapper):
    """Decode BF16 router and DeltaNet gates with the same reduction for each row."""

    ARCHS = frozenset({Arch.SM_100})
    CELLS = (
        Cell("four verifier rows and 64 DeltaNet gates", {"rows": 4, "columns": 64}),
        Cell("129 packed rows with a separate shared expert gate", {"rows": 129, "columns": 257}),
    )

    def __call__(
        self, hidden: torch.Tensor, weight: torch.Tensor, pdl: bool = False
    ) -> torch.Tensor:
        return fp8.bf16_linear(hidden, weight, pdl)

    def reference(
        self, hidden: torch.Tensor, weight: torch.Tensor, pdl: bool = False
    ) -> torch.Tensor:
        del pdl
        with true_fp32_matmul():
            return F.linear(hidden.float(), weight.float()).to(hidden.dtype)


rowwise_projection = RowwiseProjection()


class ContextProjection(RowwiseProjection):
    """Tile long prefill projections with fixed BF16 accumulation order."""

    CELLS = (
        Cell("two long DeltaNet prompt chunks", {"rows": 256, "columns": 64}),
        Cell("unaligned long router batch", {"rows": 411, "columns": 257}),
    )

    def __call__(
        self, hidden: torch.Tensor, weight: torch.Tensor, pdl: bool = False
    ) -> torch.Tensor:
        del pdl
        rows, width = hidden.shape
        columns = weight.shape[0]
        tile = 16
        out = hidden.new_empty((rows, columns))
        bf16.project[(tr.cdiv(rows, tile), tr.cdiv(columns, tile))](
            hidden, weight, out, rows, columns, width, tile, tile, 128, num_warps=4
        )
        return out


context_projection = ContextProjection()


def _context_spans(starts: tuple[int, ...]) -> list[tuple[int, int, bool]]:
    """Coalesce neighboring long prompts for one tiled projection."""
    spans = []
    for start, end in zip(starts, starts[1:]):
        tiled = end - start >= 128
        if tiled and spans and spans[-1][2]:
            start, _, _ = spans.pop()
        spans.append((start, end, tiled))
    return spans


def packed_delta_gates(
    hidden: torch.Tensor, weight: torch.Tensor, context_starts: tuple[int, ...]
) -> torch.Tensor:
    """Project context spans and the rowwise decode suffix in request order."""
    context_rows = context_starts[-1]
    if not context_rows:
        return rowwise_projection(hidden, weight)
    gates = [
        context_projection(hidden[a:b], weight) if tiled else F.linear(hidden[a:b], weight)
        for a, b, tiled in _context_spans(context_starts)
    ]
    if context_rows < hidden.shape[0]:
        gates.append(rowwise_projection(hidden[context_rows:], weight))
    return gates[0] if len(gates) == 1 else torch.cat(gates)


class DenseProjection(OpWrapper):
    """Multiply group-128 activations by a block-128 FP8 projection."""

    ARCHS = frozenset({Arch.SM_100})
    CELLS = (
        Cell("MX low-latency GDN output", {"rows": 1, "columns": 2048, "width": 4096}),
        Cell("bundled DeepGEMM for prompt rows", {"rows": 63, "columns": 1024, "width": 2048}),
    )
    note = "FlashInfer MX GEMM for up to eight rows, bundled DeepGEMM for larger matrices."

    def __call__(self, activation: Activation, weight: DenseWeight) -> torch.Tensor:
        if activation[0].shape[0] <= 8:
            return mx_decode.matmul(activation, weight.codes, weight.mx)
        return dg_prefill.project(activation, weight.codes, weight.deepgemm)

    def reference(self, activation: Activation, weight: DenseWeight) -> torch.Tensor:
        """Multiply the expanded scaled operands in FP32."""
        codes, scales = activation[:2]
        x = codes.float() * scales.repeat_interleave(128, dim=1)
        w = weight.codes.float() * weight.scales.repeat_interleave(128, 0).repeat_interleave(128, 1)
        with true_fp32_matmul():
            return F.linear(x, w).to(torch.bfloat16)


dense_projection = DenseProjection()


class RoutedWeights(NamedTuple):
    up: tuple[torch.Tensor, torch.Tensor]
    down: tuple[torch.Tensor, torch.Tensor]


class RoutedExperts(OpWrapper):
    """Compute the gate/up, SwiGLU requantization and down projection per route."""

    ARCHS = frozenset({Arch.SM_100})
    CELLS = (
        Cell("small decode routed by device expert IDs", {"rows": 1}),
        Cell("short packed prefill grouped by expert", {"rows": 11}),
    )
    note = "Standalone TCGen05 expert kernels for short batches and FlashInfer groups for prefill."

    def __call__(
        self,
        incoming: Activation,
        route: torch.Tensor,
        weights: RoutedWeights,
        small: bool,
        routing: tuple[torch.Tensor, torch.Tensor] | None,
        pdl: bool,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        rows = route.shape[0]
        if small:
            split = 4 if rows == 1 else 2
            partials = tc_decode.matmul(
                incoming,
                weights.up,
                route,
                group=4,
                split=split,
                routing=routing,
                exact_partials=True,
                interleave=rows > 1,
            )
            mid = fp8.activation_partials(
                partials,
                incoming,
                incoming,
                weights.up,
                route,
                8,
                split,
                True,
                False,
                False,
                True,
                pdl=pdl,
                live=None,
            )
            result = tc_decode.matmul(
                mid,
                weights.down,
                route,
                down=True,
                group=4,
                routing=routing,
                pdl=pdl,
                early=pdl,
            )
            return result, None
        if rows <= 256:
            grouping = fp8.group_routes(route)
            tile = 64 if rows >= 32 else 128
            up = prefill_tc.project(incoming, weights.up, grouping, rows, col_tile=tile, group=4)
            mid = fp8.activation(up, False, False)
            return prefill_tc.project(
                mid, weights.down, grouping, rows, down=True, col_tile=tile, group=4
            ), None
        return prefill_moe.grouped_experts(
            incoming,
            route,
            weights,
            align=4,
            mma_sm=1,
            rounded=False,
            fp8_up=False,
            grouped_glu=True,
            backend="trtllm",
        )

    def reference(
        self,
        incoming: Activation,
        route: torch.Tensor,
        weights: RoutedWeights,
        small: bool,
        routing: tuple[torch.Tensor, torch.Tensor] | None,
        pdl: bool,
    ) -> torch.Tensor:
        """Evaluate the routed dense products and per-128 SwiGLU quantization."""
        del small, routing, pdl
        codes, scales = incoming[:2]
        x = codes.float() * scales.repeat_interleave(128, dim=1)
        result = torch.empty((route.numel(), 2048), device=x.device, dtype=torch.bfloat16)
        flat_routes = route.flatten()
        with true_fp32_matmul():
            for expert in flat_routes.unique().tolist():
                pairs = (flat_routes == expert).nonzero().flatten()
                up, down = (
                    codes[expert].float()
                    * scales[expert].repeat_interleave(128, 0).repeat_interleave(128, 1)
                    for (codes, scales) in weights
                )
                gate, value = (
                    F.linear(x[pairs // 8], up).to(torch.bfloat16).float().chunk(2, dim=-1)
                )
                activated = (F.silu(gate) * value).view(-1, 4, 128)
                scale = activated.abs().amax(-1, keepdim=True).clamp_min(1e-10) / 448
                middle = (activated / scale).clamp(-448, 448).to(torch.float8_e4m3fn)
                result[pairs] = F.linear((middle.float() * scale).flatten(1), down).to(result.dtype)
        return result


routed_experts = RoutedExperts()


def shared_expert(activation: Activation, layer: LayerWeights) -> torch.Tensor:
    """Evaluate Qwen's separately gated shared expert."""
    up = dense_projection(activation, layer.shared_up)
    mid = fp8.activation(up, True, True, mx=up.shape[0] <= 8)
    return dense_projection(mid, layer.shared_down)


def route_experts(
    hidden: torch.Tensor,
    router: torch.Tensor,
    small: bool,
    pdl: bool,
    *,
    decode: bool = False,
    tiled: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, tuple[torch.Tensor, torch.Tensor] | None]:
    """Return sorted top-eight per-token expert IDs, weights, and shared gates."""
    rows = hidden.shape[0]
    split_gate = not (small or decode or tiled) and rows > 8
    weight = router[:256] if split_gate else router
    if tiled:
        logits = context_projection(hidden, weight)
    elif small or decode:
        logits = rowwise_projection(hidden, weight, pdl)
    else:
        logits = F.linear(hidden, weight)
    ids = hidden.new_empty((rows, 8), dtype=torch.int32)
    probs = hidden.new_empty((rows, 8))
    shared_gate = hidden.new_empty((rows,), dtype=torch.float32)
    routing = None
    if small and rows > 1:
        experts = hidden.new_empty((rows * 8,), dtype=torch.int32)
        pairs = hidden.new_empty((rows * 8, rows), dtype=torch.int32)
        fp8.routing_union[(1,)](
            logits,
            ids,
            probs,
            shared_gate,
            experts,
            pairs,
            ids,
            logits.stride(0),
            rows,
            tr.next_power_of_2(rows),
            tr.next_power_of_2(rows * 8),
            8,
            False,
            True,
            pdl,
            num_warps=4,
            launch_pdl=pdl,
        )
        routing = experts, pairs
    else:
        fp8.routing[(rows,)](
            logits,
            ids,
            probs,
            hidden,
            router[-1],
            shared_gate,
            logits.stride(0),
            2 if split_gate else 0,
            8,
            num_warps=4,
        )
    return ids, probs, shared_gate, routing


def route_packed(
    hidden: torch.Tensor, router: torch.Tensor, context_starts: tuple[int, ...], small: bool
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, tuple[torch.Tensor, torch.Tensor] | None]:
    """Apply the same routing projections whether a request runs alone or in a batch."""
    context_rows = context_starts[-1]
    if not context_rows:
        return route_experts(hidden, router, small, small, decode=True)
    spans = _context_spans(context_starts)
    if len(spans) == 1 and context_rows == hidden.shape[0]:
        return route_experts(hidden, router, False, False, tiled=spans[0][2])
    routing = [
        route_experts(hidden[a:b], router, False, False, tiled=tiled)[:3] for a, b, tiled in spans
    ]
    if context_rows < hidden.shape[0]:
        routing.append(route_experts(hidden[context_rows:], router, False, False, decode=True)[:3])
    route, probs, gate = (torch.cat(parts) for parts in zip(*routing, strict=True))
    return route, probs, gate, None

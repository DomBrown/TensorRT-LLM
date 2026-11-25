# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Dense and routed block-FP8 projections for the SM100 target."""

from __future__ import annotations

import torch
from torch.nn import functional as F

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, Cell, OpWrapper

from ..kernels import bf16_decode, fp8_decode, fp8_expert_panel, fp8_panel, kernels
from ..kernels.prefill_gemm import PrefillGemm
from ..weights import BF16, FP8, HIDDEN_SIZE, SCALE_BLOCK, Fp8Weight

ExpertOutputs = tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]


def uses_cutlass(rows: int, decode: bool) -> bool:
    """Dense CUTLASS case uses at least 128 rows and aligned prefill scales."""
    return rows >= 128 and (decode or rows % 4 == 0)


def _dequantize_activation(a: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return (a.float().view(a.shape[0], -1, SCALE_BLOCK) * scale[..., None]).reshape_as(a)


def _dequantize_weight(w: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    n, k = w.shape
    view = w.float().view(n // SCALE_BLOCK, SCALE_BLOCK, k // SCALE_BLOCK, SCALE_BLOCK)
    return (view * scale[:, None, :, None].float()).reshape(n, k)


class Fp8Projection(OpWrapper):
    """Project quantized activation rows against E4M3 weights and FP32/BF16 scales."""

    ARCHS = frozenset({Arch.SM_100})
    CELLS = (
        Cell(
            "narrow SM100 TMA input projection", {"rows": 1, "n": 9216, "k": 2048, "decode": True}
        ),
        Cell(
            "ordered FP8 group accumulation for output projection",
            {"rows": 4, "n": 2048, "k": 4096, "decode": True},
        ),
        Cell("medium-batch decode", {"rows": 32, "n": 12288, "k": 2048, "decode": True}),
        Cell("large-batch CUTLASS decode", {"rows": 128, "n": 9216, "k": 2048, "decode": True}),
        Cell("short-context 64-row tile", {"rows": 83, "n": 9216, "k": 2048, "decode": False}),
        Cell(
            "CUTLASS prefill using prepared MN-major scales",
            {"rows": 512, "n": 2048, "k": 4096, "decode": False},
        ),
    )

    def __call__(
        self, quantized: Fp8Weight, weight: Fp8Weight, decode: bool, gemm: PrefillGemm
    ) -> torch.Tensor:
        a, scale = quantized
        rows = a.shape[0]
        if uses_cutlass(rows, decode):
            return gemm.dense(a, scale, *weight)
        if decode:
            if rows <= 8 and a.shape[1] == HIDDEN_SIZE:
                return fp8_panel.dense(a, scale, *weight)
            return fp8_decode.dense(a, scale, *weight, split=True, prefetch_groups=True)
        return kernels.fp8_linear(a, scale, *weight)

    def is_valid(
        self, quantized: Fp8Weight, weight: Fp8Weight, decode: bool, gemm: PrefillGemm
    ) -> None:
        a, scale = quantized
        if not uses_cutlass(a.shape[0], decode) and not scale.is_contiguous():
            raise ValueError("the Triton FP8 projection requires row-major activation scales")

    def reference(
        self, quantized: Fp8Weight, weight: Fp8Weight, decode: bool, gemm: PrefillGemm
    ) -> torch.Tensor:
        return F.linear(_dequantize_activation(*quantized), _dequantize_weight(*weight)).to(BF16)


class DecodeGates(OpWrapper):
    """Project and apply the Gated DeltaNet gates for decode-sized batches."""

    ARCHS = frozenset({Arch.SM_100})
    CELLS = (
        Cell("split-K gate projection for batch 1", {"rows": 1}),
        Cell("batched BF16 gate projection", {"rows": 32}),
        Cell("wide partitioned gate projection", {"rows": 128}),
        Cell("same reduction for packed short contexts", {"rows": 39, "fixed_reduction": True}),
    )

    def __call__(
        self,
        x: torch.Tensor,
        weight: torch.Tensor,
        a_log: torch.Tensor,
        dt_bias: torch.Tensor,
        query_rows: int = 1,
        fixed_reduction: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch = 1 if fixed_reduction else x.shape[0] // query_rows
        return bf16_decode.gates(x, weight, a_log, dt_bias, True, batch)

    def reference(
        self,
        x: torch.Tensor,
        weight: torch.Tensor,
        a_log: torch.Tensor,
        dt_bias: torch.Tensor,
        query_rows: int = 1,
        fixed_reduction: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        ab = F.linear(x.float(), weight.float()).to(BF16).float()
        a, b = ab.chunk(2, dim=-1)
        decay = (-a_log.float().exp() * F.softplus(a + dt_bias.float())).exp()
        return decay, b.sigmoid().bfloat16().float()


class RoutedExperts(OpWrapper):
    """Route eight experts plus the shared expert, then apply their W8A8 MLPs.

    Returns expert rows or grouped rows with a permutation, and the router's
    state used by the fixed-order residual epilogue.
    """

    ARCHS = frozenset({Arch.SM_100})
    CELLS = (
        Cell("batch 1 fused route and TMA expert panels", {"rows": 1, "decode": True}),
        Cell("small decode with expert tile lists", {"rows": 4, "decode": True}),
        Cell("batch 32 fused route scheduling", {"rows": 32, "decode": True}),
        Cell("batch 128 grouped decode", {"rows": 128, "decode": True}),
        Cell("short prefill", {"rows": 27, "decode": False}),
        Cell("short-prompt packed prefill with TMA expert tiles", {"rows": 83, "decode": False}),
        Cell(
            "wide packed prefill and compact expert scatter",
            {"rows": 512, "decode": False, "fixed_router": True},
        ),
    )
    note = "Composes the reference engine's stable top-k, expert schedule, W8A8 GEMMs and SwiGLU."

    def __call__(
        self,
        x: torch.Tensor,
        quantized: Fp8Weight,
        router: torch.Tensor,
        gate_up: Fp8Weight,
        down: Fp8Weight,
        decode: bool,
        fixed_router: bool,
        route_counts: torch.Tensor,
        gemm: PrefillGemm,
        query_rows: int = 1,
    ) -> ExpertOutputs:
        rows = x.shape[0]
        batch = rows // query_rows
        wide_prefill = not decode and rows >= 512
        small_prefill = not decode and 64 <= rows < 512
        fast = decode or small_prefill
        tile_rows = (16 if rows >= 256 else fp8_decode.ROUTE_BLOCK) if fast else 64
        schedule = slots = counts = None
        if decode and batch >= 32:
            logits, routes, probabilities, schedule = bf16_decode.router_topk(
                x, router, batch, True, route_counts, tile_rows, width=query_rows
            )
        elif decode and batch == 1:
            logits, routes, probabilities = bf16_decode.single_router(x, router, True, True)
        else:
            if fixed_router:
                logits = bf16_decode.prefill_router(x, router, True)
            elif decode:
                logits = bf16_decode.router(x, router, batch)
            else:
                logits = F.linear(x, router)
            if wide_prefill:
                counts = torch.zeros(256, device=x.device, dtype=torch.int32)
                routes, probabilities, slots = kernels.route_logits(logits, True, counts)
            else:
                routes, probabilities = kernels.route_logits(logits, True)

        if schedule is None and not wide_prefill and rows > 1:
            schedule = kernels.make_expert_lists(routes, tile_rows)
        a, scale = quantized
        permutation = None
        if wide_prefill:
            out, permutation = gemm.experts(
                a, scale, gate_up, down, routes, schedule, slots, counts, defer_gather=True
            )
        elif fast:
            if decode and rows == 1:
                a, scale = fp8_expert_panel.gate_swiglu(a, scale, *gate_up, routes)
            else:
                if decode and rows <= 8:
                    up = fp8_expert_panel.gate_up(a, scale, *gate_up, schedule, tile_rows)
                else:
                    up = fp8_decode.routed(
                        a,
                        scale,
                        *gate_up,
                        routes,
                        schedule,
                        up=True,
                        tile_rows=tile_rows,
                        tuned=True,
                    )
                a, scale = kernels.quantize(up, "swiglu")
            out = fp8_decode.routed(
                a,
                scale,
                *down,
                routes,
                schedule,
                up=False,
                tile_rows=tile_rows,
                tuned=True,
            )
        else:
            up = kernels.fp8_linear(a, scale, *gate_up, route=routes, schedule=schedule, up=True)
            a, scale = kernels.quantize(up, "swiglu")
            out = kernels.fp8_linear(a, scale, *down, route=routes, schedule=schedule)
        return out, probabilities, logits, permutation

    def reference(
        self,
        x: torch.Tensor,
        quantized: Fp8Weight,
        router: torch.Tensor,
        gate_up: Fp8Weight,
        down: Fp8Weight,
        decode: bool,
        fixed_router: bool,
        route_counts: torch.Tensor,
        gemm: PrefillGemm,
        query_rows: int = 1,
    ) -> ExpertOutputs:
        logits = F.linear(x.float(), router.float()).to(BF16)
        sorted_logits, sorted_experts = logits[:, :256].float().sort(descending=True, stable=True)
        probabilities = sorted_logits[:, :8].softmax(dim=-1).to(BF16)
        shared_id = torch.full_like(sorted_experts[:, :1], 256)
        routes = torch.cat((sorted_experts[:, :8], shared_id), dim=-1).flatten()
        activation = _dequantize_activation(*quantized)
        out = x.new_empty((routes.numel(), x.shape[1]))
        for expert in routes.unique().tolist():
            pairs = torch.where(routes == expert)[0]
            expert_input = activation[pairs // 9]
            packed_weight = _dequantize_weight(gate_up[0][expert], gate_up[1][expert])
            g, u = F.linear(expert_input, packed_weight).to(BF16).float().chunk(2, dim=-1)
            expanded = (F.silu(g) * u).to(BF16).float()
            grouped = expanded.view(expanded.shape[0], -1, 128)
            maximum = grouped.abs().amax(dim=-1, keepdim=True).clamp_min(1e-10)
            a_fp8 = (grouped * (448.0 / maximum)).clamp(-448, 448).to(FP8).float()
            hidden = (a_fp8 * (maximum / 448.0)).reshape_as(expanded)
            output_weight = _dequantize_weight(down[0][expert], down[1][expert])
            out[pairs] = F.linear(hidden, output_weight).to(BF16)
        return out, probabilities, logits, None


fp8_projection = Fp8Projection()
decode_gates = DecodeGates()
routed_experts = RoutedExperts()

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Block FP8 activation quantization and Qwen-specific normalization epilogues."""

from __future__ import annotations

import torch
from torch.nn import functional as F

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, Cell, OpWrapper

from ..kernels import kernels, norms
from ..weights import FP8, Fp8Weight


def _quantize(x: torch.Tensor, mn_major: bool) -> Fp8Weight:
    groups = x.float().view(x.shape[0], -1, 128)
    maximum = groups.abs().amax(-1).clamp_min(1e-10)
    scale = maximum / 448.0
    quant = (groups * (448.0 / maximum[:, :, None])).clamp(-448, 448).to(FP8)
    if mn_major:
        scale = scale.T.contiguous().T
    return quant.reshape_as(x), scale


class BlockQuantize(OpWrapper):
    """Quantize activation blocks of 128 channels, optionally after expert SwiGLU."""

    ARCHS = frozenset({Arch.SM_100})
    CELLS = (
        Cell(
            "decode SwiGLU for one token's nine experts", {"rows": 9, "width": 512, "swiglu": True}
        ),
        Cell(
            "prefill dense projection MN-major scales",
            {"rows": 132, "width": 2048, "mn_major": True},
        ),
    )

    def __call__(self, x: torch.Tensor, swiglu: bool = False, mn_major: bool = False) -> Fp8Weight:
        return kernels.quantize(x, "swiglu" if swiglu else "", mn_major=mn_major)

    def reference(self, x: torch.Tensor, swiglu: bool = False, mn_major: bool = False) -> Fp8Weight:
        if swiglu:
            gate, up = x.float().chunk(2, dim=-1)
            x = (F.silu(gate) * up).to(x.dtype)
        return _quantize(x, mn_major)


class ResidualNormQuantize(OpWrapper):
    """Add a mixer or expert delta, normalize, and stage the next projection."""

    ARCHS = frozenset({Arch.SM_100})
    CELLS = (
        Cell("embedding RMSNorm", {"rows": 1, "mode": "rms"}),
        Cell("attention residual and normalization", {"rows": 31, "mode": "add"}),
        Cell(
            "packed expert result and next FP8 projection",
            {"rows": 33, "mode": "moe", "packed": True, "mn_major": True},
        ),
        Cell(
            "nine decode expert slots and final normalization",
            {"rows": 32, "mode": "moe", "quantize": False},
        ),
    )

    def __call__(
        self,
        x: torch.Tensor,
        weight: torch.Tensor,
        delta: torch.Tensor | None = None,
        probability: torch.Tensor | None = None,
        router: torch.Tensor | None = None,
        permutation: torch.Tensor | None = None,
        quantize: bool = True,
        mn_major: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, Fp8Weight | None]:
        residual, y, scaled = norms.norm_quant(
            x,
            weight,
            delta=delta,
            probability=probability,
            router=router,
            permutation=permutation,
            quantize=quantize,
            mn_major=mn_major,
        )
        return residual, y, scaled if quantize else None

    def reference(
        self,
        x: torch.Tensor,
        weight: torch.Tensor,
        delta: torch.Tensor | None = None,
        probability: torch.Tensor | None = None,
        router: torch.Tensor | None = None,
        permutation: torch.Tensor | None = None,
        quantize: bool = True,
        mn_major: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, Fp8Weight | None]:
        if probability is not None:
            if permutation is not None:
                delta = delta[permutation.long().flatten()]
            experts = delta.view(x.shape[0], 9, x.shape[1]).float()
            routed = (experts[:, :8] * probability.float()[:, :, None]).bfloat16().float()
            shared_gate = router[:, 256].float().sigmoid().bfloat16().float()
            shared = (experts[:, 8] * shared_gate[:, None]).bfloat16().float()
            delta = (routed.sum(1).bfloat16().float() + shared).bfloat16()
        y = x.float() if delta is None else x.float() + delta.float()
        residual = x if delta is None else y.to(x.dtype)
        normalized = (
            y * (y.square().mean(-1, keepdim=True) + 1e-6).rsqrt() * (1 + weight.float())
        ).to(x.dtype)
        return residual, normalized, _quantize(normalized, mn_major) if quantize else None


class GatedOutputQuantize(OpWrapper):
    """Apply the mixer gate and quantize one 128-channel group at a time."""

    ARCHS = frozenset({Arch.SM_100})
    CELLS = (
        Cell("GDN decode", {"rows": 1, "recurrent": True}),
        Cell(
            "GDN verification with a separate output gate",
            {"rows": 5, "recurrent": True, "gate_only": True},
        ),
        Cell("GDN grouped prefill", {"rows": 128, "recurrent": True, "mn_major": True}),
        Cell("GQA decode", {"rows": 32, "recurrent": False}),
        Cell("GQA grouped prefill", {"rows": 132, "recurrent": False, "mn_major": True}),
    )

    def __call__(
        self,
        core: torch.Tensor,
        projection: torch.Tensor,
        weight: torch.Tensor | None,
        mn_major: bool = False,
    ) -> Fp8Weight:
        if weight is not None:
            return kernels.gdn_output(core, projection, weight, False, mn_major, grouped=True)
        return kernels.attention_output(core, projection, False, False, mn_major, grouped=True)

    def reference(
        self,
        core: torch.Tensor,
        projection: torch.Tensor,
        weight: torch.Tensor | None,
        mn_major: bool = False,
    ) -> Fp8Weight:
        rows = core.shape[0]
        if weight is not None:
            x = core.float().reshape(rows, 32, 128)
            x = x * (x.square().mean(-1, keepdim=True) + 1e-6).rsqrt() * weight.float()
            z = projection[:, -4096:].float().reshape(rows, 32, 128)
            output = x * F.silu(z)
        else:
            z = projection[:, :8192].float().reshape(rows, 16, 512)[..., 256:]
            output = core.float().reshape(rows, 16, 256) * z.sigmoid()
        return _quantize(output.reshape(rows, 4096).to(core.dtype), mn_major)


class GdnGates(OpWrapper):
    """Compute the decay and BF16-rounded update gate from BF16 A/B projections."""

    ARCHS = frozenset({Arch.SM_100})
    CELLS = (Cell("mixed magnitude A/B gate inputs", {"rows": 39}),)

    def __call__(
        self, ab: torch.Tensor, a_log: torch.Tensor, dt_bias: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return kernels.gdn_gates(ab, a_log, dt_bias, True)

    def reference(
        self, ab: torch.Tensor, a_log: torch.Tensor, dt_bias: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        a, b = ab.float().chunk(2, dim=-1)
        decay = (-a_log.float().exp() * F.softplus(a + dt_bias.float())).exp()
        return decay, b.sigmoid().bfloat16().float()


class MtpInput(OpWrapper):
    """Gather token embeddings and normalize both inputs to the MTP projection."""

    ARCHS = frozenset({Arch.SM_100})
    CELLS = (
        Cell("single-token drafting", {"rows": 1}),
        Cell("batched verification rows and repeated token IDs", {"rows": 128}),
    )

    def __call__(
        self,
        input_ids: torch.Tensor,
        embedding: torch.Tensor,
        hidden: torch.Tensor,
        embedding_norm: torch.Tensor,
        hidden_norm: torch.Tensor,
    ) -> torch.Tensor:
        return norms.mtp_input(input_ids, embedding, hidden, embedding_norm, hidden_norm)

    def reference(
        self,
        input_ids: torch.Tensor,
        embedding: torch.Tensor,
        hidden: torch.Tensor,
        embedding_norm: torch.Tensor,
        hidden_norm: torch.Tensor,
    ) -> torch.Tensor:
        return torch.cat(
            (
                residual_norm_quantize.reference(
                    F.embedding(input_ids, embedding), embedding_norm, quantize=False
                )[1],
                residual_norm_quantize.reference(hidden, hidden_norm, quantize=False)[1],
            ),
            dim=-1,
        )


block_quantize = BlockQuantize()
residual_norm_quantize = ResidualNormQuantize()
gated_output_quantize = GatedOutputQuantize()
gdn_gates = GdnGates()
mtp_input = MtpInput()

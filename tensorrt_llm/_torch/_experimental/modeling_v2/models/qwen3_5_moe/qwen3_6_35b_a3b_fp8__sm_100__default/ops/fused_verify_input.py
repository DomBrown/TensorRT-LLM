# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Fuse single-request GDN projection and speculative convolution."""

import torch
from torch.nn import functional as F

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, Cell, OpWrapper

from ..kernels.fp8_panel import projection_and_verify
from ..weights import Fp8Weight
from .projection import _dequantize_activation, _dequantize_weight
from .recurrent import verify_conv


class FusedVerifyInput(OpWrapper):
    """Project QKV and the output gate, then convolve and stage one draft chain."""

    ARCHS = frozenset({Arch.SM_100})
    CELLS = (
        Cell("staged rolling windows for four projected tokens", {"width": 4}),
        Cell("raw replay inputs for five projected tokens", {"width": 5, "raw_only": True}),
    )

    def __call__(
        self,
        activations: Fp8Weight,
        weight: Fp8Weight,
        conv_weight: torch.Tensor,
        history: torch.Tensor,
        slots: torch.Tensor,
        staged: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return projection_and_verify(*activations, *weight, conv_weight, history, slots, staged)

    def reference(
        self,
        activations: Fp8Weight,
        weight: Fp8Weight,
        conv_weight: torch.Tensor,
        history: torch.Tensor,
        slots: torch.Tensor,
        staged: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        projection = F.linear(
            _dequantize_activation(*activations), _dequantize_weight(*weight)
        ).bfloat16()
        mixed = verify_conv.reference(
            projection, conv_weight, slots, history, staged, projection.shape[0]
        )
        return projection[:, -4096:].contiguous(), mixed


fused_verify_input = FusedVerifyInput()

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Load the official checkpoint's TP1 layout without dequantizing its weights."""

from collections.abc import Mapping
from dataclasses import dataclass

import torch

FP8 = torch.float8_e4m3fn
BF16 = torch.bfloat16
EXPERT_COUNT = 256
HIDDEN_SIZE = 2048
EXPERT_WIDTH = 512
SCALE_BLOCK = 128
Fp8Weight = tuple[torch.Tensor, torch.Tensor]


@dataclass(frozen=True, slots=True)
class LayerWeights:
    input_norm: torch.Tensor
    post_attention_norm: torch.Tensor
    input_projection: Fp8Weight
    output_projection: Fp8Weight
    router: torch.Tensor
    expert_gate_up: Fp8Weight
    expert_down: Fp8Weight
    linear_attention: bool
    attention_params: tuple[torch.Tensor, ...]


def _expert_weights(
    weights: Mapping[str, torch.Tensor], prefix: str, device: torch.device
) -> tuple[Fp8Weight, Fp8Weight]:
    count = EXPERT_COUNT + 1
    width = EXPERT_WIDTH
    hidden = HIDDEN_SIZE
    block = SCALE_BLOCK
    # Stack the shared expert after the routed experts, with identical layouts.
    gate_up = torch.empty((count, 2 * width, hidden), dtype=torch.uint8, device="cpu")
    down = torch.empty((count, hidden, width), dtype=torch.uint8, device="cpu")
    gate_up_scale = torch.empty(
        (count, 2 * width // block, hidden // block), dtype=BF16, device="cpu"
    )
    down_scale = torch.empty((count, hidden // block, width // block), dtype=BF16, device="cpu")
    for expert in range(count):
        stem = prefix + (f"mlp.experts.{expert}" if expert < EXPERT_COUNT else "mlp.shared_expert")
        for half, projection in enumerate(("gate_proj", "up_proj")):
            gate_up[expert, half * width : (half + 1) * width].copy_(
                weights[f"{stem}.{projection}.weight"].view(torch.uint8)
            )
            scale_width = width // block
            gate_up_scale[expert, half * scale_width : (half + 1) * scale_width].copy_(
                weights[f"{stem}.{projection}.weight_scale_inv"]
            )
        down[expert].copy_(weights[f"{stem}.down_proj.weight"].view(torch.uint8))
        down_scale[expert].copy_(weights[f"{stem}.down_proj.weight_scale_inv"])
    return (
        (gate_up.to(device).view(FP8), gate_up_scale.to(device)),
        (down.to(device).view(FP8), down_scale.to(device)),
    )


def load_layer(
    weights: Mapping[str, torch.Tensor],
    prefix: str,
    linear_attention: bool,
    device: torch.device,
) -> LayerWeights:
    """Fuse activation-independent projection weights and their block scales."""

    def read(name: str) -> torch.Tensor:
        return weights[prefix + name]

    def gpu(name: str) -> torch.Tensor:
        return read(name).to(device)

    def projection(*names: str) -> Fp8Weight:
        weight = torch.cat([read(name + ".weight").view(torch.uint8) for name in names])
        scale = torch.cat([read(name + ".weight_scale_inv") for name in names])
        return weight.to(device).view(FP8), scale.to(device)

    if linear_attention:
        stem = "linear_attn."
        input_projection = projection(stem + "in_proj_qkv", stem + "in_proj_z")
        output_projection = projection(stem + "out_proj")
        attention_params = (
            torch.cat([read(stem + f"in_proj_{gate}.weight") for gate in ("a", "b")]).to(device),
            gpu(stem + "conv1d.weight"),
            gpu(stem + "A_log"),
            gpu(stem + "dt_bias"),
            gpu(stem + "norm.weight"),
        )
    else:
        stem = "self_attn."
        input_projection = projection(*(stem + name + "_proj" for name in ("q", "k", "v")))
        output_projection = projection(stem + "o_proj")
        attention_params = gpu(stem + "q_norm.weight"), gpu(stem + "k_norm.weight")
    expert_gate_up, expert_down = _expert_weights(weights, prefix, device)
    return LayerWeights(
        gpu("input_layernorm.weight"),
        gpu("post_attention_layernorm.weight"),
        input_projection,
        output_projection,
        torch.cat([read(f"mlp.{gate}.weight") for gate in ("gate", "shared_expert_gate")]).to(
            device
        ),
        expert_gate_up,
        expert_down,
        linear_attention,
        attention_params,
    )

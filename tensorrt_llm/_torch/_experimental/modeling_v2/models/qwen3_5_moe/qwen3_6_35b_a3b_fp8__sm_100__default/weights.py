# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Place official Qwen3.6-35B-A3B FP8 tensors in their execution layouts."""

from collections.abc import Mapping
from typing import NamedTuple

import torch
import torch.nn as nn

from .kernels import dg_prefill, fp8, mx_decode


class DenseWeight(NamedTuple):
    codes: torch.Tensor
    scales: torch.Tensor
    mx: torch.Tensor
    deepgemm: torch.Tensor


class LayerWeights(nn.Module):
    """One block's packed checkpoint weights and dense FP8 layouts."""

    def __init__(self, weights: Mapping[str, torch.Tensor], prefix: str, linear: bool) -> None:
        super().__init__()
        self.linear = linear

        def read(suffix: str) -> torch.Tensor:
            return weights[prefix + suffix]

        def store(name: str, value: torch.Tensor) -> None:
            self.register_buffer(name, value.cuda())

        def dense(name: str, sources: list[str]) -> DenseWeight:
            codes = [read(source + ".weight").view(torch.uint8) for source in sources]
            scales = [read(source + ".weight_scale_inv") for source in sources]
            store(name + "_checkpoint", torch.cat(codes).view(torch.float8_e4m3fn))
            store(name + "_checkpoint_scale", torch.cat(scales))
            source = self.get_buffer(name + "_checkpoint")
            block_scales = self.get_buffer(name + "_checkpoint_scale").float()
            codes = torch.empty_like(source)
            scales = torch.empty_like(block_scales)
            rows, width = source.shape
            fp8.requant_weights[(rows // 128, width // 128)](
                source, block_scales, codes, scales, width, num_warps=8
            )
            mx = mx_decode.swizzle(scales, rows, width, row_group=128)
            dg = dg_prefill.encode(scales, rows, width, row_group=128)
            for suffix, buffer in (("codes", codes), ("scales", scales), ("mx", mx), ("dg", dg)):
                store(name + "_" + suffix, buffer)
            return DenseWeight(codes, scales, mx, dg)

        def expert(projection_names: tuple[str, ...]) -> tuple[torch.Tensor, torch.Tensor]:
            packed = []
            for suffix, as_bytes in ((".weight", True), (".weight_scale_inv", False)):
                groups = []
                for part in projection_names:
                    tensors = [
                        read(f"mlp.experts.{expert_id}.{part}{suffix}") for expert_id in range(256)
                    ]
                    if as_bytes:
                        tensors = [value.view(torch.uint8) for value in tensors]
                    groups.append(torch.stack(tensors).cuda())
                result = groups[0] if len(groups) == 1 else torch.cat(groups, dim=1)
                packed.append(result.view(torch.float8_e4m3fn) if as_bytes else result.float())
            return packed[0], packed[1]

        store("input_norm", read("input_layernorm.weight"))
        store("post_norm", read("post_attention_layernorm.weight"))
        if linear:
            self.qkv = dense("qkv", ["linear_attn.in_proj_qkv", "linear_attn.in_proj_z"])
            self.proj = dense("proj", ["linear_attn.out_proj"])
            store(
                "ba",
                torch.cat(
                    [read("linear_attn.in_proj_b.weight"), read("linear_attn.in_proj_a.weight")]
                ),
            )
            store("conv", read("linear_attn.conv1d.weight").view(8192, 4))
            store("decay", read("linear_attn.A_log"))
            store("dt", read("linear_attn.dt_bias"))
            store("gdn_norm", read("linear_attn.norm.weight"))
        else:
            self.qkv = dense("qkv", [f"self_attn.{name}_proj" for name in ("q", "k", "v")])
            self.proj = dense("proj", ["self_attn.o_proj"])
            store("q_norm", read("self_attn.q_norm.weight"))
            store("k_norm", read("self_attn.k_norm.weight"))
        store("router", torch.cat([read("mlp.gate.weight"), read("mlp.shared_expert_gate.weight")]))
        shared = "mlp.shared_expert."
        self.shared_up = dense("shared_up", [shared + "gate_proj", shared + "up_proj"])
        self.shared_down = dense("shared_down", [shared + "down_proj"])
        self.expert_up = expert(("gate_proj", "up_proj"))
        self.expert_down = expert(("down_proj",))
        for name, values in (("expert_up", self.expert_up), ("expert_down", self.expert_down)):
            store(name + "_codes", values[0])
            store(name + "_scales", values[1])


def load_text_weights(
    decoder: nn.Module,
    lm_head: nn.Module,
    weights: Mapping[str, torch.Tensor],
    layer_types: list[str],
    max_position: int,
) -> None:
    """Load the 40 decoder blocks and prepare the partial rotary table."""
    decoder.embed_tokens.weight = nn.Parameter(
        weights["model.language_model.embed_tokens.weight"].cuda(), requires_grad=False
    )
    lm_head.weight = nn.Parameter(weights["lm_head.weight"].cuda(), requires_grad=False)
    decoder.final_norm = weights["model.language_model.norm.weight"].cuda()
    decoder.layers.extend(
        LayerWeights(weights, f"model.language_model.layers.{i}.", kind == "linear_attention")
        for i, kind in enumerate(layer_types)
    )
    frequency = 1.0 / (
        10_000_000.0 ** (torch.arange(0, 64, 2, device="cuda", dtype=torch.float32) / 64)
    )
    phases = torch.arange(max_position, device="cuda", dtype=torch.float32)[:, None] * frequency
    decoder.rope = torch.cat((phases.cos(), phases.sin()), dim=-1)

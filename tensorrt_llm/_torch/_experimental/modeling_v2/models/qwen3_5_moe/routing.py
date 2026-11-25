# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import TYPE_CHECKING

from tensorrt_llm.quantization.mode import QuantAlgo

from ..._router_index import NoTarget, routes

if TYPE_CHECKING:
    from tensorrt_llm._torch.model_config import ModelConfig
    from tensorrt_llm.llmapi.llm_args import TorchLlmArgs

# Also fixes the geometry of the target's projection, recurrent and expert kernels.
_TEXT_SHAPE = {
    "hidden_size": 2048,
    "num_hidden_layers": 40,
    "num_experts": 256,
    "num_experts_per_tok": 8,
    "moe_intermediate_size": 512,
    "shared_expert_intermediate_size": 512,
    "num_attention_heads": 16,
    "num_key_value_heads": 2,
    "head_dim": 256,
    "attn_output_gate": True,
    "linear_num_key_heads": 16,
    "linear_num_value_heads": 32,
    "linear_key_head_dim": 128,
    "linear_value_head_dim": 128,
    "linear_conv_kernel_dim": 4,
    "vocab_size": 248320,
    "tie_word_embeddings": False,
    "rms_norm_eps": 1e-6,
}


@routes("Qwen3_5MoeForConditionalGeneration")
def route(
    model_config: ModelConfig, llm_args: TorchLlmArgs, compute_capability: tuple[int, int]
) -> type:
    """Resolve the text-only Qwen3.6-35B-A3B block-FP8 target on SM100."""
    config = model_config.pretrained_config
    text = getattr(config, "text_config", None)
    if text is None:
        raise NoTarget("expected the Qwen3.6-35B-A3B text config")
    text = text if isinstance(text, dict) else text.to_dict()
    quant = getattr(config, "quantization_config", None) or text.get("quantization_config", {})
    if (
        compute_capability != (10, 0)
        or model_config.quant_config.quant_algo != QuantAlgo.FP8_BLOCK_SCALES
        or quant.get("fmt") != "e4m3"
        or tuple(quant.get("weight_block_size") or ()) != (128, 128)
        or quant.get("activation_scheme") != "dynamic"
        or any(text.get(key) != value for key, value in _TEXT_SHAPE.items())
        or text.get("layer_types") != (["linear_attention"] * 3 + ["full_attention"]) * 10
    ):
        raise NoTarget("expected the Qwen3.6-35B-A3B dynamic 128x128 block-FP8 checkpoint on SM100")
    rope = text.get("rope_parameters") or text.get("rope_scaling") or {}
    if rope.get("partial_rotary_factor", text.get("partial_rotary_factor")) != 0.25:
        raise NoTarget("expected 64 rotary channels per attention head")

    mapping = model_config.mapping
    if mapping.world_size != 1 or mapping.enable_attention_dp:
        raise NoTarget("Qwen3.6-35B-A3B FP8 requires single-rank TP1")
    if not llm_args.disable_mm_encoder or llm_args.mm_encoder_only:
        raise NoTarget("Qwen3.6-35B-A3B FP8 is text-only. Set disable_mm_encoder=True.")
    if model_config.spec_config is not None:
        raise NoTarget("Qwen3.6-35B-A3B FP8 currently serves without speculation")
    if model_config.lora_config is not None or llm_args.max_beam_width != 1:
        raise NoTarget("Qwen3.6-35B-A3B FP8 requires beam width 1 and no LoRA")
    cache = llm_args.kv_cache_config
    if (
        cache.enable_block_reuse
        or cache.use_kv_cache_manager_v2 is False
        or cache.dtype not in ("auto", "bfloat16")
        or cache.max_attention_window is not None
        or cache.mamba_ssm_cache_dtype not in ("auto", "float32")
        or llm_args.enable_return_routed_experts
    ):
        raise NoTarget("Qwen3.6-35B-A3B FP8 requires BF16 paged KV and FP32 recurrent state")
    if llm_args.attn_backend != "TRTLLM":
        raise NoTarget("Qwen3.6-35B-A3B FP8 requires TRTLLM attention metadata")

    from .qwen3_6_35b_a3b_fp8__sm_100__default.modeling import ModelingV2Qwen3635BA3BFp8Sm100Default

    return ModelingV2Qwen3635BA3BFp8Sm100Default

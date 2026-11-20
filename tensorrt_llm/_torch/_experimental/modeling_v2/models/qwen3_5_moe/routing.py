# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Route the Qwen3.6-35B-A3B block-FP8 text decoder."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ..._router_index import NoTarget, engine_kv_cache_quant_algo, routes

if TYPE_CHECKING:
    from tensorrt_llm._torch.model_config import ModelConfig
    from tensorrt_llm.llmapi.llm_args import TorchLlmArgs


@routes("Qwen3_5MoeForConditionalGeneration")
def route(
    model_config: ModelConfig, llm_args: TorchLlmArgs, compute_capability: tuple[int, int]
) -> type:
    """Select the text-only, single-device GB200 target."""
    config = model_config.pretrained_config
    text = getattr(config, "text_config", None)
    expected = {
        "hidden_size": 2048,
        "vocab_size": 248320,
        "num_hidden_layers": 40,
        "num_attention_heads": 16,
        "num_key_value_heads": 2,
        "head_dim": 256,
        "num_experts": 256,
        "num_experts_per_tok": 8,
        "moe_intermediate_size": 512,
        "shared_expert_intermediate_size": 512,
        "linear_conv_kernel_dim": 4,
        "linear_key_head_dim": 128,
        "linear_value_head_dim": 128,
        "linear_num_key_heads": 16,
        "linear_num_value_heads": 32,
        "rms_norm_eps": 1e-6,
        "hidden_act": "silu",
        "attention_bias": False,
        "attn_output_gate": True,
        "tie_word_embeddings": False,
    }
    if any(getattr(text, field, None) != size for field, size in expected.items()):
        raise NoTarget("the Qwen3.6 target requires the 35B-A3B text configuration")
    schedule = ("linear_attention",) * 3 + ("full_attention",)
    if tuple(getattr(text, "layer_types", None) or ()) != schedule * 10:
        raise NoTarget("the Qwen3.6 target requires the 3:1 GDN/full-attention layer schedule")
    rope = getattr(text, "rope_parameters", None) or getattr(text, "rope_scaling", None) or {}
    theta = rope.get("rope_theta", getattr(text, "rope_theta", None))
    fraction = rope.get("partial_rotary_factor", getattr(text, "partial_rotary_factor", None))
    if theta != 10_000_000 or fraction != 0.25 or rope.get("rope_type", "default") != "default":
        raise NoTarget("the Qwen3.6 target requires its checkpoint's partial text RoPE")
    quant = getattr(config, "quantization_config", None)
    if not isinstance(quant, dict) or (
        quant.get("quant_method"),
        quant.get("fmt"),
        tuple(quant.get("weight_block_size") or ()),
    ) != ("fp8", "e4m3", (128, 128)):
        raise NoTarget("the Qwen3.6 target requires checkpoint E4M3 FP8 with 128x128 block scales")
    if compute_capability != (10, 0) or model_config.mapping.world_size != 1:
        raise NoTarget("the Qwen3.6 FP8 target requires one SM100 GPU")
    if not llm_args.disable_mm_encoder or llm_args.mm_encoder_only:
        raise NoTarget(
            "the Qwen3.6 FP8 target requires disable_mm_encoder=True for text-only serving"
        )
    cache = llm_args.kv_cache_config
    if str(model_config.torch_dtype) != "torch.bfloat16":
        raise NoTarget("the Qwen3.6 target requires BF16 activations")
    if engine_kv_cache_quant_algo(model_config, llm_args) not in (None, "bfloat16"):
        raise NoTarget("the Qwen3.6 target requires BF16 KV pages")
    if cache.use_kv_cache_manager_v2 is False or cache.enable_block_reuse:
        raise NoTarget("the Qwen3.6 target requires KV cache manager V2 with block reuse disabled")
    if cache.mamba_ssm_cache_dtype != "float32":
        raise NoTarget("the Qwen3.6 target requires kv_cache_config.mamba_ssm_cache_dtype=float32")
    spec = model_config.spec_config
    if spec is not None and (
        not spec.spec_dec_mode.is_mtp_eagle_one_model()
        or spec.speculative_model is not None
        or not 1 <= spec.max_draft_len <= 3
        or spec.use_dynamic_tree
        or spec.use_relaxed_acceptance_for_thinking
        or spec.draft_lm_head_dtype != "auto"
        or spec.draft_moe_dtype != "auto"
        or getattr(text, "mtp_num_hidden_layers", 0) != 1
        or getattr(text, "mtp_use_dedicated_embeddings", False)
    ):
        raise NoTarget(
            "the Qwen3.6 target supports only the embedded linear MTP-Eagle head at depth 1–3"
        )
    if (
        model_config.lora_config is not None
        or llm_args.cache_transceiver_config is not None
        or llm_args.max_beam_width != 1
    ):
        raise NoTarget("the Qwen3.6 target supports beam width one without LoRA or disaggregation")
    if llm_args.attn_backend != "TRTLLM":
        raise NoTarget("the Qwen3.6 target expects TRTLLM scheduler attention metadata")
    if llm_args.torch_compile_config is not None:
        raise NoTarget("the Qwen3.6 target uses eager kernels and runtime CUDA graphs")
    prefill_graph = llm_args.prefill_cuda_graph_backend.value
    if prefill_graph != "disabled" and (
        prefill_graph != "breakable"
        or llm_args.max_batch_size != 1
        or llm_args.enable_chunked_prefill
    ):
        raise NoTarget(
            "Qwen3.6 prefill graphs require breakable CUDA graphs and unchunked batch one"
        )

    from .qwen3_6_35b_a3b_fp8__sm_100__default.modeling import ModelingV2Qwen36Fp8Sm100

    return ModelingV2Qwen36Fp8Sm100

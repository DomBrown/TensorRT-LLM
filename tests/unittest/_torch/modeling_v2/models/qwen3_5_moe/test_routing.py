# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Check the Qwen3.6 FP8 target's hardware and deployment selection on CPU."""

from types import SimpleNamespace

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2._router_index import NoTarget
from tensorrt_llm._torch._experimental.modeling_v2.models.qwen3_5_moe.routing import route
from tensorrt_llm.llmapi.llm_args import PrefillCudaGraphBackend

pytestmark = pytest.mark.cpu_only


@pytest.fixture
def checkpoint():
    text = SimpleNamespace(
        hidden_size=2048,
        vocab_size=248320,
        num_hidden_layers=40,
        num_attention_heads=16,
        num_key_value_heads=2,
        head_dim=256,
        num_experts=256,
        num_experts_per_tok=8,
        moe_intermediate_size=512,
        shared_expert_intermediate_size=512,
        linear_conv_kernel_dim=4,
        linear_key_head_dim=128,
        linear_value_head_dim=128,
        linear_num_key_heads=16,
        linear_num_value_heads=32,
        rms_norm_eps=1e-6,
        hidden_act="silu",
        attention_bias=False,
        attn_output_gate=True,
        tie_word_embeddings=False,
        layer_types=("linear_attention", "linear_attention", "linear_attention", "full_attention")
        * 10,
        rope_parameters={"rope_theta": 10_000_000, "partial_rotary_factor": 0.25},
    )
    return SimpleNamespace(
        pretrained_config=SimpleNamespace(
            text_config=text,
            quantization_config={
                "quant_method": "fp8",
                "fmt": "e4m3",
                "weight_block_size": [128, 128],
            },
        ),
        mapping=SimpleNamespace(world_size=1),
        torch_dtype=torch.bfloat16,
        quant_config=SimpleNamespace(kv_cache_quant_algo=None),
        spec_config=None,
        lora_config=None,
    )


@pytest.fixture
def deployment():
    return SimpleNamespace(
        disable_mm_encoder=True,
        mm_encoder_only=False,
        kv_cache_config=SimpleNamespace(
            dtype="auto",
            use_kv_cache_manager_v2=True,
            enable_block_reuse=False,
            mamba_ssm_cache_dtype="float32",
        ),
        cache_transceiver_config=None,
        max_beam_width=1,
        attn_backend="TRTLLM",
        torch_compile_config=None,
        prefill_cuda_graph_backend=PrefillCudaGraphBackend.DISABLED,
        max_batch_size=128,
        enable_chunked_prefill=False,
    )


@pytest.mark.parametrize("cache_dtype", ["auto", "bfloat16"])
def test_route_qwen36_checkpoint(checkpoint, deployment, cache_dtype):
    deployment.kv_cache_config.dtype = cache_dtype
    target = route(checkpoint, deployment, (10, 0))
    assert target.__name__ == "ModelingV2Qwen36Fp8Sm100"


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("disable_mm_encoder", False, "text-only"),
        ("max_beam_width", 2, "beam width one"),
        ("attn_backend", "FLASHINFER", "TRTLLM scheduler"),
        ("torch_compile_config", {}, "runtime CUDA graphs"),
        ("prefill_cuda_graph_backend", PrefillCudaGraphBackend.BREAKABLE, "unchunked batch one"),
    ],
)
def test_decline_unsupported_deployment(checkpoint, deployment, field, value, message):
    setattr(deployment, field, value)
    with pytest.raises(NoTarget, match=message):
        route(checkpoint, deployment, (10, 0))


def test_decline_other_arch_or_checkpoint(checkpoint, deployment):
    with pytest.raises(NoTarget, match="one SM100"):
        route(checkpoint, deployment, (10, 3))
    checkpoint.pretrained_config.quantization_config["weight_block_size"] = (64, 64)
    with pytest.raises(NoTarget, match="128x128"):
        route(checkpoint, deployment, (10, 0))

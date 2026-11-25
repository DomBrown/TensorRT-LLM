# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from copy import deepcopy
from types import SimpleNamespace

import pytest

from tensorrt_llm._torch._experimental.modeling_v2 import NoTarget
from tensorrt_llm._torch._experimental.modeling_v2.models.qwen3_5_moe.routing import (
    _TEXT_SHAPE,
    route,
)
from tensorrt_llm.quantization.mode import QuantAlgo


@pytest.fixture
def deployment():
    text = deepcopy(_TEXT_SHAPE)
    text["layer_types"] = (["linear_attention"] * 3 + ["full_attention"]) * 10
    text["rope_parameters"] = {"partial_rotary_factor": 0.25, "rope_theta": 10_000_000}
    quantization = {"fmt": "e4m3", "weight_block_size": [128, 128], "activation_scheme": "dynamic"}
    config = SimpleNamespace(
        pretrained_config=SimpleNamespace(text_config=text, quantization_config=quantization),
        quant_config=SimpleNamespace(
            quant_algo=QuantAlgo.FP8_BLOCK_SCALES, kv_cache_quant_algo=None
        ),
        mapping=SimpleNamespace(world_size=1, enable_attention_dp=False),
        spec_config=None,
        lora_config=None,
    )
    args = SimpleNamespace(
        disable_mm_encoder=True,
        mm_encoder_only=False,
        max_beam_width=1,
        kv_cache_config=SimpleNamespace(
            enable_block_reuse=False,
            use_kv_cache_manager_v2=True,
            dtype="auto",
            max_attention_window=None,
            mamba_ssm_cache_dtype="float32",
        ),
        enable_return_routed_experts=False,
        attn_backend="TRTLLM",
    )
    return config, args


@pytest.mark.cpu_only
def test_routes_the_official_fp8_text_geometry(deployment) -> None:
    target = route(*deployment, (10, 0))
    assert target.__name__ == "ModelingV2Qwen3635BA3BFp8Sm100Default"
    assert ".qwen3_6_35b_a3b_fp8__sm_100__default." in target.__module__


@pytest.mark.cpu_only
@pytest.mark.parametrize(
    "part,key,value,capability",
    [
        ("text", "num_experts", 128, (10, 0)),
        ("quant", "weight_block_size", [64, 128], (10, 0)),
        ("args", "disable_mm_encoder", False, (10, 0)),
        ("args", "max_beam_width", 2, (10, 0)),
        ("cache", "enable_block_reuse", True, (10, 0)),
        ("cache", "dtype", "fp8", (10, 0)),
        ("mapping", "world_size", 2, (10, 0)),
        ("mapping", "world_size", 1, (10, 3)),
    ],
)
def test_declines_unsupported_deployment(deployment, part, key, value, capability) -> None:
    config, args = deployment
    object_to_update = {
        "text": config.pretrained_config.text_config,
        "quant": config.pretrained_config.quantization_config,
        "args": args,
        "cache": args.kv_cache_config,
        "mapping": config.mapping,
    }[part]
    if isinstance(object_to_update, dict):
        object_to_update[key] = value
    else:
        setattr(object_to_update, key, value)
    with pytest.raises(NoTarget):
        route(config, args, capability)

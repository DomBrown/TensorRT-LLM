# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Qwen3.6-35B-A3B text decoder for SM100 with block-FP8 weights."""

from __future__ import annotations

from copy import copy
from dataclasses import replace
from typing import TYPE_CHECKING, NamedTuple
from weakref import ref

import torch
from torch import nn
from torch.nn import functional as F

from tensorrt_llm._torch.models.modeling_utils import DecoderModel, DecoderModelForCausalLM

from .... import disable_v2_by_default
from .kernels.prefill_gemm import PrefillGemm
from .ops.attention import attention_step, gqa, paged_decode_attention, qkv_projection
from .ops.epilogue import gated_output_quantize, gdn_gates, residual_norm_quantize
from .ops.projection import decode_gates, fp8_projection, routed_experts, uses_cutlass
from .ops.recurrent import SHORT_CONTEXT, causal_convolution, gated_delta_rule
from .weights import Fp8Weight, LayerWeights, load_layer

if TYPE_CHECKING:
    from collections.abc import Mapping

    from tensorrt_llm._torch.attention.backends.trtllm import TrtllmAttentionMetadata
    from tensorrt_llm._torch.model_config import ModelConfig


class _LinearStep(NamedTuple):
    context_rows: int
    slots: torch.Tensor
    offsets: torch.Tensor | None
    row_map: torch.Tensor | None
    initial: torch.Tensor | None
    read_slots: torch.Tensor | None
    lengths: tuple[int, ...]


def _linear_step(md: TrtllmAttentionMetadata) -> _LinearStep:
    metadata = md.mamba_metadata
    num_ctx = md.num_contexts
    num_ctx_tokens = md.num_ctx_tokens
    slots = metadata.state_indices[: md.num_seqs]
    if num_ctx == 0:
        return _LinearStep(0, slots, None, None, None, None, ())
    lengths = tuple(md.seq_lens[:num_ctx].tolist())
    initial = metadata.has_initial_states[:num_ctx]
    # FlashInfer reads -1 as a zero state. Destination slots always belong to the request.
    read_slots = torch.where(initial, slots[:num_ctx], -1) if metadata.use_initial_states else None
    return _LinearStep(
        num_ctx_tokens,
        slots,
        metadata.query_start_loc[: num_ctx + 1],
        metadata.seq_idx.reshape(-1),
        initial,
        read_slots,
        lengths,
    )


def _small_bf16(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    # Keep the reduction shape independent of the number of rows in the small decode path.
    rows = x.shape[0]
    if rows < 4:
        return F.linear(F.pad(x, (0, 0, 0, 4 - rows)), weight)[:rows]
    return F.linear(x, weight)


class _Decoder(DecoderModel):
    supports_prefill_cuda_graph = False

    def __init__(self, model_config: ModelConfig) -> None:
        super().__init__(model_config)
        text = model_config.pretrained_config
        self.layer_types = tuple(text.layer_types)
        self.hidden_size = text.hidden_size
        self.embed_tokens = nn.Embedding(
            text.vocab_size, text.hidden_size, device="cuda", dtype=self.dtype
        )
        self.layers: list[LayerWeights] = []
        self.final_norm: torch.Tensor
        self.cosine: torch.Tensor
        self.sine: torch.Tensor
        self.gemm: PrefillGemm
        self.route_counts: torch.Tensor
        self._bound_cache = None
        self._attention_pools: dict[int, tuple[torch.Tensor, int]] = {}

    def initialize_runtime_buffers(self, device: torch.device) -> None:
        """Allocate scratch after the loader leaves the meta-device context."""
        text = self.model_config.pretrained_config
        theta = text.rope_parameters["rope_theta"]
        inverse = theta ** (-torch.arange(0, 64, 2, device=device, dtype=torch.float32) / 64)
        phase = torch.arange(text.max_position_embeddings, device=device)[:, None] * inverse
        self.cosine = phase.cos()
        self.sine = phase.sin()
        self.gemm = PrefillGemm(device)
        self.route_counts = torch.zeros(257, dtype=torch.int32, device=device)

    def _bind_attention_pools(self, cache: object) -> None:
        """Take zero-copy HND views of the current executor's paged KV pools."""
        if self._bound_cache is not None and self._bound_cache() is cache:
            return
        self._attention_pools = {
            i: (
                cache.get_buffers(i, kv_layout="HND"),
                int(cache.kv_cache_pool_mapping[cache.layer_offsets[i], 0]),
            )
            for i, kind in enumerate(self.layer_types)
            if kind == "full_attention"
        }
        self._bound_cache = ref(cache)

    @staticmethod
    def _prefill_gdn(
        projection: torch.Tensor,
        ab: torch.Tensor,
        conv_weight: torch.Tensor,
        a_log: torch.Tensor,
        dt_bias: torch.Tensor,
        conv: torch.Tensor,
        state: torch.Tensor,
        step: _LinearStep,
    ) -> torch.Tensor:
        rows = step.context_rows
        ctx_slots = step.slots[: len(step.lengths)]
        max_length = max(step.lengths)
        chunked = max_length > SHORT_CONTEXT
        mixed = causal_convolution(
            projection[:rows],
            conv_weight,
            conv,
            ctx_slots,
            step.offsets,
            step.row_map,
            step.initial,
            split=chunked,
        )
        alpha, beta = gdn_gates(ab, a_log, dt_bias)
        return gated_delta_rule(
            mixed,
            alpha,
            beta,
            state,
            ctx_slots,
            step.offsets,
            step.initial,
            step.read_slots,
            step.lengths,
        )

    @staticmethod
    def _decode_gdn(
        x: torch.Tensor,
        projection: torch.Tensor,
        weights: tuple[torch.Tensor, ...],
        conv: torch.Tensor,
        state: torch.Tensor,
        slots: torch.Tensor,
    ) -> torch.Tensor:
        ab_weight, conv_weight, a_log, dt_bias, _ = weights
        alpha, beta = decode_gates(x, ab_weight, a_log, dt_bias)
        mixed = causal_convolution(projection, conv_weight, conv, slots, None, None, None)
        return gated_delta_rule(mixed, alpha, beta, state, slots, None, None, None, ())

    def _mix(
        self,
        layer: LayerWeights,
        index: int,
        x: torch.Tensor,
        quantized: Fp8Weight,
        md: TrtllmAttentionMetadata,
        attention: dict,
        recurrent: _LinearStep,
        position: torch.Tensor,
        decode: bool,
        mn_major: bool,
    ) -> torch.Tensor:
        projection = fp8_projection(quantized, layer.input_projection, decode, self.gemm)
        cache = md.kv_cache_manager
        if layer.linear_attention:
            conv = cache.get_conv_states(index)
            state = cache.get_ssm_states(index)
            ab_weight, conv_weight, a_log, dt_bias, norm = layer.attention_params
            ctx_rows = recurrent.context_rows
            ctx_count = len(recurrent.lengths)
            if decode:
                core = self._decode_gdn(
                    x, projection, layer.attention_params, conv, state, recurrent.slots
                )
            else:
                core = self._prefill_gdn(
                    projection,
                    F.linear(x[:ctx_rows], ab_weight),
                    conv_weight,
                    a_log,
                    dt_bias,
                    conv,
                    state,
                    recurrent,
                )
                if ctx_rows < x.shape[0]:
                    decode_core = self._decode_gdn(
                        x[ctx_rows:],
                        projection[ctx_rows:],
                        layer.attention_params,
                        conv,
                        state,
                        recurrent.slots[ctx_count:],
                    )
                    core = torch.cat((core, decode_core))
            gated = gated_output_quantize(core, projection, norm, mn_major=mn_major)
        else:
            qkv = qkv_projection(
                projection, *layer.attention_params, self.cosine, self.sine, position
            )
            if decode and x.shape[0] == 1:
                pool, pool_id = self._attention_pools[index]
                pages = md.kv_cache_block_offsets[pool_id, : md.num_seqs, 0]
                core = paged_decode_attention(qkv, pool, position, pages, md.max_seq_len)
            else:
                core = gqa(qkv, cache.layer_offsets[index], attention)
            gated = gated_output_quantize(core, projection, None, mn_major=mn_major)
        return fp8_projection(gated, layer.output_projection, decode, self.gemm)

    def forward(
        self,
        attn_metadata: TrtllmAttentionMetadata,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> torch.Tensor:
        rows = input_ids.numel()
        position = position_ids.view(-1, rows)[0]
        recurrent = _linear_step(attn_metadata)
        attention = attention_step(attn_metadata)
        decode = attn_metadata.num_contexts == 0
        if decode and rows == 1:
            self._bind_attention_pools(attn_metadata.kv_cache_manager)
        mn_major = uses_cutlass(rows, decode) and (not decode or rows == 128)
        fixed_router = max(recurrent.lengths, default=0) > SHORT_CONTEXT
        residual = self.embed_tokens(input_ids)
        _, hidden, attention_input = residual_norm_quantize(
            residual, self.layers[0].input_norm, mn_major=mn_major
        )
        for i, layer in enumerate(self.layers):
            delta = self._mix(
                layer,
                i,
                hidden,
                attention_input,
                attn_metadata,
                attention,
                recurrent,
                position,
                decode,
                mn_major,
            )
            residual, hidden, expert_input = residual_norm_quantize(
                residual, layer.post_attention_norm, delta=delta
            )
            out, probabilities, router, permutation = routed_experts(
                hidden,
                expert_input,
                layer.router,
                layer.expert_gate_up,
                layer.expert_down,
                decode,
                fixed_router,
                self.route_counts,
                self.gemm,
            )
            last = i + 1 == len(self.layers)
            weight = self.final_norm if last else self.layers[i + 1].input_norm
            residual, hidden, attention_input = residual_norm_quantize(
                residual,
                weight,
                delta=out,
                probability=probabilities,
                router=router,
                quantize=not last,
                permutation=permutation,
                mn_major=mn_major,
            )
        return hidden


@disable_v2_by_default
class ModelingV2Qwen3635BA3BFp8Sm100Default(DecoderModelForCausalLM):
    """Text-only FP8 target, with recurrent and paged cache state owned by TRT-LLM."""

    moe_impls_honored = frozenset()

    def __init__(self, model_config: ModelConfig) -> None:
        outer = model_config.pretrained_config
        text = copy(outer.text_config)
        # The cache manager dispatches on the text architecture when the model owns no encoder.
        text.architectures = ["Qwen3_5MoeForCausalLM"]
        text._name_or_path = outer._name_or_path
        text.intermediate_size = (
            text.num_experts_per_tok * text.moe_intermediate_size
            + text.shared_expert_intermediate_size
        )
        runtime_config = replace(model_config, pretrained_config=text)
        super().__init__(
            _Decoder(runtime_config),
            config=runtime_config,
            hidden_size=text.hidden_size,
            vocab_size=text.vocab_size,
        )

    def load_weights(self, weights: Mapping[str, torch.Tensor]) -> None:
        """Populate the target-owned TP1 buffers from raw safetensors."""
        stem = "model.language_model."
        device = self.lm_head.weight.device
        self.model.embed_tokens.weight.data.copy_(weights[stem + "embed_tokens.weight"])
        self.model.final_norm = weights[stem + "norm.weight"].to(device)
        self.lm_head.weight.data.copy_(weights["lm_head.weight"])
        self.model.initialize_runtime_buffers(device)
        self.model.layers = [
            load_layer(weights, f"{stem}layers.{i}.", layer_type == "linear_attention", device)
            for i, layer_type in enumerate(self.model.layer_types)
        ]

    def forward(
        self,
        attn_metadata: TrtllmAttentionMetadata,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        return_context_logits: bool = False,
        **kwargs: object,
    ) -> torch.Tensor:
        """Evaluate packed context and generation rows in scheduler order."""
        hidden = self.model(attn_metadata, input_ids, position_ids)
        if not return_context_logits:
            if attn_metadata.num_contexts:
                ends = torch.cumsum(attn_metadata.seq_lens_cuda, dim=0, dtype=torch.long) - 1
                hidden = hidden[ends]
            else:
                return _small_bf16(hidden, self.lm_head.weight).float()
        return F.linear(hidden, self.lm_head.weight).float()

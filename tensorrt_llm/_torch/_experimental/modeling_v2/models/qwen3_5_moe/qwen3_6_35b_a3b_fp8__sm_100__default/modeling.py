# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Modeling V2 Qwen3.6-35B-A3B FP8 decoder for SM100 and TP1."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from itertools import accumulate
from typing import TYPE_CHECKING, NamedTuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from tensorrt_llm._torch.models.modeling_utils import DecoderModel, DecoderModelForCausalLM

from .... import disable_v2_by_default
from .kernels import fp8
from .ops.attention import KvPages, full_attention
from .ops.linear_attention import (
    GdnWeights,
    StatePool,
    context_gdn,
    gated_delta_output,
    gated_delta_step,
)
from .ops.projections import (
    RoutedWeights,
    dense_projection,
    route_experts,
    routed_experts,
    shared_expert,
)
from .weights import load_text_weights

if TYPE_CHECKING:
    from tensorrt_llm._torch.attention.backends.interface import AttentionMetadata
    from tensorrt_llm._torch.model_config import ModelConfig


class VocabularyHead(nn.Linear):
    """Replicated BF16 vocabulary projection for TP1."""

    num_embeddings = 248320
    vocab_size_padded = 248320
    gather_output = True
    tp_size = 1


class Step(NamedTuple):
    positions: torch.Tensor
    requests: torch.Tensor
    lengths: tuple[int, ...]
    cached: tuple[int, ...]
    cu_cpu: tuple[int, ...]
    context_rows: int
    generation_requests: int
    generation_steps: int


def make_step(metadata: AttentionMetadata, positions: torch.Tensor, rows: int) -> Step:
    """Bind packed context segments and uniform generation rows to requests."""
    contexts = metadata.num_contexts
    context_rows = metadata.num_ctx_tokens
    batch = metadata.seq_lens.shape[0]
    generation = batch - contexts
    steps = (rows - context_rows) // generation if generation else 0
    mamba = metadata.mamba_metadata
    ids = torch.arange(contexts, batch, dtype=torch.int32, device=positions.device)
    gen_requests = (
        ids.repeat_interleave(steps, output_size=rows - context_rows) if steps > 1 else ids
    )
    if not contexts:
        return Step(positions, gen_requests, (), (), (0,), 0, generation, steps)
    lengths = tuple(metadata.seq_lens[:contexts].tolist())
    cached = tuple(int(n) for n in metadata.kv_cache_params.num_cached_tokens_per_seq[:contexts])
    request_ids = mamba.seq_idx.view(-1)
    if generation:
        request_ids = torch.cat((request_ids, gen_requests))
    return Step(
        positions,
        request_ids,
        lengths,
        cached,
        tuple(accumulate(lengths, initial=0)),
        context_rows,
        generation,
        steps,
    )


class Qwen36Decoder(DecoderModel):
    """Decode packed text tokens using model-local SM100 operations."""

    supports_prefill_cuda_graph = False

    def __init__(self, model_config: ModelConfig) -> None:
        super().__init__(model_config)
        self.embed_tokens = nn.Embedding(248320, 2048, dtype=torch.bfloat16, device="meta")
        self.layers = nn.ModuleList()
        self.register_buffer("final_norm", None)
        self.register_buffer("rope", None)
        self.side_stream = torch.cuda.Stream()
        self._cache_manager = None
        self._attention_views = {}
        self._state_views = {}

    def _bind_pools(self, metadata: AttentionMetadata) -> None:
        manager = metadata.kv_cache_manager
        if manager is self._cache_manager:
            return
        attention_views = {}
        state_views = {}
        for index, layer in enumerate(self.layers):
            if layer.linear:
                state_views[index] = manager.mamba_layer_cache(index)
            else:
                pool_id = int(manager.kv_cache_pool_mapping[index, 0])
                attention_views[index] = (pool_id, manager.get_buffers(index, kv_layout="NHD"))
        self._attention_views = attention_views
        self._state_views = state_views
        self._cache_manager = manager

    def _linear_attention(
        self,
        projected: torch.Tensor,
        ba: torch.Tensor,
        layer: nn.Module,
        metadata: AttentionMetadata,
        step: Step,
        index: int,
        mx: bool,
    ) -> tuple[torch.Tensor, ...]:
        cache = self._state_views[index]
        mamba = metadata.mamba_metadata
        contexts = len(step.lengths)
        context_rows = step.context_rows
        weights = GdnWeights(layer.conv, layer.decay, layer.dt)
        if contexts:
            pool = StatePool(
                cache.conv,
                cache.temporal,
                mamba.state_indices[:contexts],
                mamba.has_initial_states,
            )
            prefill = context_gdn(
                projected[:context_rows],
                ba[:context_rows],
                weights,
                pool,
                step.requests[:context_rows],
                mamba.query_start_loc_long[: contexts + 1],
                step.cu_cpu,
                mamba.use_initial_states,
            )
        if step.generation_requests:
            pool = StatePool(
                cache.conv,
                cache.temporal,
                mamba.state_indices[contexts : contexts + step.generation_requests],
                mamba.has_initial_states,
            )
            decode = gated_delta_step(
                projected[context_rows:],
                ba[context_rows:],
                weights,
                pool,
                step.generation_steps,
                False,
                mx and not contexts,
            )
        if contexts and step.generation_requests:
            values = torch.cat((prefill, decode))
        else:
            values = prefill if contexts else decode
        return gated_delta_output(values, projected, layer.gdn_norm, mx)

    def forward(
        self,
        attn_metadata: AttentionMetadata,
        input_ids: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        """Return the final normalized hidden rows in scheduler order."""
        if inputs_embeds is not None or input_ids is None or position_ids is None:
            raise ValueError("the Qwen3.6 FP8 target expects text token IDs and positions")
        self._bind_pools(attn_metadata)
        ids = input_ids.flatten()
        rows = ids.shape[0]
        positions = position_ids.reshape(-1, rows)[0]
        step = make_step(attn_metadata, positions, rows)
        small = step.context_rows == 0 and rows <= 8
        mx = rows <= 8

        residual = self.embed_tokens(ids)
        hidden = fp8.rms(residual, self.layers[0].input_norm)
        quantized = fp8.quant(hidden, True)
        current = torch.cuda.current_stream()
        for index, layer in enumerate(self.layers):
            if layer.linear:
                self.side_stream.wait_stream(current)
                with torch.cuda.stream(self.side_stream):
                    ba = fp8.bf16_linear(hidden, layer.ba) if small else F.linear(hidden, layer.ba)
            projected = dense_projection(quantized, layer.qkv)
            if layer.linear:
                current.wait_stream(self.side_stream)
                ba.record_stream(current)
                attended = self._linear_attention(
                    projected, ba, layer, attn_metadata, step, index, mx
                )
            else:
                pool, cache = self._attention_views[index]
                pages = KvPages(cache, attn_metadata.kv_cache_block_offsets[pool])
                attended = full_attention(
                    projected,
                    layer.q_norm,
                    layer.k_norm,
                    self.rope,
                    step.positions,
                    step.requests,
                    pages,
                    step.lengths,
                    step.cached,
                    step.context_rows,
                    mx,
                )
            attention_output = dense_projection(attended, layer.proj)
            residual, hidden, routed, shared = fp8.add_rms_quant(
                residual,
                attention_output,
                layer.post_norm,
                False,
                False,
                True,
                mx=mx,
                exact=not small,
                pdl=small,
            )

            self.side_stream.wait_stream(current)
            with torch.cuda.stream(self.side_stream):
                shared_value = shared_expert(shared, layer)
            routes, probs, shared_gate, grouping = route_experts(hidden, layer.router, small, small)
            experts, inverse = routed_experts(
                routed,
                routes,
                RoutedWeights(layer.expert_up, layer.expert_down),
                small,
                grouping,
                small,
            )
            current.wait_stream(self.side_stream)
            shared_value.record_stream(current)
            final = index + 1 == len(self.layers)
            next_norm = self.final_norm if final else self.layers[index + 1].input_norm
            residual, hidden, quantized = fp8.combine_next_norm(
                experts,
                probs,
                shared_value,
                shared_gate,
                residual,
                next_norm,
                False,
                8,
                False,
                not final,
                True,
                mx,
                inverse,
                None,
                None,
                pdl=small,
            )
        return hidden


@disable_v2_by_default
class ModelingV2Qwen36Fp8Sm100(DecoderModelForCausalLM):
    """Text-only Qwen3.6 causal LM using the TensorRT-LLM request executor."""

    use_fx_for_pcg_fallback = False

    def __init__(self, model_config: ModelConfig) -> None:
        self.text_config = model_config.pretrained_config.text_config
        # The runtime resolves GDN state and full-attention geometry from the text config.
        model_config = replace(model_config, pretrained_config=self.text_config)
        super().__init__(
            Qwen36Decoder(model_config),
            config=model_config,
            hidden_size=2048,
            vocab_size=248320,
        )
        self.lm_head = VocabularyHead(2048, 248320, bias=False, device="meta", dtype=torch.bfloat16)
        self.epilogue = [self.lm_head]

    def load_weights(self, weights: Mapping[str, torch.Tensor]) -> None:
        """Load text checkpoint tensors directly into the local model."""
        load_text_weights(
            self.model,
            self.lm_head,
            weights,
            self.text_config.layer_types,
            self.text_config.max_position_embeddings,
        )

    def infer_max_seq_len(self) -> int:
        """Use the text checkpoint position limit."""
        return self.text_config.max_position_embeddings

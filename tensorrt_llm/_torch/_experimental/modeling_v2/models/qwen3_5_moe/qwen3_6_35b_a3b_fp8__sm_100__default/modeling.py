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

from tensorrt_llm._torch.models.modeling_speculative import SpecDecOneEngineForCausalLM
from tensorrt_llm._torch.models.modeling_utils import DecoderModel
from tensorrt_llm._torch.pyexecutor.breakable_cuda_graph.context import is_in_breakable_cuda_graph

from .... import disable_v2_by_default
from .cache import Qwen36HybridCacheManager, VerifierLayerCache, clear_pending
from .kernels import fp8
from .mtp import DraftShortlist, Qwen36MtpLayer
from .ops.attention import KvPages, full_attention
from .ops.linear_attention import (
    GdnWeights,
    ReplayState,
    StatePool,
    context_gdn,
    gated_delta_output,
    gated_delta_step,
)
from .ops.projections import (
    RoutedWeights,
    dense_projection,
    packed_delta_gates,
    route_packed,
    routed_experts,
    shared_expert,
)
from .weights import load_text_weights

if TYPE_CHECKING:
    from tensorrt_llm._torch.attention.backends.interface import AttentionMetadata
    from tensorrt_llm._torch.model_config import ModelConfig
    from tensorrt_llm._torch.speculative.interface import SpecMetadata


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

    prefill_graph_requires_exact_rows = True
    mtp_layer_cls = Qwen36MtpLayer

    def __init__(self, model_config: ModelConfig) -> None:
        super().__init__(model_config)
        self.embed_tokens = nn.Embedding(248320, 2048, dtype=torch.bfloat16, device="meta")
        self.layers = nn.ModuleList()
        self.num_hidden_layers = model_config.pretrained_config.num_hidden_layers
        self.register_buffer("final_norm", None)
        self.register_buffer("rope", None)
        self.side_stream = torch.cuda.Stream()
        self.aux_stream_dict = {"draft": self.side_stream}
        self._cache_manager = None
        self._attention_views = {}
        self._state_views = {}
        self._prefill_pages = {}
        self._prefill_slot = None
        self._prefill_fresh = None

    def _bind_pools(self, metadata: AttentionMetadata) -> None:
        manager = metadata.kv_cache_manager
        if manager is self._cache_manager:
            return
        attention_views = {}
        state_views = {}
        for index in range(self.num_hidden_layers):
            layer = self.layers[index]
            if layer.linear:
                state_views[index] = manager.mamba_layer_cache(index)
            else:
                pool_id = int(manager.kv_cache_pool_mapping[index, 0])
                attention_views[index] = (pool_id, manager.get_buffers(index, kv_layout="NHD"))
        self._attention_views = attention_views
        self._state_views = state_views
        self._cache_manager = manager

    def stage_prefill(self, metadata: AttentionMetadata) -> None:
        """Stage mutable page and state addresses before an exact-size graph runs."""
        self._bind_pools(metadata)
        state_indices = metadata.mamba_metadata.state_indices[:1]
        if self._prefill_slot is None:
            self._prefill_slot = torch.empty_like(state_indices)
            self._prefill_fresh = torch.zeros((1,), device=state_indices.device, dtype=torch.bool)
        self._prefill_slot.copy_(state_indices)
        for pool, _ in self._attention_views.values():
            if pool not in self._prefill_pages:
                table = metadata.kv_cache_block_offsets[pool][:1]
                self._prefill_pages[pool] = torch.empty_like(table)
        for pool, table in self._prefill_pages.items():
            table.copy_(metadata.kv_cache_block_offsets[pool][:1])

    def _linear_attention(
        self,
        projected: torch.Tensor,
        ba: torch.Tensor,
        layer: nn.Module,
        metadata: AttentionMetadata,
        step: Step,
        index: int,
        mx: bool,
        graph_cu: torch.Tensor | None,
    ) -> tuple[torch.Tensor, ...]:
        cache = self._state_views[index]
        mamba = metadata.mamba_metadata
        contexts = len(step.lengths)
        context_rows = step.context_rows
        weights = GdnWeights(layer.conv, layer.decay, layer.dt)
        if graph_cu is not None:
            pool = StatePool(cache.conv, cache.temporal, self._prefill_slot, self._prefill_fresh)
            prefill = context_gdn(
                projected, ba, weights, pool, step.requests, graph_cu, step.cu_cpu, False
            )
        elif contexts:
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
            replay = None
            if isinstance(cache, VerifierLayerCache):
                replay = ReplayState(
                    cache.factors,
                    cache.pending,
                    mamba.replay_state_indices[contexts : contexts + step.generation_requests],
                    cache.raw_conv,
                )
            decode = gated_delta_step(
                projected[context_rows:],
                ba[context_rows:],
                weights,
                pool,
                step.generation_steps,
                False,
                mx and not contexts,
                replay,
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
        graph_prefill = is_in_breakable_cuda_graph() and attn_metadata.num_contexts == 1
        if graph_prefill:
            requests = torch.zeros(rows, device=positions.device, dtype=torch.int32)
            step = Step(positions, requests, (rows,), (0,), (0, rows), rows, 0, 0)
            graph_cu = torch.arange(2, dtype=torch.int64, device=positions.device) * rows
        else:
            step = make_step(attn_metadata, positions, rows)
            graph_cu = None
        small = step.context_rows == 0 and rows <= 8
        mx = rows <= 8

        residual = self.embed_tokens(ids)
        hidden = fp8.rms(residual, self.layers[0].input_norm)
        quantized = fp8.quant(hidden, True)
        current = torch.cuda.current_stream()
        for index in range(self.num_hidden_layers):
            layer = self.layers[index]
            if layer.linear:
                self.side_stream.wait_stream(current)
                with torch.cuda.stream(self.side_stream):
                    ba = packed_delta_gates(hidden, layer.ba, step.cu_cpu)
            projected = dense_projection(quantized, layer.qkv)
            if layer.linear:
                current.wait_stream(self.side_stream)
                ba.record_stream(current)
                attended = self._linear_attention(
                    projected, ba, layer, attn_metadata, step, index, mx, graph_cu
                )
            else:
                pool, cache = self._attention_views[index]
                table = (
                    self._prefill_pages[pool]
                    if graph_prefill
                    else attn_metadata.kv_cache_block_offsets[pool]
                )
                pages = KvPages(cache, table)
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
            routes, probs, shared_gate, grouping = route_packed(
                hidden, layer.router, step.cu_cpu, small
            )
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
            final = index + 1 == self.num_hidden_layers
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
        first_cache = self._state_views[0]
        if step.generation_steps == 1 and isinstance(first_cache, VerifierLayerCache):
            start = len(step.lengths)
            slots = attn_metadata.mamba_metadata.replay_state_indices[
                start : start + step.generation_requests
            ]
            clear_pending(first_cache.pending, slots)
        return hidden


@disable_v2_by_default
class ModelingV2Qwen36Fp8Sm100(SpecDecOneEngineForCausalLM):
    """Text-only Qwen3.6 causal LM using the TensorRT-LLM request executor."""

    use_fx_for_pcg_fallback = False
    hybrid_kv_cache_manager_cls = Qwen36HybridCacheManager

    @classmethod
    def supports_spec_decode_fast_paths(cls, model_config: ModelConfig) -> bool:
        return True

    def __init__(self, model_config: ModelConfig) -> None:
        self.text_config = model_config.pretrained_config.text_config
        # The runtime resolves GDN state and full-attention geometry from the text config.
        model_config = replace(model_config, pretrained_config=self.text_config)
        if model_config.spec_config is not None:
            self.text_config.num_nextn_predict_layers = 1
        super().__init__(
            Qwen36Decoder(model_config),
            model_config,
            hidden_size=2048,
            vocab_size=248320,
        )
        self.lm_head = VocabularyHead(2048, 248320, bias=False, device="meta", dtype=torch.bfloat16)
        self.epilogue[0] = self.lm_head
        if self.draft_model is not None:
            self.draft_model.lm_head = self.lm_head

    def load_weights(self, weights: Mapping[str, torch.Tensor]) -> None:
        """Load text checkpoint tensors directly into the local model."""
        load_text_weights(
            self.model,
            self.lm_head,
            weights,
            self.text_config.layer_types,
            self.text_config.max_position_embeddings,
        )
        if self.draft_model is not None:
            draft = self.draft_model.mtp_layers[0]
            draft.load_checkpoint(weights, self.model.rope)
            self.model.layers.append(draft)

    def post_load_weights(self) -> None:
        """Share the vocabulary weights with the greedy draft shortlist."""
        super().post_load_weights()
        if self.spec_worker is not None:
            self.spec_worker.draft_head = DraftShortlist(self.lm_head.weight)

    def forward(
        self,
        attn_metadata: AttentionMetadata,
        input_ids: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        return_context_logits: bool = False,
        spec_metadata: SpecMetadata | None = None,
        resource_manager=None,
        **kwargs,
    ) -> torch.Tensor:
        if is_in_breakable_cuda_graph() and attn_metadata.num_contexts == 1:
            self.model.stage_prefill(attn_metadata)
        return super().forward(
            attn_metadata,
            input_ids,
            position_ids,
            inputs_embeds,
            return_context_logits,
            spec_metadata,
            resource_manager,
            **kwargs,
        )

    def infer_max_seq_len(self) -> int:
        """Use the text checkpoint position limit."""
        return self.text_config.max_position_embeddings

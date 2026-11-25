# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Qwen3.6-35B-A3B text decoder for SM100 with block-FP8 weights."""

from __future__ import annotations

from copy import copy
from dataclasses import replace
from itertools import accumulate
from typing import TYPE_CHECKING, NamedTuple
from weakref import WeakKeyDictionary, ref

import torch
from torch import nn
from torch.nn import functional as F

from tensorrt_llm._torch.models.modeling_utils import DecoderModel, DecoderModelForCausalLM
from tensorrt_llm._torch.pyexecutor.breakable_cuda_graph import is_in_breakable_cuda_graph
from tensorrt_llm._torch.pyexecutor.runtime_hooks import RuntimeHooks
from tensorrt_llm._torch.speculative.greedy_argmax import greedy_argmax
from tensorrt_llm._torch.speculative.interface import should_use_separate_draft_kv_cache
from tensorrt_llm._torch.speculative.utils import get_spec_worker

from .... import disable_v2_by_default
from .cache_manager import Qwen36HybridCacheManager
from .kernels.prefill_gemm import PrefillGemm
from .ops.attention import (
    attention_step,
    context_attention_step,
    gqa,
    paged_decode_attention,
    qkv_projection,
    uses_native_decode,
)
from .ops.epilogue import gated_output_quantize, gdn_gates, mtp_input, residual_norm_quantize
from .ops.projection import decode_gates, fp8_projection, routed_experts, uses_cutlass
from .ops.recurrent import (
    SHORT_CONTEXT,
    causal_convolution,
    gated_delta_rule,
    replay_gdn,
    verify_conv,
    verify_gdn,
)
from .weights import Fp8Weight, LayerWeights, load_layer

if TYPE_CHECKING:
    from collections.abc import Mapping

    from tensorrt_llm._torch.attention.backends.trtllm import TrtllmAttentionMetadata
    from tensorrt_llm._torch.model_config import ModelConfig
    from tensorrt_llm._torch.speculative.interface import SpecMetadata


_SPEC_DECODE_HOOKS = RuntimeHooks(
    loop_on_execution_stream=True,
    pack_step_inputs=True,
    pack_parallel_scatter=True,
    skip_greedy_rng_upload=True,
    spec_small_h2d=True,
)


class _ContextGroup(NamedTuple):
    rows: torch.Tensor | None
    slots: torch.Tensor
    offsets: torch.Tensor
    row_map: torch.Tensor
    initial: torch.Tensor
    read_slots: torch.Tensor | None
    lengths: tuple[int, ...]


class _LinearStep(NamedTuple):
    context_rows: int
    slots: torch.Tensor
    replay_slots: torch.Tensor | None
    contexts: tuple[_ContextGroup, ...]


def _partition_context(group: _ContextGroup) -> tuple[_ContextGroup, ...]:
    # Use the same recurrence for a sequence whether or not a longer prompt is
    # in its prefill batch. In a homogeneous batch the runtime views suffice.
    short = [i for i, length in enumerate(group.lengths) if length <= SHORT_CONTEXT]
    if not short or len(short) == len(group.lengths):
        return (group,)

    long = [i for i, length in enumerate(group.lengths) if length > SHORT_CONTEXT]
    offsets = list(accumulate(group.lengths, initial=0))
    device = group.slots.device
    result = []
    for sequences in (short, long):
        selected = torch.tensor(sequences, dtype=torch.long, device=device)
        lengths = tuple(group.lengths[index] for index in sequences)
        rows = [row for index in sequences for row in range(offsets[index], offsets[index + 1])]
        row_map = [index for index, length in enumerate(lengths) for _ in range(length)]
        result.append(
            _ContextGroup(
                torch.tensor(rows, dtype=torch.long, device=device),
                group.slots.index_select(0, selected),
                torch.tensor(
                    list(accumulate(lengths, initial=0)), dtype=group.offsets.dtype, device=device
                ),
                torch.tensor(row_map, dtype=group.row_map.dtype, device=device),
                group.initial.index_select(0, selected),
                group.read_slots.index_select(0, selected)
                if group.read_slots is not None
                else None,
                lengths,
            )
        )
    return tuple(result)


def _linear_step(md: TrtllmAttentionMetadata) -> _LinearStep:
    metadata = md.mamba_metadata
    num_ctx = md.num_contexts
    slots = metadata.state_indices[: md.num_seqs]
    replay_slots = None
    if md.kv_cache_manager.get_replay_state_update_metadata() is not None:
        replay_slots = metadata.replay_state_indices[num_ctx : md.num_seqs]
    if num_ctx == 0:
        return _LinearStep(0, slots, replay_slots, ())
    lengths = tuple(md.seq_lens[:num_ctx].tolist())
    initial = metadata.has_initial_states[:num_ctx]
    context_slots = slots[:num_ctx]
    # FlashInfer reads -1 as a zero state. Destination slots always belong to the request.
    read_slots = torch.where(initial, context_slots, -1) if metadata.use_initial_states else None
    context = _ContextGroup(
        None,
        context_slots,
        metadata.query_start_loc[: num_ctx + 1],
        metadata.seq_idx.reshape(-1),
        initial,
        read_slots,
        lengths,
    )
    return _LinearStep(md.num_ctx_tokens, slots, replay_slots, _partition_context(context))


class _PrefillGraphStep(NamedTuple):
    input_ids: torch.Tensor
    position_ids: torch.Tensor
    recurrent: _LinearStep
    attention: dict


class _PrefillGraphBuffers:
    """Address-stable operands refreshed before each exact-length prefill replay."""

    def __init__(self) -> None:
        self.tensors: dict[str, torch.Tensor] = {}

    def _update(self, name: str, source: torch.Tensor) -> torch.Tensor:
        target = self.tensors.get(name)
        if target is None:
            target = source.clone()
            self.tensors[name] = target
        else:
            target.copy_(source)
        return target

    def prepare(
        self, md: TrtllmAttentionMetadata, input_ids: torch.Tensor, position_ids: torch.Tensor
    ) -> _PrefillGraphStep:
        step = _linear_step(md)
        (group,) = step.contexts
        static_group = group._replace(
            slots=self._update("state_slots", group.slots),
            offsets=self._update("query_offsets", group.offsets),
            row_map=self._update("query_requests", group.row_map),
            initial=self._update("initial_state", group.initial),
        )
        recurrent = _LinearStep(step.context_rows, static_group.slots, None, (static_group,))
        attention = context_attention_step(md)
        for field, value in attention.items():
            if field != "workspace" and isinstance(value, torch.Tensor) and value.is_cuda:
                attention[field] = self._update("attention." + field, value)
        return _PrefillGraphStep(
            self._update("input_ids", input_ids),
            self._update("position_ids", position_ids),
            recurrent,
            attention,
        )


def _small_bf16(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    # Keep the reduction shape independent of the number of rows in the small decode path.
    rows = x.shape[0]
    if rows < 4:
        return F.linear(F.pad(x, (0, 0, 0, 4 - rows)), weight)[:rows]
    return F.linear(x, weight)


class _Decoder(DecoderModel):
    def __init__(self, model_config: ModelConfig) -> None:
        super().__init__(model_config)
        self.supports_prefill_cuda_graph = model_config.spec_config is None
        self._prefill_graph_buffers: dict[int, _PrefillGraphBuffers] = {}
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
        self.gate_stream: torch.cuda.Stream
        self._attention_pools: WeakKeyDictionary = WeakKeyDictionary()

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
        self.gate_stream = torch.cuda.Stream(device)

    @staticmethod
    def _single_fresh_context(md: TrtllmAttentionMetadata, rows: int) -> bool:
        return (
            md.num_contexts == md.num_seqs == 1
            and md.num_ctx_tokens == rows
            and not md.mamba_metadata.use_initial_states
        )

    def accepts_prefill_graph(
        self, md: TrtllmAttentionMetadata, rows: int, buckets: list[int] | None
    ) -> bool:
        # Each captured recurrence specializes the length and request boundaries.
        return bool(buckets and rows in buckets and self._single_fresh_context(md, rows))

    def prepare_prefill_graph(
        self, md: TrtllmAttentionMetadata, input_ids: torch.Tensor, position_ids: torch.Tensor
    ) -> _PrefillGraphStep | None:
        rows = input_ids.numel()
        if not self._single_fresh_context(md, rows):
            return None
        buffers = self._prefill_graph_buffers.get(rows)
        if buffers is None:
            buffers = self._prefill_graph_buffers[rows] = _PrefillGraphBuffers()
        return buffers.prepare(md, input_ids, position_ids)

    def _attention_pool(self, cache: object, index: int) -> tuple[torch.Tensor, int]:
        """Take a zero-copy HND view of the executor's paged KV pool."""
        layers = self._attention_pools.setdefault(cache, {})
        if index not in layers:
            layers[index] = (
                cache.get_buffers(index, kv_layout="HND"),
                int(cache.kv_cache_pool_mapping[cache.layer_offsets[index], 0]),
            )
        return layers[index]

    @staticmethod
    def _prefill_gdn(
        projection: torch.Tensor,
        hidden: torch.Tensor,
        weights: tuple[torch.Tensor, ...],
        conv: torch.Tensor,
        state: torch.Tensor,
        group: _ContextGroup,
    ) -> torch.Tensor:
        ab_weight, conv_weight, a_log, dt_bias, _ = weights
        chunked = max(group.lengths) > SHORT_CONTEXT
        mixed = causal_convolution(
            projection,
            conv_weight,
            conv,
            group.slots,
            group.offsets,
            group.row_map,
            group.initial,
            split=chunked,
        )
        if chunked:
            alpha, beta = gdn_gates(F.linear(hidden, ab_weight), a_log, dt_bias)
        else:
            alpha, beta = decode_gates(hidden, ab_weight, a_log, dt_bias)
        return gated_delta_rule(
            mixed,
            alpha,
            beta,
            state,
            group.slots,
            group.offsets,
            group.initial,
            group.read_slots,
            group.lengths,
        )

    @staticmethod
    def _decode_gdn(
        x: torch.Tensor,
        projection: torch.Tensor,
        weights: tuple[torch.Tensor, ...],
        conv: torch.Tensor,
        state: torch.Tensor,
        slots: torch.Tensor,
        query_rows: int,
        cache: object,
        index: int,
        replay_slots: torch.Tensor | None,
        pending_gates: tuple[torch.cuda.Stream, tuple[torch.Tensor, torch.Tensor]] | None = None,
    ) -> torch.Tensor:
        ab_weight, conv_weight, a_log, dt_bias, _ = weights
        if pending_gates is None:
            gates = decode_gates(x, ab_weight, a_log, dt_bias)
        buffers = (
            cache.mamba_layer_cache(index) if query_rows > 1 or replay_slots is not None else None
        )
        if query_rows > 1:
            mixed = verify_conv(
                projection, conv_weight, slots, conv, buffers.intermediate_conv_window, query_rows
            )
        else:
            mixed = causal_convolution(projection, conv_weight, conv, slots, None, None, None)
        if pending_gates is not None:
            stream, gates = pending_gates
            current = torch.cuda.current_stream()
            current.wait_stream(stream)
            for gate in gates:
                gate.record_stream(current)
        alpha, beta = gates
        if replay_slots is not None:
            return replay_gdn(
                mixed,
                alpha,
                beta,
                state,
                slots,
                replay_slots,
                buffers.old_B,
                buffers.old_x,
                buffers.old_dt,
                buffers.prev_num_accepted_tokens,
                buffers.cache_buf_idx,
                query_rows,
                cache.spec_config.tokens_per_gen_step,
            )
        if query_rows > 1:
            return verify_gdn(
                mixed, alpha, beta, slots, state, buffers.intermediate_ssm, query_rows
            )
        return gated_delta_rule(mixed, alpha, beta, state, slots, None, None, None, ())

    def _mix(
        self,
        layer: LayerWeights,
        index: int,
        x: torch.Tensor,
        quantized: Fp8Weight,
        md: TrtllmAttentionMetadata,
        attention: dict,
        recurrent: _LinearStep | None,
        position: torch.Tensor,
        decode: bool,
        mn_major: bool,
        query_rows: int,
    ) -> torch.Tensor:
        pending_gates = None
        if layer.linear_attention and decode and x.shape[0] <= 8:
            # The two projections read the same normalized rows independently.
            ab_weight, _, a_log, dt_bias, _ = layer.attention_params
            self.gate_stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(self.gate_stream):
                pending_gates = (self.gate_stream, decode_gates(x, ab_weight, a_log, dt_bias))
        projection = fp8_projection(quantized, layer.input_projection, decode, self.gemm)
        cache = md.kv_cache_manager
        if layer.linear_attention:
            assert recurrent is not None
            conv = cache.get_conv_states(index)
            state = cache.get_ssm_states(index)
            norm = layer.attention_params[-1]
            ctx_rows = recurrent.context_rows
            if decode:
                core = self._decode_gdn(
                    x,
                    projection,
                    layer.attention_params,
                    conv,
                    state,
                    recurrent.slots,
                    query_rows,
                    cache,
                    index,
                    recurrent.replay_slots,
                    pending_gates,
                )
            else:
                if len(recurrent.contexts) == 1:
                    core = self._prefill_gdn(
                        projection[:ctx_rows],
                        x[:ctx_rows],
                        layer.attention_params,
                        conv,
                        state,
                        recurrent.contexts[0],
                    )
                else:
                    core = projection.new_empty((ctx_rows, 4096))
                    for group in recurrent.contexts:
                        indices = group.rows
                        assert indices is not None
                        core.index_copy_(
                            0,
                            indices,
                            self._prefill_gdn(
                                projection.index_select(0, indices),
                                x.index_select(0, indices),
                                layer.attention_params,
                                conv,
                                state,
                                group,
                            ),
                        )
                if ctx_rows < x.shape[0]:
                    decode_core = self._decode_gdn(
                        x[ctx_rows:],
                        projection[ctx_rows:],
                        layer.attention_params,
                        conv,
                        state,
                        recurrent.slots[md.num_contexts :],
                        query_rows,
                        cache,
                        index,
                        recurrent.replay_slots,
                    )
                    core = torch.cat((core, decode_core))
            gated = gated_output_quantize(core, projection, norm, mn_major=mn_major)
        else:
            qkv = qkv_projection(
                projection, *layer.attention_params, self.cosine, self.sine, position
            )
            native = uses_native_decode(md)
            if decode:
                native_output = (
                    gqa(qkv, cache.layer_offsets[index], attention, query_rows) if native else None
                )
                pool, pool_id = self._attention_pool(cache, index)
                pages = md.kv_cache_block_offsets[pool_id, : md.num_seqs, 0]
                gated = paged_decode_attention(
                    qkv,
                    pool,
                    position,
                    pages,
                    md.max_seq_len,
                    query_rows,
                    projection=projection,
                    mn_major=mn_major,
                    native_output=native_output,
                )
                return fp8_projection(gated, layer.output_projection, decode, self.gemm)
            if native:
                core = gqa(qkv, cache.layer_offsets[index], attention, query_rows)
            else:
                ctx_count = md.num_contexts
                ctx_rows = md.num_ctx_tokens
                core = gqa(qkv[:ctx_rows], cache.layer_offsets[index], attention)
                if ctx_rows < qkv.shape[0]:
                    pool, pool_id = self._attention_pool(cache, index)
                    pages = md.kv_cache_block_offsets[pool_id, ctx_count : md.num_seqs, 0]
                    gen_core = paged_decode_attention(
                        qkv[ctx_rows:], pool, position[ctx_rows:], pages, md.max_seq_len, query_rows
                    )
                    core = torch.cat((core, gen_core))
            gated = gated_output_quantize(core, projection, None, mn_major=mn_major)
        return fp8_projection(gated, layer.output_projection, decode, self.gemm)

    def forward(
        self,
        attn_metadata: TrtllmAttentionMetadata,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        prefill_graph: _PrefillGraphStep | None = None,
    ) -> torch.Tensor:
        rows = input_ids.numel()
        if prefill_graph is None:
            recurrent = _linear_step(attn_metadata)
            attention = (
                attention_step(attn_metadata)
                if uses_native_decode(attn_metadata)
                else context_attention_step(attn_metadata)
            )
        else:
            input_ids, position_ids, recurrent, attention = prefill_graph
        position = position_ids.view(-1, rows)[0]
        decode = attn_metadata.num_contexts == 0
        generation_count = attn_metadata.num_seqs - attn_metadata.num_contexts
        query_rows = (rows - recurrent.context_rows) // generation_count if generation_count else 1
        mn_major = uses_cutlass(rows, decode) and (not decode or rows == 128)
        short_group = (
            recurrent.contexts[0]
            if recurrent.contexts and max(recurrent.contexts[0].lengths) <= SHORT_CONTEXT
            else None
        )
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
                query_rows,
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
                self.route_counts,
                self.gemm,
                query_rows=query_rows if decode else 1,
                short_context=short_group is not None and short_group.rows is None,
                short_rows=short_group.rows if short_group is not None else None,
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
        if query_rows == 1 and generation_count and recurrent.replay_slots is not None:
            # No acceptance update follows a step that committed its only input.
            replay = attn_metadata.kv_cache_manager.get_replay_state_update_metadata()
            replay.prev_num_accepted_tokens.index_fill_(0, recurrent.replay_slots, 0)
            replay.cache_buf_idx.index_fill_(0, recurrent.replay_slots, 0)
        return hidden


class _MtpLayer(nn.Module):
    supports_pick_rows = True

    def __init__(self, decoder: _Decoder) -> None:
        super().__init__()
        self._decoder = ref(decoder)
        self.weights: LayerWeights
        self.fc: torch.Tensor
        self.embedding_norm: torch.Tensor
        self.hidden_norm: torch.Tensor
        self.final_norm: torch.Tensor

    def load(self, weights: Mapping[str, torch.Tensor], device: torch.device) -> None:
        self.weights = load_layer(weights, "mtp.layers.0.", False, device)
        self.fc = weights["mtp.fc.weight"].to(device)
        self.embedding_norm = weights["mtp.pre_fc_norm_embedding.weight"].to(device)
        self.hidden_norm = weights["mtp.pre_fc_norm_hidden.weight"].to(device)
        self.final_norm = weights["mtp.norm.weight"].to(device)

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        hidden_states: torch.Tensor,
        attn_metadata: TrtllmAttentionMetadata,
        embed_tokens: nn.Embedding,
        pick_rows: torch.Tensor | None = None,
        **kwargs: object,
    ) -> torch.Tensor:
        decoder = self._decoder()
        layer = self.weights
        rows = input_ids.numel()
        position = position_ids.reshape(-1, rows)[0].contiguous()
        decode = attn_metadata.num_contexts == 0
        context_count = attn_metadata.num_contexts
        context_rows = attn_metadata.num_ctx_tokens if context_count else 0
        generation_count = attn_metadata.num_seqs - context_count
        query_rows = (rows - context_rows) // generation_count if generation_count else 1
        pair = mtp_input(
            input_ids, embed_tokens.weight, hidden_states, self.embedding_norm, self.hidden_norm
        )
        residual = _small_bf16(pair, self.fc) if decode else F.linear(pair, self.fc)
        mn_major = uses_cutlass(rows, decode) and (not decode or rows == 128)
        _, hidden, attention_input = residual_norm_quantize(
            residual, layer.input_norm, mn_major=mn_major
        )
        delta = decoder._mix(
            layer,
            len(decoder.layers),
            hidden,
            attention_input,
            attn_metadata,
            (
                attention_step(attn_metadata)
                if uses_native_decode(attn_metadata)
                else context_attention_step(attn_metadata)
            ),
            None,
            position,
            decode,
            mn_major,
            query_rows,
        )
        if pick_rows is not None:
            residual = residual[pick_rows]
            delta = delta[pick_rows]
        expert_decode = decode or pick_rows is not None
        residual, hidden, expert_input = residual_norm_quantize(
            residual, layer.post_attention_norm, delta=delta
        )
        out, probabilities, router, permutation = routed_experts(
            hidden,
            expert_input,
            layer.router,
            layer.expert_gate_up,
            layer.expert_down,
            expert_decode,
            decoder.route_counts,
            decoder.gemm,
            query_rows=query_rows if decode and pick_rows is None else 1,
            short_context=context_rows <= SHORT_CONTEXT,
        )
        _, hidden, _ = residual_norm_quantize(
            residual,
            self.final_norm,
            delta=out,
            probability=probabilities,
            router=router,
            quantize=False,
            permutation=permutation,
        )
        return hidden

    @staticmethod
    def shared_head(
        hidden: torch.Tensor,
        lm_head: nn.Module,
        attn_metadata: TrtllmAttentionMetadata,
        return_context_logits: bool,
    ) -> torch.Tensor:
        return _small_bf16(hidden, lm_head.weight)


class _DraftHead:
    """Reduced vocabulary for MTP proposals, verified by the complete target head."""

    def __init__(self, target_weight: torch.Tensor) -> None:
        common = 65536
        special = 248000
        vocab_size, hidden = target_weight.shape
        columns = common + vocab_size - special
        aligned = (columns + 255) // 256 * 256
        self.weight = target_weight.new_empty((aligned, hidden))
        self.common = common
        self.weight[:common].copy_(target_weight[:common])
        self.weight[common:columns].copy_(target_weight[special:])
        self.columns = columns
        self.special_offset = special - common

    def supports(self, rows: int) -> bool:
        return rows <= 32

    def sample(self, hidden: torch.Tensor, out: torch.Tensor | None = None) -> torch.Tensor:
        logits = _small_bf16(hidden, self.weight)[:, : self.columns]
        tokens = greedy_argmax(logits, small_rows=True)
        tokens = torch.where(tokens < self.common, tokens, tokens + self.special_offset)
        if out is not None:
            out.copy_(tokens)
            return out
        return tokens


class _MtpDraft(nn.Module):
    def __init__(self, decoder: _Decoder, lm_head: nn.Module) -> None:
        super().__init__()
        self.embed_tokens = decoder.embed_tokens
        self.lm_head = lm_head
        self.mtp_layers = nn.ModuleList([_MtpLayer(decoder)])


@disable_v2_by_default
class ModelingV2Qwen3635BA3BFp8Sm100Default(DecoderModelForCausalLM):
    """Text-only FP8 target, with recurrent and paged cache state owned by TRT-LLM."""

    moe_impls_honored = frozenset()
    kv_cache_manager_cls = Qwen36HybridCacheManager

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
        self.spec_worker = None
        self.draft_model = None
        spec_config = runtime_config.spec_config
        self.spec_decode_fast_paths = spec_config is not None
        self.runtime_hooks = _SPEC_DECODE_HOOKS if self.spec_decode_fast_paths else None
        if spec_config is not None:
            self.draft_model = _MtpDraft(self.model, self.lm_head)
            self.spec_worker = get_spec_worker(
                spec_config,
                runtime_config,
                runtime_config.mapping,
                use_separate_draft_kv_cache=should_use_separate_draft_kv_cache(spec_config),
                spec_decode_fast_paths=self.spec_decode_fast_paths,
            )
            self.spec_worker.set_draft_model(self.draft_model)
            self.spec_worker.set_target_model(self)

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
        if self.draft_model is not None:
            self.draft_model.mtp_layers[0].load(weights, device)
            self.spec_worker.draft_head = _DraftHead(self.lm_head.weight)

    def forward(
        self,
        attn_metadata: TrtllmAttentionMetadata,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        return_context_logits: bool = False,
        spec_metadata: SpecMetadata | None = None,
        resource_manager: object = None,
        **kwargs: object,
    ) -> torch.Tensor | dict:
        """Evaluate packed context and generation rows in scheduler order."""
        prefill_graph = None
        if self.model.supports_prefill_cuda_graph and is_in_breakable_cuda_graph():
            prefill_graph = self.model.prepare_prefill_graph(attn_metadata, input_ids, position_ids)
        hidden = self.model(attn_metadata, input_ids, position_ids, prefill_graph)
        if self.spec_worker is not None:
            selected = hidden[spec_metadata.gather_ids]
            logits = _small_bf16(selected, self.lm_head.weight)
            if (
                not spec_metadata.is_all_greedy_sample
                or self.spec_worker.guided_decoder is not None
            ):
                logits = logits.float()
            return self.spec_worker(
                input_ids=input_ids,
                position_ids=position_ids,
                hidden_states=hidden,
                logits=logits,
                attn_metadata=attn_metadata,
                spec_metadata=spec_metadata,
                draft_model=self.draft_model,
                resource_manager=resource_manager,
            )
        if not return_context_logits:
            if attn_metadata.num_contexts:
                ends = torch.cumsum(attn_metadata.seq_lens_cuda, dim=0, dtype=torch.long) - 1
                hidden = hidden[ends]
            else:
                return _small_bf16(hidden, self.lm_head.weight).float()
        return F.linear(hidden, self.lm_head.weight).float()

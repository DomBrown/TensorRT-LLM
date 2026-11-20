# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Qwen3.6 FP8 MTP layer for the runtime's linear-chain draft worker."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING

import torch
import torch.nn as nn
import torch.nn.functional as F

from .kernels import fp8
from .ops.attention import KvPages, full_attention
from .ops.multi_token import draft_argmax, draft_input
from .ops.projections import (
    RoutedWeights,
    dense_projection,
    route_experts,
    routed_experts,
    shared_expert,
)
from .weights import LayerWeights

if TYPE_CHECKING:
    from tensorrt_llm._torch.attention.backends.interface import AttentionMetadata
    from tensorrt_llm._torch.model_config import ModelConfig


class DraftOutput(nn.Module):
    """Share the target's full BF16 head when the worker needs draft logits."""

    def forward(
        self,
        hidden: torch.Tensor,
        lm_head: nn.Module,
        attn_metadata: AttentionMetadata,
        return_context_logits: bool,
    ) -> torch.Tensor:
        if not return_context_logits:
            hidden = hidden[attn_metadata.seq_lens_cuda.cumsum(0).long() - 1]
        return lm_head(hidden)


class DraftShortlist:
    """Propose greedy tokens from a prefix of the shared vocabulary matrix."""

    def __init__(self, weight: torch.Tensor) -> None:
        self.weight = weight[:65536]

    def supports(self, rows: int) -> bool:
        return rows <= 8

    def sample(self, hidden: torch.Tensor, out: torch.Tensor | None = None) -> torch.Tensor:
        return draft_argmax(F.linear(hidden, self.weight), out)


class Qwen36MtpLayer(nn.Module):
    """One recurrently applied MTP block using the decoder's FP8 operators."""

    def __init__(
        self, model_config: ModelConfig, layer_idx: int, streams: dict[str, torch.cuda.Stream]
    ) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        self.shared_head = DraftOutput()
        self.side_stream = streams["draft"]
        self._cache_manager = None
        self._cache_view = None
        self._table_index = 0

    def load_checkpoint(self, weights: Mapping[str, torch.Tensor], rope: torch.Tensor) -> None:
        """Place the MTP FC, norms and FP8 decoder block."""
        self.layer = LayerWeights(weights, "mtp.layers.0.", linear=False)
        self.register_buffer("embedding_norm", weights["mtp.pre_fc_norm_embedding.weight"].cuda())
        self.register_buffer("hidden_norm", weights["mtp.pre_fc_norm_hidden.weight"].cuda())
        self.register_buffer("fc", weights["mtp.fc.weight"].cuda())
        self.register_buffer("final_norm", weights["mtp.norm.weight"].cuda())
        self.rope = rope

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        hidden_states: torch.Tensor,
        embed_tokens: nn.Embedding,
        attn_metadata: AttentionMetadata,
        **kwargs,
    ) -> torch.Tensor:
        """Run one draft pass and return hidden rows in request order."""
        from .modeling import make_step

        ids = input_ids.flatten()
        rows = ids.shape[0]
        positions = position_ids.reshape(-1, rows)[0]
        batch = make_step(attn_metadata, positions, rows)
        layer = self.layer
        stack = draft_input(
            ids, embed_tokens.weight, hidden_states, self.embedding_norm, self.hidden_norm
        )
        residual = F.linear(stack, self.fc)
        hidden = fp8.rms(residual, layer.input_norm)
        qkv = dense_projection(fp8.quant(hidden, True), layer.qkv)

        manager = attn_metadata.kv_cache_manager
        if manager is not self._cache_manager:
            self._cache_view = manager.get_buffers(self.layer_idx, kv_layout="NHD")
            self._table_index = int(
                manager.kv_cache_pool_mapping[manager.layer_offsets[self.layer_idx], 0]
            )
            self._cache_manager = manager
        pages = KvPages(self._cache_view, attn_metadata.kv_cache_block_offsets[self._table_index])
        small = batch.context_rows == 0 and rows <= 8
        gate = full_attention(
            qkv,
            layer.q_norm,
            layer.k_norm,
            self.rope,
            positions,
            batch.requests,
            pages,
            batch.lengths,
            batch.cached,
            batch.context_rows,
            rows <= 8,
        )
        attention = dense_projection(gate, layer.proj)
        residual, hidden, incoming, shared_in = fp8.add_rms_quant(
            residual,
            attention,
            layer.post_norm,
            False,
            False,
            True,
            mx=rows <= 8,
            exact=not small,
            pdl=small,
        )
        current = torch.cuda.current_stream()
        self.side_stream.wait_stream(current)
        with torch.cuda.stream(self.side_stream):
            shared = shared_expert(shared_in, layer)
        route, prob, shared_gate, grouping = route_experts(hidden, layer.router, small, small)
        experts, inverse = routed_experts(
            incoming,
            route,
            RoutedWeights(layer.expert_up, layer.expert_down),
            small,
            grouping,
            small,
        )
        current.wait_stream(self.side_stream)
        shared.record_stream(current)
        _, hidden, _ = fp8.combine_next_norm(
            experts,
            prob,
            shared,
            shared_gate,
            residual,
            self.final_norm,
            False,
            8,
            False,
            False,
            True,
            False,
            inverse,
            None,
            None,
            pdl=small,
        )
        return hidden

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Bind the model's Q/K/V rows to the runtime's paged BF16 attention."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, Cell, OpWrapper
from tensorrt_llm._torch._experimental.modeling_v2.catalog.attention.thop_attention import (
    thop_attention,
)

from ..kernels.kernels import prepare_qkv
from ..kernels.paged_decode import paged_decode

if TYPE_CHECKING:
    from tensorrt_llm._torch.attention.backends.trtllm import TrtllmAttentionMetadata


_CALL_CONSTANTS = dict(
    output_sf=None,
    update_kv_cache=True,
    attention_input_type=0,
    is_mla_enable=False,
    mask_type=1,
    q_scaling=1.0,
    attention_sinks=None,
    position_embedding_type=0,
    rotary_inv_freq=None,
    rotary_cos_sin=None,
    rope_dim=0,
    rope_base=10000.0,
    rope_scale_type=0,
    rope_scale=1.0,
    rope_short_m_scale=1.0,
    rope_long_m_scale=1.0,
    rope_max_positions=1024,
    rope_original_max_positions=1024,
    out_scale=None,
    latent_cache=None,
    q_pe=None,
    q_lora_rank=None,
    kv_lora_rank=None,
    qk_nope_head_dim=None,
    qk_rope_head_dim=None,
    v_head_dim=None,
    rope_append=None,
    chunked_prefill_buffer_batch_size=1,
    attention_chunk_size=None,
    softmax_stats_tensor=None,
    sparse_kv_indices=None,
    sparse_kv_offsets=None,
    sparse_attn_indices=None,
    sparse_attn_offsets=None,
    sparse_attn_indices_block_size=0,
    mrope_rotary_cos_sin=None,
    mrope_position_deltas=None,
    helix_position_offsets=None,
    helix_is_inactive_rank=None,
    cross_kv=None,
    relative_attention_bias=None,
    relative_attention_max_distance=0,
    kv_norm_weight=None,
    kv_norm_eps=1e-6,
    skip_correction_threshold=0.0,
)


def attention_step(md: TrtllmAttentionMetadata) -> dict:
    """Read the prepared paged-attention buffers once for the whole stack."""
    return dict(
        sequence_length=md.kv_lens_cuda_runtime,
        host_past_key_value_lengths=md.kv_lens_runtime,
        host_total_kv_lens=md.host_total_kv_lens,
        context_lengths=md.prompt_lens_cuda_runtime,
        host_context_lengths=md.prompt_lens_cpu_runtime,
        host_request_types=md.host_request_types_runtime,
        kv_cache_block_offsets=md.kv_cache_block_offsets,
        host_kv_cache_pool_pointers=md.host_kv_cache_pool_pointers,
        host_kv_cache_pool_mapping=md.host_kv_cache_pool_mapping,
        workspace=md.effective_workspace,
        tokens_per_block=md.tokens_per_block,
        max_num_requests=md.max_num_requests,
        max_context_length=md.max_context_length,
        max_seq_len=md.max_seq_len,
        # No sliding window: the full admitted length on every attention layer.
        attention_window_size=md.max_seq_len,
        num_contexts=md.num_contexts,
        num_ctx_tokens=md.num_ctx_tokens,
        trtllm_gen_jit_warmup=md.trtllm_gen_jit_warmup,
        use_paged_context_fmha=md.use_paged_context_fmha,
        beam_width=md.effective_beam_width,
        cache_indirection=md.cache_indirection,
        block_ids_per_seq=md.block_ids_per_seq,
        max_context_q_len_override=md.max_context_q_len_override,
        is_cross=md.is_cross,
        is_spec_decoding_enabled=md.is_spec_decoding_enabled,
        use_spec_decoding=md.use_spec_decoding,
        is_spec_dec_tree=md.is_spec_dec_tree,
        spec_decoding_generation_lengths=md.spec_decoding_generation_lengths,
        spec_decoding_position_offsets=None,
        spec_decoding_packed_mask=md.spec_decoding_packed_mask,
        spec_decoding_bl_tree_mask_offset=md.spec_decoding_bl_tree_mask_offset,
        spec_decoding_bl_tree_mask=md.spec_decoding_bl_tree_mask,
        spec_decoding_target_max_draft_tokens=md.max_total_draft_tokens,
        spec_bl_tree_first_sparse_mask_offset_kv=md.spec_bl_tree_first_sparse_mask_offset_kv,
        num_sparse_topk=md.num_sparse_topk,
        flash_mla_tile_scheduler_metadata=md.flash_mla_tile_scheduler_metadata,
        flash_mla_num_splits=md.flash_mla_num_splits,
        max_num_sequences=md.max_num_sequences,
        force_prepare_spec_dec_tree_mask=md.force_prepare_spec_dec_tree_mask,
    )


def gqa(
    packed_qkv: torch.Tensor, layer_offset: int, metadata: dict, query_rows: int = 1
) -> torch.Tensor:
    """Append packed [rows, (16 + 2 + 2) * 256] BF16 QKV and return attended rows."""
    return thop_attention(
        q=packed_qkv,
        k=None,
        v=None,
        output=packed_qkv.new_empty((packed_qkv.shape[0], 4096)),
        is_fused_qkv=True,
        predicted_tokens_per_seq=query_rows,
        local_layer_idx=layer_offset,
        num_heads=16,
        num_kv_heads=2,
        head_size=256,
        quant_mode=0,
        kv_scale_orig_quant=None,
        kv_scale_quant_orig=None,
        **metadata,
        **_CALL_CONSTANTS,
    )


class PrepareQkv(OpWrapper):
    """Normalize Q/K and rotate the first 64 channels at each logical position."""

    ARCHS = frozenset({Arch.SM_100})
    CELLS = (
        Cell("context rows over multiple positions", {"rows": 129}),
        Cell("decode rows with long prefixes", {"rows": 32, "packed": False}),
    )

    def __call__(
        self,
        projection: torch.Tensor,
        qw: torch.Tensor,
        kw: torch.Tensor,
        cosine: torch.Tensor,
        sine: torch.Tensor,
        position: torch.Tensor,
        packed: bool = True,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return prepare_qkv(projection, qw, kw, cosine, sine, position, packed)

    def reference(
        self,
        projection: torch.Tensor,
        qw: torch.Tensor,
        kw: torch.Tensor,
        cosine: torch.Tensor,
        sine: torch.Tensor,
        position: torch.Tensor,
        packed: bool = True,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        rows = projection.shape[0]
        q = projection[:, :8192].reshape(rows, 16, 512)[..., :256]
        k = projection[:, 8192:8704].reshape(rows, 2, 256)
        v = projection[:, 8704:].reshape(rows, 2, 256)
        cos = cosine[position.long(), None]
        sin = sine[position.long(), None]

        def normalize_and_rotate(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
            xf = x.float()
            xf = xf * (xf.square().mean(-1, keepdim=True) + 1e-6).rsqrt() * (1 + weight.float())
            left, right, tail = xf.split((32, 32, 192), dim=-1)
            return torch.cat((cos * left - sin * right, cos * right + sin * left, tail), -1).to(
                x.dtype
            )

        q = normalize_and_rotate(q, qw)
        k = normalize_and_rotate(k, kw)
        if packed:
            return torch.cat((q.flatten(1), k.flatten(1), v.flatten(1)), dim=-1)
        return q, k, v


qkv_projection = PrepareQkv()


class PagedDecodeAttention(OpWrapper):
    """Append BF16 KV and compute one causal GQA query per request."""

    ARCHS = frozenset({Arch.SM_100})
    CELLS = (
        Cell("single-token history in a nonzero page", {"positions": (0,)}),
        Cell("history spans uneven physical pages and split boundaries", {"positions": (286,)}),
        Cell(
            "coalesced layers and varied history lengths",
            {"positions": (143, 31, 161), "layers": 3},
        ),
    )

    def __call__(
        self,
        qkv: torch.Tensor,
        pool: torch.Tensor,
        positions: torch.Tensor,
        key_pages: torch.Tensor,
        max_seq_len: int,
    ) -> torch.Tensor:
        return paged_decode(qkv, pool, positions, key_pages, max_seq_len)

    def reference(
        self,
        qkv: torch.Tensor,
        pool: torch.Tensor,
        positions: torch.Tensor,
        key_pages: torch.Tensor,
        max_seq_len: int,
    ) -> torch.Tensor:
        page_size = pool.shape[-2]
        row_stride = pool.stride(0) // pool.stride(1)
        q, k, v = qkv.split((4096, 512, 512), dim=-1)
        output = torch.empty_like(q)
        for i, position in enumerate(positions.tolist()):
            logical_page, offset = divmod(position, page_size)
            ids = key_pages[i, : logical_page + 1].long() // row_stride
            page = int(ids[-1])
            pool[page, 0, :, offset].copy_(k[i].view(2, 256))
            pool[page, 1, :, offset].copy_(v[i].view(2, 256))
            key, value = pool[ids].permute(1, 2, 0, 3, 4).flatten(2, 3)
            key = key[:, : position + 1].float()
            value = value[:, : position + 1].float()
            query = q[i].view(2, 8, 256).float()
            probs = ((query @ key.transpose(-1, -2)) / 16.0).softmax(dim=-1)
            output[i].copy_((probs @ value).flatten())
        return output


paged_decode_attention = PagedDecodeAttention()

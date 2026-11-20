# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Full attention over TensorRT-LLM KV pages."""

from typing import NamedTuple

import torch
import torch.nn.functional as F
import triton as tr
from torch.nn.attention.bias import causal_lower_right

from .....catalog._op import Arch, Cell, OpWrapper
from ..kernels import attention, fp8


class KvPages(NamedTuple):
    cache: torch.Tensor
    table: torch.Tensor


class ProjectAndStore(OpWrapper):
    """Split QKV, normalize Q/K, rotate 64 channels and store paged K/V."""

    ARCHS = frozenset({Arch.SM_100})
    CELLS = (
        Cell("decode addresses a nonzero request and page", {"context_rows": 0}),
        Cell("packed projections retain only context rows", {"context_rows": 2}),
    )

    def __call__(
        self,
        projected: torch.Tensor,
        q_norm: torch.Tensor,
        k_norm: torch.Tensor,
        rope: torch.Tensor,
        positions: torch.Tensor,
        requests: torch.Tensor,
        pages: KvPages,
        context_rows: int,
        pdl: bool,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Split QKV, apply text RoPE and publish K/V to the scheduler's pages."""
        rows = projected.shape[0]
        q = projected.new_empty((rows, 16, 256))
        k = projected.new_empty((context_rows, 2, 256))
        v = torch.empty_like(k)
        gate = projected.new_empty((rows, 4096))
        attention.qkv_rope_paged[(rows, 18)](
            projected,
            q_norm,
            k_norm,
            rope,
            positions,
            requests,
            pages.table,
            pages.cache,
            q,
            k,
            v,
            gate,
            pages.table.stride(1),
            pages.cache.stride(1),
            pages.cache.shape[2],
            context_rows,
            pdl,
            num_warps=1,
            enable_fp_fusion=False,
            launch_pdl=pdl,
        )
        return q, k, v, gate

    def reference(
        self,
        projected: torch.Tensor,
        q_norm: torch.Tensor,
        k_norm: torch.Tensor,
        rope: torch.Tensor,
        positions: torch.Tensor,
        requests: torch.Tensor,
        pages: KvPages,
        context_rows: int,
        pdl: bool,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute partial rotary embeddings and publish the paged cache."""
        del pdl
        rows = projected.shape[0]
        query_gate, key, value = projected.split((8192, 512, 512), dim=-1)
        query, gate = query_gate.view(rows, 16, 512).chunk(2, dim=-1)
        key, value = key.reshape(rows, 2, 256), value.reshape(rows, 2, 256)

        def rotate(x: torch.Tensor, norm: torch.Tensor) -> torch.Tensor:
            x = x.float()
            x = (
                (x * torch.rsqrt(x.square().mean(-1, keepdim=True) + 1e-6) * (1.0 + norm.float()))
                .to(projected.dtype)
                .float()
            )
            cos, sin = rope[positions.long(), None, :].chunk(2, dim=-1)
            half_a, half_b, unrotated = x.split((32, 32, 192), dim=-1)
            return torch.cat(
                (half_a * cos - half_b * sin, half_b * cos + half_a * sin, unrotated), dim=-1
            ).to(projected.dtype)

        query = rotate(query, q_norm)
        key = rotate(key, k_norm)
        page_size = pages.cache.shape[2]
        planes = pages.cache.as_strided(
            (pages.cache.shape[0] * 2, *pages.cache.shape[2:]),
            (pages.cache.stride(1), *pages.cache.stride()[2:]),
        )
        for row, (position, request) in enumerate(zip(positions.tolist(), requests.tolist())):
            key_page, value_page = pages.table[request, :, position // page_size]
            planes[key_page, position % page_size].copy_(key[row])
            planes[value_page, position % page_size].copy_(value[row])
        return query, key[:context_rows], value[:context_rows], gate.reshape(rows, 4096)


project_and_store = ProjectAndStore()


class PagedDecodeAttention(OpWrapper):
    """Compensated split-KV GQA and gate followed by FP8 quantization."""

    ARCHS = frozenset({Arch.SM_100})
    CELLS = (
        Cell("one short decode row with nonsequential pages", {"lengths": (63,), "page_size": 32}),
        Cell(
            "batched decode crossing the long-context tile switch",
            {"lengths": (129, 2051), "page_size": 32},
        ),
        Cell("packed split-KV decode", {"lengths": (63, 129, 271) * 6, "page_size": 16}),
    )
    note = "The Triton attention and FP8 epilogue preserve BF16 gate and FP32 softmax rounding."

    def __call__(
        self,
        q: torch.Tensor,
        gate: torch.Tensor,
        positions: torch.Tensor,
        requests: torch.Tensor,
        pages: KvPages,
        encode_mx: bool,
        pdl: bool,
    ) -> tuple[torch.Tensor, ...]:
        rows = q.shape[0]
        splits = 16
        part = q.new_empty((rows, 16, splits, 256), dtype=torch.float32)
        lse = q.new_empty((rows, 16, splits), dtype=torch.float32)
        attention.attention_paged[(2, splits, rows)](
            q,
            pages.table,
            pages.cache,
            positions,
            requests,
            part,
            lse,
            pages.table.stride(1),
            pages.cache.stride(1),
            pages.cache.shape[2],
            splits,
            pdl,
            num_warps=4,
            launch_pdl=pdl,
        )
        quantized = q.new_empty((rows, 4096), dtype=torch.float8_e4m3fn)
        scales = q.new_empty((rows, 32), dtype=torch.float32)
        mx = fp8.mx_buffer(quantized, 4096) if encode_mx else scales
        fp8.attn_merge_fp8[(16, 2, rows)](
            part,
            lse,
            gate,
            quantized,
            scales,
            mx,
            splits,
            True,
            encode_mx,
            rows,
            pdl,
            num_warps=4,
            launch_pdl=pdl,
        )
        return (quantized, scales, mx.view(torch.uint8)) if encode_mx else (quantized, scales)

    def reference(
        self,
        q: torch.Tensor,
        gate: torch.Tensor,
        positions: torch.Tensor,
        requests: torch.Tensor,
        pages: KvPages,
        encode_mx: bool,
        pdl: bool,
    ) -> torch.Tensor:
        """Return the BF16 gated result before quantization."""
        del encode_mx, pdl
        outputs = []
        for row, (position, request) in enumerate(zip(positions.tolist(), requests.tolist())):
            length = position + 1
            layers = pages.cache.as_strided(
                (pages.cache.shape[0] * 2, *pages.cache.shape[2:]),
                (pages.cache.stride(1), *pages.cache.stride()[2:]),
            )
            count = tr.cdiv(length, pages.cache.shape[2])
            keys = layers[pages.table[request, 0, :count]].flatten(0, 1)[:length]
            values = layers[pages.table[request, 1, :count]].flatten(0, 1)[:length]
            keys = keys.repeat_interleave(8, dim=1).float()
            values = values.repeat_interleave(8, dim=1).float()
            scores = torch.einsum("hd,lhd->hl", q[row].float(), keys) / 16
            result = torch.einsum("hl,lhd->hd", scores.softmax(-1), values)
            outputs.append(result.to(q.dtype).flatten().float() * torch.sigmoid(gate[row].float()))
        return torch.stack(outputs).to(q.dtype)


paged_decode_attention = PagedDecodeAttention()


def context_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    pages: KvPages,
    lengths: tuple[int, ...],
    cached: tuple[int, ...],
) -> torch.Tensor:
    """Causal SDPA for fresh prompts and continuation chunks in a mixed step."""
    output = torch.empty_like(q)
    begin = 0
    for request, (length, prefix) in enumerate(zip(lengths, cached)):
        end = begin + length
        if prefix:
            total = prefix + length
            keys = k.new_empty((total, 2, 256))
            values = torch.empty_like(keys)
            attention.gather_paged[(tr.cdiv(total * 512, 1024),)](
                pages.table,
                pages.cache,
                keys,
                values,
                request,
                pages.table.stride(1),
                pages.cache.stride(1),
                pages.cache.shape[2],
                total,
                1024,
            )
            mask = causal_lower_right(length, total)
        else:
            keys, values = k[begin:end], v[begin:end]
            mask = None
        result = F.scaled_dot_product_attention(
            q[begin:end].transpose(0, 1).unsqueeze(0),
            keys.transpose(0, 1).unsqueeze(0),
            values.transpose(0, 1).unsqueeze(0),
            attn_mask=mask,
            is_causal=mask is None,
            enable_gqa=True,
        )
        output[begin:end].copy_(result.squeeze(0).transpose(0, 1))
        begin = end
    return output


def full_attention(
    projected: torch.Tensor,
    q_norm: torch.Tensor,
    k_norm: torch.Tensor,
    rope: torch.Tensor,
    positions: torch.Tensor,
    requests: torch.Tensor,
    pages: KvPages,
    lengths: tuple[int, ...],
    cached: tuple[int, ...],
    context_rows: int,
    encode_mx: bool,
) -> tuple[torch.Tensor, ...]:
    """Execute context and generation rows using their own attention schedules."""
    pdl = context_rows == 0 and projected.shape[0] <= 8
    q, k, v, gate = project_and_store(
        projected, q_norm, k_norm, rope, positions, requests, pages, context_rows, pdl
    )
    if not context_rows:
        return paged_decode_attention(q, gate, positions, requests, pages, encode_mx, pdl)
    output = context_attention(q[:context_rows], k, v, pages, lengths, cached)
    quantized = projected.new_empty((context_rows, 4096), dtype=torch.float8_e4m3fn)
    scales = projected.new_empty((context_rows, 32), dtype=torch.float32)
    fp8.softmax_prefill_output[(context_rows, 8)](
        output, gate, quantized, scales, scales, True, 4, False, 0, num_warps=4
    )
    if context_rows == projected.shape[0]:
        return quantized, scales
    decode_quant, decode_scales = paged_decode_attention(
        q[context_rows:],
        gate[context_rows:],
        positions[context_rows:],
        requests[context_rows:],
        pages,
        False,
        False,
    )
    quantized = torch.cat((quantized.view(torch.uint8), decode_quant.view(torch.uint8)))
    return quantized.view(torch.float8_e4m3fn), torch.cat((scales, decode_scales))

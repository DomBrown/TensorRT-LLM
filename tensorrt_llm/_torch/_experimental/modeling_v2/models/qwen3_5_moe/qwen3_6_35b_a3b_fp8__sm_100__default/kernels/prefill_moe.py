# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Device-side expert grouping for Blackwell FP8 prefill GEMMs."""

import torch
import triton as tr
import triton.language as tl
from flashinfer.gemm import group_gemm_fp8_nt_groupwise

from . import fp8 as kernels
from . import grouped_fp8_runtime


@tr.jit
def offsets(Count, Indptr, Pairs, ALIGN: tl.constexpr):
    e = tl.arange(0, 256)
    counts = tl.load(Count + e)
    lengths = tl.cdiv(counts, ALIGN) * ALIGN
    end = tl.cumsum(lengths)
    tl.store(Indptr + e, end - lengths)
    tl.store(Indptr + 256, tl.sum(lengths, 0))
    pad = tl.arange(0, ALIGN)
    tl.store(
        Pairs + (end[:, None] - 1 - pad[None, :]), -1, pad[None, :] < (lengths - counts)[:, None]
    )


@tr.jit(do_not_specialize=["N"])
def scatter(Route, Rank, Indptr, Pairs, Inverse, Start, N, B: tl.constexpr, WINDOW: tl.constexpr):
    i = tl.program_id(0) * B + tl.arange(0, B)
    valid = i < N
    if WINDOW:
        valid &= i >= 8 * tl.load(Start)
    expert = tl.load(Route + i, valid, 0)
    rank = tl.load(Rank + i, valid, 0)
    dest = tl.load(Indptr + expert) + rank
    tl.store(Pairs + dest, i, valid)
    tl.store(Inverse + i, dest, valid)


@tr.jit
def gather_rows(
    A, S, Pairs, Indptr, Sorted, SortedS, K: tl.constexpr, WIDE: tl.constexpr, BR: tl.constexpr
):
    dest = tl.program_id(0) * BR + tl.arange(0, BR)
    active = dest < tl.load(Indptr + 256)
    pair = tl.load(Pairs + dest, active, -1)
    src = tl.maximum(pair, 0) // 8
    d = tl.arange(0, K)
    x = tl.load(A + src[:, None] * K + d[None, :], pair[:, None] >= 0, 0)
    row = dest.to(tl.int64) if WIDE else dest
    tl.store(Sorted + row[:, None] * K + d[None, :], x, active[:, None])
    g = tl.arange(0, K // 128)
    scale = tl.load(S + src[:, None] * (K // 128) + g[None, :], pair[:, None] >= 0, 1.0)
    tl.store(SortedS + dest[:, None] * (K // 128) + g[None, :], scale, active[:, None])


def grouped_experts(
    activation,
    routes,
    weights,
    *,
    align,
    mma_sm,
    rounded,
    fp8_up,
    grouped_glu,
    backend="trtllm",
    start=None,
):
    x, sx = activation[:2]
    m, k = x.shape
    size = m * 8
    capacity = tr.cdiv(size, align) * align + 256 * (align - 1)
    count = torch.zeros(256, device=x.device, dtype=torch.int32)
    rank = torch.empty_like(routes)
    indptr = torch.empty(257, device=x.device, dtype=torch.int32)
    pairs = torch.empty(capacity, device=x.device, dtype=torch.int32)
    inverse = torch.empty_like(routes)
    first = routes if start is None else start
    kernels.count_routes[(tr.cdiv(size, 128),)](
        routes, count, rank, first, size, 128, start is not None
    )
    offsets[(1,)](count, indptr, pairs, align, num_warps=4)
    scatter[(tr.cdiv(size, 128),)](
        routes, rank, indptr, pairs, inverse, first, size, 128, start is not None
    )
    packed = torch.empty((capacity, k), device=x.device, dtype=x.dtype)
    scales = torch.empty((capacity, k // 128), device=x.device, dtype=torch.float32)
    gather_rows[(tr.cdiv(capacity, 8),)](
        x.view(torch.uint8),
        sx,
        pairs,
        indptr,
        packed.view(torch.uint8),
        scales,
        k,
        capacity * k >= 2**31,
        8,
        num_warps=4,
    )

    def project(a, scale, parameter):
        weight, weight_scale = parameter
        if m <= 1536 and align == 4 and mma_sm == 1 and backend == "trtllm":
            output = grouped_fp8_runtime.project(a, weight[:256], scale, weight_scale[:256], indptr)
            if output is not None:
                return output
        return group_gemm_fp8_nt_groupwise(
            a,
            weight[:256],
            scale,
            weight_scale[:256],
            indptr,
            scale_major_mode="K",
            mma_sm=mma_sm,
            out_dtype=torch.bfloat16,
            backend=backend,
            segment_alignment=align,
        )

    up = project(packed, scales, weights.up)
    if grouped_glu and not fp8_up:
        mid, mid_scale = kernels.grouped_activation(up, indptr[256:], rounded)
    else:
        mid, mid_scale = kernels.activation(up, False, rounded, fp8_up, limit=indptr[256:])
    down = project(mid, mid_scale, weights.down)
    return down, inverse

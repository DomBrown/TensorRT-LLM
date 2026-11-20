# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Map group-128 power-of-two FP8 scales to Blackwell's MX scale layout."""

import torch
import triton as tr
import triton.language as tl
from flashinfer import mm_mxfp8


@tr.jit
def pack_scales(S, Output, M, K: tl.constexpr, ROW_GROUP: tl.constexpr, B: tl.constexpr):
    i = tl.program_id(0) * B + tl.arange(0, B)
    columns: tl.constexpr = K // 32
    r = i // columns
    c = i % columns
    off = (r // 128) * (128 * columns) + (c // 4) * 512 + (r % 32) * 16 + (r // 32 % 4) * 4 + c % 4
    scale = tl.load(S + (r // ROW_GROUP) * (K // 128) + c // 4, r < M, 1.0)
    exponent = (scale.to(tl.int32, bitcast=True) >> 23) & 255
    tl.store(Output + off, exponent)


def swizzle(scales, rows, width, row_group=1):
    size = tr.cdiv(rows, 128) * 128 * (width // 32)
    encoded = torch.empty((size,), dtype=torch.uint8, device=scales.device)
    pack_scales[(tr.cdiv(size, 512),)](scales, encoded, rows, width, row_group, 512, num_warps=4)
    return encoded


def matmul(activation, weight, weight_scales, *, backend=None):
    q, scales = activation[:2]
    n, width = weight.shape
    m = q.shape[0]
    if backend is None:
        backend = "cutedsl_low_latency" if m <= 8 and n >= 4096 else "cute-dsl"
    encoded = activation[2] if len(activation) == 3 else swizzle(scales, m, width)
    return mm_mxfp8(q, weight.T, encoded, weight_scales, backend=backend)

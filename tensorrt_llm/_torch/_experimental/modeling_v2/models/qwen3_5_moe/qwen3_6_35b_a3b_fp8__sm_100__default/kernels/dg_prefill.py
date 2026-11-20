# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Pack scales for the bundled DeepGEMM SM100 FP8 projection."""

import torch
import triton as tr
import triton.language as tl

from tensorrt_llm.deep_gemm import fp8_gemm_nt


@tr.jit(do_not_specialize=["M"])
def pack(
    Scales, Packed, M, PITCH, K_GROUPS: tl.constexpr, ROW_GROUP: tl.constexpr, B: tl.constexpr
):
    row = tl.program_id(0) * B + tl.arange(0, B)
    group = tl.program_id(1)
    bits = tl.full((B,), 0, tl.uint32)
    for byte in tl.static_range(4):
        k = group * 4 + byte
        s = tl.load(Scales + (row // ROW_GROUP) * K_GROUPS + k, (row < M) & (k < K_GROUPS), 1.0)
        exponent = (s.to(tl.uint32, bitcast=True) >> 23) & 255
        bits |= exponent << (8 * byte)
    tl.store(Packed + group * PITCH + row, bits, row < M)


def encode(scales, rows, width, row_group=1):
    words = tr.cdiv(width, 512)
    pitch = tr.cdiv(rows, 4) * 4
    storage = torch.empty((words, pitch), device=scales.device, dtype=torch.int32)
    pack[(tr.cdiv(rows, 128), words)](
        scales, storage, rows, pitch, width // 128, row_group, 128, num_warps=4
    )
    return storage[:, :rows].T


def project(activation, weight, scales):
    a, a_scale = activation[:2]
    m, k = a.shape
    n = weight.shape[0]
    encoded = encode(a_scale, m, k)
    out = torch.empty((m, n), dtype=torch.bfloat16, device=a.device)
    fp8_gemm_nt((a, encoded), (weight, scales), out, recipe=(1, 1, 128))
    return out

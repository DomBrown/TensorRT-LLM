# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""BF16 projections with fixed per-row FP32 accumulation."""

import triton as tr
import triton.language as tl


@tr.jit(do_not_specialize=["rows"])
def project(
    X,
    W,
    Y,
    rows,
    N: tl.constexpr,
    K: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
):
    """Use the same K tile and reduction order for solo and packed rows."""
    row = tl.program_id(0) * BM + tl.arange(0, BM)
    column = tl.program_id(1) * BN + tl.arange(0, BN)
    offset = tl.arange(0, BK)
    total = tl.full((BM, BN), 0.0, tl.float32)
    for block in range(K // BK):
        k = block * BK + offset
        a = tl.load(X + row[:, None] * K + k[None, :], row[:, None] < rows, 0)
        b = tl.load(W + column[None, :] * K + k[:, None], column[None, :] < N, 0)
        total = tl.dot(a, b, total)
    tl.store(
        Y + row[:, None] * N + column[None, :], total, (row[:, None] < rows) & (column[None, :] < N)
    )

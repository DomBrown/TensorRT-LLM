# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""SM100 CUTLASS block-FP8 kernels for wide prefills."""

from functools import lru_cache
from pathlib import Path

import torch
import triton
import triton.language as tl

from . import kernels as ops


@lru_cache(maxsize=1)
def _kernels():
    from flashinfer.jit import env as jit_env
    from flashinfer.jit import gen_jit_spec

    return gen_jit_spec(
        "modeling_v2_qwen_sm100_groupwise_fp8",
        [Path(__file__).with_name("fp8_cutlass.cu")],
        extra_include_paths=[jit_env.FLASHINFER_CSRC_DIR],
        extra_cuda_cflags=["-gencode=arch=compute_100a,code=sm_100a"],
    ).build_and_load()


@lru_cache(maxsize=640)
def _mn_major_scales(scales):
    return scales.transpose(-1, -2).contiguous().float()


@triton.jit(do_not_specialize=["CAP"])
def _pack_expert_rows(A, S, ROUTES, SORTED, START, EXTENT, PACK, PS, PERM, CAP, K: tl.constexpr):
    sorted_index = tl.program_id(0)
    pair = tl.load(SORTED + sorted_index).to(tl.int32)
    expert = tl.load(ROUTES + pair)
    start = tl.load(START + expert)
    destination = tl.load(EXTENT + expert) + sorted_index - start
    row = pair // 9
    c = tl.arange(0, K)
    groups = tl.arange(0, K // 128)
    a = tl.load(A + row * K + c)
    scale = tl.load(S + row * (K // 128) + groups)
    tl.store(PACK + destination * K + c, a)
    tl.store(PS + groups * CAP + destination, scale)
    tl.store(PERM + pair, destination)


@triton.jit(do_not_specialize=["ROWS"])
def _scan_expert_counts(COUNTS, EXTENT, ROWS):
    e = tl.arange(0, 512)
    count = tl.load(COUNTS + e, e < 256, 0)
    count = tl.where(e == 256, ROWS, count)
    padded = tl.cdiv(count, 4) * 4
    tl.store(EXTENT, 0)
    tl.store(EXTENT + 1 + e, tl.cumsum(padded, 0), e < 257)


@triton.jit(do_not_specialize=["CAP"])
def _pack_scattered_rows(A, S, ROUTES, SLOTS, EXTENT, PACK, PS, PERM, CAP, K: tl.constexpr):
    row = tl.program_id(0)
    i = tl.arange(0, K)
    g = tl.arange(0, K // 128)
    a = tl.load(A + row * K + i)
    scale = tl.load(S + row * (K // 128) + g)
    for j in tl.static_range(9):
        slot = row * 9 + j
        expert = tl.load(ROUTES + slot).to(tl.int32)
        dst = tl.load(EXTENT + expert) + tl.load(SLOTS + slot)
        tl.store(PACK + dst * K + i, a)
        tl.store(PS + g * CAP + dst, scale)
        tl.store(PERM + slot, dst)


@triton.jit(do_not_specialize=["CAP", "ROWS"])
def _clear_expert_padding(PACK, PS, COUNTS, EXTENT, CAP, ROWS, K: tl.constexpr):
    expert = tl.program_id(0)
    count = tl.where(expert == 256, ROWS, tl.load(COUNTS + expert, expert < 256, 0))
    padding = (4 - count % 4) % 4
    if padding > 0:
        start = tl.load(EXTENT + expert) + count
        c = tl.arange(0, K)
        g = tl.arange(0, K // 128)
        for j in tl.static_range(3):
            tl.store(PACK + (start + j) * K + c, 0.0, j < padding)
            tl.store(PS + g * CAP + start + j, 0.0, j < padding)


class PrefillGemm:
    def __init__(self, device):
        self.kernels = _kernels()
        self.multiprocessors = torch.cuda.get_device_properties(device).multi_processor_count
        self.workspace = torch.empty(64 << 20, dtype=torch.uint8, device=device)
        self.descriptors = torch.empty(8 << 20, dtype=torch.uint8, device=device)

    def dense(self, a, scale, weight, weight_scale):
        out = torch.empty((a.shape[0], weight.shape[0]), dtype=torch.bfloat16, device=a.device)
        self.kernels.dense(
            a, weight, scale.T.contiguous(), _mn_major_scales(weight_scale), out, self.workspace
        )
        return out

    def _grouped(self, a, mn_scale, weight, weight_scale, extent, small=False):
        out = torch.empty((a.shape[0], weight.shape[-2]), dtype=torch.bfloat16, device=a.device)
        gemm = self.kernels.grouped_small if small else self.kernels.grouped
        gemm(
            a,
            weight,
            mn_scale,
            _mn_major_scales(weight_scale),
            out,
            extent,
            self.descriptors,
            self.workspace,
        )
        return out

    def expert_in(self, a, scale, up, routes, schedule, slots=None, counts=None):
        rows, k = a.shape
        cap = triton.cdiv(rows * 9 + 257 * 3, 4) * 4
        allocate = torch.empty if slots is not None else torch.zeros
        packed = allocate((cap, k), dtype=a.dtype, device=a.device)
        packed_scale = allocate((k // 128, cap), dtype=torch.float32, device=a.device)
        permutation = torch.empty(routes.numel(), dtype=torch.int32, device=a.device)
        if slots is not None:
            extent = torch.empty(258, dtype=torch.int32, device=a.device)
            _scan_expert_counts[(1,)](counts, extent, rows, num_warps=4)
            _clear_expert_padding[(257,)](
                packed,
                packed_scale,
                counts,
                extent,
                cap,
                rows,
                k,
                num_warps=4,
            )
            _pack_scattered_rows[(rows,)](
                a,
                scale,
                routes,
                slots,
                extent,
                packed,
                packed_scale,
                permutation,
                cap,
                k,
                num_warps=4,
            )
        else:
            sorted_rows, starts, ends, _ = schedule
            counts = ends - starts[:257]
            padded = triton.cdiv(counts, 4) * 4
            zeros = torch.zeros(1, device=a.device, dtype=torch.int32)
            extent = torch.cat((zeros, padded.cumsum(0, dtype=torch.int32)))
            _pack_expert_rows[(routes.numel(),)](
                a,
                scale,
                routes,
                sorted_rows,
                starts,
                extent,
                packed,
                packed_scale,
                permutation,
                cap,
                k,
                num_warps=4,
            )
        out = self._grouped(packed, packed_scale, *up, extent, small=rows < 4096)
        return out, extent, permutation

    def experts(
        self, a, scale, up, down, routes, schedule, slots=None, counts=None, *, defer_gather=False
    ):
        gate_up, extent, permutation = self.expert_in(
            a,
            scale,
            up,
            routes,
            schedule,
            slots,
            counts,
        )
        a, scale = ops.quantize(gate_up, "swiglu", mn_major=True)
        result = self._grouped(
            a, scale.T.contiguous(), *down, extent, small=routes.shape[0] <= 4096
        )
        if defer_gather:
            return result, permutation
        return result.index_select(0, permutation), None

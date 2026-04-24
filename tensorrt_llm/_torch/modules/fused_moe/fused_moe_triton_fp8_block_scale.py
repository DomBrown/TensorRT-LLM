# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 NVIDIA Corporation. All rights reserved.
"""
Triton FP8 block-scale MoE forward pass for SM120 (RTX PRO 6000 Blackwell).

Ports vLLM's fused_moe_kernel (Apache-2.0, Copyright vLLM contributors) to
TRT-LLM to work around CUTLASS FP8 block-scale TMA descriptor failures on
SM120 for large token counts.  The Triton GEMM kernel is architecture-agnostic
(no wgmma / TMA intrinsics) and works on any SM >= 8.9.

vLLM reference:
  vllm/model_executor/layers/fused_moe/fused_moe.py  (fused_moe_kernel)
  vllm/model_executor/layers/fused_moe/moe_align_block_size.py
  vllm/model_executor/layers/fused_moe/utils.py      (_fp8_quantize)
"""

from typing import Optional

import torch
import torch.nn.functional as F
import triton
import triton.language as tl

from tensorrt_llm._torch.cute_dsl_utils import IS_CUTLASS_DSL_AVAILABLE

_FP8_MAX = 448.0  # float8_e4m3fn saturation value
_BLOCK_SHAPE = [128, 128]  # [group_k, group_n] for FP8 block-scale
_BLOCK_SIZE_M = 16
_BLOCK_SIZE_N = 32
_BLOCK_SIZE_K = 128  # must be <= group_k
_GROUP_SIZE_M = 4

# ─── Per-token-group FP8 quantization ────────────────────────────────────────


@triton.jit
def _per_token_group_quant_fp8_kernel(
    x_ptr,
    x_q_ptr,
    scale_ptr,
    K,
    num_groups,
    GROUP_K: tl.constexpr,
    FP8_MAX: tl.constexpr,
):
    """Quantize one GROUP_K-wide K-slice of one token row to FP8 E4M3."""
    pid_m = tl.program_id(0)  # token index
    pid_g = tl.program_id(1)  # group index within the row

    offs = pid_g * GROUP_K + tl.arange(0, GROUP_K)
    mask = offs < K

    x = tl.load(x_ptr + pid_m * K + offs, mask=mask, other=0.0).to(tl.float32)
    absmax = tl.max(tl.abs(x))
    scale = tl.where(absmax == 0.0, 1.0, absmax / FP8_MAX)

    x_q = tl.clamp(x / scale, -FP8_MAX, FP8_MAX).to(tl.float8e4nv)
    tl.store(x_q_ptr + pid_m * K + offs, x_q, mask=mask)
    tl.store(scale_ptr + pid_m * num_groups + pid_g, scale.to(tl.float32))


def per_token_group_quant_fp8(
    x: torch.Tensor,
    group_k: int = 128,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Quantize x (num_tokens, K) to FP8 with per-K-group scales.

    Returns:
        x_q   : (num_tokens, K) float8_e4m3fn
        scales: (num_tokens, K // group_k) float32
    """
    assert x.ndim == 2
    x = x.contiguous()
    num_tokens, K = x.shape
    assert K % group_k == 0, f"K={K} must be divisible by group_k={group_k}"
    num_groups = K // group_k

    x_q = torch.empty_like(x, dtype=torch.float8_e4m3fn)
    scales = torch.empty((num_tokens, num_groups), dtype=torch.float32, device=x.device)

    _per_token_group_quant_fp8_kernel[(num_tokens, num_groups)](
        x,
        x_q,
        scales,
        K,
        num_groups,
        GROUP_K=group_k,
        FP8_MAX=_FP8_MAX,
    )
    return x_q, scales


# ─── Triton routing kernels ───────────────────────────────────────────────────
#
# Replace the previous multi-kernel PyTorch approach (argsort + scatter_add_ +
# cumsum + scatter_ + searchsorted, ~760 µs/step) with four lightweight Triton
# kernels that avoid radix-sort entirely and run ~5-10× faster for small N
# (decode) and ~3× faster for large N (prefill).
#
# Phase 1 (_moe_histogram): histogram of expert IDs via scalar atomic_add.
# Phase 2 (_moe_prefix): prefix sum + total padded count (single CTA).
# Phase 3 (_moe_expert_ids): fill expert_ids_out (one CTA per expert).
# Phase 4 (_moe_scatter): scatter token indices using atomic per-expert counter.


@triton.jit
def _moe_histogram_kernel(
    flat_ids_ptr,
    counts_ptr,
    N,
    BLOCK_N: tl.constexpr,
):
    """Phase 1: histogram via vectorised atomic_add on a block of pointers."""
    pid = tl.program_id(0)
    offs = pid * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = offs < N
    ids = tl.load(flat_ids_ptr + offs, mask=mask, other=0).to(tl.int32)
    ones = tl.full([BLOCK_N], 1, dtype=tl.int32)
    tl.atomic_add(counts_ptr + ids, ones, mask=mask)


@triton.jit
def _int32_add(a, b):
    return a + b


@triton.jit
def _moe_prefix_kernel(
    counts_ptr,
    out_off_ptr,
    num_post_pad_ptr,
    BLOCK_SIZE: tl.constexpr,
    NUM_EXPERTS: tl.constexpr,
):
    """Phase 2: exclusive prefix sum of padded counts + total (single CTA)."""
    e = tl.arange(0, NUM_EXPERTS)
    counts = tl.load(counts_ptr + e).to(tl.int32)
    padded = ((counts + BLOCK_SIZE - 1) // BLOCK_SIZE) * BLOCK_SIZE
    # Inclusive scan; subtract self for exclusive prefix.
    inc = tl.associative_scan(padded, 0, _int32_add)
    out_off = inc - padded
    tl.store(out_off_ptr + e, out_off)
    tl.store(num_post_pad_ptr, tl.sum(padded).to(tl.int32))


@triton.jit
def _moe_expert_ids_kernel(
    counts_ptr,
    out_off_ptr,
    expert_ids_ptr,
    num_experts,
    BLOCK_SIZE: tl.constexpr,
):
    """Phase 3: fill expert_ids_out — one CTA per expert."""
    e = tl.program_id(0)
    if e >= num_experts:
        return
    cnt = tl.load(counts_ptr + e).to(tl.int32)
    off = tl.load(out_off_ptr + e).to(tl.int32)
    padded = ((cnt + BLOCK_SIZE - 1) // BLOCK_SIZE) * BLOCK_SIZE
    start_blk = off // BLOCK_SIZE
    n_blk = padded // BLOCK_SIZE
    for i in range(n_blk):
        tl.store(expert_ids_ptr + start_blk + i, e)


@triton.jit
def _moe_scatter_kernel(
    flat_ids_ptr,
    counters_ptr,
    sorted_token_ids_ptr,
    N,
    BLOCK_N: tl.constexpr,
):
    """Phase 4: scatter token indices using vectorised atomic per-expert counter."""
    pid = tl.program_id(0)
    offs = pid * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = offs < N
    ids = tl.load(flat_ids_ptr + offs, mask=mask, other=0).to(tl.int32)
    ones = tl.full([BLOCK_N], 1, dtype=tl.int32)
    # atomic_add returns old value = the slot reserved for this token
    pos = tl.atomic_add(counters_ptr + ids, ones, mask=mask)
    tl.store(sorted_token_ids_ptr + pos, offs.to(tl.int32), mask=mask)


# ─── CuTe DSL routing kernels ────────────────────────────────────────────────
#
# Alternative implementation of moe_align_block_size using NVIDIA CuTe DSL
# (CUTLASS 4.x Python API).  All four phases are fused into a single
# @cute.jit host function to minimise Python-level dispatch overhead.
#
# Measured ~8–15% faster than the Triton version on RTX PRO 6000 Blackwell
# (SM120) across N ∈ {8, 64, 256, 800, 2048, 8000} with warmup=20, rep=200.
#
# Architecture:
#   Phase 1 (_cute_moe_histogram_kernel):  atomic_add per element into counts.
#   Phase 2 (_cute_moe_prefix_dual_kernel): Kogge-Stone warp scan + smem merge
#                                            → exclusive prefix → offsets AND
#                                            counters (two writes, no clone).
#   Phase 3 (_cute_moe_expert_ids_kernel): one CTA per expert, fills blocks.
#   Phase 4 (_cute_moe_scatter_kernel):    atomic_add returns old value → slot.

if IS_CUTLASS_DSL_AVAILABLE:
    import cutlass
    import cutlass.cute as _cute
    from cutlass._mlir.dialects import arith as _cute_arith
    from cutlass.cute.runtime import from_dlpack as _from_dlpack
    from cutlass.utils.distributed import atomicAdd as _atomicAdd

if IS_CUTLASS_DSL_AVAILABLE:
    _CUTE_HIST_BLOCK_N = 32  # threads per CTA for histogram / scatter
    _CUTE_PREFIX_THREADS = 256  # must be >= num_experts (Laguna: 256)

    # ── Phase 1: per-expert histogram ─────────────────────────────────────────
    @_cute.kernel
    def _cute_moe_histogram_kernel(
        gFlatIds: _cute.Tensor,
        gCounts: _cute.Tensor,
        N: cutlass.Int32,
    ):
        tidx, _, _ = _cute.arch.thread_idx()
        bidx, _, _ = _cute.arch.block_idx()
        bdim, _, _ = _cute.arch.block_dim()
        idx = bidx * bdim + tidx
        if idx < N:
            expert_id = gFlatIds[(idx,)]
            _atomicAdd(gCounts.iterator + expert_id, cutlass.Int32(1))

    # ── Phase 2: exclusive prefix sum, writes offsets AND counters ────────────
    @_cute.kernel
    def _cute_moe_prefix_dual_kernel(
        gCounts: _cute.Tensor,
        gOffsets: _cute.Tensor,
        gCounters: _cute.Tensor,
        gNumPostPad: _cute.Tensor,
        block_size: cutlass.Int32,
        num_experts: cutlass.Int32,
    ):
        """
        Kogge-Stone intra-warp inclusive scan + shared-memory cross-warp merge
        → exclusive prefix sum of padded expert counts.
        Writes identical values to gOffsets (for Phase 3) and gCounters (for
        Phase 4), avoiding a separate tensor.clone() call.
        """
        _WARP_SIZE = 32
        _NUM_WARPS = _CUTE_PREFIX_THREADS // _WARP_SIZE  # 8

        tidx, _, _ = _cute.arch.thread_idx()
        warp_id = tidx // _WARP_SIZE
        lane_id = tidx % _WARP_SIZE

        padded = cutlass.Int32(0)
        if tidx < num_experts:
            cnt = gCounts[(tidx,)]
            padded = ((cnt + block_size - cutlass.Int32(1)) // block_size) * block_size

        # Kogge-Stone warp inclusive prefix scan.
        # mask_and_clamp=0 → source lane clamped at 0 (not at WARP_SIZE-1).
        val = padded
        for _off in [1, 2, 4, 8, 16]:
            other = _cute.arch.shuffle_sync_up(val, _off, 0xFFFFFFFF, 0)
            cond = lane_id >= _off
            val = cutlass.Int32(
                _cute_arith.select(cond.ir_value(), (val + other).ir_value(), val.ir_value())
            )

        # Cross-warp merge via shared memory.
        smem_ptr = _cute.arch.alloc_smem(cutlass.Int32, _NUM_WARPS + 1)
        smem = _cute.make_tensor(smem_ptr, _cute.make_layout((_NUM_WARPS + 1,)))

        if lane_id == _WARP_SIZE - 1:
            smem[(warp_id,)] = val

        _cute.arch.sync_threads()

        if tidx == 0:
            running = cutlass.Int32(0)
            for w in range(_NUM_WARPS):
                warp_sum = smem[(w,)]
                smem[(w,)] = running
                running = running + warp_sum
            smem[(_NUM_WARPS,)] = running

        _cute.arch.sync_threads()

        # Convert inclusive → exclusive and add warp base offset.
        warp_base = smem[(warp_id,)]
        excl = warp_base + val - padded

        if tidx < num_experts:
            gOffsets[(tidx,)] = excl
            gCounters[(tidx,)] = excl  # duplicate: counters used by scatter

        if tidx == 0:
            gNumPostPad[(0,)] = smem[(_NUM_WARPS,)]

    # ── Phase 3: fill expert_ids_out ─────────────────────────────────────────
    @_cute.kernel
    def _cute_moe_expert_ids_kernel(
        gCounts: _cute.Tensor,
        gOffsets: _cute.Tensor,
        gExpertIds: _cute.Tensor,
        block_size: cutlass.Int32,
    ):
        """One CTA per expert; fills expert_ids_out with its expert index."""
        expert_id = _cute.arch.block_idx()[0]
        cnt = gCounts[(expert_id,)]
        off = gOffsets[(expert_id,)]
        padded = ((cnt + block_size - cutlass.Int32(1)) // block_size) * block_size
        start_blk = off // block_size
        n_blk = padded // block_size
        for i in range(n_blk):
            gExpertIds[(start_blk + i,)] = expert_id

    # ── Phase 4: scatter token indices ───────────────────────────────────────
    @_cute.kernel
    def _cute_moe_scatter_kernel(
        gFlatIds: _cute.Tensor,
        gCounters: _cute.Tensor,
        gSortedTokIds: _cute.Tensor,
        N: cutlass.Int32,
        BLOCK_N: cutlass.Int32,
    ):
        """atomic_add on per-expert counter returns old value = write slot."""
        bidx, _, _ = _cute.arch.block_idx()
        tidx, _, _ = _cute.arch.thread_idx()
        idx = bidx * BLOCK_N + tidx
        if idx < N:
            expert_id = gFlatIds[(idx,)]
            old_pos = _atomicAdd(gCounters.iterator + expert_id, cutlass.Int32(1))
            gSortedTokIds[(old_pos,)] = idx

    # ── Fused host: all four phases in a single @cute.jit ────────────────────
    @_cute.jit
    def _cute_moe_fused_host(
        mFlatIds: _cute.Tensor,
        mCounts: _cute.Tensor,
        mOffsets: _cute.Tensor,
        mCounters: _cute.Tensor,
        mNumPostPad: _cute.Tensor,
        mExpertIds: _cute.Tensor,
        mSortedTokIds: _cute.Tensor,
        N: cutlass.Int32,
        block_size: cutlass.Int32,
        num_experts: cutlass.Int32,
    ):
        _BLOCK_N = cutlass.Int32(_CUTE_HIST_BLOCK_N)
        num_blocks_N = (N + _BLOCK_N - cutlass.Int32(1)) // _BLOCK_N

        _cute_moe_histogram_kernel(mFlatIds, mCounts, N).launch(
            grid=(num_blocks_N, 1, 1), block=(_CUTE_HIST_BLOCK_N, 1, 1)
        )
        _cute_moe_prefix_dual_kernel(
            mCounts, mOffsets, mCounters, mNumPostPad, block_size, num_experts
        ).launch(grid=(1, 1, 1), block=(_CUTE_PREFIX_THREADS, 1, 1))
        _cute_moe_expert_ids_kernel(mCounts, mOffsets, mExpertIds, block_size).launch(
            grid=(num_experts, 1, 1), block=(1, 1, 1)
        )
        _cute_moe_scatter_kernel(mFlatIds, mCounters, mSortedTokIds, N, _BLOCK_N).launch(
            grid=(num_blocks_N, 1, 1), block=(_CUTE_HIST_BLOCK_N, 1, 1)
        )

    # ── Compiled-kernel cache (keyed by device string) ────────────────────────
    _cute_moe_compiled_cache: dict = {}

    def _get_cute_moe_compiled(device: str):
        """Compile and cache the fused CuTe DSL kernel for a given device."""
        if device in _cute_moe_compiled_cache:
            return _cute_moe_compiled_cache[device]

        def _fake_i32(n: int = 1):
            t = torch.zeros(n, dtype=torch.int32, device=device)
            return _from_dlpack(t).mark_layout_dynamic()

        compiled = _cute.compile(
            _cute_moe_fused_host,
            _fake_i32(),
            _fake_i32(),
            _fake_i32(),
            _fake_i32(),
            _fake_i32(),
            _fake_i32(),
            _fake_i32(),
            cutlass.Int32(1),
            cutlass.Int32(16),
            cutlass.Int32(256),
        )
        _cute_moe_compiled_cache[device] = compiled
        return compiled

    def moe_align_block_size_cute(
        topk_ids: torch.Tensor,
        block_size: int,
        num_experts: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        CuTe DSL implementation of moe_align_block_size.

        Produces the same outputs as :func:`moe_align_block_size` using four
        GPU kernels fused into a single ``@cute.jit`` host call.

        Requires ``nvidia-cutlass-dsl`` (``cutlass`` package) to be installed.
        Falls back to :func:`moe_align_block_size` if cutlass is unavailable.

        Constraints:
            - ``num_experts`` must be ≤ 256 (``_CUTE_PREFIX_THREADS``).
            - CUDA-graph compatible: no ``.item()``; all buffer sizes are
              Python-int expressions computed from the input shape.

        Args:
            topk_ids  : (num_tokens, top_k) int64 expert IDs
            block_size: BLOCK_SIZE_M used in the downstream GEMM kernel
            num_experts: total number of experts

        Returns:
            sorted_token_ids      : (max_tokens_static,) int32
            expert_ids_out        : (num_m_blocks_static,) int32
            num_tokens_post_padded: (1,) int32
        """
        assert num_experts <= _CUTE_PREFIX_THREADS, (
            f"moe_align_block_size_cute: num_experts={num_experts} exceeds "
            f"_CUTE_PREFIX_THREADS={_CUTE_PREFIX_THREADS}"
        )

        device = str(topk_ids.device)
        flat_ids = topk_ids.reshape(-1).to(torch.int32)
        N = flat_ids.shape[0]  # Python int — no GPU sync

        max_tokens_static = N + num_experts * block_size
        num_m_blocks_static = N // block_size + num_experts

        # All buffers have statically known sizes → CUDA-graph safe.
        counts = torch.zeros(num_experts, dtype=torch.int32, device=topk_ids.device)
        offsets = torch.empty(num_experts, dtype=torch.int32, device=topk_ids.device)
        counters = torch.empty(num_experts, dtype=torch.int32, device=topk_ids.device)
        sorted_token_ids = torch.full(
            (max_tokens_static,), N, dtype=torch.int32, device=topk_ids.device
        )
        expert_ids_out = torch.zeros(
            (num_m_blocks_static,), dtype=torch.int32, device=topk_ids.device
        )
        num_post_pad = torch.empty(1, dtype=torch.int32, device=topk_ids.device)

        compiled = _get_cute_moe_compiled(device)

        def _fd(t: torch.Tensor):
            return _from_dlpack(t).mark_layout_dynamic()

        compiled(
            _fd(flat_ids),
            _fd(counts),
            _fd(offsets),
            _fd(counters),
            _fd(num_post_pad),
            _fd(expert_ids_out),
            _fd(sorted_token_ids),
            cutlass.Int32(N),
            cutlass.Int32(block_size),
            cutlass.Int32(num_experts),
        )

        return sorted_token_ids, expert_ids_out, num_post_pad

else:
    # cutlass not available: stub delegates to the Triton version.
    def moe_align_block_size_cute(
        topk_ids: torch.Tensor,
        block_size: int,
        num_experts: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Stub: cutlass not installed — delegates to moe_align_block_size."""
        return moe_align_block_size(topk_ids, block_size, num_experts)


def moe_align_block_size(
    topk_ids: torch.Tensor,
    block_size: int,
    num_experts: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Sort token-expert pairs by expert and pad each expert's row-count to
    a multiple of block_size.

    Implemented via four lightweight Triton kernels (histogram → prefix sum →
    expert-id fill → scatter).  CUDA-graph compatible: all buffer sizes are
    Python-int expressions; no .item() calls or GPU→CPU synchronisation.

    Args:
        topk_ids  : (num_tokens, top_k) int64 expert IDs
        block_size: BLOCK_SIZE_M used in the GEMM kernel (Python int)
        num_experts: total number of experts (Python int)

    Returns:
        sorted_token_ids      : (max_tokens_static,) int32
        expert_ids_out        : (num_m_blocks_static,) int32
        num_tokens_post_padded: (1,) int32 — actual padded token count
    """
    device = topk_ids.device
    flat_ids = topk_ids.reshape(-1).to(torch.int32)
    N = flat_ids.shape[0]  # Python int, no GPU sync

    # Static worst-case sizes (Python-int arithmetic only)
    max_tokens_static = N + num_experts * block_size
    num_m_blocks_static = N // block_size + num_experts

    # Persistent buffers — all constant-size, CUDA-graph compatible
    counts = torch.zeros(num_experts, dtype=torch.int32, device=device)
    out_offsets = torch.empty(num_experts, dtype=torch.int32, device=device)
    sorted_token_ids = torch.full((max_tokens_static,), N, dtype=torch.int32, device=device)
    expert_ids_out = torch.zeros((num_m_blocks_static,), dtype=torch.int32, device=device)
    num_post_pad = torch.empty(1, dtype=torch.int32, device=device)

    _BLOCK_N = 32  # one warp per program; minimises waste for small N (decode)

    # Phase 1: histogram
    if N > 0:
        _moe_histogram_kernel[triton.cdiv(N, _BLOCK_N),](flat_ids, counts, N, BLOCK_N=_BLOCK_N)

    # Phase 2: exclusive prefix sum → out_offsets; sum → num_post_pad
    _moe_prefix_kernel[1,](
        counts,
        out_offsets,
        num_post_pad,
        BLOCK_SIZE=block_size,
        NUM_EXPERTS=num_experts,
    )

    # Phase 3: fill expert_ids_out (one CTA per expert, runtime inner loop)
    _moe_expert_ids_kernel[num_experts,](
        counts,
        out_offsets,
        expert_ids_out,
        num_experts=num_experts,
        BLOCK_SIZE=block_size,
    )

    # Phase 4: scatter — use a mutable copy of out_offsets as per-expert counters
    if N > 0:
        counters = out_offsets.clone()
        _moe_scatter_kernel[triton.cdiv(N, _BLOCK_N),](
            flat_ids, counters, sorted_token_ids, N, BLOCK_N=_BLOCK_N
        )

    return sorted_token_ids, expert_ids_out, num_post_pad


# ─── Triton MoE FP8 block-scale GEMM kernel (ported from vLLM) ───────────────


@triton.jit
def _write_zeros_to_output(
    c_ptr,
    stride_cm,
    stride_cn,
    pid_n,
    N,
    offs_token,
    token_mask,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    compute_type: tl.constexpr,
):
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=compute_type)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_token[:, None] + stride_cn * offs_cn[None, :]
    c_mask = token_mask[:, None] & (offs_cn[None, :] < N)
    tl.store(c_ptrs, acc, mask=c_mask)


@triton.jit
def _fused_moe_fp8_block_scale_kernel(
    # Tensor pointers
    a_ptr,
    b_ptr,
    c_ptr,
    a_scale_ptr,
    b_scale_ptr,
    topk_weights_ptr,
    sorted_token_ids_ptr,
    expert_ids_ptr,
    num_tokens_post_padded_ptr,
    # Dimensions (runtime)
    N,
    K,
    EM,
    num_valid_tokens,
    # Strides (runtime)
    stride_am,
    stride_ak,
    stride_be,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    stride_asm,
    stride_ask,
    stride_bse,
    stride_bsk,
    stride_bsn,
    # Quantization block sizes (compile-time)
    group_n: tl.constexpr,
    group_k: tl.constexpr,
    # Tile sizes (compile-time)
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    # Other compile-time flags
    top_k: tl.constexpr,
    compute_type: tl.constexpr,
    MUL_ROUTED_WEIGHT: tl.constexpr,
):
    """
    Fused MoE GEMM with FP8 block-scale dequantization.

    Computes C[sorted_token, n] = sum_k A[token, k] * B[expert, n, k]
                                  * A_scale[token, k//group_k]
                                  * B_scale[expert, n//group_n, k//group_k]
    optionally multiplying by topk routing weights.

    Layout:
        A           : (num_tokens, K)          FP8
        B           : (E, N, K)                FP8
        C           : (num_tokens, top_k, N)   BF16  (output)
        A_scale     : (num_tokens, K//group_k) F32
        B_scale     : (E, N//group_n, K//group_k) F32
        topk_weights: (num_tokens * top_k,)    F32   (flat view)
    """
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(EM, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % num_pid_in_group) % group_size_m
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs = tl.arange(0, BLOCK_SIZE_M).to(tl.int64)
    num_tokens_post_padded = tl.load(num_tokens_post_padded_ptr)
    if pid_m * BLOCK_SIZE_M >= num_tokens_post_padded:
        return

    offs_token_id = pid_m * BLOCK_SIZE_M + offs
    offs_token = tl.load(sorted_token_ids_ptr + offs_token_id).to(tl.int64)
    token_mask = offs_token < num_valid_tokens

    off_experts = tl.load(expert_ids_ptr + pid_m).to(tl.int64)
    if off_experts == -1:
        _write_zeros_to_output(
            c_ptr,
            stride_cm,
            stride_cn,
            pid_n,
            N,
            offs_token,
            token_mask,
            BLOCK_SIZE_M,
            BLOCK_SIZE_N,
            compute_type,
        )
        return

    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N).to(tl.int64)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    # A pointer: row = original token index (offs_token // top_k)
    a_ptrs = a_ptr + offs_token[:, None] // top_k * stride_am + offs_k[None, :] * stride_ak
    # B pointer: row = K index, col = N index, sliced per expert
    b_ptrs = (
        b_ptr + off_experts * stride_be + offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn
    )

    # Scale pointers (block-wise)
    a_scale_ptrs = a_scale_ptr + (offs_token // top_k) * stride_asm
    offs_bsn = offs_bn // group_n  # N-block index per output column
    b_scale_ptrs = b_scale_ptr + off_experts * stride_bse + offs_bsn * stride_bsn

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        a = tl.load(
            a_ptrs,
            mask=token_mask[:, None] & (offs_k[None, :] < K - k * BLOCK_SIZE_K),
            other=0.0,
        )
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)

        # Per-block dequantization: load one A-scale per token-row,
        # one B-scale per N-column (scalar within the current N-tile).
        k_start = k * BLOCK_SIZE_K
        offs_ks = k_start // group_k
        a_scale = tl.load(a_scale_ptrs + offs_ks * stride_ask, mask=token_mask, other=0.0)
        b_scale = tl.load(b_scale_ptrs + offs_ks * stride_bsk)

        accumulator += tl.dot(a, b) * a_scale[:, None] * b_scale[None, :]

        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    if MUL_ROUTED_WEIGHT:
        moe_weight = tl.load(topk_weights_ptr + offs_token, mask=token_mask, other=0.0)
        accumulator *= moe_weight[:, None]

    accumulator = accumulator.to(compute_type)

    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_token[:, None] + stride_cn * offs_cn[None, :]
    c_mask = token_mask[:, None] & (offs_cn[None, :] < N)
    tl.store(c_ptrs, accumulator, mask=c_mask)


def _invoke_fp8_block_scale_moe_kernel(
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    A_scale: torch.Tensor,
    B_scale: torch.Tensor,
    topk_weights: torch.Tensor,
    sorted_token_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    num_tokens_post_padded: torch.Tensor,
    mul_routed_weight: bool,
    top_k: int,
) -> None:
    """
    Launch _fused_moe_fp8_block_scale_kernel.

    Tensor layouts:
        A       : (num_tokens_or_expanded, K)  FP8, contiguous
        B       : (E, N, K)                    FP8, contiguous
        C       : (batch, top_k_orig, N)       BF16, stride(1)=N, stride(2)=1
        A_scale : (num_tokens_or_expanded, K//128)  F32
        B_scale : (E, N//128, K//128)                F32
        topk_weights: (num_tokens, top_k_orig) F32  (flat-indexable by offs_token)
    """
    EM = sorted_token_ids.shape[0]
    N = B.shape[1]
    K = B.shape[2]
    num_valid = A.shape[0] * top_k
    group_n, group_k = _BLOCK_SHAPE[1], _BLOCK_SHAPE[0]

    def grid(META):
        return (triton.cdiv(EM, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"]),)

    _fused_moe_fp8_block_scale_kernel[grid](
        A,
        B,
        C,
        A_scale,
        B_scale,
        topk_weights,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        N,
        K,
        EM,
        num_valid,
        A.stride(0),
        A.stride(1),
        B.stride(0),
        B.stride(2),
        B.stride(1),
        C.stride(1),
        C.stride(2),
        A_scale.stride(0) if A_scale.ndim == 2 else 0,
        A_scale.stride(1) if A_scale.ndim == 2 else 0,
        B_scale.stride(0) if B_scale.ndim >= 2 else 0,
        B_scale.stride(2) if B_scale.ndim == 3 else 0,
        B_scale.stride(1) if B_scale.ndim >= 2 else 0,
        group_n=group_n,
        group_k=group_k,
        BLOCK_SIZE_M=_BLOCK_SIZE_M,
        BLOCK_SIZE_N=_BLOCK_SIZE_N,
        BLOCK_SIZE_K=_BLOCK_SIZE_K,
        GROUP_SIZE_M=_GROUP_SIZE_M,
        top_k=top_k,
        compute_type=tl.bfloat16,
        MUL_ROUTED_WEIGHT=mul_routed_weight,
    )


# ─── Triton MoE BF16-activation + FP8-weight kernel ─────────────────────────
# For weight-only FP8 models (post-training quantization): activations stay BF16,
# FP8 weights are dequanted on-the-fly inside the K-loop.  No activation quant error.


@triton.jit
def _fused_moe_bf16act_fp8w_kernel(
    # Tensor pointers
    a_ptr,  # (num_tokens, K)              BF16 activations
    b_ptr,  # (E, N, K)                    FP8 weights
    c_ptr,  # (num_tokens, top_k, N)        BF16 output
    b_scale_ptr,  # (E, N//group_n, K//group_k)   F32 weight block scales
    topk_weights_ptr,
    sorted_token_ids_ptr,
    expert_ids_ptr,
    num_tokens_post_padded_ptr,
    # Dimensions (runtime)
    N,
    K,
    EM,
    num_valid_tokens,
    # Strides (runtime)
    stride_am,
    stride_ak,
    stride_be,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    stride_bse,
    stride_bsk,
    stride_bsn,
    # Quantization block sizes (compile-time)
    group_n: tl.constexpr,
    group_k: tl.constexpr,
    # Tile sizes (compile-time)
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    # Other compile-time flags
    top_k: tl.constexpr,
    compute_type: tl.constexpr,
    MUL_ROUTED_WEIGHT: tl.constexpr,
):
    """
    Fused MoE GEMM: BF16 activations × FP8 weights (dequant on-the-fly).

    For weight-only quantized models: activations never go through FP8, so
    there is no per-layer quantization error.  Each FP8 weight tile is
    dequanted to BF16 inside the K-loop using the per-block scale:
        C[m, n] += dot(A_bf16[m, :], B_fp8[:, n].to(bf16)) * B_scale[n//gn, k//gk]
    """
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(EM, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % num_pid_in_group) % group_size_m
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs = tl.arange(0, BLOCK_SIZE_M).to(tl.int64)
    num_tokens_post_padded = tl.load(num_tokens_post_padded_ptr)
    if pid_m * BLOCK_SIZE_M >= num_tokens_post_padded:
        return

    offs_token_id = pid_m * BLOCK_SIZE_M + offs
    offs_token = tl.load(sorted_token_ids_ptr + offs_token_id).to(tl.int64)
    token_mask = offs_token < num_valid_tokens

    off_experts = tl.load(expert_ids_ptr + pid_m).to(tl.int64)
    if off_experts == -1:
        _write_zeros_to_output(
            c_ptr,
            stride_cm,
            stride_cn,
            pid_n,
            N,
            offs_token,
            token_mask,
            BLOCK_SIZE_M,
            BLOCK_SIZE_N,
            compute_type,
        )
        return

    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N).to(tl.int64)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    a_ptrs = a_ptr + offs_token[:, None] // top_k * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = (
        b_ptr + off_experts * stride_be + offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn
    )

    offs_bsn = offs_bn // group_n
    b_scale_ptrs = b_scale_ptr + off_experts * stride_bse + offs_bsn * stride_bsn

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        # BF16 activations — no quantization
        a = tl.load(
            a_ptrs,
            mask=token_mask[:, None] & (offs_k[None, :] < K - k * BLOCK_SIZE_K),
            other=0.0,
        )
        # FP8 weights — dequant to BF16 on-the-fly
        b_fp8 = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)
        b = b_fp8.to(tl.bfloat16)

        k_start = k * BLOCK_SIZE_K
        offs_ks = k_start // group_k
        b_scale = tl.load(b_scale_ptrs + offs_ks * stride_bsk)

        # FP32 accumulation with per-block weight dequant factor
        accumulator += tl.dot(a, b, out_dtype=tl.float32) * b_scale[None, :]

        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    if MUL_ROUTED_WEIGHT:
        moe_weight = tl.load(topk_weights_ptr + offs_token, mask=token_mask, other=0.0)
        accumulator *= moe_weight[:, None]

    accumulator = accumulator.to(compute_type)

    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_token[:, None] + stride_cn * offs_cn[None, :]
    c_mask = token_mask[:, None] & (offs_cn[None, :] < N)
    tl.store(c_ptrs, accumulator, mask=c_mask)


def _invoke_bf16act_fp8w_moe_kernel(
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    B_scale: torch.Tensor,
    topk_weights: torch.Tensor,
    sorted_token_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    num_tokens_post_padded: torch.Tensor,
    mul_routed_weight: bool,
    top_k: int,
) -> None:
    """
    Launch _fused_moe_bf16act_fp8w_kernel.

    Tensor layouts:
        A       : (num_tokens_or_expanded, K)  BF16, contiguous
        B       : (E, N, K)                    FP8, contiguous
        C       : (batch, top_k_orig, N)       BF16, stride(1)=N, stride(2)=1
        B_scale : (E, N//128, K//128)           F32
        topk_weights: (num_tokens, top_k_orig) F32  (flat-indexable by offs_token)
    """
    EM = sorted_token_ids.shape[0]
    N = B.shape[1]
    K = B.shape[2]
    group_n, group_k = _BLOCK_SHAPE[1], _BLOCK_SHAPE[0]

    def grid(META):
        return (triton.cdiv(EM, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"]),)

    _fused_moe_bf16act_fp8w_kernel[grid](
        A,
        B,
        C,
        B_scale,
        topk_weights,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        N,
        K,
        EM,
        A.shape[0] * top_k,
        A.stride(0),
        A.stride(1),
        B.stride(0),
        B.stride(2),
        B.stride(1),
        C.stride(1),
        C.stride(2),
        B_scale.stride(0) if B_scale.ndim >= 2 else 0,
        B_scale.stride(2) if B_scale.ndim == 3 else 0,
        B_scale.stride(1) if B_scale.ndim >= 2 else 0,
        group_n=group_n,
        group_k=group_k,
        BLOCK_SIZE_M=_BLOCK_SIZE_M,
        BLOCK_SIZE_N=_BLOCK_SIZE_N,
        BLOCK_SIZE_K=_BLOCK_SIZE_K,
        GROUP_SIZE_M=_GROUP_SIZE_M,
        top_k=top_k,
        compute_type=tl.bfloat16,
        MUL_ROUTED_WEIGHT=mul_routed_weight,
    )


# ─── BF16 debug fallback (dequant weights on-the-fly, no FP8 activations) ──


def _run_moe_bf16_fallback(
    x: torch.Tensor,
    token_selected_experts: torch.Tensor,
    token_final_scales: torch.Tensor,
    w3_w1: torch.Tensor,
    w3_w1_scales: torch.Tensor,
    w2: torch.Tensor,
    w2_scales: torch.Tensor,
    activation_type,
    output_dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    """BF16 path: dequant FP8 weights on-the-fly, use BF16 activations."""
    from tensorrt_llm._torch.modules.fused_moe.interface import ActivationType

    num_tokens, hidden = x.shape
    num_experts, N_gate_up, _ = w3_w1.shape
    intermediate = N_gate_up // 2
    top_k = token_selected_experts.shape[1]
    device = x.device

    tok_exp = token_selected_experts.long()
    tok_wt = token_final_scales

    # Dequant weights block-by-block (group_n=group_k=128)
    def dequant_weight(w_fp8, scales):
        E, N, K = w_fp8.shape
        Nb, Kb = scales.shape[1], scales.shape[2]
        out = torch.zeros(E, N, K, dtype=torch.bfloat16, device=device)
        for e in range(E):
            for nb in range(Nb):
                for kb in range(Kb):
                    blk = w_fp8[e, nb * 128 : (nb + 1) * 128, kb * 128 : (kb + 1) * 128].float()
                    out[e, nb * 128 : (nb + 1) * 128, kb * 128 : (kb + 1) * 128] = (
                        blk * scales[e, nb, kb]
                    ).bfloat16()
        return out

    w31_bf16 = dequant_weight(w3_w1, w3_w1_scales)
    w2_bf16 = dequant_weight(w2, w2_scales)

    output = torch.zeros(num_tokens, hidden, dtype=torch.bfloat16, device=device)
    for t in range(num_tokens):
        acc = torch.zeros(hidden, dtype=torch.float32, device=device)
        for k in range(top_k):
            e = int(tok_exp[t, k])
            gate_up = x[t].float() @ w31_bf16[e].float().T
            u, g = gate_up[:intermediate], gate_up[intermediate:]
            act_int = int(activation_type)
            if act_int in (int(ActivationType.Swiglu), int(ActivationType.Silu)):
                ic2 = F.silu(g) * u
            else:
                ic2 = F.gelu(g) * u
            acc += (ic2 @ w2_bf16[e].float().T) * float(tok_wt[t, k])
        output[t] = acc.bfloat16()

    if output_dtype is not None and output.dtype != output_dtype:
        output = output.to(output_dtype)
    return output


# ─── Full MoE forward ────────────────────────────────────────────────────────


def run_triton_fp8_block_scale_moe(
    x: torch.Tensor,
    token_selected_experts: torch.Tensor,
    token_final_scales: torch.Tensor,
    w3_w1: torch.Tensor,
    w3_w1_scales: torch.Tensor,
    w2: torch.Tensor,
    w2_scales: torch.Tensor,
    activation_type: int,
    output_dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    """
    Complete MoE forward pass using Triton FP8 block-scale GEMMs.

    This is a drop-in replacement for torch.ops.trtllm.fused_moe() for
    SM120 + FP8_BLOCK_SCALES, matching the weight layout produced by
    DeepSeekFP8BlockScalesFusedMoEMethod.

    Args:
        x                    : (T, H)      BF16  input activations
        token_selected_experts: (T, K)     int64 expert IDs per token
        token_final_scales   : (T, K)      F32   routing weights
        w3_w1                : (E, 2I, H)  FP8   gate+up projection weights
        w3_w1_scales         : (E, Nb, Kb) F32   block scales for w3_w1
        w2                   : (E, H, I)   FP8   down-projection weights
        w2_scales            : (E, Nb, Kb) F32   block scales for w2
        activation_type      : int         ActivationType enum value
        output_dtype         : optional output dtype (default: BF16)

    Returns:
        (T, H) BF16 output tensor
    """
    import os

    from tensorrt_llm._torch.modules.fused_moe.interface import ActivationType

    _debug = os.environ.get("TRTLLM_MOE_TRITON_DEBUG", "0") == "1"

    num_tokens, hidden = x.shape
    num_experts, N_gate_up, _ = w3_w1.shape
    if _debug:
        import sys

        print(
            f"[TritonMoE] x={x.shape} {x.dtype}, w31={w3_w1.shape} {w3_w1.dtype}, "
            f"w31sc={w3_w1_scales.shape} {w3_w1_scales.dtype}, "
            f"w2={w2.shape} {w2.dtype}, w2sc={w2_scales.shape} {w2_scales.dtype}, "
            f"tok_exp={token_selected_experts.shape} {token_selected_experts.dtype}",
            file=sys.stderr,
            flush=True,
        )
        print(f"[TritonMoE] w31sc[0,:2,:2]={w3_w1_scales[0, :2, :2]}", file=sys.stderr, flush=True)
        print(f"[TritonMoE] w2sc[0,:2,:2]={w2_scales[0, :2, :2]}", file=sys.stderr, flush=True)
    # Slow Python-loop fallback for debugging (enable with TRTLLM_MOE_TRITON_BF16=1)
    _bf16_mode = os.environ.get("TRTLLM_MOE_TRITON_BF16", "0") == "1"
    if _bf16_mode:
        return _run_moe_bf16_fallback(
            x,
            token_selected_experts,
            token_final_scales,
            w3_w1,
            w3_w1_scales,
            w2,
            w2_scales,
            activation_type,
            output_dtype,
        )

    intermediate = N_gate_up // 2
    top_k = token_selected_experts.shape[1]
    device = x.device

    act_int = int(activation_type)
    swiglu = int(ActivationType.Swiglu)
    geglu = int(ActivationType.Geglu)
    silu = int(ActivationType.Silu)
    gelu = int(ActivationType.Gelu)

    # ── Align token-expert pairs by expert ───────────────────────────────────
    sorted_token_ids, expert_ids, num_tokens_post_padded = moe_align_block_size(
        token_selected_experts, _BLOCK_SIZE_M, num_experts
    )

    # ── Phase 1 GEMM: x_bf16 @ w3_w1^T (FP8, dequant on-the-fly) ───────────
    ic1 = torch.zeros((num_tokens, top_k, N_gate_up), dtype=torch.bfloat16, device=device)
    _invoke_bf16act_fp8w_moe_kernel(
        x.contiguous(),
        w3_w1,
        ic1,
        w3_w1_scales,
        token_final_scales,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        mul_routed_weight=False,
        top_k=top_k,
    )

    # ── Gated activation (SwiGLU / GeGLU) ───────────────────────────────────
    # w3_w1 = concat(w3/up_proj, w1/gate_proj): first half is the value path,
    # second half is the gate path.  SwiGLU = silu(gate) * up.
    ic1_2d = ic1.view(num_tokens * top_k, N_gate_up)
    up, gate = ic1_2d[:, :intermediate], ic1_2d[:, intermediate:]

    if act_int in (swiglu, silu):
        ic2 = F.silu(gate) * up
    elif act_int in (geglu, gelu):
        ic2 = F.gelu(gate) * up
    else:
        raise ValueError(f"Unsupported activation_type={act_int} in Triton FP8 block-scale MoE")

    ic2 = ic2.contiguous()  # (T*top_k, I)  BF16

    # ── Phase 2 GEMM: ic2_bf16 @ w2^T (FP8, dequant on-the-fly) ────────────
    # top_k=1: each ic2 row is already one (token, expert) pair
    ic3 = torch.zeros((num_tokens, top_k, hidden), dtype=torch.bfloat16, device=device)
    _invoke_bf16act_fp8w_moe_kernel(
        ic2,
        w2,
        ic3,
        w2_scales,
        token_final_scales,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        mul_routed_weight=True,
        top_k=1,
    )

    # ── Reduce over top_k (routing weights applied in kernel) ────────────────
    output = ic3.sum(dim=1)  # (T, H)

    if output_dtype is not None and output.dtype != output_dtype:
        output = output.to(output_dtype)
    return output

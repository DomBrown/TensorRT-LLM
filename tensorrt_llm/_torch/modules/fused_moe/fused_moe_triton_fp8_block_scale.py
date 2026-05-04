# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 NVIDIA Corporation. All rights reserved.
"""
Triton FP8 block-scale MoE forward pass for SM120 (RTX PRO 6000 Blackwell).

Works around CUTLASS FP8 block-scale TMA descriptor failures on SM120 for
large token counts.  The Triton GEMM kernel is architecture-agnostic
(no wgmma / TMA intrinsics) and works on any SM >= 8.9.
"""

from typing import Optional

import torch
import torch.nn.functional as F
import triton
import triton.language as tl

_BLOCK_SHAPE = [128, 128]  # [group_k, group_n] for FP8 block-scale
_BLOCK_SIZE_M = 16
_BLOCK_SIZE_N = 32
_BLOCK_SIZE_K = 128  # must be <= group_k
_GROUP_SIZE_M = 4

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


# ─── Shared MoE GEMM helper ──────────────────────────────────────────────────


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
    from tensorrt_llm._torch.modules.fused_moe.interface import ActivationType

    num_tokens, hidden = x.shape
    num_experts, N_gate_up, _ = w3_w1.shape
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

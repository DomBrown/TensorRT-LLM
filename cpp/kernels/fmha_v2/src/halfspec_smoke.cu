/*
 * SPDX-FileCopyrightText: Copyright (c) 2011-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 *
 * Smoke test for the halfspec sm_120 / sm_121 warp-specialized FMHA.
 *
 * This file exists to give us a compile-time feedback loop on the halfspec
 * headers (Kernel_traits_halfspec_sm120, ws_sm120::Compute, ws_sm120::DMA)
 * without going through the full fmha_v2/setup.py code-gen machinery. The
 * instantiation below targets the shape we actually care about for
 * Qwen3.6-35B-A3B prefill on GB10: BF16, head_dim=256, head_dim_v=256,
 * STEP_Q=64, STEP_KV=64, causal mask.
 *
 * To compile this against the production CMake build:
 *
 *   cp cpp/kernels/fmha_v2/src/halfspec_smoke.cu \
 *      cpp/tensorrt_llm/kernels/contextFusedMultiHeadAttention/fmha_v2_cu/halfspec_smoke_sm120.cu
 *   cd cpp/build_RelWithDebInfo && cmake . && \
 *     cmake --build . --target _context_attention_kernels_120
 *
 * The fmha_v2_cu/ dir is gitignored (regenerated from setup.py at build
 * time) so the file has to LIVE there to be picked up, but is TRACKED here
 * in src/. Once phase 4 lands a proper setup.py halfspec branch this manual
 * copy step goes away.
 *
 * Status (2026-05-28): COMPILES cleanly for sm_120 / sm_121 / BF16 /
 * head_dim=256 / causal. Object is at
 * tensorrt_llm/kernels/contextFusedMultiHeadAttention/CMakeFiles/
 *   _context_attention_kernels_120.dir/fmha_v2_cu/halfspec_smoke_sm120.cu.o
 * (~4 MB). Correctness validation is phase 6 (runtime gen_token compare).
 */

#include <cstdio>
#include <cstdlib> // std::getenv

#include <cuda.h>  // CUtensorMap

#include <fmha/traits.h>
#include <fmha/warpspec_sm120/kernel_traits.h>
#include <fused_multihead_attention.h>
#include <fused_multihead_attention_kernel.h>

#include "fused_multihead_flash_attention_kernel_ws_sm120.h"

namespace fmha_smoke
{

// NOTE: the `S` template arg is the *kv loop step* (per-iter KV tile size),
// not the runtime maximum sequence length. setup.py uses kv_loop_step=128.
// The runtime kv seqlen is read from binfo.actual_kv_seqlen and the kv loop
// iterates in chunks of S.
//
// Also: with TMA box size capped at 256 elements per axis, S must be <=256
// (we load STEP_KV = Cta_tile_p::N = S elements per TMA box call).
//
// Head-dim-parameterized halfspec traits. HEAD_DIM must be a multiple of the
// 64-element (= 128-byte BF16) TMA chunk width so the Q/K head-dim chunks and
// the 64-wide V dv-chunks keep 128-byte smem rows -- the layout that matches
// the TMA 128B hardware swizzle. head_dim 128 and 256 both satisfy this; a
// non-multiple-of-64 head dim would break the swizzle invariant.
template <int HEAD_DIM>
using Halfspec_ktraits = fmha::ws_sm120::Kernel_traits_halfspec_sm120<
    /*Traits_=*/fmha::Ampere_hmma_bf16_traits,
    /*S=*/128,
    /*VALID_D_=*/HEAD_DIM,
    /*VALID_DV_=*/HEAD_DIM,
    /*STEP_Q_=*/64,
    /*WARPS_M_=*/4,
    /*WARPS_N_=*/1,
    /*VERSION_=*/2,
    /*MASK_VERSION_=*/3, // 3 = causal
    /*RING_DEPTH_=*/1,
    /*ENABLE_SKIP_SOFTMAX_=*/true,
    /*NUM_PRODUCER_WARPS_=*/1>;

// Back-compat alias for the original d=256 smoke instantiation.
using Smoke_Ktraits = Halfspec_ktraits<256>;

} // namespace fmha_smoke

// Templated entry kernel -- one instantiation per head dim (128, 256). THREADS
// is head-dim-independent (1 producer + WARPS_M*WARPS_N consumer warps), so the
// launch_bounds value is identical across instantiations.
template <typename Ktraits>
__global__ __launch_bounds__(Ktraits::THREADS, 1) void halfspec_kernel(
    bert::Fused_multihead_attention_params_v2 const params, __grid_constant__ const CUtensorMap tma_q,
    __grid_constant__ const CUtensorMap tma_k, __grid_constant__ const CUtensorMap tma_v)
{
    // The CUtensorMaps live in const/param space (grid_constant); passing
    // their addresses to cp.async.bulk.tensor is a valid tensormap operand.
    fused_multihead_attention::device_flash_attention_ws_sm120<Ktraits>(params, &tma_q, &tma_k, &tma_v);
}

////////////////////////////////////////////////////////////////////////////////////////////////////
//
// Host-side launcher for the halfspec smoke kernel.
//
// Sets up the TMA descriptors via DMA::Host::init_params, sizes shared memory,
// configures the launch attribute that lets the kernel use >48KB smem, and
// launches the kernel on the given stream.
//
// Returns the cudaError_t from the launch -- the caller is responsible for
// checking it (and running a separate sync to surface launch-time aborts).
////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Ktraits>
static cudaError_t launch_halfspec(bert::Fused_multihead_attention_params_v2 params,
    bert::Fused_multihead_attention_launch_params const& launch_params, cudaStream_t stream)
{
    // 1. Build the three TMA descriptors host-side (cuTensorMapEncodeTiled).
    //    These are passed to the kernel as __grid_constant__ params.
    CUtensorMap tma_q{}, tma_k{}, tma_v{};
    typename fmha::ws_sm120::DMA<Ktraits>::Host dma_host;
    dma_host.init_params(params, launch_params, tma_q, tma_k, tma_v);

    // 2. Size the smem allocation. If it exceeds the 48 KB default, raise the
    //    cudaFuncAttributeMaxDynamicSharedMemorySize cap before launch. (head_dim
    //    128 needs roughly half the head_dim 256 footprint.)
    constexpr int smem_bytes = static_cast<int>(Ktraits::BYTES_PER_SMEM);
    // Per-launch diagnostic, off by default (this fires once per attention layer
    // per forward and is very noisy). Set TRTLLM_HALFSPEC_VERBOSE=1 to confirm
    // halfspec is dispatching for the prefill.
    static bool const kVerbose = (std::getenv("TRTLLM_HALFSPEC_VERBOSE") != nullptr);
    if (kVerbose)
    {
        std::fprintf(stderr, "[halfspec] BYTES_PER_SMEM = %d\n", smem_bytes);
    }
    if (smem_bytes >= 48 * 1024)
    {
        cudaError_t err
            = cudaFuncSetAttribute(halfspec_kernel<Ktraits>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes);
        if (err != cudaSuccess)
        {
            return err;
        }
    }

    // 3. Grid = (Q-tiles, H, B). One CTA per (Q-tile, head, batch). Each CTA
    //    has THREADS threads (1 producer warp + WARPS_M*WARPS_N consumer
    //    warps = 5 warps total = 160 threads).
    int const q_tiles = (params.s + Ktraits::STEP_Q - 1) / Ktraits::STEP_Q;
    dim3 const grid(q_tiles, params.h, params.b);
    dim3 const block(Ktraits::THREADS);

    halfspec_kernel<Ktraits><<<grid, block, smem_bytes, stream>>>(params, tma_q, tma_k, tma_v);
    return cudaGetLastError();
}

// Back-compat extern "C" entry for the standalone validation harness, which
// links against `launch_halfspec_smoke` (the d=256 instantiation).
extern "C" cudaError_t launch_halfspec_smoke(bert::Fused_multihead_attention_params_v2 params,
    bert::Fused_multihead_attention_launch_params const& launch_params, cudaStream_t stream)
{
    return launch_halfspec<fmha_smoke::Smoke_Ktraits>(params, launch_params, stream);
}

// d=128 instantiation for the standalone bench/validation harness (e.g.
// halfspec_bench.cu). Mirrors launch_halfspec_smoke; head_dim 128 is the other
// shape the in-engine halfspec dispatch supports (Qwen3-30B-A3B etc.).
extern "C" cudaError_t launch_halfspec_smoke_d128(bert::Fused_multihead_attention_params_v2 params,
    bert::Fused_multihead_attention_launch_params const& launch_params, cudaStream_t stream)
{
    return launch_halfspec<fmha_smoke::Halfspec_ktraits<128>>(params, launch_params, stream);
}

////////////////////////////////////////////////////////////////////////////////////////////////////
//
// In-engine dispatch bridge (phase 6d).
//
// The production fmhaRunner uses `tensorrt_llm::kernels::Fused_multihead_attention_params_v2`
// (defined in contextFusedMultiHeadAttention/fused_multihead_attention_common.h),
// which is a separate but ABI-compatible struct from the fmha_v2 `bert::` one --
// the generated kernels bridge them with reinterpret_cast, and we do the same.
// Only the production build (which copies this file into fmha_v2_cu/ and compiles
// with GENERATE_CUBIN=1) sees the common header; the standalone validation build
// (nvcc -I fmha_v2/src) compiles without it, so guard the bridge accordingly.
////////////////////////////////////////////////////////////////////////////////////////////////////

#if defined(GENERATE_CUBIN)
#include "../fused_multihead_attention_common.h"
#include "tensorrt_llm/common/config.h"

TRTLLM_NAMESPACE_BEGIN

namespace kernels
{

// Launchers with the signature the runner's dispatch hook expects (mirrors the
// generated `run_fmha_v2_..._sm90` convention: kernels:: param/launch types,
// reinterpret_cast to bert:: for the actual launch). One per supported head
// dim; both share the templated launch_halfspec<> path.
//
// The bert launch_params is NOT ABI-identical to kernels::Launch_params, so we
// copy the one field Host::init_params reads (total_q_seqlen) rather than
// reinterpret_cast it.
void run_halfspec_bf16_d256_causal_sm120(
    Fused_multihead_attention_params_v2& params, Launch_params const& launch_params, cudaStream_t stream)
{
    bert::Fused_multihead_attention_launch_params blp{};
    blp.total_q_seqlen = launch_params.total_q_seqlen;
    blp.attention_input_layout = fmha::Attention_input_layout::PACKED_QKV;

    ::launch_halfspec<::fmha_smoke::Halfspec_ktraits<256>>(
        reinterpret_cast<bert::Fused_multihead_attention_params_v2&>(params), blp, stream);
}

void run_halfspec_bf16_d128_causal_sm120(
    Fused_multihead_attention_params_v2& params, Launch_params const& launch_params, cudaStream_t stream)
{
    bert::Fused_multihead_attention_launch_params blp{};
    blp.total_q_seqlen = launch_params.total_q_seqlen;
    blp.attention_input_layout = fmha::Attention_input_layout::PACKED_QKV;

    ::launch_halfspec<::fmha_smoke::Halfspec_ktraits<128>>(
        reinterpret_cast<bert::Fused_multihead_attention_params_v2&>(params), blp, stream);
}

} // namespace kernels

TRTLLM_NAMESPACE_END
#endif // GENERATE_CUBIN

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

#include <cuda.h>  // CUtensorMap

#include <fused_multihead_attention.h>
#include <fused_multihead_attention_kernel.h>
#include <fmha/traits.h>
#include <fmha/warpspec_sm120/kernel_traits.h>

#include "fused_multihead_flash_attention_kernel_ws_sm120.h"

namespace fmha_smoke
{

// NOTE: the `S` template arg is the *kv loop step* (per-iter KV tile size),
// not the runtime maximum sequence length. For head_dim=256 the existing
// setup.py uses kv_loop_step=128. The runtime kv seqlen is read from
// binfo.actual_kv_seqlen and the kv loop iterates in chunks of S.
//
// Also: with TMA box size capped at 256 elements per axis, S must be <=256
// (we load STEP_KV = Cta_tile_p::N = S elements per TMA box call).
using Smoke_Ktraits = fmha::ws_sm120::Kernel_traits_halfspec_sm120<
    /*Traits_=*/fmha::Ampere_hmma_bf16_traits,
    /*S=*/128,
    /*VALID_D_=*/256,
    /*VALID_DV_=*/256,
    /*STEP_Q_=*/64,
    /*WARPS_M_=*/4,
    /*WARPS_N_=*/1,
    /*VERSION_=*/2,
    /*MASK_VERSION_=*/3, // 3 = causal
    /*RING_DEPTH_=*/1,
    /*ENABLE_SKIP_SOFTMAX_=*/false,
    /*NUM_PRODUCER_WARPS_=*/1>;

} // namespace fmha_smoke

extern "C" __global__ __launch_bounds__(fmha_smoke::Smoke_Ktraits::THREADS, 1)
void halfspec_smoke_kernel(bert::Fused_multihead_attention_params_v2 const params,
    __grid_constant__ const CUtensorMap tma_q,
    __grid_constant__ const CUtensorMap tma_k,
    __grid_constant__ const CUtensorMap tma_v)
{
    // The CUtensorMaps live in const/param space (grid_constant); passing
    // their addresses to cp.async.bulk.tensor is a valid tensormap operand.
    fused_multihead_attention::device_flash_attention_ws_sm120<fmha_smoke::Smoke_Ktraits>(
        params, &tma_q, &tma_k, &tma_v);
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

extern "C" cudaError_t launch_halfspec_smoke(
    bert::Fused_multihead_attention_params_v2 params,
    bert::Fused_multihead_attention_launch_params const& launch_params,
    cudaStream_t stream)
{
    // 1. Build the three TMA descriptors host-side (cuTensorMapEncodeTiled).
    //    These are passed to the kernel as __grid_constant__ params.
    CUtensorMap tma_q{}, tma_k{}, tma_v{};
    fmha::ws_sm120::DMA<fmha_smoke::Smoke_Ktraits>::Host dma_host;
    dma_host.init_params(params, launch_params, tma_q, tma_k, tma_v);

    // 2. Size the smem allocation. With RING_DEPTH=3 and head_dim=256 BF16 +
    //    STEP_Q/KV=64, the Shared struct is ~150 KB, comfortably above the
    //    48 KB cudaFuncAttributeMaxDynamicSharedMemorySize default.
    constexpr int smem_bytes = static_cast<int>(fmha_smoke::Smoke_Ktraits::BYTES_PER_SMEM);
    std::fprintf(stderr, "[halfspec-runtest] BYTES_PER_SMEM = %d\n", smem_bytes);
    if (smem_bytes >= 48 * 1024)
    {
        cudaError_t err = cudaFuncSetAttribute(halfspec_smoke_kernel,
            cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes);
        if (err != cudaSuccess)
        {
            return err;
        }
    }

    // 3. Grid = (Q-tiles, H, B). One CTA per (Q-tile, head, batch). Each CTA
    //    has THREADS threads (1 producer warp + WARPS_M*WARPS_N consumer
    //    warps = 5 warps total = 160 threads).
    int const q_tiles = (params.s + fmha_smoke::Smoke_Ktraits::STEP_Q - 1)
                      / fmha_smoke::Smoke_Ktraits::STEP_Q;
    dim3 const grid(q_tiles, params.h, params.b);
    dim3 const block(fmha_smoke::Smoke_Ktraits::THREADS);

    halfspec_smoke_kernel<<<grid, block, smem_bytes, stream>>>(params, tma_q, tma_k, tma_v);
    return cudaGetLastError();
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

// Launcher with the signature the runner's dispatch hook expects (mirrors the
// generated `run_fmha_v2_..._sm90` convention: kernels:: param/launch types,
// reinterpret_cast to bert:: for the actual launch).
void run_halfspec_bf16_d256_causal_sm120(
    Fused_multihead_attention_params_v2& params, Launch_params const& launch_params, cudaStream_t stream)
{
    // The bert launch_params is NOT ABI-identical to kernels::Launch_params, so
    // copy the one field Host::init_params reads (total_q_seqlen) rather than
    // reinterpret_cast it.
    bert::Fused_multihead_attention_launch_params blp{};
    blp.total_q_seqlen = launch_params.total_q_seqlen;
    blp.attention_input_layout = fmha::Attention_input_layout::PACKED_QKV;

    launch_halfspec_smoke(
        reinterpret_cast<bert::Fused_multihead_attention_params_v2&>(params), blp, stream);
}

} // namespace kernels
TRTLLM_NAMESPACE_END
#endif // GENERATE_CUBIN

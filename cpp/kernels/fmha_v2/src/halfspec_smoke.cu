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

#include <fused_multihead_attention.h>
#include <fused_multihead_attention_kernel.h>
#include <fmha/traits.h>
#include <fmha/warpspec_sm120/kernel_traits.h>

#include "fused_multihead_flash_attention_kernel_ws_sm120.h"

namespace fmha_smoke
{

using Smoke_Ktraits = fmha::ws_sm120::Kernel_traits_halfspec_sm120<
    /*Traits_=*/fmha::Ampere_hmma_bf16_traits,
    /*S=*/8192,
    /*VALID_D_=*/256,
    /*VALID_DV_=*/256,
    /*STEP_Q_=*/64,
    /*WARPS_M_=*/4,
    /*WARPS_N_=*/1,
    /*VERSION_=*/2,
    /*MASK_VERSION_=*/3, // 3 = causal
    /*RING_DEPTH_=*/3,
    /*ENABLE_SKIP_SOFTMAX_=*/false,
    /*NUM_PRODUCER_WARPS_=*/1>;

} // namespace fmha_smoke

extern "C" __global__ __launch_bounds__(fmha_smoke::Smoke_Ktraits::THREADS, 1)
void halfspec_smoke_kernel(bert::Fused_multihead_attention_params_v2 const params)
{
    fused_multihead_attention::device_flash_attention_ws_sm120<fmha_smoke::Smoke_Ktraits>(params);
}

/*
 * SPDX-FileCopyrightText: Copyright (c) 2011-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

// =====================================================================
// Warp-specialized flash-attention prefill kernel for sm_120 / sm_121.
// =====================================================================
//
// Differences from `noloop_tiled.h` (the current sm_120 path):
//
// 1. Producer / consumer warp split.
//    - 1 producer warp (32 threads) issues `cp.async.bulk.tensor.2d` for
//      Q (once) and K, V (every kv iter) into a multi-buffer smem ring.
//    - The remaining warps consume the ring and run the same BMM1 +
//      softmax + skip-softmax + BMM2 body as noloop_tiled.h.
//
// 2. mbarrier producer/consumer handshake instead of CTA-wide
//    __syncthreads(). Consumers unblock as soon as their tile's
//    cp.async.bulk.tensor.* completes; the producer can fly ahead by
//    `CIRCULAR_BUFFER_DEPTH` tiles.
//
// 3. `setmaxnreg.sync.aligned` register-budget split: producer warp gets
//    a small register cap, consumer warps get the rest. This frees
//    physical registers for the consumers to keep larger acc_o slices live.
//
// Same as before:
// - sync `mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32` for both
//   BMM1 and BMM2 (via fmha::gemm + Fragment_accumulator::mma in
//   fmha/fragment.h). The compute body is unchanged from noloop_tiled.h.
// - Per-warp skip-softmax vote with log-threshold predicate and BMM2
//   split (skip vs no-skip), exactly as committed on
//   dcampora/skip-softmax-sm120 branch.
//
// Status: SKELETON.  See fmha/warpspec_sm120/README.md for the
// TODO checklist and the path from here to a buildable kernel.

#include <fmha/utils.h>
#include <fmha/warpspec_sm120/compute_sync_mma.h>
#include <fmha/warpspec_sm120/dma_sync_mma.h>
#include <fused_multihead_attention_kernel.h>

namespace fused_multihead_attention
{

////////////////////////////////////////////////////////////////////////////////////////////////////

namespace
{

// sm_120 register-budget split. Numbers below are starting points;
// final values depend on the launched warps and the per-tile working set.
constexpr int DMA_NREG = 40;       // producer: TMA issue + coord math only
constexpr int COMPUTE_NREG = 232;  // consumers: acc_o + softmax + frag_p live

inline __device__ void setmaxnreg_dma()
{
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    asm volatile("setmaxnreg.dec.sync.aligned.u32 %0;\n" ::"n"(DMA_NREG));
#endif
}

inline __device__ void setmaxnreg_compute()
{
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    asm volatile("setmaxnreg.inc.sync.aligned.u32 %0;\n" ::"n"(COMPUTE_NREG));
#endif
}

} // namespace

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Kernel_traits, typename Params>
inline __device__ void device_flash_attention_ws_sm120(Params const& params)
{
    using Shared = typename Kernel_traits::Shared;

    // The shared struct contains:
    //   - smem_q[CIRCULAR_BUFFER_DEPTH] aligned tiles for Q
    //   - smem_k[CIRCULAR_BUFFER_DEPTH] aligned tiles for K
    //   - smem_v[CIRCULAR_BUFFER_DEPTH] aligned tiles for V
    //   - q_barriers / k_barriers / v_barriers (entry-produced /
    //     entry-consumed mbarrier pairs)
    extern __shared__ char smem_[];
    char* smem_aligned = fmha::align_1024(smem_);
    Shared* shared = reinterpret_cast<Shared*>(&smem_aligned[0]);
    shared->init(threadIdx.x == 0);
    __syncthreads();

    // 32-thread warps. Warp 0 = TMA producer. Warps 1+ = sync-MMA consumers.
    int const warp_id = threadIdx.x / 32;
    int const lane = threadIdx.x % 32;
    int const tidx_in_compute_group = threadIdx.x - 32;

    if (warp_id == 0)
    {
        setmaxnreg_dma();
        uint32_t const elect_one = (lane == 0) ? 1u : 0u;
        fmha::ws_sm120::DMA<Kernel_traits> dma(elect_one);
        dma.run(params, shared);
    }
    else
    {
        setmaxnreg_compute();
        fmha::ws_sm120::Compute<Kernel_traits> compute;
        compute.run(tidx_in_compute_group, shared, params);
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace fused_multihead_attention

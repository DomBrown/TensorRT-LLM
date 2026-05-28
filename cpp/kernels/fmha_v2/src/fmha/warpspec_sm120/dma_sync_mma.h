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

// Producer (TMA-loader) half of the sm_120 / sm_121 warp-specialized FMHA.
//
// Single dedicated warp issues `cp.async.bulk.tensor.2d` for Q (once) and
// K / V (every kv iter), arrives on the entry-produced mbarriers, and waits
// on the entry-consumed mbarriers before recycling a ring slot.
//
// Reuses the existing TMA descriptor + utmaldg helpers from
// fmha/hopper/utils_tma.h (which compile under __CUDA_ARCH__ >= 900, inclusive
// of sm_120 / sm_121) and the CircularBufferWriter from
// fmha/warpspec/circular_buffer.h (Arrive_wait-based, also CC >= 9.0).

#include <fmha/hopper/arrive_wait.h>
#include <fmha/hopper/tma_descriptor.h>
#include <fmha/hopper/tma_types.h>
#include <fmha/hopper/utils_tma.h>
#include <fmha/utils.h>
#include <fmha/warpspec/circular_buffer.h>

namespace fmha
{
namespace ws_sm120
{

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Kernel_traits>
struct DMA
{
    using Shared = typename Kernel_traits::Shared;

    using Cbw_q = typename Kernel_traits::Circular_buffer_q_writer;
    using Cbw_k = typename Kernel_traits::Circular_buffer_k_writer;
    using Cbw_v = typename Kernel_traits::Circular_buffer_v_writer;

    enum
    {
        STEP_Q = Kernel_traits::STEP_Q
    };
    enum
    {
        STEP_KV = Kernel_traits::STEP_KV
    };
    enum
    {
        D = Kernel_traits::D
    };
    enum
    {
        DV = Kernel_traits::DV
    };
    enum
    {
        ELEMENT_BYTES = Kernel_traits::ELEMENT_BYTES
    };

    // Bytes per K-tile TMA load = STEP_KV rows x D cols x dtype_bytes.
    enum
    {
        TX_BYTES_K = STEP_KV * D * ELEMENT_BYTES
    };
    enum
    {
        TX_BYTES_V = STEP_KV * DV * ELEMENT_BYTES
    };
    enum
    {
        TX_BYTES_Q = STEP_Q * D * ELEMENT_BYTES
    };

    explicit inline __device__ DMA(uint32_t elect_one)
        : elect_one_(elect_one)
    {
    }

    // Runs on a single warp (32 threads); only thread 0 (`elect_one`) issues
    // the cp.async.bulk.tensor.* instructions. The other 31 threads still
    // participate in mbarrier arrives where needed.
    template <typename Params>
    inline __device__ void run(Params const& params, Shared* shared)
    {
        Block_info_padded<1> const binfo(params, blockIdx.y, blockIdx.z, blockIdx.x);
        if (binfo.stop_early(params.is_s_padded))
        {
            return;
        }

        int const kv_loop_end = binfo.actual_kv_seqlen;

        // The TMA descriptors are built once on the host and passed via
        // params. See fmha_v2 setup.py for the existing
        // `setup_tma_descriptors_*` helpers (we reuse those for sm_120).
        cudaTmaDesc const* desc_q = &params.tma_desc_q;
        cudaTmaDesc const* desc_k = &params.tma_desc_k;
        cudaTmaDesc const* desc_v = &params.tma_desc_v;

        Cbw_q cbw_q(&shared->q_barriers);
        Cbw_k cbw_k(&shared->k_barriers);
        Cbw_v cbw_v(&shared->v_barriers);

        // --- Q load: once per CTA -------------------------------------------
        //
        // We issue a single 2D TMA load into ring slot 0. Coordinate args are
        // (col_offset_in_features, row_offset_in_seq). The exact coord pattern
        // depends on Gmem_tile_q's TMA-descriptor layout; the existing
        // ws::DMA in fmha/warpspec/dma.h has the canonical setup we should
        // mirror.
        {
            int const q_slot = cbw_q.tmaReserve(elect_one_, TX_BYTES_Q);
            uint32_t const q_smem = __nvvm_get_smem_pointer(&shared->smem_q[q_slot]);
            uint32_t const q_bar = cbw_q.barrierAddr(q_slot);
            int32_t const coord[2] = {0, blockIdx.y * STEP_Q + binfo.bidh * 0 /* TODO */};
            // 2D, TILED, no multicast.
            fmha::utmaldg<2, fmha::cudaTmaDescType::TILED, false>(desc_q, q_smem, q_bar, coord, elect_one_);
        }

        // --- KV loop: one (K, V) load per iter ------------------------------
        for (int kv_loop = 0; kv_loop < kv_loop_end; kv_loop += STEP_KV)
        {
            // K
            int const k_slot = cbw_k.tmaReserve(elect_one_, TX_BYTES_K);
            uint32_t const k_smem = __nvvm_get_smem_pointer(&shared->smem_k[k_slot]);
            uint32_t const k_bar = cbw_k.barrierAddr(k_slot);
            int32_t const k_coord[2] = {0, kv_loop};
            fmha::utmaldg<2, fmha::cudaTmaDescType::TILED, false>(desc_k, k_smem, k_bar, k_coord, elect_one_);

            // V
            int const v_slot = cbw_v.tmaReserve(elect_one_, TX_BYTES_V);
            uint32_t const v_smem = __nvvm_get_smem_pointer(&shared->smem_v[v_slot]);
            uint32_t const v_bar = cbw_v.barrierAddr(v_slot);
            int32_t const v_coord[2] = {0, kv_loop};
            fmha::utmaldg<2, fmha::cudaTmaDescType::TILED, false>(desc_v, v_smem, v_bar, v_coord, elect_one_);
        }

        // Producer exits. The consumers' cbw_*.complete() acks have already
        // been gated on entry-consumed mbarriers via the ring's wait() each
        // iter; nothing else to do.
    }

    uint32_t elect_one_;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace ws_sm120
} // namespace fmha

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
#include <fused_multihead_attention_kernel.h>  // Block_info_padded

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
        fused_multihead_attention::Single_cta<Kernel_traits::VERSION> const binfo(
            params, blockIdx.z, blockIdx.y, 0, /*tidx=*/0);
        if (binfo.stop_early(0))
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
            uint32_t const q_smem = __nvvm_get_smem_pointer(&shared->smem_q[q_slot][0]);
            uint32_t const q_bar = __nvvm_get_smem_pointer(cbw_q.barrier_ptr(q_slot));
            int32_t const coord[2] = {0, static_cast<int32_t>(blockIdx.y * STEP_Q)};
            // 2D, TILED, no multicast.
            fmha::utmaldg<2, fmha::cudaTmaDescType::TILED, false>(desc_q, q_smem, q_bar, coord, elect_one_);
        }

        // --- KV loop: one (K, V) load per iter ------------------------------
        for (int kv_loop = 0; kv_loop < kv_loop_end; kv_loop += STEP_KV)
        {
            // K
            int const k_slot = cbw_k.tmaReserve(elect_one_, TX_BYTES_K);
            uint32_t const k_smem = __nvvm_get_smem_pointer(&shared->smem_k[k_slot][0]);
            uint32_t const k_bar = __nvvm_get_smem_pointer(cbw_k.barrier_ptr(k_slot));
            int32_t const k_coord[2] = {0, kv_loop};
            fmha::utmaldg<2, fmha::cudaTmaDescType::TILED, false>(desc_k, k_smem, k_bar, k_coord, elect_one_);

            // V
            int const v_slot = cbw_v.tmaReserve(elect_one_, TX_BYTES_V);
            uint32_t const v_smem = __nvvm_get_smem_pointer(&shared->smem_v[v_slot][0]);
            uint32_t const v_bar = __nvvm_get_smem_pointer(cbw_v.barrier_ptr(v_slot));
            int32_t const v_coord[2] = {0, kv_loop};
            fmha::utmaldg<2, fmha::cudaTmaDescType::TILED, false>(desc_v, v_smem, v_bar, v_coord, elect_one_);
        }

        // Producer exits. The consumers' cbw_*.complete() acks have already
        // been gated on entry-consumed mbarriers via the ring's wait() each
        // iter; nothing else to do.
    }

    uint32_t elect_one_;

    ////////////////////////////////////////////////////////////////////////////////////////////////
    //
    // Host-side TMA descriptor setup.
    //
    // Called once per LLM forward (or once per layer if descriptors are layer-
    // varying) before the kernel launch. Populates params.tma_desc_q / _k / _v
    // with cudaTmaDesc values that:
    //   * Use SWIZZLE_128B for the smem layout (tensor-core-friendly, matches
    //     what ldmatrix.x4 expects on the consumer side).
    //   * Use F16_RN format (BF16 is treated as 2-byte ints by the TMA engine;
    //     F16_RN is the right opaque format for 2-byte types -- the engine
    //     does no element-wise conversion, just moves bytes).
    //   * Use 3D tensor (D, H_or_HKV, total_seqlen) and a 3D box of
    //     (D, 1, STEP_Q / STEP_KV).
    //
    // Scope of v0:
    //   * BF16 (or FP16) only -- FP8 (E4M3) needs a separate Host that uses
    //     U8 desc_format and the V-transpose path, mirroring the Hopper
    //     Kernel_traits_Hopper_qgmma_e4m3_fp32 variant.
    //   * PACKED_QKV input layout only ([total_seqlen, H, D] + KV after Q).
    //     CONTIGUOUS_Q_KV, SEPARATE_Q_K_V, Q_PAGED_KV are TODOs.
    //   * No TMA store -- the epilogue still uses the existing Gmem_tile_o
    //     scalar STG path.
    ////////////////////////////////////////////////////////////////////////////////////////////////

    struct Host
    {
        Host() = default;

        template <typename Params, typename Launch_params>
        void init_params(Params& params, Launch_params const& launch_params, cudaStream_t /*stream*/) const
        {
            uint32_t const d = params.d;
            uint32_t const dv = params.dv;
            uint32_t const h = params.h;
            uint32_t const h_kv = params.h_kv;

            // Total sequence length across the batch.
            uint32_t const total_seqlen = params.is_s_padded
                ? static_cast<uint32_t>(params.b * params.s)
                : static_cast<uint32_t>(launch_params.total_q_seqlen);

            // ---- Constants shared by all 3 descriptors -------------------------
            uint32_t const traversal_stride[3] = {1, 1, 1};
            uint32_t const oob_fill = 0;
            uint32_t const fp32_to_tf32 = 0;

            // SWIZZLE_128B: the standard tensor-core-friendly smem layout.
            // Both the non-Hopper Smem_tile_a/b (used by halfspec) and the
            // Hopper Smem_tile_hopper_a/b (used by Hopper warpspec) are
            // bank-conflict-free when fed by a 128B-swizzled TMA, so this
            // choice is safe across both consumer styles.
            static constexpr fmha::cudaTmaDescSwizzle swizzle_mode
                = fmha::cudaTmaDescSwizzle::SWIZZLE_128B;

            // F16_RN is the right opaque format for 2-byte element types
            // (BF16 / FP16). The TMA engine does no element-wise conversion;
            // it just routes bytes through the swizzle.
            static_assert(Kernel_traits::ELEMENT_BYTES == 2,
                "halfspec v0 only supports BF16 / FP16 (2-byte elements). "
                "FP8 needs a separate Host that uses U8 format and the V "
                "transpose path.");
            static constexpr fmha::cudaTmaDescFormat desc_format
                = fmha::cudaTmaDescFormat::F16_RN;

            static_assert(STEP_Q <= 256 && STEP_KV <= 256,
                "TMA box dimensions are capped at 256 elements per axis.");

            // ---- Q descriptor -------------------------------------------------
            //
            // Tensor layout in gmem: [total_seqlen, H, D]
            // Tensor size (elements, fastest-varying first): (D, H, total_seqlen)
            // Box size:                                       (D, 1, STEP_Q)
            //
            // Strides (in bytes, slowest-varying first, excluding the
            // implicit byte-per-element axis 0):
            //   stride[0] = D * ELEMENT_BYTES         (within-head row)
            //   stride[1] = params.q_stride_in_bytes  (across heads -- this
            //                                          accounts for the
            //                                          packed QKV layout
            //                                          where (H_q + H_kv +
            //                                          H_kv) heads of D
            //                                          elements lie along
            //                                          axis 1)
            uint32_t const tensor_size_q[3] = {d, h, total_seqlen};
            uint64_t const tensor_stride_q[2]
                = {d * Kernel_traits::ELEMENT_BYTES, static_cast<uint64_t>(params.q_stride_in_bytes)};
            uint32_t const box_size_q[3] = {static_cast<uint32_t>(Kernel_traits::D), 1, STEP_Q};
            char* const q_ptr = reinterpret_cast<char*>(params.qkv_ptr);

            fmha::Multiple_tma_descriptor<3> q_desc;
            q_desc.set_tma_desctriptor(q_ptr, desc_format,
                fmha::cudaTmaDescInterleave::INTERLEAVE_DISABLED, swizzle_mode,
                fmha::cudaTmaDescPromotion::PROMOTION_DISABLED, tensor_size_q, tensor_stride_q,
                traversal_stride, box_size_q, oob_fill, fp32_to_tf32, &params.tma_desc_q);

            // ---- K, V descriptors ---------------------------------------------
            //
            // PACKED_QKV layout, MQA/GQA-style (H_q + H_kv + H_kv heads):
            //   k_ptr = qkv_ptr + h     * d  * sizeof(elt)
            //   v_ptr = k_ptr   + h_kv  * d  * sizeof(elt)
            //
            // K tensor size:    (D,  H_kv, total_seqlen)
            // V tensor size:    (DV, H_kv, total_seqlen)
            // Box for both K/V: (D or DV, 1, STEP_KV)
            uint32_t const tensor_size_k[3] = {d, h_kv, total_seqlen};
            uint32_t const tensor_size_v[3] = {dv, h_kv, total_seqlen};
            uint64_t const tensor_stride_k[2]
                = {d * Kernel_traits::ELEMENT_BYTES, static_cast<uint64_t>(params.k_stride_in_bytes)};
            uint64_t const tensor_stride_v[2]
                = {dv * Kernel_traits::ELEMENT_BYTES, static_cast<uint64_t>(params.v_stride_in_bytes)};
            uint32_t const box_size_k[3] = {static_cast<uint32_t>(Kernel_traits::D), 1, STEP_KV};
            uint32_t const box_size_v[3] = {static_cast<uint32_t>(Kernel_traits::DV), 1, STEP_KV};

            char* const k_ptr = q_ptr + h * d * Kernel_traits::ELEMENT_BYTES;
            char* const v_ptr = k_ptr + h_kv * d * Kernel_traits::ELEMENT_BYTES;

            fmha::Multiple_tma_descriptor<3> kv_desc;
            kv_desc.set_tma_desctriptor(k_ptr, desc_format,
                fmha::cudaTmaDescInterleave::INTERLEAVE_DISABLED, swizzle_mode,
                fmha::cudaTmaDescPromotion::PROMOTION_DISABLED, tensor_size_k, tensor_stride_k,
                traversal_stride, box_size_k, oob_fill, fp32_to_tf32, &params.tma_desc_k);
            kv_desc.set_tma_desctriptor(v_ptr, desc_format,
                fmha::cudaTmaDescInterleave::INTERLEAVE_DISABLED, swizzle_mode,
                fmha::cudaTmaDescPromotion::PROMOTION_DISABLED, tensor_size_v, tensor_stride_v,
                traversal_stride, box_size_v, oob_fill, fp32_to_tf32, &params.tma_desc_v);
        }
    };
};

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace ws_sm120
} // namespace fmha

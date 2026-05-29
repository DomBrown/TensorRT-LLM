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

#include <cstdio>

#include <cuda.h>  // CUtensorMap + cuTensorMapEncodeTiled (driver API)

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
//
// halfspec-local TMA load helper using `shared::cta` (single-CTA) variant.
//
// The Hopper fmha::utmaldg<...> uses `shared::cluster` qualifier, which
// requires cluster launch (cluster_dim > 0) to be valid PTX. sm_120 / sm_121
// kernels are launched without an explicit cluster attribute, so the
// shared::cluster variant emits an Illegal Instruction at runtime even
// though it assembles successfully. The `shared::cta` variant is the
// single-CTA-no-cluster form and is what we need on consumer Blackwell.
//
// CRITICAL (proven 2026-05-29): the descriptor passed here MUST be a driver
// API `CUtensorMap` built by `cuTensorMapEncodeTiled` (128 bytes). The
// fmha_v2 hand-rolled `fmha::cudaTmaDesc` (64 bytes, Hopper-era bit layout)
// is REJECTED by Blackwell's TMA engine -- UTMALDG.3D faults with an
// "Illegal Instruction" at runtime even though the PTX assembles. A minimal
// reproducer confirmed: hand-rolled desc -> illegal instruction; encode-tiled
// CUtensorMap -> loads correct data on GB10 (sm_121).
////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ void utmaldg_3d_cta(void const* p_desc,
    uint32_t smem_ptr, uint32_t smem_barrier, int32_t const (&coord)[3], uint32_t elect_one)
{
    if (elect_one)
    {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
        // Note the `.tile` qualifier: required on sm_120 / sm_121 PTX (omitting
        // it emits an illegal-instruction at runtime). The trtllmGenKernels FMHA
        // shipping today uses the same `.shared::cta.global.tile` variant
        // (cpp/tensorrt_llm/kernels/trtllmGenKernels/fmha/cuda_ptx/cuda_ptx.h:2073).
        asm volatile(
            "cp.async.bulk.tensor.3d.shared::cta.global.tile.mbarrier::complete_tx::bytes "
            "[%0], [%1, {%2, %3, %4}], [%5];\n"
            :
            : "r"(smem_ptr), "l"(reinterpret_cast<uint64_t>(p_desc)),
              "r"(coord[0]), "r"(coord[1]), "r"(coord[2]),
              "r"(smem_barrier)
            : "memory");
#endif
    }
}

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
    //
    // The three TMA descriptors are CUtensorMaps built host-side by
    // Host::init_params (cuTensorMapEncodeTiled) and passed in as
    // __grid_constant__ kernel params (their addresses live in const/param
    // space, which is a valid tensormap operand space for cp.async.bulk.tensor).
    template <typename Params>
    inline __device__ void run(Params const& params, Shared* shared,
        CUtensorMap const* desc_q, CUtensorMap const* desc_k, CUtensorMap const* desc_v)
    {
        fused_multihead_attention::Single_cta<Kernel_traits::VERSION> const binfo(
            params, blockIdx.z, blockIdx.y, 0, /*tidx=*/0);
        if (binfo.stop_early(0))
        {
            return;
        }

        int const kv_loop_end = binfo.actual_kv_seqlen;

        Cbw_q cbw_q(&shared->q_barriers);
        Cbw_k cbw_k(&shared->k_barriers);
        Cbw_v cbw_v(&shared->v_barriers);

        // --- Q load: once per CTA -------------------------------------------
        //
        // The Host::init_params CUtensorMaps are 3D over tensors of shape
        // (D, H_or_HKV, total_seqlen), so the cp.async.bulk.tensor call is 3D
        // with a 3-element coord array. Coord ordering matches that layout:
        //   coord[0] = D offset (always 0; the box currently covers the full
        //              head dim -- see phase 6c: this must become chunked)
        //   coord[1] = head index (this CTA's bidh)
        //   coord[2] = seq position
        int const bidh    = static_cast<int>(blockIdx.y);
        int const bidh_kv = bidh / static_cast<int>(params.h_q_per_kv);

        {
            int const q_slot = cbw_q.tmaReserve(elect_one_, TX_BYTES_Q);
            uint32_t const q_smem = __nvvm_get_smem_pointer(&shared->smem_q[q_slot][0]);
            uint32_t const q_bar = __nvvm_get_smem_pointer(cbw_q.barrier_ptr(q_slot));
            // Q starts at q_loop * STEP_Q rows; for now this kernel handles
            // one Q-tile per CTA so the row offset is blockIdx.x * STEP_Q.
            int32_t const coord[3] = {0, bidh, static_cast<int32_t>(blockIdx.x * STEP_Q)};
            utmaldg_3d_cta(desc_q, q_smem, q_bar, coord, elect_one_);
        }

        // --- KV loop: one (K, V) load per iter ------------------------------
        for (int kv_loop = 0; kv_loop < kv_loop_end; kv_loop += STEP_KV)
        {
            // K
            int const k_slot = cbw_k.tmaReserve(elect_one_, TX_BYTES_K);
            uint32_t const k_smem = __nvvm_get_smem_pointer(&shared->smem_k[k_slot][0]);
            uint32_t const k_bar = __nvvm_get_smem_pointer(cbw_k.barrier_ptr(k_slot));
            int32_t const k_coord[3] = {0, bidh_kv, kv_loop};
            utmaldg_3d_cta(desc_k, k_smem, k_bar, k_coord, elect_one_);

            // V
            int const v_slot = cbw_v.tmaReserve(elect_one_, TX_BYTES_V);
            uint32_t const v_smem = __nvvm_get_smem_pointer(&shared->smem_v[v_slot][0]);
            uint32_t const v_bar = __nvvm_get_smem_pointer(cbw_v.barrier_ptr(v_slot));
            int32_t const v_coord[3] = {0, bidh_kv, kv_loop};
            utmaldg_3d_cta(desc_v, v_smem, v_bar, v_coord, elect_one_);
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
    // varying) before the kernel launch. Builds three driver-API CUtensorMaps
    // (cuTensorMapEncodeTiled) for Q / K / V. These are passed into the kernel
    // as __grid_constant__ params (see halfspec_smoke.cu).
    //
    // Why the driver API (not the fmha_v2 hand-rolled fmha::cudaTmaDesc):
    //   The 64-byte hand-rolled descriptor uses a Hopper-era bit layout that
    //   Blackwell's TMA engine rejects -- UTMALDG faults at runtime. The
    //   128-byte CUtensorMap from cuTensorMapEncodeTiled is the only portable,
    //   Blackwell-valid form (it is what the shipping trtllmGenKernels FMHA
    //   uses). Proven via reproducer on GB10 (sm_121).
    //
    // Each descriptor:
    //   * 3D tensor (D, H_or_HKV, total_seqlen), fastest-varying axis = head
    //     dim (contiguous), and a 3D box of (D, 1, STEP_Q / STEP_KV).
    //   * Element format BFloat16 / Float16 (2-byte).
    //
    // Swizzle note: cuTensorMapEncodeTiled requires the leading box dim *in
    // bytes* (boxDim[0] * ELEMENT_BYTES) to be <= the swizzle width. For
    // head_dim=256 BF16 that is 512 bytes, which exceeds the 128B max, so the
    // whole-head-dim box only encodes with SWIZZLE_NONE. Matching a 128B
    // swizzle (what the consumer Smem_tile ldmatrix wants) requires splitting
    // the head-dim load into <=64-element chunks -- that is the remaining
    // producer/consumer layout-matching work (see README phase plan). For now
    // we pick the widest swizzle the leading dim permits.
    //
    // Scope of v0:
    //   * BF16 / FP16 only (2-byte). FP8 needs U8 format + V-transpose.
    //   * PACKED_QKV input layout only.
    //   * No TMA store -- epilogue still uses the scalar STG Gmem_tile_o path.
    ////////////////////////////////////////////////////////////////////////////////////////////////

    struct Host
    {
        Host() = default;

        // Pick the widest swizzle the leading box dim (bytes) permits.
        static CUtensorMapSwizzle pick_swizzle(uint32_t lead_bytes)
        {
            if (lead_bytes % 128 == 0 && lead_bytes <= 128)
                return CU_TENSOR_MAP_SWIZZLE_128B;
            if (lead_bytes % 64 == 0 && lead_bytes <= 64)
                return CU_TENSOR_MAP_SWIZZLE_64B;
            if (lead_bytes % 32 == 0 && lead_bytes <= 32)
                return CU_TENSOR_MAP_SWIZZLE_32B;
            return CU_TENSOR_MAP_SWIZZLE_NONE;
        }

        // Encode one 3D tiled descriptor. tensor_size / box_size are in
        // elements (fastest-varying first); the global stride of dim>=1 is in
        // bytes (dim 0 is implicitly contiguous at element size).
        static void encode(CUtensorMap& out, void* gmem_ptr, uint32_t const (&tensor_size)[3],
            uint64_t seq_stride_bytes, uint32_t const (&box_size)[3])
        {
            CUtensorMapDataType fmt = (Kernel_traits::ELEMENT_BYTES == 2)
                ? CU_TENSOR_MAP_DATA_TYPE_BFLOAT16
                : CU_TENSOR_MAP_DATA_TYPE_FLOAT32;

            // globalDim fastest-first: (D, H, seq).
            uint64_t global_dim[3] = {tensor_size[0], tensor_size[1], tensor_size[2]};
            // globalStrides are for dims 1.. (dim 0 implicit). Bytes.
            //   dim1 (head)  stride = D * ELEMENT_BYTES (next head)
            //   dim2 (seq)   stride = seq_stride_bytes
            uint64_t global_stride[2]
                = {static_cast<uint64_t>(tensor_size[0]) * Kernel_traits::ELEMENT_BYTES, seq_stride_bytes};
            uint32_t box_dim[3] = {box_size[0], box_size[1], box_size[2]};
            uint32_t elem_stride[3] = {1, 1, 1};

            uint32_t const lead_bytes = box_size[0] * Kernel_traits::ELEMENT_BYTES;
            CUtensorMapSwizzle swizzle = pick_swizzle(lead_bytes);

            CUresult res = cuTensorMapEncodeTiled(&out, fmt, /*rank=*/3, gmem_ptr, global_dim,
                global_stride, box_dim, elem_stride, CU_TENSOR_MAP_INTERLEAVE_NONE, swizzle,
                CU_TENSOR_MAP_L2_PROMOTION_L2_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
            if (res != CUDA_SUCCESS)
            {
                char const* err = nullptr;
                cuGetErrorString(res, &err);
                fprintf(stderr,
                    "[halfspec] cuTensorMapEncodeTiled failed: %s "
                    "(dim=%u,%u,%u box=%u,%u,%u lead_bytes=%u swizzle=%d)\n",
                    err, tensor_size[0], tensor_size[1], tensor_size[2], box_size[0], box_size[1],
                    box_size[2], lead_bytes, static_cast<int>(swizzle));
            }
        }

        template <typename Params, typename Launch_params>
        void init_params(Params& params, Launch_params const& launch_params, CUtensorMap& tma_q,
            CUtensorMap& tma_k, CUtensorMap& tma_v) const
        {
            uint32_t const d = params.d;
            uint32_t const dv = params.dv;
            uint32_t const h = params.h;
            uint32_t const h_kv = params.h_kv;

            uint32_t const total_seqlen = params.is_s_padded
                ? static_cast<uint32_t>(params.b * params.s)
                : static_cast<uint32_t>(launch_params.total_q_seqlen);

            static_assert(Kernel_traits::ELEMENT_BYTES == 2,
                "halfspec v0 only supports BF16 / FP16 (2-byte elements).");
            static_assert(STEP_Q <= 256 && STEP_KV <= 256,
                "TMA box dimensions are capped at 256 elements per axis.");

            char* const q_ptr = reinterpret_cast<char*>(params.qkv_ptr);
            // PACKED_QKV (H_q + H_kv + H_kv heads of D elements):
            char* const k_ptr = q_ptr + h * d * Kernel_traits::ELEMENT_BYTES;
            char* const v_ptr = k_ptr + h_kv * d * Kernel_traits::ELEMENT_BYTES;

            // ---- Q ----
            uint32_t const tensor_size_q[3] = {d, h, total_seqlen};
            uint32_t const box_size_q[3] = {static_cast<uint32_t>(Kernel_traits::D), 1, STEP_Q};
            encode(tma_q, q_ptr, tensor_size_q, static_cast<uint64_t>(params.q_stride_in_bytes), box_size_q);

            // ---- K ----
            uint32_t const tensor_size_k[3] = {d, h_kv, total_seqlen};
            uint32_t const box_size_k[3] = {static_cast<uint32_t>(Kernel_traits::D), 1, STEP_KV};
            encode(tma_k, k_ptr, tensor_size_k, static_cast<uint64_t>(params.k_stride_in_bytes), box_size_k);

            // ---- V ----
            uint32_t const tensor_size_v[3] = {dv, h_kv, total_seqlen};
            uint32_t const box_size_v[3] = {static_cast<uint32_t>(Kernel_traits::DV), 1, STEP_KV};
            encode(tma_v, v_ptr, tensor_size_v, static_cast<uint64_t>(params.v_stride_in_bytes), box_size_v);
        }
    };
};

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace ws_sm120
} // namespace fmha

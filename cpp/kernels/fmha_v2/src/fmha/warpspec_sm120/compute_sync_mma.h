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

// Consumer (sync-MMA) half of the sm_120/sm_121 warp-specialized FMHA.
//
// Math is identical to fused_multihead_flash_attention_kernel_noloop_tiled.h
// (BMM1 + softmax + skip-softmax + BMM2 via fmha::gemm), the only delta is:
//   - replace gmem_q.load(smem_q) + fmha::ldgdepbar with cbr_q.wait()
//   - replace __syncthreads() (between load and compute) with the mbarrier
//     completion guaranteed by cbr_*.wait()
//   - signal cbr_*.complete() so the producer can recycle the smem entry
//
// The compute warps still issue all softmax / exp / max-reduce work
// themselves, and the per-warp skip-softmax vote (this branch's main
// contribution) lives unchanged inside the BMM1->softmax transition.
//
// Intentionally header-only and templated on Kernel_traits so the existing
// fmha_v2 setup.py machinery can stamp out instantiations per
// (D, S, mask, dtype) tuple.

#include <fmha/gemm.h>
#include <fmha/softmax.h>
#include <fmha/utils.h>
#include <fmha/warpspec/circular_buffer.h>

namespace fmha
{
namespace ws_sm120
{

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Kernel_traits>
struct Compute
{
    using Shared = typename Kernel_traits::Shared;

    // Reuse the ring readers from the warpspec dir.  These are dtype-agnostic
    // and built on fmha::Arrive_wait, which compiles to mbarrier.* PTX
    // (CC 9.0+, inclusive of sm_120 / sm_121).
    using Cbr_q = typename Kernel_traits::Circular_buffer_q_reader;
    using Cbr_k = typename Kernel_traits::Circular_buffer_k_reader;
    using Cbr_v = typename Kernel_traits::Circular_buffer_v_reader;

    using Traits_p = typename Kernel_traits::Traits_p;
    using Traits_o = typename Kernel_traits::Traits_o;
    using Cta_tile_p = typename Kernel_traits::Cta_tile_p;
    using Cta_tile_o = typename Kernel_traits::Cta_tile_o;
    using Mma_tile_p = typename Traits_p::template Mma_tile<Cta_tile_p>;
    using Mma_tile_o = typename Traits_o::template Mma_tile<Cta_tile_o>;

    using Smem_tile_q = typename Kernel_traits::Smem_tile_q;
    using Smem_tile_k = typename Kernel_traits::Smem_tile_k;
    using Smem_tile_v = typename Kernel_traits::Smem_tile_v;

    using Gmem_tile_o = typename Kernel_traits::Gmem_tile_o;
    using Smem_tile_o = typename Kernel_traits::Smem_tile_o;

    using Softmax = fmha::Softmax<Traits_p, Cta_tile_p, Kernel_traits>;

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
        CAUSAL_MASK = Kernel_traits::CAUSAL_MASK
    };
    enum
    {
        ENABLE_SKIP_SOFTMAX = Kernel_traits::ENABLE_SKIP_SOFTMAX
    };

    inline __device__ Compute() {}

    // Single consumer warp body. The producer warp(s) are in dma_sync_mma.h.
    //
    // warp_id_in_compute_group: 0..NUM_COMPUTE_WARPS-1 (NOT the global warp id;
    //   the dispatcher in the top-level kernel already subtracted the
    //   producer-warp count).
    // tidx: lane within the consumer warp(s) (0..127 for a 4-warp consumer
    //   group; or 0..31 if we go to single-warp consumers).
    template <typename Params>
    inline __device__ void run(int warp_id_in_compute_group, int tidx, Shared* shared, Params const& params)
    {
        // --- Iteration boundaries: identical to noloop_tiled.h --------------
        Block_info_padded<Cta_tile_p::WARPS_N> const binfo(params, blockIdx.y, blockIdx.z, blockIdx.x);
        if (binfo.stop_early(params.is_s_padded))
        {
            return;
        }

        int const kv_loop_start = 0;
        int const kv_loop_end = binfo.actual_kv_seqlen;

        // --- Skip-softmax: precompute log(threshold/L) once -----------------
        //
        // Same expression as noloop_tiled.h:
        //   skip_softmax_log_threshold = __logf(scale_factor / actual_kv_seqlen)
        // Per-row predicate becomes a single FP compare.
        float const skip_softmax_log_threshold = ENABLE_SKIP_SOFTMAX
            ? __logf(params.skip_softmax_threshold_scale_factor / static_cast<float>(binfo.actual_kv_seqlen))
            : 0.0f;

        // --- Ring readers ---------------------------------------------------
        //
        // The constructors hand us Arrive_wait views onto the shared barrier
        // arrays so we can .wait() and .complete() on individual ring slots.
        Cbr_q cbr_q(&shared->q_barriers);
        Cbr_k cbr_k(&shared->k_barriers);
        Cbr_v cbr_v(&shared->v_barriers);

        // --- Wait for Q tile to be produced --------------------------------
        //
        // Producer arrives on entryProducedBarriers[ptr] after its
        // cp.async.bulk.tensor.2d completes. Mbarrier completion implies a
        // memory fence, so the smem buffer is observable.
        int const q_slot = cbr_q.wait();
        Smem_tile_q smem_q(shared->smem_q[q_slot]);

        // --- BMM1 / softmax / BMM2 loop -------------------------------------
        //
        // This is structurally a verbatim copy of noloop_tiled.h's kv loop,
        // with two substitutions:
        //   * gmem_q.load() / gmem_k.load() / gmem_v.load() -> ring waits
        //   * fmha::ldgdepbar<USE_LDGSTS>() + __syncthreads() -> mbarrier
        //
        // The per-warp skip-softmax vote, the BMM2 split (skip-path /
        // no-skip-path), and the softmax tail all carry through unchanged.
        // See noloop_tiled.h lines ~470-720 for the reference body.
        //
        // TODO(dcampora,sm120-ws): port the body. The math is unchanged so
        // this is a mechanical rewrite, but the smem-pointer plumbing
        // (Smem_tile_k/v constructed from shared->smem_k[k_slot] each iter)
        // wants careful sizing of the ring entry stride.

        // ----- Accumulators / running softmax state ------------------------
        float global_max[Softmax::ROWS_PER_THREAD];
        float global_sum[Softmax::ROWS_PER_THREAD];
        // tmp[] holds the current tile's per-row max for the skip-softmax
        // predicate (see noloop_tiled.h).
        float tmp[Softmax::ROWS_PER_THREAD];

#pragma unroll
        for (int i = 0; i < Softmax::ROWS_PER_THREAD; i++)
        {
            global_max[i] = -std::numeric_limits<float>::max();
            global_sum[i] = 0.f;
        }

        bool first_step = true;
        for (int kv_loop = kv_loop_start; kv_loop < kv_loop_end; kv_loop += STEP_KV)
        {
            // Wait for this iter's K slot.
            int const k_slot = cbr_k.wait();
            Smem_tile_k smem_k(shared->smem_k[k_slot]);

            // ---- BMM1: Q x K' via fmha::gemm (sync mma.sync.aligned) ------
            // Same fragment loads + fmha::gemm pattern as noloop_tiled.h.
            //
            // TODO(dcampora,sm120-ws): copy the BMM1 main-loop body verbatim
            // from noloop_tiled.h lines ~225-310 (the
            // `for (int ki = 0; ki < Mma_tile_p::MMAS_K; ++ki)` block).
            // Fragment loads are `smem_q.load(frag_q[ki], ki); smem_k.load(...)`.

            // ---- Done with K for this iter: release the ring slot. -------
            cbr_k.complete(tidx == 0, k_slot);

            // ---- Per-warp skip-softmax vote (this branch's contribution) -
            //
            // Identical to noloop_tiled.h post-BMM2-split, post-per-warp port:
            //   bool skip = ((global_max[0] - tmp[0]) < log_threshold);
            //   for (i = 1; ...) skip = skip & (...);
            //   tile_negligible = __all_sync(0xffffffff, skip);
            //
            // (Outside first_step.)

            // ---- Softmax / mask / running max+sum update ------------------
            //   See noloop_tiled.h ~lines 490-595.

            // ---- BMM2 main + tail with BMM2-split based on tile_negligible -
            //   See noloop_tiled.h ~lines 612-720. Both the skip path
            //   (V-load pipeline + syncs only) and the no-skip path (full
            //   fmha::gemm over MMAS_K) carry through unchanged; just swap
            //   the V smem_v construction to use cbr_v.wait() + cbr_v.complete().

            int const v_slot = cbr_v.wait();
            Smem_tile_v smem_v(shared->smem_v[v_slot]);
            // ... BMM2 body here ...
            cbr_v.complete(tidx == 0, v_slot);

            first_step = false;
        }

        cbr_q.complete(tidx == 0, q_slot);

        // ---- Epilogue: write O ----------------------------------------------
        // Same as noloop_tiled.h's final epilogue; can be lifted verbatim.
        // TODO(dcampora,sm120-ws).
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace ws_sm120
} // namespace fmha

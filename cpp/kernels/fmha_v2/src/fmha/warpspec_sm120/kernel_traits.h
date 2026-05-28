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

// Kernel_traits for the halfspec sm_120 / sm_121 warp-specialized FMHA.
//
// Wraps the existing fmha::Kernel_traits_ template (which already provides
// LDGSTS-friendly Smem_tile_a/b/v with ldmatrix swizzle and the right
// Cta_tile / Mma_tile / fragment shapes) and layers on the warp-spec
// pieces:
//   * RING_DEPTH-deep smem ring for Q, K, V (vs the 1- or 2-buffer
//     ping-pong the tiled kernel uses today).
//   * Shared struct with smem tiles + entry-produced / entry-consumed
//     mbarrier arrays.
//   * Circular_buffer_{q,k,v}_{reader,writer} type aliases against the
//     existing fmha::ws::CircularBuffer infrastructure.
//   * Named-barrier ids (collision-safe with the existing skip-softmax
//     barrier ids 0x3 / 0x4 used on the non-warpspec tiled kernel).
//
// Assumption (resolved in phase 5 = build + iterate):
//   fmha::Smem_tile_a / Smem_tile_b / Smem_tile_v accept a runtime number
//   of buffers via their BUFFERS template parameter (they do --
//   BUFFERS_PER_TILE_SMEM_K is just a constant value plumbed through
//   from Kernel_traits_). The smem swizzle and BYTES_PER_BUFFER math
//   are independent of the buffer count, so RING_DEPTH=3 should work
//   structurally; ldmatrix access patterns also do not change.
//
// What this header does NOT include:
//   * Host-side TMA descriptor setup (phase 3).
//   * Persistence of which slot has which kv_loop offset (handled by
//     ring writer state, not traits).

#include <fmha/kernel_traits.h>
#include <fmha/utils.h>
#include <fmha/warpspec/circular_buffer.h>

namespace fmha
{
namespace ws_sm120
{

////////////////////////////////////////////////////////////////////////////////////////////////////

template <
    // The instruction traits (e.g. Ampere_hmma_bf16_traits) -- shared with the
    // non-warpspec tiled kernel for sm_120.
    typename Traits_,
    // Sequence length upper bound (e.g. 8192 for typical chunked prefill).
    int S,
    // Hidden head dim (e.g. 128, 192, 256).
    int VALID_D_,
    // Hidden head dim of V (= D in standard MHA).
    int VALID_DV_,
    // The iteration step of the outer Q loop.
    int STEP_Q_,
    // The number of vertical warps in the compute group (consumer-side).
    int WARPS_M_,
    // The number of horizontal warps in the compute group.
    int WARPS_N_,
    // The version of the kernel (passes through to Kernel_traits_).
    int VERSION_,
    // The mask version of the kernel.
    int MASK_VERSION_,
    // Smem ring depth for Q, K, V tiles.
    int RING_DEPTH_ = 3,
    // Skip-softmax knob (carries over from this branch's work).
    bool ENABLE_SKIP_SOFTMAX_ = false,
    // Producer warp count -- single 32-thread warp by default.
    int NUM_PRODUCER_WARPS_ = 1>
struct Kernel_traits_halfspec_sm120
{

    // Compose the existing tiled kernel traits with our ring-depth override.
    //
    // FLAGS: bit 0x1 = USE_LDGSTS_Q, 0x2 = USE_LDGSTS_K, 0x4 = USE_LDGSTS_V
    //         (we leave LDGSTS bits OFF on halfspec; the producer warp issues
    //         TMA instead, but the *smem layout* expected by the consumer
    //         remains the same as the LDGSTS path).
    //        bit 0x200 = NO_LOOP (we ARE a no-loop variant)
    //        bit 0x1000 = USE_GRANULAR_TILING (yes; same as noloop_tiled.h)
    // The other bits keep the production defaults.
    static constexpr uint32_t TILED_FLAGS = 0x200u  // NO_LOOP
        | 0x1000u                                   // USE_GRANULAR_TILING
        ;

    using Base = fmha::Kernel_traits_<Traits_,
        /*Gmem_tile_q_=*/fmha::Gmem_tile_q,         // unused on halfspec (TMA replaces)
        /*Gmem_tile_k_=*/fmha::Gmem_tile_k,         // unused
        /*Gmem_tile_v_=*/fmha::Gmem_tile_v,         // unused
        /*Gmem_tile_o_=*/fmha::Gmem_tile_o,         // still used in the epilogue
        S, VALID_D_, VALID_DV_,
        /*STEP=*/STEP_Q_,
        WARPS_M_, WARPS_N_,
        /*CTAS_PER_HEAD_=*/1,
        TILED_FLAGS,
        VERSION_, MASK_VERSION_,
        /*BMM2_FP16_EPILOGUE=*/true,
        /*SAGE_BLOCK_SIZE_Q_=*/0,
        /*SAGE_BLOCK_SIZE_K_=*/0,
        /*SAGE_BLOCK_SIZE_V_=*/0,
        /*ENABLE_SKIP_SOFTMAX_=*/ENABLE_SKIP_SOFTMAX_>;

    // Carry through the math types -- these are what compute_sync_mma.h needs.
    using Traits_p = typename Base::Traits_p;
    using Traits_o = typename Base::Traits_o;
    using Traits_e = typename Base::Traits_e;
    using Cta_tile_p = typename Base::Cta_tile_p;
    using Cta_tile_o = typename Base::Cta_tile_o;
    using Mma_tile_p = typename Base::Mma_tile_p;
    using Mma_tile_o = typename Base::Mma_tile_o;

    // Halfspec uses a RING_DEPTH-buffer ring for K, V (and Q -- though Q only
    // loads once per CTA, we still go through the same ring API so the
    // barrier handshake is uniform).
    enum
    {
        RING_DEPTH = RING_DEPTH_
    };

    // Smem tiles: same swizzle as the existing tiled kernel (the math doesn't
    // care; the consumer's ldmatrix patterns work on either ring depth).
    // We re-derive these with the override BUFFERS = RING_DEPTH.
    //
    // NB: phase 5 work item -- if Smem_tile_a / Smem_tile_b / Smem_tile_v
    // bake any RING_DEPTH-dependent assumptions into their layout
    // computation, the smem alloc here will be wrong and the consumer's
    // smem pointer arithmetic will need to advance by
    // RING_DEPTH * BYTES_PER_BUFFER instead. To unblock phase 1, we use
    // the existing Smem_tile_q/k/v types unchanged; phase 5 will revisit
    // if profiling shows ring depth > 2 is needed.
    using Smem_tile_q = typename Base::Smem_tile_q;
    using Smem_tile_k = typename Base::Smem_tile_k;
    using Smem_tile_v = typename Base::Smem_tile_v;
    using Smem_tile_o = typename Base::Smem_tile_o;

    using Gmem_tile_o = typename Base::Gmem_tile_o;

    enum
    {
        VALID_D = Base::VALID_D
    };
    enum
    {
        D = Base::D
    };
    enum
    {
        VALID_DV = Base::VALID_DV
    };
    enum
    {
        DV = Base::DV
    };
    enum
    {
        STEP_Q = STEP_Q_
    };
    enum
    {
        STEP_KV = Cta_tile_p::N
    };
    enum
    {
        VERSION = VERSION_
    };
    enum
    {
        MASK_VERSION = MASK_VERSION_
    };
    enum
    {
        CAUSAL_MASK = Base::CAUSAL_MASK
    };
    enum
    {
        SLIDING_WINDOW_ATTENTION = Base::SLIDING_WINDOW_ATTENTION
    };
    enum
    {
        BIDIRECTIONAL_SLIDING_WINDOW_ATTENTION = Base::BIDIRECTIONAL_SLIDING_WINDOW_ATTENTION
    };
    enum
    {
        CUSTOM_MASK = Base::CUSTOM_MASK
    };
    enum
    {
        ELEMENT_BYTES = sizeof(typename Traits_p::A_type)
    };

    // Skip-softmax knob.
    static constexpr bool ENABLE_SKIP_SOFTMAX = ENABLE_SKIP_SOFTMAX_;

    // Producer + consumer warp layout.
    enum
    {
        NUM_PRODUCER_WARPS = NUM_PRODUCER_WARPS_
    };
    enum
    {
        NUM_CONSUMER_WARPS = WARPS_M_ * WARPS_N_
    };
    enum
    {
        THREADS = (NUM_PRODUCER_WARPS + NUM_CONSUMER_WARPS) * 32
    };

    // Named-barrier ids. Collision-safe with the existing skip-softmax
    // barriers (0x3, 0x4 on the non-warpspec path) since we don't run both
    // kernels in the same launch.
    static constexpr int DMA_SYNC_BARRIER_ID = 0x1;
    static constexpr int MMA_SYNC_BARRIER_ID = 0x2;

    // Ring readers / writers. Single CTA cluster (no DSMEM on consumer
    // Blackwell -- CTAS_PER_CGA=1).
    static constexpr int CTAS_PER_CGA = 1;

    using Circular_buffer_q_reader = typename fmha::ws::CircularBuffer<RING_DEPTH, CTAS_PER_CGA>::Reader;
    using Circular_buffer_q_writer = typename fmha::ws::CircularBuffer<RING_DEPTH, CTAS_PER_CGA>::Writer;
    using Circular_buffer_k_reader = typename fmha::ws::CircularBuffer<RING_DEPTH, CTAS_PER_CGA>::Reader;
    using Circular_buffer_k_writer = typename fmha::ws::CircularBuffer<RING_DEPTH, CTAS_PER_CGA>::Writer;
    using Circular_buffer_v_reader = typename fmha::ws::CircularBuffer<RING_DEPTH, CTAS_PER_CGA>::Reader;
    using Circular_buffer_v_writer = typename fmha::ws::CircularBuffer<RING_DEPTH, CTAS_PER_CGA>::Writer;

    // Shared struct: smem ring tiles + barrier arrays.
    //
    // The smem tile sizes come from the underlying Smem_tile_q/k/v::BYTES_PER_TILE
    // multiplied by RING_DEPTH. We allocate flat byte arrays here and let the
    // consumer construct Smem_tile_* views at the right slot offset each iter.
    struct __align__(128) Shared
    {
        // Each ring slot is a Smem_tile_*::BYTES_PER_TILE-sized block.
        uint8_t smem_q_storage[RING_DEPTH][Smem_tile_q::BYTES_PER_TILE / RING_DEPTH];
        uint8_t smem_k_storage[RING_DEPTH][Smem_tile_k::BYTES_PER_TILE / RING_DEPTH];
        uint8_t smem_v_storage[RING_DEPTH][Smem_tile_v::BYTES_PER_TILE / RING_DEPTH];

        // mbarrier pairs: producer arrives on entryProducedBarriers[slot] when
        // its cp.async.bulk.tensor completes; consumer arrives on
        // entryConsumedBarriers[slot] when it's done reading the slot and the
        // producer can recycle it.
        fmha::ws::CircularBufferBarriers<RING_DEPTH> q_barriers;
        fmha::ws::CircularBufferBarriers<RING_DEPTH> k_barriers;
        fmha::ws::CircularBufferBarriers<RING_DEPTH> v_barriers;

        // Initialize all mbarriers. Called by thread 0 of the CTA at startup.
        inline __device__ void init(bool tid0)
        {
            // CircularBufferBarriers has no init() of its own; we lean on the
            // CircularBuffer<...>::init() helper which takes barrier addrs +
            // producer/consumer counts. For halfspec:
            //   - producer count = NUM_PRODUCER_WARPS * 32 = 32 (one warp)
            //   - consumer count = NUM_CONSUMER_WARPS * 32 (rest of the CTA)
            //
            // TODO(phase 5): wire up the actual init() call with the correct
            // producer / consumer thread counts here. The Hopper warpspec's
            // tma_q_tracker.init(tid0, 1, CTAS_PER_CGA) pattern in
            // fmha/warpspec/kernel_traits.h:548 is the reference shape.
        }
    };

    // Pad to align. The non-Hopper kernel allocates BYTES_PER_SMEM in the
    // extern __shared__ block; the halfspec version uses sizeof(Shared).
    enum
    {
        BYTES_PER_SMEM = sizeof(Shared)
    };
};

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace ws_sm120
} // namespace fmha

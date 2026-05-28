# warpspec_sm120 — TMA-load + sync-MMA warp-specialized FMHA for sm_120 / sm_121

## What this is

A *partial* port of the Hopper warp-specialized FMHA pattern to consumer/integrated
Blackwell (GeForce sm_120, Spark/GB10 sm_121).

Hopper's warpspec FMHA combines two ideas:
1. **TMA-driven async loads** with mbarrier completion (producer warp).
2. **Async MMA via `wgmma.async`** so compute warps don't block on tensor cores.

Consumer Blackwell exposes (1) — TMA + mbarrier are PTX features documented as
`CC 9.0+`, which is inclusive of sm_120 / sm_121 — but does **not** expose (2);
there is no `wgmma.async`-equivalent on sm_120 (confirmed by user). Compute warps
must use `mma.sync.aligned.m16n8k16.*` and block on the result registers.

So this port adopts the producer/consumer split and the TMA load path, but
keeps the compute body identical to the existing `noloop_tiled.h` kernel that
already uses `fmha::gemm()` (sync MMA, via `Fragment_accumulator::mma()` in
`fmha/fragment.h`).

### Expected wins on sm_120

- **Fewer load instructions**: a single TMA descriptor + `cp.async.bulk.tensor.2d`
  replaces what is currently ~`Nrows × Ncols / 16B` `LDGSTS` instructions per
  K-tile.
- **Per-buffer mbarrier wait** instead of CTA-wide `__syncthreads()` between
  load and compute, so consumer warps unblock as soon as their tile lands.
- **Multi-buffer ring** (depth ≥ 3) so the producer can stay ahead even when a
  consumer iter is long (BMM1 + softmax + BMM2).
- **`setmaxnreg.sync.aligned` register-budget split**: the producer warp gets a
  small register budget (it does no FP math), freeing physical registers for
  the compute warps to keep larger `acc_o` slices live.

### What this port does NOT do

- No async-MMA overlap. Each iter is still `mma_qk → softmax → mma_pv` on the
  consumer warps. We cannot get the same overlap Hopper does because `mma.sync`
  blocks on its output registers.
- Single-CTA only (no cluster, no DSMEM). Consumer Blackwell does not have
  cluster launch / DSMEM in the same form. We use `cp.async.bulk.tensor.*` with
  `.shared::cluster` modifier which falls back to single-CTA behaviour when
  `cluster_dim==1`.

## Files in this directory

| File | Role |
|------|------|
| `README.md` | This doc |
| `dma_sync_mma.h` | Producer-side TMA loader. Issues `cp.async.bulk.tensor.2d` for Q / K / V tiles into the circular smem ring; arrives on the entry-produced mbarrier when the descriptor completes. Reuses `fmha::hopper::utmaldg<2, TILED, false>` and `fmha::ws::CircularBufferWriter`. |
| `compute_sync_mma.h` | Consumer-side compute. Identical math to `noloop_tiled.h` (BMM1 + softmax + skip-softmax + BMM2) but waits on entry-produced mbarriers instead of `__syncthreads` and signals entry-consumed mbarriers when done. Uses `fmha::gemm()` (sync `mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32`). |

## How this kernel sits in the source tree

```
fused_multihead_flash_attention_kernel_ws_sm120.h  ← TOP-LEVEL ENTRY (see below)
  └── fmha/warpspec_sm120/
        ├── dma_sync_mma.h           (producer)
        └── compute_sync_mma.h       (consumer)
        ↓ relies on
  fmha/hopper/{tma_descriptor,tma_types,utils_tma}.h     (TMA descriptors + utmaldg)
  fmha/hopper/arrive_wait.h                              (mbarrier)
  fmha/warpspec/circular_buffer.h                        (ring)
  fmha/{gemm,fragment,softmax}.h                         (sync MMA + softmax)
```

The top-level entry header (sibling to the existing `noloop.h` /
`noloop_tiled.h`) does the warp-role dispatch:

```cpp
const int warp_group = threadIdx.x / 32;  // 32-thread warps, NOT 128-thread wg
if (warp_group == 0) {
    fmha::ws_sm120::DMA<Kernel_traits>::Device dma(elect_one);
    dma.run(params, shared);             // TMA producer
} else {
    fmha::ws_sm120::Compute<Kernel_traits>::Device compute;
    compute.run(warp_group - 1, tidx, shared, params);   // sync-MMA consumer
}
```

Note we use **32-thread warps** for role split, not 128-thread warp-groups: on
sm_120 without `wgmma`, the warp-group is not the natural unit of MMA issue.
A single 32-thread warp is enough to drive a `m16n8k16` mma.sync.

## Wiring into the build (setup.py)

The fmha_v2 `setup.py` generates kernel instantiations per
(data_type, head_dim, S, mask, sm). To enable this kernel for
sm_120 + sm_121 + BF16 + head_dim={128, 192, 256} + causal-mask,
the setup.py would need a new branch alongside the existing
warp-spec branch (currently `sm == 90` gated). That part is **not yet
written** in this prototype.

## Status

- [x] Design + skeleton (this README)
- [ ] `dma_sync_mma.h`: TMA load path — needs concrete TMA descriptor setup
      (specifically, the smem swizzle that maximises `ldmatrix.sync` BW for
      the consumer side, which is the `K_SMEM_STRIDED_8x128B` pattern that
      `noloop_tiled.h` already uses for K/V).
- [ ] `compute_sync_mma.h`: lift the BMM1+softmax+BMM2 body verbatim from
      `noloop_tiled.h`, replace `gmem_q.load` / `fmha::ldgdepbar` /
      `__syncthreads` with `cbr_q.wait()` / `cbr_q.complete()` from the ring.
- [ ] Top-level `fused_multihead_flash_attention_kernel_ws_sm120.h`: producer/
      consumer dispatch + shared-mem layout (`Shared` struct) +
      circular-buffer barrier init.
- [ ] `setup.py` integration: emit `_sm120` warpspec kernel variants.
- [ ] Smem layout finalization: pick swizzle for ldmatrix bank-conflict-free
      reads; ring depth 3 or 4 depending on smem budget on sm_120 (228 KB on
      GB10, vs 228 KB on H100).
- [ ] Correctness validation: golden-tensor diff against `noloop_tiled.h`
      output on a fixed (Q, K, V) input.
- [ ] Perf measurement against `noloop_tiled.h` baseline at L=49 152 on
      Qwen3.6-35B-A3B prefill.

## Why not use CuTe DSL / CUTLASS examples?

CUTLASS 4.x has sm_120-specific FMHA kernels in its examples tree that already
implement the TMA-load + sync-MMA pattern. Long term, the right move is to
port the TRT-LLM warpspec gate to dispatch to a CuTe-DSL implementation for
sm_120 (which is what the kernel-cute-specialist agent is for). This skeleton
exists as a stepping stone — to map out where the existing fmha_v2 infra
applies vs needs replacement — before committing to a CuTe rewrite.

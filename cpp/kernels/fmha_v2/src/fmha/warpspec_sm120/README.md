# halfspec — TMA-load + sync-MMA warp-specialized FMHA for sm_120 / sm_121

> Codename: **halfspec** = half of the Hopper warp-specialization recipe.
> TMA-driven async loads survive the port; async MMA does not (sm_120 has no
> `wgmma.async` equivalent — confirmed). Compute warps stay on `mma.sync`.

## Status (as of 2026-05-29)

**Builds, launches, and runs the full chunked TMA + sync-MMA pipeline to
completion on GB10 (sm_121) — 0 compute-sanitizer memcheck errors, 0 racecheck
hazards.** Numerics are not yet validated (epilogue is still a stub and the
runtest uses zeroed inputs); V's smem-swizzle match is still open. But the
producer/consumer chunked handshake — the structural heart of halfspec — is in
place and verified race-free. Files in this directory plus
`fused_multihead_flash_attention_kernel_ws_sm120.h` establish:

- the producer/consumer warp-role dispatch
- a Blackwell-valid TMA load: `cuTensorMapEncodeTiled` `CUtensorMap`
  descriptors (driver API) passed as `__grid_constant__` kernel params, +
  `cp.async.bulk.tensor.3d.shared::cta.global.tile` + `fmha::CircularBuffer*`
- the design rationale and what's reusable vs needs replacement

### What got fixed this session (2026-05-29)

1. **TMA descriptor format (was "Illegal Instruction" at UTMALDG.3D).**
   The fmha_v2 hand-rolled 64-byte `fmha::cudaTmaDesc` (Hopper-era bit layout)
   is **rejected by Blackwell's TMA engine**. Replaced with the driver API
   `cuTensorMapEncodeTiled` (128-byte `CUtensorMap`) — the same form the
   shipping trtllmGenKernels FMHA uses. Proven with a minimal reproducer:
   hand-rolled desc → illegal instruction; encode-tiled desc → loads correct
   data on sm_121.

2. **`setmaxnreg.{dec,inc}` unsupported on consumer Blackwell.** It is a
   Hopper / datacenter-Blackwell instruction (sm_90/100/103); ptxas
   *hard-errors* on sm_120/121 under CUDA 13.1 (older toolkits only warned).
   The producer/consumer register-budget split therefore **does not exist** on
   this hardware — guarded to `__CUDA_ARCH__ ∈ [900,1200)` (no-op on sm_120/121).

### 6c.1 — chunked producer + per-chunk granular handshake (DONE, 2026-05-29)

The consumer reuses the LDGSTS `Smem_tile_q/k/v` with `USE_GRANULAR_TILING`.
Each granular buffer holds only **`[STEP, D/2]`** worth (the contraction dim is
streamed in chunks across a 2-deep ping-pong). A single full-head-dim TMA box
(`[STEP, 256]`, 64 KB) was 2× the buffer → "Out-of-range shared address"; and
three full boxes (Q+K+V = 160 KB) exceed GB10's ~99 KB cap
(`sharedMemPerBlockOptin = 101376`).

Fixed by making the producer stream **chunks** that match the consumer's
granular buffers, and restructuring the handshake to per-chunk:

- `kernel_traits.h`: the granular double-buffer *is* the ring. `GRANULAR_DEPTH
  = Smem_tile::BUFFERS_PER_TILE (=2)`; `Shared` holds one flat
  `Smem_tile::BYTES_PER_TILE` region per tensor (`q_buf/k_buf/v_buf(slot)`
  index buffer `slot`); barriers are depth-2; the consumed barrier arrival
  count is `CONSUMER_THREADS` so every consumer thread arrives (doubles as the
  pre-recycle sync). Chunk counts: `NUM_BMM1_CHUNKS = 4` (head-dim/64),
  `NUM_BMM2_CHUNKS = 4` (kv-pos/32).
- `dma_sync_mma.h`: per kv-tile, stream Q/K head-dim chunks (box `(64,1,STEP)`,
  128B swizzle, `coord[0]=c*64`) and V kv-position chunks (box `(DV,1,32)`,
  `coord[2]=kv_loop+c*32`) into buffer `c % 2`. Producer kv-loop range matches
  the consumer's exactly (causal-aware) so the consumed-barrier handshake can't
  deadlock.
- `compute_sync_mma.h`: BMM1/BMM2 now stream chunks — per chunk: `cbr.wait()`,
  MMAs, all-thread `cbr.complete()` + `advance()`, `move_to_next_read_buffer()`.

Validated on GB10: builds, runs to completion, **0 memcheck errors, 0 racecheck
hazards**. Q/K use the proven-correct 128B swizzle; V currently uses
`SWIZZLE_NONE` (runs, but numerically unverified — see remaining work).

### Remaining for numeric correctness (6c.2 → 6d)

- **V smem-swizzle match.** `Smem_tile_v` is a different type than the Q/K
  `Smem_tile_a/b`; its expected layout must be checked the same way Q/K were
  (`tma_swizzle_verify.cu`) and the V descriptor swizzle/box adjusted to match.
- **Epilogue.** `compute_sync_mma.h` still stubs the O normalize + store; wire
  up `Smem_tile_o` + `Gmem_tile_o` (verbatim from `noloop_tiled.h`).
- **Numeric validation.** The runtest uses zeroed inputs; add non-zero inputs +
  a reference (or run in-engine) to confirm `gen_token` matches `noloop_tiled.h`.

**Viability DE-RISKED (2026-05-29): the port works with the existing consumer
smem tiles — no rewrite needed.** The make-or-break question was whether the
fmha_v2 LDGSTS `Smem_tile` XOR swizzle equals a TMA hardware swizzle mode. It
does: both Q and K granular tiles use `BYTES_PER_ROW=128`, `BYTES_PER_STS=16`,
`ROWS_PER_XOR_PATTERN=8`, `COLS_PER_XOR_PATTERN=1`, i.e. physical 16-byte chunk
`= (col/8) ^ (row % 8)` — which is **byte-identical to the TMA 128B hardware
swizzle**. Verified empirically on GB10 (`tma_swizzle_verify.cu`): a 128B-swizzle
TMA load of a known `[8, 64]` bf16 tile places every logical element at exactly
the fmha-predicted offset (e.g. phys row 1 = logical cols 8..15 then 0..7, the
`row%8` chunk swap). So a chunked 128B-swizzle TMA fills `Smem_tile_q/k`
directly and the consumer's ldmatrix reads correct data.

### Fast iteration loop (bypass the slow ~4 MB cmake build)

```
cd cpp/kernels/fmha_v2/src
nvcc -arch=sm_120f -std=c++17 -c -I . halfspec_smoke.cu -o /tmp/halfspec_smoke.o
nvcc -arch=sm_120f -std=c++17 -I . \
     /home/scratch.dcampora_gpu/projects/vllm-workspace/halfspec_runtest.cu \
     /tmp/halfspec_smoke.o -o /tmp/halfspec_runtest -lcuda
compute-sanitizer --tool memcheck /tmp/halfspec_runtest
```

(`sm_120f` is family-specific and runs on GB10/sm_121. Standalone driver-API
reproducers live next to the runtest: `tma_minrepro.cu` (hand-rolled vs
encode-tiled descriptor), `tma_encode_probe.cu` (swizzle/box constraints),
`tma_swizzle_verify.cu` (TMA-128B == fmha XOR swizzle proof), `halfspec_sizes.cu`
(smem slot / swizzle constants).)

## Why the naïve "just split warps" shortcut doesn't work

In the existing `noloop_tiled.h` kernel, `gmem_q.load(smem_q)` /
`gmem_k.load(smem_k)` / `gmem_v.load(smem_v)` are *multi-thread* LDGSTS
operations: each of the 128 threads in the CTA issues a few `LDGSTS`
instructions per load to cover the (Q-tile rows × D bytes) of data. There
is no way to "have warp 0 do the load" without rewriting the
gmem_tile/smem_tile load helpers — the partition is baked into them.

This is *exactly* what TMA fixes. A single descriptor + a single
`cp.async.bulk.tensor.2d` from one thread issues an entire tile load, and
the consumer waits on an `mbarrier`. So the v0 of halfspec **must** swap
LDGSTS → TMA. There's no LDGSTS-based stepping stone that saves work.

## Phased plan

| Phase | Scope | Realistic effort |
|------:|-------|-----------------:|
| 0 | This skeleton + design doc. | done |
| 1 | `Kernel_traits_halfspec_sm120` reusing existing tiled `Smem_tile_*` types (LDGSTS swizzle, ldmatrix-friendly). Author `Shared` struct, ring barrier arrays, named-barrier ids. | 1–2 days |
| 2 | Port the BMM1+softmax+skip-softmax+BMM2 kv-loop body verbatim from `noloop_tiled.h` into `Compute::run`. Replace `gmem_q/k/v.load + ldgdepbar + __syncthreads` calls with `cbr_*.wait()` / `cbr_*.complete()` against the ring. Math is bit-identical. | 1 day |
| 3 | Author the producer (`DMA::run`). Build the TMA descriptors host-side in `fused_multihead_attention.cpp` (existing helpers in `setup_tma_descriptors_*`); thread them through `Launch_params` into the kernel. Per-iter issues 2D `cp.async.bulk.tensor` for Q (once) and K, V (per iter). | 2 days |
| 4 | Wire into `setup.py` instantiation generator. Add a new branch alongside the existing Hopper warpspec block (~line 1700) that emits halfspec_sm120 kernels for BF16 + head_dim ∈ {128,192,256} + causal mask, gated on `sm == 120 ∥ sm == 121`. | 1 day |
| 5 | Build (`scripts/build_wheel.py`), iterate on compile errors and ptxas warnings. | done |
| 6a | TMA descriptor: hand-rolled `cudaTmaDesc` → `cuTensorMapEncodeTiled` `CUtensorMap` (Blackwell-valid). UTMALDG no longer faults. | **done (2026-05-29)** |
| 6b | Guard `setmaxnreg` off for sm_120/121 (unsupported there). | **done (2026-05-29)** |
| 6c.1 | **Chunked producer + per-chunk granular handshake.** Chunked `[STEP,64]` 128B-swizzle Q/K loads + `[DV,32]` V loads into the granular double-buffer; consumer streams chunks (per-chunk wait/complete + `move_to_next_read_buffer`). Runs to completion on GB10, 0 memcheck/racecheck. | **done (2026-05-29)** |
| 6c.2 | V smem-swizzle match (verify `Smem_tile_v` layout like Q/K; fix V descriptor) + epilogue (`Smem_tile_o`/`Gmem_tile_o` store). | 1–2 days |
| 6d | Numeric validation: non-zero inputs + reference, confirm `gen_token=13477` matches `noloop_tiled.h`. | 1 day |
| 6d | Correctness validation: 1-token gen on Qwen3.6-35B-A3B prefill at L=49 152, confirm `gen_token=13477` matches `noloop_tiled.h`. | 1 day |
| 7 | Performance sweep: `RING_DEPTH ∈ {2,3,4}`, ring placement, smem layout for ldmatrix bank-conflict-free reads. (Register-budget split is N/A on sm_120/121.) | 3–5 days |
| 8 | Compare to baseline + skip-softmax at L ∈ {8k,16k,24k,32k,40k,49k} with the same `bench_prefill_35b_trtllm.py` harness. | 1 day |
| **Total remaining** | | **~1.5–2 weeks** of focused engineering |

(Estimates assume sm_121 hardware available for iterative compile + run +
debug cycles. Without that, every phase from 5 onward stalls.)

## Files in this directory

| File | Role |
|------|------|
| `README.md` | This doc |
| `dma_sync_mma.h` | Producer. Issues `cp.async.bulk.tensor.3d.shared::cta.global.tile.mbarrier::complete_tx::bytes` (single-CTA, no cluster) for Q / K / V via the halfspec-local `utmaldg_3d_cta` helper + `CircularBufferWriter`. `DMA::Host::init_params` builds the three `CUtensorMap` descriptors with `cuTensorMapEncodeTiled`. **Current limitation:** loads one full-head-dim box per tile — must be chunked into `[STEP,64]` 128B loads (phase 6c). |
| `compute_sync_mma.h` | Consumer. The kv-loop body (BMM1 + softmax + skip-softmax + BMM2) is ported from `noloop_tiled.h`, reading the granular `Smem_tile_q/k/v` per ring slot. |

## What this port WILL win on sm_120

- **Fewer load instructions** — a single `cp.async.bulk.tensor.2d` per tile
  replaces ~N_rows × N_cols / 16B `LDGSTS` instructions.
- **Per-buffer-slot waits** (`mbarrier.try_wait.parity`) instead of
  CTA-wide `__syncthreads()` between load and compute. Consumer warps
  unblock as soon as their tile lands, not when *every* CTA load has
  drained.
- **Multi-buffer ring** (depth ≥ 3) gives the producer headroom to fly
  ahead of the consumers, hiding global memory latency more aggressively
  than the current 2-buffer ping-pong in `noloop_tiled.h`.
- ~~**`setmaxnreg.{dec,inc}.sync.aligned.u32`** register-budget split~~ —
  **NOT available on sm_120/121.** `setmaxnreg` is a Hopper / datacenter-
  Blackwell instruction; ptxas hard-errors on consumer Blackwell. The producer
  warp cannot give back registers, so the consumers do not gain a larger
  physical-register budget. (Confirmed 2026-05-29.)

## What this port WILL NOT win on sm_120

- **MMA / softmax overlap** — there's no `wgmma.async` on sm_120, so a
  consumer warp's `mma.sync` blocks its issuing thread until result
  registers are committed. The Hopper warpspec hides BMM1 MMA latency
  behind softmax issue and BMM2 MMA latency behind frag_p prep; we
  cannot do this on sm_120 with sync MMA only.

The expected uplift is therefore the load-side win plus the per-slot wait
win plus the register-budget win, NOT the MMA-overlap win. A reasonable
upper bound is the current 2-buffer ping-pong's `__syncthreads()`
overhead percentage of total kernel time, which is likely in the
single-digit-percent range at L=49 152.

## What this port is not (vs CUTLASS-CuTe alternative)

CUTLASS 4.x has Blackwell sm_120 FMHA examples that already implement the
TMA-load + sync-MMA pattern in CuTe DSL. The long-term right answer is to
port the TRT-LLM dispatcher to call into a CuTe-DSL kernel for sm_120 /
sm_121. This skeleton is **scaffolding** — it maps the relationship
between the existing fmha_v2 infra and the halfspec design, identifies
the gaps, and provides the kernel-shape decisions someone (or a
follow-up CuTe specialist agent) can build on.

## How to continue from here

1. **If iterating on this fmha_v2 path**: pick up phase 1 of the plan
   above. The first commit you should make is `Kernel_traits_halfspec_sm120`
   plus the `Shared` struct, *without* the kv-loop body or DMA filled in
   — that gives you a compile target for phase 2/3.

2. **If switching to CuTe DSL**: hand off to the
   `kernel-cute-specialist` agent with the context that you need a
   sm_120 / BF16 / head_dim=256 / causal FMHA, and that the dispatcher
   integration point is `cpp/kernels/fmha_v2/src/fused_multihead_attention.cpp`
   in `determine_launch_params` around line 462.

3. **If reverting**: this whole directory plus the entry-kernel header
   are isolated (no other code includes them) and can be deleted with no
   knock-on. The committed skip-softmax + BMM2-split + per-warp vote
   work on `noloop_tiled.h` is independent and stays valid as a
   ~−4 % win on the current sm_120 path.

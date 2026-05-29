# halfspec — TMA-load + sync-MMA warp-specialized FMHA for sm_120 / sm_121

> Codename: **halfspec** = half of the Hopper warp-specialization recipe.
> TMA-driven async loads survive the port; async MMA does not (sm_120 has no
> `wgmma.async` equivalent — confirmed). Compute warps stay on `mma.sync`.

## Status (as of 2026-05-29)

**Builds, runs, and is FULLY NUMERICALLY CORRECT on GB10 (sm_121).** A
host-reference harness (`halfspec_validate.cu`) confirms the complete kernel —
chunked-128B-TMA Q/K + dv-chunked-128B-TMA V + BMM1 + softmax + causal mask +
BMM2 + epilogue + scales — matches reference attention to bf16-rounding accuracy
(distinct-V `max_abs 6e-4`, V-const `max_abs 2e-4`), with 0 compute-sanitizer
memcheck errors and 0 racecheck hazards. The whole halfspec data path is now
TMA-driven and validated, and it **also builds cleanly in the production cmake
target** `_context_attention_kernels_120` (not just standalone nvcc). What
remains is in-engine integration (setup.py codegen, dispatcher wiring) and the
perf sweep. NB: the phase-6d end-to-end check (`gen_token=13477`) needs a real
Qwen3.6-35B-A3B run, which the current GB10 dev box can't host (no
`LLM_MODELS_ROOT`), so 6d's runtime validation must happen on a machine with the
model. Files in this directory plus
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

### V is an architectural blocker for the single-box TMA approach (verified 2026-05-29)

The V swizzle verification (the Q/K-style check applied to `Smem_tile_v`) came
back **negative**: V cannot be filled by the chunked-128B-TMA scheme that works
for Q/K. Why:

- `Smem_tile_v` (Ampere bf16) derives from the same `Smem_tile_without_skews`
  base, but with `LEAD_DIM = Cta_tile_o::N = DV = 256`, giving
  `BYTES_PER_ROW = 512` (vs 128 for Q/K) and `ROWS = 32` (kv-positions/chunk).
  The XOR swizzle (`chunk ^ (row%8)`) is still the 128B pattern, but it now
  repeats across **four** 128-byte segments within each 512-byte row.
- `cuTensorMapEncodeTiled` caps the **leading box dim at the swizzle width**
  (≤128 bytes). V's leading dim is the full DV (512 bytes), so the V box only
  encodes with `SWIZZLE_NONE` (confirmed: 128B/64B/32B all return
  `invalid argument`; only NONE is OK). `SWIZZLE_NONE` is plain row-major and
  does **not** match the consumer's XOR-swizzled read (`ldsmt` + offset XOR
  toggles), so it would compute wrong results.
- Issuing four per-segment 128B-swizzle TMAs cannot compose into the strided
  512-byte-row layout either: each `cp.async.bulk.tensor` writes a contiguous
  (128-byte-pitch) swizzled block to its smem destination; there is no
  smem-row-pitch argument to interleave segments into 512-byte rows.

So a correct V needs one of (a design fork — see "How to continue"):

1. **Consumer-side LDGSTS V (hybrid).** Keep Q/K on the TMA producer; load V in
   the consumer warps via the proven `Gmem_tile_v` + `Smem_tile_v` path
   (verbatim from `noloop_tiled.h`), using a consumer-group named barrier in
   place of `__syncthreads()` (the producer warp must not be caught in it).
   Lowest risk, correct; V just isn't TMA-accelerated (V loads are a minority
   vs the BMM1 Q/K loads).
2. **Custom non-swizzled V smem tile.** Store V row-major (`SWIZZLE_NONE`) via
   TMA and write a halfspec `Smem_tile_v` read that drops the XOR toggles.
   Correct but bank-conflicted; needs careful custom `ldsmt` addressing.
3. **Re-tile BMM2 so V uses 128-byte rows** (tile DV into 64-wide groups).
   **← chosen path.** Deepest change — ripples through `Mma_tile_o`, `frag_v`,
   the BMM2 loop and `acc_o`.
4. **CuTe DSL** (the README's long-term recommendation).

### Chosen V path: re-tile BMM2 to 64-wide DV chunks (6c.3, DONE + validated)

Instantiate the V smem tile with a `Cta_tile` whose `N = 64` (one DV chunk), so
`Smem_tile_v` gets `LEAD_DIM = 64` → `BYTES_PER_ROW = 128` — the **same
128-byte-row XOR-swizzle layout as K** (already proven == TMA 128B swizzle), and
the existing `Smem_tile_v_ampere_hmma` `N==64` read path applies unchanged (the
XOR-swizzled smem is read-method-agnostic; `ldsmt` works on it). Plan:

- Producer: per kv-tile, load V as `DV/64 = 4` dv-chunks, each a `[64 dv, STEP_KV]`
  128B-swizzle box (`coord[0] = d*64`) into a granular buffer — structurally
  identical to the K loads.
- Consumer BMM2: outer loop over the 4 dv-chunks; for each, contract all kv
  (`TOTAL_BMM2_MMAS_K` steps) accumulating into the `acc_o` columns for that
  chunk (`acc_o[*][d*4 .. d*4+4]`). Needs `fmha::gemm` into an `acc_o` sub-range
  per dv-chunk.
- Validate with `halfspec_validate.cu distinct` (currently the failing oracle).

### Status of the rest (DONE / validated)

- **mask.load bug — FIXED.** The consumer wasn't initializing the causal mask's
  query-row offset, so every Q-tile after the first masked the wrong diagonal.
- **Epilogue — DONE.** O normalize + `Smem_tile_o`/`Gmem_tile_o` store, with a
  consumer-group named barrier replacing `__syncthreads()`.
- **Numeric validation harness — DONE** (`halfspec_validate.cu`). V-const test
  PASSES at bf16 accuracy (BMM1+softmax+mask+BMM2+epilogue+scales correct);
  distinct-V test is the oracle for the V re-tile.

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
| 6c.2 | Epilogue (O store) + numeric harness; fixed the missing `mask.load` causal bug. BMM1+softmax+mask+BMM2+epilogue+scales validated (V-const PASS). V swizzle verified as a blocker (needs re-tile). | **done (2026-05-29)** |
| 6c.3 | Re-tile BMM2 to 64-wide DV chunks so V uses 128-byte rows (same proven layout as K). Producer streams 4 dv x 4 kv V sub-tiles; consumer BMM2 contracts per dv-chunk into the acc_o sub-range. distinct-V validates at bf16 accuracy. | **done (2026-05-29)** |
| 6d | In-engine integration: setup.py codegen branch + fmhaRunner dispatch (flag-gated); confirm `gen_token=13477` matches `noloop_tiled.h`. | 1–2 days |
| 7 | Perf sweep (RING_DEPTH, ring placement, ldmatrix bank conflicts). | 3–5 days |
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

# halfspec — TMA-load + sync-MMA warp-specialized FMHA for sm_120 / sm_121

> Codename: **halfspec** = half of the Hopper warp-specialization recipe.
> TMA-driven async loads survive the port; async MMA does not (sm_120 has no
> `wgmma.async` equivalent — confirmed). Compute warps stay on `mma.sync`.

## Status (as of 2026-05-28)

**Skeleton only.** Files in this directory plus
`fused_multihead_flash_attention_kernel_ws_sm120.h` establish:

- the producer/consumer warp-role dispatch
- the integration points against `fmha::Arrive_wait` (mbarrier),
  `fmha::utmaldg<2, TILED, false>` (TMA), and `fmha::CircularBuffer*` (ring)
- the design rationale and what's reusable vs needs replacement

What is **not** done: a buildable, correctness-validated kernel. The
remaining work is itemised in the **Phased plan** below with realistic
effort estimates.

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
| 5 | Build (`scripts/build_wheel.py`), iterate on compile errors and ptxas warnings. | 1–2 days |
| 6 | Correctness validation: run a 1-token gen on Qwen3.6-35B-A3B prefill at L=49 152 and confirm `gen_token=13477` matches the existing `noloop_tiled.h` output (this was the validated baseline on the skip-softmax sweep). | 1 day |
| 7 | Performance sweep: `CIRCULAR_BUFFER_DEPTH ∈ {2,3,4}`, `DMA_NREG` / `COMPUTE_NREG` budget split, ring placement, smem layout for ldmatrix bank-conflict-free reads. | 3–5 days |
| 8 | Compare to baseline + skip-softmax at L ∈ {8k,16k,24k,32k,40k,49k} with the same `bench_prefill_35b_trtllm.py` harness. | 1 day |
| **Total** | | **~2.5 weeks** of focused engineering |

(Estimates assume sm_121 hardware available for iterative compile + run +
debug cycles. Without that, every phase from 5 onward stalls.)

## Files in this directory

| File | Role |
|------|------|
| `README.md` | This doc |
| `dma_sync_mma.h` | Producer skeleton. Issues `cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes` for Q / K / V via `fmha::utmaldg<2, TILED, false>` and the existing `CircularBufferWriter`. TMA descriptor host-side setup is a TODO (phase 3). |
| `compute_sync_mma.h` | Consumer skeleton. The kv-loop shape is in place with per-slot wait/complete plumbing; the actual BMM1+softmax+BMM2 body is a TODO pointing at the noloop_tiled.h line range to lift verbatim (phase 2). |

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
- **`setmaxnreg.{dec,inc}.sync.aligned.u32`** register-budget split: the
  producer warp gets a small register cap (it only does coord math + TMA
  issue), the consumer warps inherit the freed physical registers and can
  keep more of `acc_o` live without spilling.

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

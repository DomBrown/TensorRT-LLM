Started 2026-11-25 08:16:09 UTC.

<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Qwen3.6-35B-A3B FP8 on GB200

## Modeling V1 baseline

Ran the v1.4.0rc4 PyTorch LLM API with `modeling_v2="off"`, TP1, vision
encoder disabled, CUDA graphs for batch 1, overlap scheduling, KV reuse off,
and no speculation. The official checkpoint loaded directly. First-use
FlashInfer sampling and attention compilation ran before measurement.

The dev oracle passed all 32 items. Top-20 mean KL was 0.006901 and non-tie
top-1 agreement was 99.587% over 2,181 positions. Nine continuations matched
exactly and 23 first diverged at a near-tie.

### Batch 1, ISL 128/1024/4096, OSL 1024

Medians across two prompts per ISL quoting TensorRT-LLM documentation. TTFT
is wall time for an OSL 1 request. Decode throughput differences OSL 32 and
OSL 1024 for the same prompt.

| ISL | OSL | TTFT (ms) | Decode (tokens/s) |
|---:|---:|---:|---:|
| 128 | 1024 | 89.23 | 322.1 |
| 1024 | 1024 | 88.89 | 321.2 |
| 4096 | 1024 | 85.91 | 289.9 |

### OpenAI serving, ISL 1024/4096, OSL 1024

Measured with `/task/bench_serving.py`, one server frontend, batch cap 128,
8,192-token scheduler budget, chunked prefill, and CUDA graph padding.
V1 used its automatic BF16 recurrent-state cache. Each level uses one
closed-loop wave of requests, with a minimum of two requests.

| ISL | OSL | Concurrency | Output tokens/s | Median TTFT (ms) | Median TPOT (ms) |
|---:|---:|---:|---:|---:|---:|
| 1024 | 1024 | 1 | 308.3 | 78.9 | 3.17 |
| 1024 | 1024 | 2 | 557.1 | 156.9 | 3.44 |
| 1024 | 1024 | 4 | 945.0 | 161.1 | 4.08 |
| 1024 | 1024 | 8 | 1626.2 | 166.4 | 4.76 |
| 1024 | 1024 | 16 | 2691.0 | 297.0 | 5.66 |
| 1024 | 1024 | 32 | 4477.6 | 421.9 | 6.73 |
| 1024 | 1024 | 64 | 6679.5 | 691.1 | 8.89 |
| 1024 | 1024 | 128 | 8921.2 | 1089.9 | 13.24 |
| 4096 | 1024 | 1 | 307.8 | 82.2 | 3.17 |
| 4096 | 1024 | 8 | 1521.1 | 394.0 | 4.87 |
| 4096 | 1024 | 32 | 3571.8 | 1295.2 | 7.65 |

## Target design

The route admits the text path of the official block-FP8 checkpoint on SM100
at TP1. The target opts in through `modeling_v2="require"`. It uses the
runtime's paged BF16 KV cache and FP32 Gated DeltaNet state through the hybrid
V2 cache manager. The current serving configuration enables chunked prefill
and CUDA graph decode. Weights remain E4M3 with BF16 128×128 scales, and
activations use per-token, per-128-channel dynamic FP8 scales.

The reference engine at `6780d5e` supplied the layer ordering, projection
packing, dynamic quantization, fused residual normalization, expert routing
and expert kernels. The target's `weights.py` adapts `read_weights` from
`engine.py` to the TensorRT-LLM weight loader. The Triton/Gluon code under
`kernels/` preserves the reference's arithmetic while indexing cache slots
by scheduler request. The runtime-compiled `fp8_cutlass.cu` and
`prefill_gemm.py` come from the same reference files. A full source build
would place this CUDA source under the target's `csrc/`. The catalog's
paged attention is bound using the Qwen4 target's metadata pattern. No
shared scheduler or cache implementation changes are needed.
## Initial V2 validation and performance

The first catalog-only attention version passed the KL and top-1 checks, with
two non-tie greedy divergences. Porting the reference's split-KV single-request
decode to the runtime's page table removed both failures. Catalog attention
still handles context and batches above one. The resulting target passed all
32 dev items with mean top-20 KL 0.006169 and 99.679% non-tie top-1 agreement.
Seven continuations matched exactly and 25 first diverged at a near-tie.
The OpenAI server returned HTTP 200 for the model list and a greedy token-ID
completion.

### V2 batch 1 without MTP, ISL 128/1024/4096, OSL 1024

Same two prompts per ISL and timing method as the V1 API run.

| ISL | OSL | TTFT (ms) | Decode (tokens/s) |
|---:|---:|---:|---:|
| 128 | 1024 | 33.54 | 374.0 |
| 1024 | 1024 | 32.90 | 379.1 |
| 4096 | 1024 | 53.79 | 377.5 |

### V2 OpenAI serving without MTP, ISL 1024/4096, OSL 1024

Same prompts and request counts as the V1 serving run. V2 used FP32 GDN
state to match the reference engine.

| ISL | OSL | Concurrency | Output tokens/s | Median TTFT (ms) | Median TPOT (ms) |
|---:|---:|---:|---:|---:|---:|
| 1024 | 1024 | 1 | 353.6 | 42.8 | 2.79 |
| 1024 | 1024 | 2 | 635.1 | 80.1 | 3.07 |
| 1024 | 1024 | 4 | 1131.4 | 102.3 | 3.44 |
| 1024 | 1024 | 8 | 1902.2 | 140.1 | 4.07 |
| 1024 | 1024 | 16 | 2868.2 | 202.3 | 5.38 |
| 1024 | 1024 | 32 | 4480.4 | 309.4 | 6.84 |
| 1024 | 1024 | 64 | 6344.0 | 544.0 | 9.55 |
| 1024 | 1024 | 128 | 9624.7 | 950.7 | 12.32 |
| 4096 | 1024 | 1 | 357.7 | 65.0 | 2.73 |
| 4096 | 1024 | 8 | 1728.3 | 341.2 | 4.30 |
| 4096 | 1024 | 32 | 3471.7 | 998.8 | 8.22 |

Batch-1 decode improved from 321 to 379 tokens/s at ISL 1024. The first
V2 serving configuration reached 4,480 tokens/s at concurrency 32 and
9,625 at 128. The standalone reference's published targets are 7,788
and 16,324 tokens/s at comparable static batch sizes. These V2 baseline
measurements have no MTP.

## Independent standalone reference run

I copied `/ref-engine` at `6780d5e` to `/tmp` for its JIT cache. The original
environment's FlashAttention 4 and QuACK versions were incompatible with the
installed CuTe DSL. The copied launcher ran with FlashAttention 4.0.0b41 and
QuACK 0.6.5 in a subprocess-specific import path, with the engine sources
unchanged. The dev oracle passed all 32 items with mean top-20 KL 0.006443
and 99.679% non-tie top-1 agreement. Both static batch sizes also passed
the oracle's greedy checks.

### Standalone batches, ISL 128/1024/4096, OSL 1024

Medians across two runs per point, using the same prompts and timing method
as the V1 and V2 API measurements. Static batch throughput sums all output
tokens across the batch.

| ISL | OSL | Static batch | TTFT (ms) | Decode (tokens/s) |
|---:|---:|---:|---:|---:|
| 128 | 1024 | 1 | 8.29 | 765.9 |
| 1024 | 1024 | 1 | 18.10 | 858.0 |
| 4096 | 1024 | 1 | 46.65 | 792.1 |
| 1024 | 1024 | 32 | — | 8652.4 |
| 1024 | 1024 | 128 | — | 17427.5 |

The initial target was also rerun from an isolated checkout with a cold
CUTLASS build. The build requires the FlashInfer headers before the TVM FFI
utilities introduce a global `Tensor` alias. With that order preserved, the
complete oracle again passed at 0.006169 KL and 99.679% non-tie top-1.
The server started and returned a greedy completion.

## Split MTP with staged recurrent state

The target loads the embedded checkpoint's MTP layer and shares its BF16
embedding and vocabulary head with the verifier. The runtime's one-model
MTP worker rewinds the paged draft cache and promotes the accepted
convolution history and FP32 Gated DeltaNet state. The staged GDN kernels
adapt `/ref-engine/speculation.py` to the cache manager's request slots.
The draft uses the reference engine's 65,536-token common vocabulary and
the special-token range for batches up to 32. The complete target
vocabulary verifies every proposal. The CLI uses a second executor for
teacher-forced scoring because the speculative sampler does not return
context logits. Both executors load the same FP8 checkpoint.

### Staged MTP batch 1, ISL 128/1024/4096, OSL 1024

The CLI uses three draft tokens and executes the one-token TTFT request
through the standard sampler. MTP passed all 32 oracle items with mean
KL 0.006169 and 99.679% non-tie top-1. The 32-token prefixes matched
the 1024-token requests in all six benchmark cases.

| ISL | OSL | TTFT (ms) | Decode (tokens/s) |
|---:|---:|---:|---:|
| 128 | 1024 | 32.19 | 710.9 |
| 1024 | 1024 | 32.20 | 743.0 |
| 4096 | 1024 | 54.18 | 726.2 |

### Experimental staged MTP serving, ISL 1024/4096, OSL 1024

The server started with a fixed draft length of three and CUDA graphs up
to batch 128. At concurrency 1 the dev oracle passed. At concurrency 8,
32 and 128, the concurrent request runs exposed non-tie differences.
Staging four full FP32 recurrent checkpoints also reduced throughput at
large batches. The default server retains the initial configuration.
The HTTP client disconnected during warm-up at ISL 4096, concurrency 8.

| ISL | OSL | Concurrency | Output tokens/s | Median TTFT (ms) | Median TPOT (ms) |
|---:|---:|---:|---:|---:|---:|
| 1024 | 1024 | 1 | 664.9 | 43.9 | 1.46 |
| 1024 | 1024 | 2 | 1070.3 | 86.7 | 1.74 |
| 1024 | 1024 | 4 | 1677.2 | 143.4 | 2.19 |
| 1024 | 1024 | 8 | 2660.5 | 124.8 | 2.85 |
| 1024 | 1024 | 16 | 3808.9 | 194.2 | 3.85 |
| 1024 | 1024 | 32 | 5484.0 | 320.7 | 5.30 |
| 1024 | 1024 | 64 | 6219.0 | 561.7 | 9.36 |
| 1024 | 1024 | 128 | 4896.0 | 3677.1 | 22.18 |
| 4096 | 1024 | 1 | 676.1 | 65.8 | 1.42 |

## FP32 replay for concurrent verification

The replay kernel adapts the paired FP32 updates in
`/ref-engine/gdn_decode.py`. It retains accepted keys, decays, and
innovations between full recurrent checkpoints. The target's cache
manager declares the FP32 buffers in the runtime's request-private
replay region and reuses its slot allocation, convolution promotion,
and acceptance bookkeeping. The executor has one optional class
selection hook for this cache policy. Modeling V1 and V2 targets
without that policy keep their existing cache manager selection.

An isolated rollout covered partial acceptance, slot rollover, and a
seed-only step. Its BF16 outputs matched staged verification bitwise.
Paired CUDA graph microbenchmarks reduced per-layer verify time
from 130 to 51 µs at batch 32 with four verification rows per request,
and from 387 to 201 µs at batch 128 with three rows per request.

### MTP replay batch 1, ISL 128/1024/4096, OSL 1024

The replay experiment passed all 32 dev items at 0.006169 KL and
99.679% non-tie top-1. The CLI retains staged state for batch 1
based on these measurements.

| ISL | OSL | TTFT (ms) | Decode (tokens/s) |
|---:|---:|---:|---:|
| 128 | 1024 | 35.56 | 702.6 |
| 1024 | 1024 | 34.76 | 734.8 |
| 4096 | 1024 | 54.53 | 718.4 |

### Experimental MTP replay serving, ISL 1024, OSL 1024

The replay experiment kept the staged server's fixed draft length of
three and its cache capacity, graph shapes, and scheduler settings.
The concurrent oracle produced the same greedy results as staged
verification, including the non-tie deviations. Prefill traces showed
some numerical differences before the first draft.

| ISL | OSL | Concurrency | Output tokens/s | Median TTFT (ms) | Median TPOT (ms) |
|---:|---:|---:|---:|---:|---:|
| 1024 | 1024 | 1 | 742.6 | 42.9 | 1.31 |
| 1024 | 1024 | 8 | 2889.3 | 132.1 | 2.60 |
| 1024 | 1024 | 32 | 7437.7 | 313.3 | 3.83 |
| 1024 | 1024 | 128 | 11972.7 | 1500.3 | 8.79 |

### Replay cache and draft-length trial, ISL 1024, OSL 1024

Two draft tokens, a 524,288-token KV budget, and a 2,048-token average
sequence length were tested together. Kernel JIT occurred during the run,
so the timing does not isolate either the budget or the draft width.
The concurrent oracle still had the same first divergences. The
four-row verifier remains the faster measured configuration.

| ISL | OSL | Concurrency | Output tokens/s | Median TTFT (ms) | Median TPOT (ms) |
|---:|---:|---:|---:|---:|---:|
| 1024 | 1024 | 32 | 6587.4 | 327.4 | 4.39 |
| 1024 | 1024 | 128 | 7462.4 | 2957.6 | 13.96 |

## Packed prefill and grouped paged attention experiments

Layer tracing isolated early packing-dependent differences in short-context
GDN. A difference below one millionth in a recurrent output could move a
value across an FP8 rounding boundary and grow through later layers. The
short prefill trial uses a common BF16 gate projection reduction across
packings and the same time-tiled recurrence at every short length. An
operation test checks bitwise packing parity for both the recurrent output
and state. The batch-1 oracle passed with KL 0.006457 and 99.587% non-tie
top-1 agreement.

The grouped paged tiles come from `/ref-engine/verify_attention.py`. They
share each loaded K/V tile across the query positions of a request and use
the executor's BF16 page table. With grouped tiles at all batch sizes, the
external oracle passed through concurrency 32. Concurrency 128 had
non-fragile divergences. A hybrid dispatch uses the catalog's full-width
paged attention above 32 generation requests and grouped tiles up to 32.
It passed the external checks at 1, 8 and 32. At 128 the only external
first differences were at one fragile position, repeated four times.
Direct comparison against the CLI also observed first differences with
teacher-forced margins of at least 0.5 at the larger concurrencies.

### Attention experiments, ISL 1024, OSL 1024

| Attention | ISL | OSL | Concurrency | Output tokens/s | Median TTFT (ms) | Median TPOT (ms) |
|---|---:|---:|---:|---:|---:|---:|
| Grouped paged | 1024 | 1024 | 32 | 7188.8 | 320.6 | 3.90 |
| Grouped paged | 1024 | 1024 | 128 | 12031.3 | 1199.1 | 8.81 |
| Hybrid paged | 1024 | 1024 | 32 | 7228.6 | 307.1 | 3.88 |
| Hybrid paged | 1024 | 1024 | 128 | 12944.2 | 1113.8 | 8.32 |

### Replay decode profile, ISL 1024, OSL 256

Nsight Systems captured 24 steady batch-128 speculative steps. The
transposed expert projections occupied 36.6% of kernel time, GDN
replay occupied 27.7%, and dense FP8 projections occupied 9.3%.
The recurrent update took approximately 202 µs per layer. Reordering
GDN CTAs to process adjacent tiles of one state first lost time in
isolated tests with strided recurrent slots.

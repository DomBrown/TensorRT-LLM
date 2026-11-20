Fri Nov 20 12:01:08 UTC 2026

<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Qwen3.6-35B-A3B FP8 on GB200

## Modeling V1 baseline

The baseline uses the stock Qwen3.5 MoE model with the vision encoder disabled,
BF16 attention pages, FP32 DeltaNet state, CUDA graphs, the overlap scheduler,
chunked prefill and no block reuse. The first cold start built FlashInfer
sampling and attention kernels. With the default SSM dtype, a protocol smoke
test completed `generate`, `score` and `shutdown`. The timed configuration sets
`mamba_ssm_cache_dtype: float32` for parity with the target.

The full development oracle with FP32 state gave mean KL 0.006909 and
2,172/2,181 non-tie top-1 matches (99.587%). Twenty-three first greedy
differences were near ties and one was a listed fragile position. Item 10,
output token 43, differed at a 0.5000-nat gap, so this baseline fails the full
greedy gate.

### Batch 1, ISL 128 / 1,024 / 4,096, OSL 1,024

Medians of three technical chat prompts. The external client uses a one-token
request for TTFT and subtracts 32-token latency from 1,024-token latency to
measure decode. The passages come from TensorRT-LLM documentation because the
standalone reference's documentation mounts are absent here.

| ISL | OSL | V1 TTFT (ms) | V1 TPOT (ms) | V1 decode (tokens/s) |
| ---: | ---: | ---: | ---: | ---: |
| 128 | 1,024 | 66.02 | 3.237 | 309.0 |
| 1,024 | 1,024 | 61.32 | 3.241 | 308.5 |
| 4,096 | 1,024 | 60.22 | 3.281 | 304.8 |

### Serving baseline, ISL 1,024 and 4,096, OSL 1,024

`bench_serving.py` ran against `trtllm-serve` with `modeling_v2: "off"`, the
same cache settings and `--minimum 4 --multiplier 1`. Warm-up requests had
OSL 16. The workload used technical passages from the TensorRT-LLM docs.

| ISL | OSL | Concurrency | Requests | Output tokens/s | Median TTFT (ms) | Median TPOT (ms) |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1,024 | 1,024 | 1 | 4 | 288.4 | 77.38 | 3.38 |
| 1,024 | 1,024 | 2 | 4 | 530.7 | 150.77 | 3.62 |
| 1,024 | 1,024 | 4 | 4 | 930.8 | 140.75 | 4.16 |
| 1,024 | 1,024 | 8 | 8 | 1,591.2 | 151.91 | 4.88 |
| 1,024 | 1,024 | 16 | 16 | 2,560.3 | 235.81 | 6.02 |
| 1,024 | 1,024 | 32 | 32 | 3,927.8 | 363.53 | 7.79 |
| 1,024 | 1,024 | 64 | 64 | 6,137.9 | 500.21 | 9.92 |
| 1,024 | 1,024 | 128 | 128 | 8,435.6 | 828.39 | 14.31 |
| 4,096 | 1,024 | 1 | 4 | 284.4 | 74.43 | 3.44 |
| 4,096 | 1,024 | 8 | 8 | 1,486.7 | 311.36 | 5.08 |
| 4,096 | 1,024 | 32 | 32 | 3,541.3 | 837.92 | 8.19 |

## Modeling V2 implementation

The route selects SM100 TP1 for the text decoder of the official
`Qwen3_5MoeForConditionalGeneration` E4M3 checkpoint. The target is opt-in
and uses the release's Modeling V2 package. The target hands the runtime
the checkpoint's normalized text config so it allocates 30 GDN state layers
and 10 full-attention KV layers. It uses BF16 KV pages, BF16 convolution
history and FP32 recurrent state. It obtains the pages and recurrent slots
from the existing hybrid cache manager. Requests are packed by the scheduler,
with context rows first. Continuation chunks read the retained state and
cached KV. Decode uses runtime CUDA graph replay and the overlap scheduler.

The port starts from standalone reference commit `4e6990d`.

- `engine.py` supplies checkpoint placement, block ordering and rounding in `weights.py` and `modeling.py`.
- `kernels.py`, `norm_exact.py` supplies target-local per-128 quantization, RMSNorm, activation/gate and MoE epilogues.
- `mx_decode.py`, `dg_prefill.py` supplies sM100 MX-scaled dense decode and packed DeepGEMM prefill scales.
- `tc_decode.py`, `prefill_tc.py`, `prefill_moe.py` supplies native FP8 tensor-core expert products and device-side grouping.
- `spec_kernels.py`, `kernels.py` supplies compensated grouped attention, causal convolution and FP32 GDN recurrence adapted to scheduler pages and slots.

The canonical weights stay in E4M3 with their block scales. Dense and
shared-expert projections also use the reference's E4M3 power-of-two packing
for the MX and DeepGEMM paths. Routed experts retain the original FP8 values
and scales. No dequantized weight copy is stored. The paged GQA, recurrent
verify and small FP8 expert kernels are candidates for a target `csrc/`
library in a native build. This deployment uses Triton, Gluon and the installed
FlashInfer and DeepGEMM libraries.

### First passing target

`python -m aegis_engine.oracle` exercised the real JSONL entry point on all 32
development cases. Mean KL was **0.006259**, with **2,171/2,181 (99.541%)**
non-tie top-1 matches and zero disallowed greedy divergences. Twenty-three
first greedy differences were near ties. `serve.sh /model` served
`/v1/models` and an eight-token greedy completion on SM100. The target-local
GPU operation checks and the Modeling V2 stale-claim check passed (13 tests).

A first run with a token-only cache quota of 8,256 failed during graph
warm-up. The hybrid pool charged recurrent slots to that budget and reduced
the allocatable sequence length to 5,824. The batch-1 adapter reserves room
for both kinds of state, and the final CUDA graph captured with the full
8,192-token limit.

## Standalone reference on this GB200

I ran a temporary copy of `/ref-engine/engine.sh` at `4e6990d`. Its local
`/refs` tree is unavailable here. The copy points the intermediate CUTLASS
tile at FlashInfer's installed headers and calls TensorRT-LLM's bundled
DeepGEMM with the same packed power-of-two scales. Both changes are limited
to the temporary comparator. It took 49.36 s to report ready with cold local
kernel caches. The three prompt sets and the timing method match the V1 run.

| ISL | OSL | Reference TTFT (ms) | Reference TPOT (ms) | Reference decode (tokens/s) |
| ---: | ---: | ---: | ---: | ---: |
| 128 | 1,024 | 9.27 | 1.153 | 867.1 |
| 1,024 | 1,024 | 17.95 | 1.092 | 915.5 |
| 4,096 | 1,024 | 42.02 | 1.149 | 870.3 |

### Serial target serving, ISL 1,024 and 4,096, OSL 1,024

The first V2 commit completed the same closed-loop sweep as V1. A frontend
disconnect interrupted the original run at ISL 4,096 and concurrency 8.
The two remaining measurements succeeded on retry with the same server.

| ISL | OSL | Concurrency | V1 output tokens/s | V2 output tokens/s | V2 median TTFT (ms) |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1,024 | 1,024 | 1 | 288.4 | 441.5 | 51.41 |
| 1,024 | 1,024 | 2 | 530.7 | 780.6 | 100.25 |
| 1,024 | 1,024 | 4 | 930.8 | 1,306.3 | 139.17 |
| 1,024 | 1,024 | 8 | 1,591.2 | 1,917.8 | 122.10 |
| 1,024 | 1,024 | 16 | 2,560.3 | 2,347.4 | 179.37 |
| 1,024 | 1,024 | 32 | 3,927.8 | 3,785.6 | 258.91 |
| 1,024 | 1,024 | 64 | 6,137.9 | 5,754.3 | 395.03 |
| 1,024 | 1,024 | 128 | 8,435.6 | 7,895.3 | 679.43 |
| 4,096 | 1,024 | 1 | 284.4 | 416.4 | 59.29 |
| 4,096 | 1,024 | 8 | 1,486.7 | 1,764.6 | 248.36 |
| 4,096 | 1,024 | 32 | 3,541.3 | 3,248.8 | 685.97 |

## Three-token MTP

The batch-1 adapter uses the runtime's embedded Eagle MTP worker with three
drafts. The MTP layer shares the checkpoint's token embedding and output
head. Its FP8 projections and paged attention use the same target operations
as the trunk. The BF16 draft input fusion comes from the reference's
`spec_kernels.py`. The greedy draft head searches the first 65,536 IDs, as in
the standalone engine. The verifier always uses the full vocabulary. The
adapter uses a second, non-speculative executor for teacher-forced score
requests because the speculative sampler does not expose that operation.

The hybrid manager allocates FP32 rank-one updates for rejected or accepted
draft rows. It commits the first row's full GDN state and replays only the
accepted updates at the next step. It uses the runtime's convolution window
promotion and slot lifecycle. The shared executor change lets a selected
Modeling V2 target provide its hybrid manager. All other targets keep the
existing manager.

The three-draft adapter passed all 32 development cases with mean KL
**0.006259**, non-tie top-1 **2,171/2,181 (99.541%)**, 23 near ties and zero
disallowed differences. In the short smoke requests the verifier produced
2.857 tokens per iteration on average. Engine startup took 45.08 s. The
serving wrapper continued to use the serial target configuration for this
revision.

| ISL | OSL | V2 MTP TTFT (ms) | V2 MTP TPOT (ms) | V2 MTP decode (tokens/s) |
| ---: | ---: | ---: | ---: | ---: |
| 128 | 1,024 | 62.38 | 1.213 | 824.2 |
| 1,024 | 1,024 | 63.74 | 1.181 | 847.0 |
| 4,096 | 1,024 | 65.24 | 1.286 | 777.4 |

## Exact-size prefill graphs and draft argmax

The batch-one generator captures the decoder body at powers of two from
128 to 4,096 tokens with the runtime's breakable prefill CUDA graphs. It
stages the current recurrent-state slot and KV page table at each replay.
An executor capability check keeps the original row count for models that
require an exact graph size. The scorer continues to use eager prefill.
The draft shortlist also uses a tiled argmax from the reference's
`kernels.py`, packing BF16 scores and 16-bit vocabulary indices into
32-bit keys. It writes directly to the runtime's draft-token buffer.

| ISL | OSL | V2 graph TTFT (ms) | V2 graph TPOT (ms) | V2 graph decode (tokens/s) |
| ---: | ---: | ---: | ---: | ---: |
| 128 | 1,024 | 12.04 | 1.184 | 844.5 |
| 1,024 | 1,024 | 22.04 | 1.152 | 868.0 |
| 4,096 | 1,024 | 47.30 | 1.259 | 794.3 |

The graph adapter took 46.69 s to report ready. An Nsight Systems trace
of the preceding eager adapter showed about 69 ms between the first
GPU prefill work and the first decode graph, with about 18 ms of prefill
kernel work. The same trace showed approximately 1.0 ms per decode step
in the small FP8 routed products and 0.37 ms in recurrent GDN.

For the initial serial serving target, a token-level check retained the
actual single-request decode top-two scores. All 32 prompts passed the
near-tie rule at concurrency 8. At concurrency 32, two prompts diverged
outside the rule at gaps of 0.75 and 0.50 nats. The first mismatch at
0.50 occurred on the first generated token, implicating packed prefill.

## MXFP8 draft vocabulary

The draft-only shortlist now uses a per-32-channel MXFP8 copy of the
checkpoint's BF16 vocabulary prefix and FlashInfer's CuTe DSL GEMM. The
target head remains BF16 over all 248,320 tokens. In a rotating-weight
batch-one CUDA graph microbenchmark, shortlist projection plus argmax
took 28.9 µs with CuTe DSL, 33.8 µs with FlashInfer's low-latency tactic
and 47.6 µs with the BF16 head. The CUTLASS tactic took 143.5 µs.

All 32 development cases passed with KL 0.006259 and 99.541% non-tie
top-1. Captured and eager prefill in the preceding revision also produced
identical 256-token continuations at nine lengths from 127 to 4,096,
including exact buckets and adjacent lengths. The SM100 CI test list now
includes the mirrored operations and an oracle integration test that uses
the local checkpoint and reference file.

| ISL | OSL | V2 MX draft TTFT (ms) | V2 MX draft TPOT (ms) | V2 MX draft decode (tokens/s) |
| ---: | ---: | ---: | ---: | ---: |
| 128 | 1,024 | 12.74 | 1.171 | 854.2 |
| 1,024 | 1,024 | 22.23 | 1.136 | 880.2 |
| 4,096 | 1,024 | 47.58 | 1.240 | 806.3 |

## Packed-request greedy parity

A layer trace found batch-size-dependent BF16 projections before the first
packed prefill divergence. Using a fixed tensor-core accumulation order
for both prefill and decode gave mean KL 0.005498 with three disallowed
greedy differences. Using the rowwise FP32 reduction for all prefill
projections gave mean KL 0.005852 with two disallowed differences.

The passing path keeps each prompt's BF16 projection shape in packed
prefill and uses the reference's rowwise FP32 reduction in decode. It
preserves the batch-one oracle result at KL 0.006259 and 99.541% non-tie
top-1. Against single-request serving, 32 requests at concurrency 8,
32 at concurrency 32, and 128 at concurrency 128 all passed the
near-tie rule. The comparisons used the top-two scores from the original
single-request decode. There were 5, 21, and 84 near-tie differences,
respectively, across those trials. Prompt lengths ranged from 6 to 35
tokens, with OSL 64.

Packing each prompt independently increased eager projection launches. At
concurrency 128, median TTFT increased from 0.68 s to 20.44 s in the
single-wave run. The median decode interval was 16.39 ms compared with
15.49 ms in the first V2 revision.

| ISL | OSL | Concurrency | Packed parity revision output tokens/s | Median TTFT (ms) |
| ---: | ---: | ---: | ---: | ---: |
| 1,024 | 1,024 | 8 | 1,154.3 | 379.16 |
| 1,024 | 1,024 | 32 | 1,685.3 | 1,943.54 |
| 1,024 | 1,024 | 128 | 3,514.9 | 20,436.15 |

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

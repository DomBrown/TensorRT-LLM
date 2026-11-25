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

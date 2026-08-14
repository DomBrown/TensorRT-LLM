<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Laguna FP8 CUDA Graph Illegal Memory Access Investigation

## Status

The failure is reproducible on B200 and has been localized to a TRTLLM-Gen FP8
MoE GEMM2 kernel replayed from a CUDA graph.

A candidate change that places all FP8 MoE temporary storage in one aligned
arena passes the complete Laguna FP8 test. The initial candidate combined two
changes:

1. It consolidates many allocations into one aligned arena.
2. It retains that arena on the C++ runner.

A controlled variant kept the same contiguous packing but made the arena
transient. It passed initialization four consecutive times, including three
executor build/shutdown cycles in one process. This rules out runner ownership
as a requirement.

Further allocation bisection showed that moving only the activation-output
allocation is sufficient to pass, while moving only its scale allocation is
not. With the original failing allocation order held fixed, three tileN=8
configurations faulted and a tileN=16 configuration passed. The leading
hypothesis is now an address-sensitive defect in the tileN=8 GEMM2 TMA path,
not missing workspace ownership. The arena remains a validated workaround, not
a final root-cause fix.

The current WIP restores the original separate workspace allocations and
excludes tileN=8 tactics for non-fused FP8 block-scale MoE. It passed the
initialization reproduction and complete Laguna accuracy test. This is a
low-memory workaround, not yet a submission-ready fix: the restriction applies
to every non-fused SM100/SM103 FP8 block-scale MoE workload and may regress
decode performance where tileN=8 is optimal.

## Timeline

- `6fec819bcf` introduced a two-pass CUDA graph warmup:
  - Warm every graph shape first.
  - Capture every graph shape in a second pass.
  - This fixed attention workspace resizing, but changed allocator ordering and
    exposed the Laguna failure.
- `60e7fcaeaf4` kept a strong reference to every captured graph output.
  - Laguna passed.
  - The change retained full-vocabulary FP32 logits for every graph key.
- `9a2ff2ebc53` reverted the strong-reference change.
  - The H100 multi-GPU CI stage had increased from roughly 70 minutes to more
    than 250 minutes.
  - The Laguna illegal memory access returned.

## Reproduction

### Integration test

```bash
LLM_MODELS_ROOT=/llm-models/ \
pytest -v -s \
tests/integration/defs/accuracy/test_llm_api_pytorch.py::TestLagunaXS::test_fp8
```

The test configures:

- Model: `/llm-models/Laguna-XS.2-FP8`
- `max_seq_len=4096`
- `max_num_tokens=4096`
- `max_batch_size=128`
- FP8 block-scaled weights
- FP8 KV cache
- Default CUDA graph batch sizes

### Faster initialization-only reproduction

The failure occurs during executor initialization, before accuracy evaluation.
It can therefore be reproduced without the MMLU and GSM8K workloads:

```bash
CUDA_LAUNCH_BLOCKING=1 \
LLM_MODELS_ROOT=/llm-models/ \
python3 -c '
from tensorrt_llm._torch.pyexecutor.py_executor_creator import create_py_executor
from tensorrt_llm.llmapi import KvCacheConfig
from tensorrt_llm.llmapi.llm_args import TorchLlmArgs

path = "/llm-models/Laguna-XS.2-FP8"
executor = create_py_executor(
    TorchLlmArgs(
        model=path,
        max_seq_len=4096,
        max_num_tokens=4096,
        max_batch_size=128,
        kv_cache_config=KvCacheConfig(free_gpu_memory_fraction=0.9),
    ),
    checkpoint_dir=path,
)
executor.shutdown()
'
```

### Failure point

The memory-profiling executor, which has only a small temporary KV cache,
captures all graph sizes successfully.

The final executor allocates approximately 128 GiB of KV cache, then fails while
capturing generation graphs. With synchronous CUDA error reporting, the failure
consistently appears during the immediate replay of the batch-size-3 graph:

```text
PyTorchModelEngine.warmup
  -> _capture_generation_cuda_graphs
  -> forward
  -> CUDAGraphRunner.replay
  -> torch.cuda.CUDAGraph.replay
  -> CUDA error: an illegal memory access was encountered
```

## GPU fault evidence

Nsight Systems showed that the failure occurs after the batch-size-3 graph is
captured and launched.

A CUDA GPU core dump identified:

```text
CUDA Exception: Warp MMU Fault

Kernel:
bmm_Bfloat16_E4m3E4m3_Fp32_t128x8x128u2_s8_...
_schedS_..._dynB_sm100f

Faulting instruction:
UTMALDG.4D
```

This is the TRTLLM-Gen FP8 MoE GEMM2 kernel. `UTMALDG` is a TMA load, so the
kernel attempted to load through a TMA descriptor containing an invalid encoded
address.

The stable model parameters used by GEMM2 are:

- GEMM2 weights
- GEMM2 weight scales

The temporary inputs and outputs associated with GEMM2 are:

- Activation output
- Activation output scales
- GEMM2 output
- GEMM2 BMM workspace

The core dump establishes that a TMA descriptor was invalid. It does not, by
itself, identify which encoded A, scale, or output address was wrong.

## The original workspace behavior

`run_fp8_block_scale_moe()` created separate local tensors for:

- Routing:
  - Token counts per expert
  - Total padded token count
  - Expanded-to-permuted token map
  - Permuted-to-token map
  - Expert weights and indexes
  - Expert-count histogram
  - CTA-to-batch and CTA-limit maps
  - Non-exiting CTA count
- Compute:
  - GEMM1 output
  - GEMM1 output scales
  - Activation output
  - Activation output scales
  - GEMM2 output
- BMM scratch:
  - GEMM1 workspace
  - GEMM2 workspace

The tensors' raw pointers were passed to the TRTLLM-Gen kernels through
`MoEWorkspace`.

These allocations are graph-local. Their C++ tensor handles do not need to
survive graph capture under normal CUDA graph semantics: PyTorch's graph memory
pool preserves the backing virtual memory and replay uses the recorded
addresses. Therefore, locality alone is not evidence of a dangling pointer.

The unresolved question is why the selected small-batch GEMM2 tactic obtained a
bad TMA descriptor when these buffers were independently allocated and reused
from the shared graph pool.

## Changes that did not fix the issue

### Keeping the graph output alive through first replay

`CUDAGraphRunner.capture()` was changed to return the owning output tensor so
the caller retained it through the immediate first replay.

Result: the same batch-size-3 replay failed.

Conclusion: the immediate lifetime of final logits is not sufficient to explain
the failure.

### Retaining only GEMM1 and GEMM2 BMM workspaces

The two TRTLLM-Gen BMM scratch tensors were retained during graph capture.
Variants included:

- One workspace pair shared by compatible captures.
- Workspace pairs keyed more narrowly by stream, layer weights, input address,
  token count, tactic, and workspace sizes.

Result: GEMM2 still failed at the same TMA load.

Conclusion: the BMM scratch allocation alone is not the only memory placement
involved in the fault.

### Providing additional free GPU memory

The KV-cache free-memory fraction was reduced from `0.9` to `0.895`.

Result: the same failure occurred after allocating approximately 0.5 GiB less
KV cache.

Conclusion: the failure is not simply an out-of-memory condition at the original
memory fraction.

### Compute Sanitizer

Compute Sanitizer followed the child process and reached the failure, but the
instrumented process crashed at `cuGraphLaunch` without identifying an
individual invalid access.

Conclusion: this run was inconclusive. The CUDA GPU core dump provided the
useful kernel-level evidence.

## Changes that made the test pass

### Full MoE workspace arena

The initial successful candidate change:

1. Computes the sizes of every routing, intermediate, scale, GEMM output, and
   BMM scratch region.
2. Allocates one aligned byte arena.
3. Slices raw pointers for each region using the repository workspace-alignment
   helpers.
4. Uses a transient arena during eager execution.
5. During CUDA graph capture, stores arenas on the C++
   `FP8BlockScaleMoeRunner`.
6. Reuses an arena on the same device and capture stream when it has sufficient
   capacity.
7. Adds an arena instead of replacing an existing one when a larger capacity is
   required, because an earlier graph may still reference the old address.
Production graph-teardown wiring is intentionally not part of the current
diagnostic patch. Captured arenas remain owned by the process-global cached
runner.

The controlled follow-up removed steps 5-7. Every invocation instead owns one
local contiguous arena, including during CUDA graph capture. PyTorch's graph
memory pool preserves the allocation backing captured addresses after the local
tensor handle is destroyed.

### Why it works operationally

The change removes allocator-dependent placement and reuse for all raw workspace
pointers used by the captured FP8 MoE operation.

Both variants prevent:

- Workspace regions from being repacked between graph captures.
- Individual workspace allocations from being recycled independently.

This is sufficient to prevent the observed invalid TMA descriptor.

The transient variant proves that persistence until graph teardown is not the
important property. It is not yet known whether the important property is:

- Contiguous arena layout.
- Different alignment or relative offsets.
- Avoidance of a specific shared-pool reuse pattern.
- Keeping an out-of-bounds access within mapped arena storage.

### Validation

The candidate passed:

- Initialization-only Laguna reproduction with `CUDA_LAUNCH_BLOCKING=1`.
- Complete `TestLagunaXS::test_fp8`.
  - MMLU: `74.46`, threshold `73.373`.
  - GSM8K: `86.96`, threshold `83.947`.
- Focused multi-key CUDA graph workspace lifetime test.
- Existing TRTLLM FP8 block-scale MoE backend tests:
  - `13 passed`.
- Incremental C++ build.
- Targeted pre-commit checks.

The transient-arena variant additionally passed:

- Initialization-only Laguna reproduction once in a fresh process.
- Three consecutive executor build/shutdown cycles in one process.

The cleaned tileN workaround, with the original separate allocations, passed:

- Initialization-only Laguna reproduction with `CUDA_LAUNCH_BLOCKING=1`.
- Complete `TestLagunaXS::test_fp8`.
  - MMLU: `74.46`, threshold `73.373`.
  - GSM8K: `86.96`, threshold `83.947`.
  - `1 passed` in approximately three minutes.
- Incremental C++ build and extension installation.

### Controlled allocation bisection

A temporary workspace mask allowed each of the 17 workspace regions to be
allocated either separately or from one transient arena without changing tensor
sizes or tactics.

The allocation order matters. The exact baseline order is:

1. Routing and GEMM/activation intermediate allocations.
2. Routing launch.
3. Final output allocation.
4. BMM scratch allocations.
5. MoE compute launch.

Results with that order:

- All 17 regions separate: fails at the batch-size-3 replay.
- All 17 regions in one arena: passes.
- Routing regions in an arena, compute regions separate: passes.
- Compute regions in an arena, routing regions separate: passes.
- Only `num_tokens_per_expert` moved into the arena: fails.
- GEMM1 output and GEMM1 output scales together: passes.
- GEMM1 output alone: fails.
- GEMM1 output scales alone: fails.
- Activation output alone: passes.
- Activation output scales alone: fails.

Moving final output allocation before all workspaces also passes with every
workspace region separate. That is another allocator-order perturbation, not a
root-cause fix.

The activation-only result is the most specific: GEMM2 consumes that buffer as
its FP8 activation input, and changing its placement changes the TMA descriptor
that the faulting kernel uses.

The selected batch-size-3 config reports zero bytes of GEMM2 BMM scratch.
Therefore, the observed fault does not depend on a GEMM2 scratch allocation.

### Controlled tactic comparison

With all workspaces separate and the exact failing allocation order preserved:

- tileN=8, config 15: fails.
- tileN=8, config 14: fails.
- tileN=8, config 0: fails.
- tileN=16, config 0: passes.
- tileN=16, config 15: passes.

The tileN=8 configs cover different schedule and unroll variants. This rules out
config 15 alone. Config 15 uses the corresponding unrolled `schedS` variant at
both tile sizes, so its tileN=16 pass further isolates the failure to tileN=8 or
the stage-count/layout changes associated with tileN=8.

Restricting CUDA graph capture to batch sizes `[3, 4, 5]` did not reproduce the
failure. The full preceding capture sequence is required to produce the
problematic allocator state.

### Memory impact

The baseline run allocated approximately `128.03 GiB` of final KV cache.
The retained-arena candidate allocated approximately `126.22 GiB`.
The transient-arena variant allocated approximately `127.78 GiB` on its first
run and `127.69-127.72 GiB` during repeated executor construction.
The cleaned tactic workaround allocated approximately `127.76 GiB` while using
the original separate workspace allocations.

The measured retained-memory cost of runner ownership is therefore approximately
`1.8 GiB` for this configuration. Removing runner ownership recovers most of
that memory; the remaining difference from baseline is approximately
`0.25-0.34 GiB`.

This does not have the reverted output-reference scaling:

```text
reverted change:
    memory proportional to vocab_size * sum(all graph batch sizes) * ranks

retained arena candidate:
    memory proportional to MoE workspace high-water capacities per capture stream

transient arena variant:
    memory proportional to captured graph-pool arena allocations
```

Starting capture with the largest batch size normally allows smaller graphs to
reuse the first arena. If a smaller graph selects a tactic requiring a larger
workspace, a second arena is added rather than invalidating the earlier graph.

## Why the strong graph-output references appeared to fix it

The reverted change pinned one FP32 full-vocabulary output for every graph key.
Those allocations changed the shared graph pool's free-list and placement of
later allocations.

Laguna passing with that change demonstrates memory-layout sensitivity, but it
does not prove that graph outputs were the faulty pointers.

The memory cost was much larger and scaled with every graph batch size, which
caused severe H100 multi-GPU CI regressions. It is therefore not an acceptable
general fix.

## Open questions

1. Which activation addresses or address relationships make tileN=8 GEMM2's TMA
   descriptor invalid?
2. Is the workspace size returned for the selected GEMM2 tactic correct?
3. Is the defect shared by all tileN=8 dynamic-batch cubins, or only a subset of
   schedule/unroll variants?
4. Does tileN=8 access beyond the activation extent for the batch-size-3 routing
   distribution?
5. Is descriptor state prepared by an operation that is not reproduced during
   graph replay?
6. Why does the failure appear only after the two-pass warmup changed allocator
   ordering?
7. Is the SM120 failure the same class of issue? SM120 uses a separate
   Cutlass/Triton FP8 MoE path, so the current C++ arena does not cover it.

## Possible next steps

### 1. Separate layout from persistence (completed)

The contiguous but transient variant passed four consecutive initialization
runs. Runner ownership is therefore not required.

The persistent-separate-allocation experiment is no longer needed to establish
whether persistence is necessary. It may still be useful if allocator placement
needs to be compared directly.

### 2. Identify the tileN=8 address condition

Keep the failing graph sequence and tileN=8 tactic, then sweep:

- Activation allocation order.
- Aligned leading offsets within a padded allocation.
- Address alignment above the required 16-byte minimum.
- Placement relative to a virtual-memory allocation boundary.

Log and compare the encoded `tmaB` descriptor for failing and passing activation
addresses.

### 3. Verify a focused regression fails without the candidate

Create a multi-key MoE CUDA graph test that reliably fails with separate
allocations and passes with the transient contiguous arena. The earlier focused
test was removed because it passed both variants and therefore did not cover the
Laguna failure.

### 4. Inspect and compare the tileN=8 and tileN=16 GEMM2 tactics

Record for batch sizes around the failure:

- Tile size.
- TRTLLM-Gen config index.
- Reported BMM workspace size.
- Actual region offsets and alignment.
- GEMM2 input, scale, output, and BMM workspace addresses.

Compare batch sizes `5`, `4`, and `3`, with particular attention to the shared
tileN=8 TMA setup and loop bounds. Config 15's `schedS` cubin is the one
identified by the core dump, but configs 14 and 0 also fail.

### 5. Check workspace bounds

Add temporary canary padding around:

- Activation output.
- Activation output scales.
- GEMM2 output.
- GEMM2 BMM workspace.

After eager execution, check whether any kernel writes outside its reported
workspace region. CUDA graph replay cannot perform host checks, so this should
first be done in an equivalent eager run using the same tactic.

### 6. Compare valid tactics (partially completed)

Replay the same batch-size-3 problem with each valid GEMM2 tactic.

All tested tileN=8 tactics fault and the tested tileN=16 tactic passes. Continue
with tileN=32 and additional tileN=16 configs if needed to establish the exact
capability boundary.

If tileN=8 is confirmed as the boundary, the fix belongs in:

- Tactic validation.
- TMA descriptor setup or loop bounds shared by those cubins.
- The affected cubins/kernel generator.

A tileN=8 exclusion may be useful as an emergency workaround but should not be
presented as the root-cause fix without understanding the kernel defect.

Before broadening or submitting that exclusion, benchmark tileN=8 versus
tileN=16 on representative non-fused FP8 MoE decode workloads. The autotuner
selected tileN=8 for Laguna's small-token case, so a global minimum tileN of 16
has credible latency-regression risk. Prefer a kernel fix or a restriction
scoped to a proven unsafe capability boundary.

### 7. Investigate SM120 independently

Collect a CUDA GPU core dump for the SM120 Laguna failure.

SM120 routes FP8 block-scale MoE through
`run_triton_fp8_block_scale_moe()`, not the SM100 TRTLLM-Gen runner. A separate
faulting-kernel identification is required before assuming the B200 arena change
applies.

### 8. Run memory and distributed coverage before submission

Before proposing a final patch:

- Measure retained memory for representative large-vocabulary and high-batch
  models.
- Run the H100 multi-GPU stage that regressed under strong output references.
- Run B200/B300 Laguna FP8.
- Run an SM120 reproduction.
- Exercise repeated executor build, shutdown, and rebuild in one process.

### 9. Add production lifecycle wiring only if the final fix retains resources

Runner ownership is not required by the current evidence. If a later final fix
does retain backend resources, release them before resetting the graphs that
reference them. Keep this integration behind a generic backend or graph-resource
lifecycle hook; do not import `FP8BlockScaleMoERunner` directly from
`PyTorchModelEngine`.

## Submission recommendation

Do not submit either the transient arena or the broad non-fused tileN=8
exclusion as a proven root-cause fix yet.

It is appropriate as:

- A diagnostic patch.
- An emergency workaround if the additional memory is acceptable and the
  limitation is stated explicitly.
- A WIP tactic workaround while its performance impact and safe scope are
  measured.

For a normal fix, identify the tileN=8 TMA address or bounds defect and fix the
kernel generator, descriptor setup, or tactic validation. The current evidence
does not support retaining runner-owned resources.

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compile a 64×256 SM100 tile for groupwise FP8 expert projections."""

import ctypes
import subprocess
from functools import cache
from hashlib import sha256
from pathlib import Path
from tempfile import TemporaryDirectory

import flashinfer
import torch
from filelock import FileLock
from torch.utils.cpp_extension import CUDA_HOME

from tensorrt_llm.logger import logger

# The FlashInfer template already handles the device-side group offsets and
# per-128 scales. The tile changes the work distribution across output columns.
_DEFAULT_TILE = "using MmaTileShape_MNK = Shape<cute::Int<MmaSM * 128>, _128, _128>;"
_INTEGER_BYTES = 1 << 20
_WORKSPACE_BYTES = 32 << 20
_WRAPPER = r"""
#include "groupwise.cuh"

extern "C" int project_sm100(
    void* integers, int integer_bytes, void* scratch, int scratch_bytes,
    void* a, void* b, void* a_scales, void* b_scales, void* output,
    void* offsets, int max_m, int n, int k, int groups, void* stream) {
    using flashinfer::group_gemm::CutlassFP8GroupwiseScaledGroupGEMMSM100;
    using Fp8 = cutlass::float_e4m3_t;
    return CutlassFP8GroupwiseScaledGroupGEMMSM100<1, 128, 128, true, 1>(
        integers, integer_bytes, scratch, scratch_bytes,
        static_cast<Fp8*>(a), static_cast<Fp8*>(b),
        static_cast<float*>(a_scales), static_cast<float*>(b_scales),
        static_cast<cutlass::bfloat16_t*>(output), static_cast<int*>(offsets),
        max_m, n, k, groups, static_cast<cudaStream_t>(stream));
}
"""


@cache
def _load_project():
    if CUDA_HOME is None:
        return None
    base = Path(flashinfer.__file__).resolve().parent / "data"
    include = base / "include"
    header = include / "flashinfer/gemm/group_gemm_fp8_groupwise_sm100.cuh"
    nvcc = Path(CUDA_HOME) / "bin/nvcc"
    if not header.is_file() or not nvcc.is_file():
        return None
    code = header.read_text()
    if _DEFAULT_TILE not in code:
        return None
    code = code.replace(_DEFAULT_TILE, "using MmaTileShape_MNK = Shape<_64, _256, _128>;")
    for sibling in ("allocator.h", "cutlass_utils.cuh", "utils.cuh"):
        code = code.replace(f'"../{sibling}"', f"<flashinfer/{sibling}>")
    key = sha256(
        f"{code}\n{_WRAPPER}\n{flashinfer.__version__}\n{nvcc}\n{torch.version.cuda}".encode()
    ).hexdigest()[:16]
    root = Path.home() / ".cache" / "torch_extensions" / "qwen36_fp8_groupwise"
    root.mkdir(parents=True, exist_ok=True)
    library = root / f"sm100a-{key}.so"
    with FileLock(str(library) + ".lock"):
        if not library.exists():
            with TemporaryDirectory(dir=root) as tmp:
                build = Path(tmp)
                (build / "groupwise.cuh").write_text(code)
                (build / "projection.cu").write_text(_WRAPPER)
                output = build / "projection.so"
                try:
                    subprocess.run(
                        [
                            str(nvcc),
                            "-std=c++17",
                            "-O3",
                            "-arch=sm_100a",
                            "--expt-relaxed-constexpr",
                            "--expt-extended-lambda",
                            "-shared",
                            "-Xcompiler=-fPIC",
                            f"-I{include}",
                            f"-I{base / 'cutlass/include'}",
                            f"-I{base / 'cutlass/tools/util/include'}",
                            str(build / "projection.cu"),
                            "-o",
                            str(output),
                        ],
                        check=True,
                        capture_output=True,
                        text=True,
                    )
                except subprocess.CalledProcessError as exc:
                    logger.warning("Qwen3.6 FP8 prefill tile compilation failed: %s", exc.stderr)
                    return None
                output.replace(library)
    project = ctypes.CDLL(str(library)).project_sm100
    ptr, integer = ctypes.c_void_p, ctypes.c_int
    project.argtypes = [ptr, integer, ptr, integer, *([ptr] * 6), *([integer] * 4), ptr]
    project.restype = integer
    return project


def project(
    a: torch.Tensor,
    weight: torch.Tensor,
    a_scales: torch.Tensor,
    weight_scales: torch.Tensor,
    offsets: torch.Tensor,
) -> torch.Tensor | None:
    """Project one packed expert matrix with scratch scoped to the current stream."""
    launch = _load_project()
    if launch is None:
        return None
    groups, n, k = weight.shape
    int_workspace = a.new_empty((_INTEGER_BYTES,), dtype=torch.uint8)
    workspace = a.new_empty((_WORKSPACE_BYTES,), dtype=torch.uint8)
    output = a.new_empty((a.shape[0], n), dtype=torch.bfloat16)
    stream = torch.cuda.current_stream(a.device)
    status = launch(
        int_workspace.data_ptr(),
        _INTEGER_BYTES,
        workspace.data_ptr(),
        _WORKSPACE_BYTES,
        a.data_ptr(),
        weight.data_ptr(),
        a_scales.data_ptr(),
        weight_scales.data_ptr(),
        output.data_ptr(),
        offsets.data_ptr(),
        a.shape[0],
        n,
        k,
        groups,
        stream.cuda_stream,
    )
    if status:
        raise RuntimeError(f"Qwen3.6 groupwise FP8 projection returned CUDA error {status}")
    return output

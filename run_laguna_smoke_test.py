#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 NVIDIA Corporation. All rights reserved.
"""Smoke test for Laguna-XS with TensorRT-LLM PyTorch backend.

Usage:
    python run_laguna_smoke_test.py [--model PATH] [--tp N]
"""

import argparse
import json
import os
import sys
import tempfile
import time

MODEL_PATH = "/home/scratch.dcampora_gpu/laguna-xs-drop-16-04-2026/laguna-xs-fp8-hf"
MODELING_DIR = "/home/scratch.dcampora_gpu/projects/poolside/dan_bench_serving"


def make_shadow_dir(model_dir: str) -> str:
    """Temp dir with quantization_config / compression_config stripped from config.json.

    If the checkpoint has fp8 block-scale quantization, writes hf_quant_config.json
    so TRT-LLM's ModelConfig.from_pretrained auto-detects fp8.
    """
    shadow_dir = tempfile.mkdtemp(prefix="laguna_smoke_")
    with open(os.path.join(model_dir, "config.json")) as f:
        cfg = json.load(f)

    # Detect fp8 before stripping quant fields
    import modeling_laguna_trtllm as _m

    quant_config = _m._detect_quant_config(model_dir)

    for key in ("quantization_config", "compression_config"):
        cfg.pop(key, None)
    with open(os.path.join(shadow_dir, "config.json"), "w") as f:
        json.dump(cfg, f, indent=2)

    if quant_config is not None:
        print(f"Detected quantization: {quant_config.quant_algo}")
        exclude_modules = _m._get_fp8_exclude_modules(model_dir)
        hf_quant = {
            "quantization": {
                "quant_algo": "fp8_pb_wo",
                "group_size": 128,
                "exclude_modules": exclude_modules,
            }
        }
        with open(os.path.join(shadow_dir, "hf_quant_config.json"), "w") as f:
            json.dump(hf_quant, f, indent=2)

    for entry in os.listdir(model_dir):
        if entry == "config.json":
            continue
        os.symlink(
            os.path.join(model_dir, entry),
            os.path.join(shadow_dir, entry),
        )
    return shadow_dir


SMOKE_PROMPTS = [
    "poolside is an AI startup that offers",
    "The capital of France is",
    "In Python, to reverse a list you can",
    "def fibonacci(n):",
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=MODEL_PATH)
    parser.add_argument("--tp", type=int, default=1)
    parser.add_argument("--max-tokens", type=int, default=128)
    args = parser.parse_args()

    # Register the out-of-tree model before importing LLM
    sys.path.insert(0, MODELING_DIR)
    import modeling_laguna_trtllm  # noqa: F401 — side-effect: registers model

    from tensorrt_llm import LLM, SamplingParams

    shadow = make_shadow_dir(args.model)
    print(f"Shadow config dir: {shadow}")
    print(f"Loading model (TP={args.tp})...")
    t0 = time.time()
    llm = LLM(
        model=shadow, tensor_parallel_size=args.tp, trust_remote_code=True, cuda_graph_config=None
    )
    print(f"Model loaded in {time.time() - t0:.1f}s\n")

    sampling = SamplingParams(max_tokens=args.max_tokens, temperature=0.0)

    print("=" * 60)
    print("SMOKE TEST RESULTS")
    print("=" * 60)
    t1 = time.time()
    outputs = llm.generate(SMOKE_PROMPTS, sampling)
    elapsed = time.time() - t1
    total_tokens = sum(len(o.outputs[0].token_ids) for o in outputs)

    all_ok = True
    for i, output in enumerate(outputs):
        prompt = output.prompt
        text = output.outputs[0].text
        n_tok = len(output.outputs[0].token_ids)
        finish = output.outputs[0].finish_reason
        ok = n_tok > 0 and text.strip() != ""
        status = "OK" if ok else "FAIL (empty output)"
        if not ok:
            all_ok = False
        print(f"\n[{i + 1}] {status}")
        print(f"     Prompt: {prompt!r}")
        print(f"     Generated ({n_tok} tok, finish={finish}): {text!r}")

    print(f"\nTotal: {total_tokens} tokens in {elapsed:.2f}s ({total_tokens / elapsed:.1f} tok/s)")

    if all_ok:
        print("\nSmoke test PASSED.")
        sys.exit(0)
    else:
        print("\nSmoke test FAILED.")
        sys.exit(1)


if __name__ == "__main__":
    main()

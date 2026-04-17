#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 NVIDIA Corporation. All rights reserved.
"""
GSM8K evaluation for Laguna-XS with TensorRT-LLM PyTorch backend.

Runs 8-shot chain-of-thought evaluation on GSM8K test set.
Extracts the final numerical answer and reports accuracy.

Usage:
    python run_laguna_gsm8k.py [--model PATH] [--tp N] [--limit N] [--batch-size N]
"""
import argparse
import json
import os
import re
import sys
import tempfile
import time
from typing import Optional

MODEL_PATH = "/home/scratch.dcampora_gpu/laguna-xs-drop-16-04-2026/laguna-xs-fp8-hf"
MODELING_DIR = "/home/scratch.dcampora_gpu/projects/poolside/dan_bench_serving"

# Standard 8-shot GSM8K CoT examples (from the original paper / widely used)
FEW_SHOT_EXAMPLES = [
    {
        "question": "There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today?",
        "answer": "There are 15 trees originally. Then there were 21 trees after some more were planted. So there must have been 21 - 15 = 6. The answer is 6.",
    },
    {
        "question": "If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot?",
        "answer": "There are originally 3 cars. 2 more cars arrive. 3 + 2 = 5. The answer is 5.",
    },
    {
        "question": "Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total?",
        "answer": "Originally, Leah had 32 chocolates. Her sister had 42. So in total they had 32 + 42 = 74. After eating 35, they had 74 - 35 = 39. The answer is 39.",
    },
    {
        "question": "Jason had 20 lollipops. He gave Denny some lollipops. Now Jason has 12 lollipops. How many lollipops did Jason give to Denny?",
        "answer": "Jason started with 20 lollipops. Then he gave some to Denny. Now he has 12 lollipops. So he gave Denny 20 - 12 = 8 lollipops. The answer is 8.",
    },
    {
        "question": "Shawn has five toys. For Christmas, he got two toys each from his mom and dad. How many toys does he have now?",
        "answer": "Shawn started with 5 toys. If he got 2 toys each from his mom and dad, then that is 4 more toys. 5 + 4 = 9. The answer is 9.",
    },
    {
        "question": "There were nine computers in the server room. Five more computers were installed each day, from monday to thursday. How many computers are now in the server room?",
        "answer": "There were originally 9 computers. For each of 4 days, 5 more computers were added. So 5 * 4 = 20 computers were added. 9 + 20 is 29. The answer is 29.",
    },
    {
        "question": "Michael had 58 golf balls. On tuesday, he lost 23 golf balls. On wednesday, he lost 2 more. How many golf balls did he have at the end of wednesday?",
        "answer": "Michael started with 58 golf balls. After losing 23 on tuesday, he had 58 - 23 = 35. After losing 2 more on wednesday, he had 35 - 2 = 33 golf balls. The answer is 33.",
    },
    {
        "question": "Olivia has $23. She bought five bagels for $3 each. How much money does she have left?",
        "answer": "Olivia had 23 dollars. 5 bagels for 3 dollars each will be 5 x 3 = 15 dollars. So she has 23 - 15 dollars left. 23 - 15 is 8. The answer is 8.",
    },
]


def build_few_shot_prefix() -> str:
    lines = []
    for ex in FEW_SHOT_EXAMPLES:
        lines.append(f"Question: {ex['question']}")
        lines.append(f"Answer: {ex['answer']}")
        lines.append("")
    return "\n".join(lines)


FEW_SHOT_PREFIX = build_few_shot_prefix()


def build_prompt(question: str) -> str:
    return FEW_SHOT_PREFIX + f"Question: {question}\nAnswer:"


def extract_answer(text: str) -> Optional[float]:
    """Extract the final numerical answer from model output."""
    # Look for "The answer is X" pattern first
    m = re.search(r"[Tt]he answer is[:\s]*(-?[\d,]+(?:\.\d+)?)", text)
    if m:
        try:
            return float(m.group(1).replace(",", ""))
        except ValueError:
            pass

    # Fall back to last number in text
    numbers = re.findall(r"-?[\d,]+(?:\.\d+)?", text)
    if numbers:
        try:
            return float(numbers[-1].replace(",", ""))
        except ValueError:
            pass
    return None


def extract_ground_truth(answer_str: str) -> Optional[float]:
    """Extract ground truth from GSM8K answer string (after ####)."""
    m = re.search(r"####\s*(-?[\d,]+(?:\.\d+)?)", answer_str)
    if m:
        try:
            return float(m.group(1).replace(",", ""))
        except ValueError:
            pass
    return None


def make_shadow_dir(model_dir: str) -> str:
    """Temp dir with quantization_config / compression_config stripped from config.json.

    If the checkpoint has fp8 block-scale quantization, writes hf_quant_config.json
    so TRT-LLM's ModelConfig.from_pretrained auto-detects fp8.
    """
    shadow_dir = tempfile.mkdtemp(prefix="laguna_gsm8k_")
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=MODEL_PATH)
    parser.add_argument("--tp", type=int, default=1)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit number of test examples (default: all 1319)")
    parser.add_argument("--output", default=None,
                        help="Path to save per-example results as JSONL")
    args = parser.parse_args()

    # Register the out-of-tree model
    sys.path.insert(0, MODELING_DIR)
    import modeling_laguna_trtllm  # noqa: F401 — side-effect: registers model

    from datasets import load_dataset
    from tensorrt_llm import LLM, SamplingParams

    print("Loading GSM8K test set...")
    dataset = load_dataset("gsm8k", "main", split="test")
    if args.limit:
        dataset = dataset.select(range(args.limit))
    print(f"Examples: {len(dataset)}")

    shadow = make_shadow_dir(args.model)
    print(f"Loading model from {shadow} (TP={args.tp})...")
    t0 = time.time()
    llm = LLM(model=shadow, tensor_parallel_size=args.tp, trust_remote_code=True,
              cuda_graph_config=None)
    print(f"Model loaded in {time.time() - t0:.1f}s")

    sampling = SamplingParams(
        max_tokens=args.max_tokens,
        temperature=0.0,
        stop=["Question:", "\n\nQuestion:"],
    )

    questions = [ex["question"] for ex in dataset]
    answers = [ex["answer"] for ex in dataset]
    prompts = [build_prompt(q) for q in questions]

    print(f"\nRunning inference on {len(prompts)} examples...")
    t1 = time.time()
    outputs = llm.generate(prompts, sampling)
    elapsed = time.time() - t1
    total_tokens = sum(len(o.outputs[0].token_ids) for o in outputs)
    print(f"Inference: {total_tokens} tokens in {elapsed:.1f}s "
          f"({total_tokens/elapsed:.1f} tok/s)\n")

    correct = 0
    results = []
    for i, (output, gt_str) in enumerate(zip(outputs, answers)):
        pred_text = output.outputs[0].text
        pred_val = extract_answer(pred_text)
        gt_val = extract_ground_truth(gt_str)
        is_correct = (pred_val is not None and gt_val is not None
                      and abs(pred_val - gt_val) < 1e-6)
        if is_correct:
            correct += 1
        results.append({
            "idx": i,
            "question": questions[i],
            "prediction": pred_text,
            "pred_val": pred_val,
            "gt_val": gt_val,
            "correct": is_correct,
        })

    accuracy = correct / len(results) * 100
    print("=" * 60)
    print(f"GSM8K Results ({len(results)} examples)")
    print("=" * 60)
    print(f"Correct:  {correct} / {len(results)}")
    print(f"Accuracy: {accuracy:.2f}%")

    # Show a few wrong examples
    wrong = [r for r in results if not r["correct"]]
    if wrong:
        print(f"\nFirst 3 incorrect examples:")
        for r in wrong[:3]:
            print(f"  [{r['idx']}] Q: {r['question'][:80]}...")
            print(f"       pred={r['pred_val']}  gt={r['gt_val']}")
            print(f"       output: {r['prediction'][:150]!r}")

    if args.output:
        with open(args.output, "w") as f:
            for r in results:
                f.write(json.dumps(r) + "\n")
        print(f"\nResults saved to {args.output}")

    return accuracy


if __name__ == "__main__":
    main()

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Validate engine.sh on SM100 using an independently generated token oracle."""

import json
import math
import os
import subprocess
from pathlib import Path

import pytest
import torch

_REPOSITORY = Path(__file__).resolve().parents[7]


def test_engine_oracle() -> None:
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 0):
        pytest.skip("The Qwen3.6 FP8 target requires SM100")
    root = Path(os.environ.get("LLM_MODELS_ROOT", "/"))
    model = next(
        (
            path
            for path in (root / "Qwen3.6-35B-A3B-FP8", root / "model", root)
            if (path / "config.json").exists()
        ),
        None,
    )
    oracle_path = Path(os.environ.get("TRTLLM_QWEN36_FP8_DEV_ORACLE", "/task/oracle_dev.json"))
    if model is None or not oracle_path.is_file():
        pytest.skip("The official FP8 checkpoint and independent token oracle are required")

    oracle = json.loads(oracle_path.read_text())
    fragile = {(entry["prompt"], entry["token"]) for entry in oracle["fragile"]}
    score_count = top1_count = non_tie_count = 0
    kl_sum = 0.0
    divergences = []
    with subprocess.Popen(
        [_REPOSITORY / "engine.sh", model],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        cwd=_REPOSITORY,
        text=True,
    ) as proc:
        assert proc.stdin is not None and proc.stdout is not None

        def request(**payload: object) -> dict:
            proc.stdin.write(json.dumps(payload) + "\n")
            proc.stdin.flush()
            line = proc.stdout.readline()
            assert line, f"engine stopped during {payload['op']}"
            result = json.loads(line)
            assert "error" not in result, result
            return result

        try:
            assert json.loads(proc.stdout.readline()) == {"ready": True}
            for index, item in enumerate(oracle["items"]):
                reference = item["top_k"]
                score = request(
                    op="score",
                    ids=item["seq_ids"],
                    top_k=2,
                    logprob_ids=[[token for token, _ in row] for row in reference],
                )
                for answer, logprobs, row in zip(
                    score["top1"], score["logprobs"], reference, strict=True
                ):
                    kl_sum += sum(
                        math.exp(p) * (p - q) for (_, p), q in zip(row, logprobs, strict=True)
                    )
                    score_count += 1
                    if row[0][1] - row[1][1] >= 0.5:
                        non_tie_count += 1
                        top1_count += answer == row[0][0]
                generated = request(
                    op="generate",
                    ids=item["prompt_ids"],
                    max_new_tokens=len(item["gen_ids"]),
                    ignore_eos=True,
                )["ids"]
                for step, (answer, expected) in enumerate(
                    zip(generated, item["gen_ids"], strict=True)
                ):
                    if answer == expected:
                        continue
                    position = len(item["prompt_ids"]) + step - 1
                    row = reference[position]
                    gap = row[0][1] - row[1][1]
                    if gap >= 0.5 and (
                        (index, step) not in fragile or answer != score["top1"][position]
                    ):
                        divergences.append((index, step, gap, answer, expected))
                    break
            assert request(op="shutdown") == {"ok": True}
            assert proc.wait(timeout=30) == 0
        finally:
            if proc.poll() is None:
                proc.terminate()
    assert kl_sum / score_count <= 0.01
    assert top1_count / non_tie_count >= 0.99
    assert not divergences, divergences

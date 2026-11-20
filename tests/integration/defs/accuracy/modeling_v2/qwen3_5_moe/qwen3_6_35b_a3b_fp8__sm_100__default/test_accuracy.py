# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Whole-model score and greedy-generation gate for the SM100 target."""

import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch


def test_development_oracle():
    """Check the checkpoint and reference outputs through the JSONL entry point."""
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 0):
        pytest.skip("Qwen3.6 FP8 target requires SM100")
    root = Path(os.environ.get("LLM_MODELS_ROOT", ""))
    model = Path(os.environ.get("QWEN36_FP8_MODEL_DIR", root / "Qwen" / "Qwen3.6-35B-A3B-FP8"))
    oracle = Path(os.environ.get("QWEN36_FP8_ORACLE", model / "oracle_dev.json"))
    if not (model / "config.json").is_file() or not oracle.is_file():
        pytest.skip("Set QWEN36_FP8_MODEL_DIR and QWEN36_FP8_ORACLE to the checkpoint and oracle")
    repo = Path(__file__).resolve().parents[7]
    subprocess.run(
        [
            sys.executable,
            "-m",
            "aegis_engine.oracle",
            "--model",
            str(model.resolve()),
            "--oracle",
            str(oracle.resolve()),
        ],
        cwd=repo,
        check=True,
        timeout=900,
    )

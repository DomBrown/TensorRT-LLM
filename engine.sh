#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail
if [[ $# != 1 ]]; then
    printf 'Usage: %s MODEL_DIR\n' "$0" >&2
    exit 2
fi
cd -- "$(dirname -- "$0")"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export TLLM_WORKER_USE_SINGLE_PROCESS=1
exec python3 -u -m aegis_engine.protocol "$1"

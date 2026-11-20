#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail
if [[ $# != 1 && ( $# != 3 || ${2-} != --port ) ]]; then
    printf 'Usage: %s MODEL_DIR [--port N]\n' "$0" >&2
    exit 2
fi
script_dir=$(dirname -- "$(realpath -- "$0")")
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export TLLM_OPENAI_KEEP_ALIVE_TIMEOUT="${TLLM_OPENAI_KEEP_ALIVE_TIMEOUT:-30}"
exec trtllm-serve "$1" --host 0.0.0.0 --port "${3:-8000}" \
    --served_model_name qwen3.6-35b-a3b-fp8 --config "$script_dir/aegis_engine/serve.yaml"

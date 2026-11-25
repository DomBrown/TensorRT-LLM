#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail
if [[ $# -ne 1 ]] && { [[ $# -ne 3 ]] || [[ $2 != --port ]]; }; then
    echo 'usage: serve.sh MODEL_DIRECTORY [--port N]' >&2
    exit 2
fi
root="$(dirname -- "$(realpath -- "$0")")"
export PYTHONPATH="$root${PYTHONPATH:+:$PYTHONPATH}"
export TLLM_WORKER_USE_SINGLE_PROCESS=1
export TRTLLM_USE_GDN_REPLAY="${TRTLLM_USE_GDN_REPLAY:-1}"
exec trtllm-serve "$1" --config "$root/aegis_engine/serve.yaml" --port "${3:-8000}" \
    --num_serve_frontends 1 --served_model_name qwen3.6-35b-a3b-fp8

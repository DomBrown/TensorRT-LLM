#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail
if [[ $# -ne 1 ]]; then
    echo 'usage: engine.sh MODEL_DIRECTORY' >&2
    exit 2
fi
export PYTHONPATH="$(dirname -- "$(realpath -- "$0")")${PYTHONPATH:+:$PYTHONPATH}"
export TLLM_WORKER_USE_SINGLE_PROCESS=1
export TRTLLM_USE_GDN_REPLAY=0
exec python3 -u -m aegis_engine "$1"

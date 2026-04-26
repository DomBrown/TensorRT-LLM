#!/usr/bin/env bash
# Run trtllm-eval on a HuggingFace model checkpoint against gsm8k or mmlu.
#
# Usage:
#   run_trtllm_eval.sh --model <path> --task <gsm8k|mmlu> [extra trtllm-eval args...]
#
# Examples:
#   run_trtllm_eval.sh --model /poolside/models/old/laguna-xs-bf16 --task gsm8k
#   run_trtllm_eval.sh -m /poolside/models/old/laguna-xs-bf16 -t mmlu --num_samples 200

set -euo pipefail

MODEL=""
TASK=""
CONFIG=""

usage() {
    cat <<EOF
Usage: $(basename "$0") --model <path> --task <gsm8k|mmlu> [--config YAML] [extra args...]

Required:
  -m, --model PATH      Path to HF checkpoint (or model name).
  -t, --task  TASK      Either 'gsm8k' or 'mmlu'.

Optional:
  -c, --config YAML     Path to a YAML config that overrides LLM API options
                        (e.g. moe_config.backend). Forwarded to trtllm-eval as
                        a top-level --config flag (must precede the subcommand).

Any additional flags after the recognized options are forwarded to the
trtllm-eval subcommand (e.g. --num_samples, --dataset_path, --apply_chat_template).
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        -m|--model)
            MODEL="$2"; shift 2 ;;
        -t|--task)
            TASK="$2"; shift 2 ;;
        -c|--config)
            CONFIG="$2"; shift 2 ;;
        -h|--help)
            usage; exit 0 ;;
        --)
            shift; break ;;
        *)
            break ;;
    esac
done

if [[ -z "$MODEL" || -z "$TASK" ]]; then
    echo "ERROR: --model and --task are required." >&2
    usage >&2
    exit 1
fi

case "$TASK" in
    gsm8k|mmlu) ;;
    *)
        echo "ERROR: --task must be 'gsm8k' or 'mmlu' (got '$TASK')." >&2
        exit 1
        ;;
esac

if [[ ! -d "$MODEL" && ! -f "$MODEL" ]]; then
    echo "WARN: '$MODEL' is not a local path; will be passed to trtllm-eval as-is." >&2
fi

# Trust remote code is required for models that ship custom modeling files
# in their checkpoint (e.g. Laguna's auto_map -> modeling_laguna.LagunaForCausalLM).
TRUST_REMOTE_CODE_FLAG="--trust_remote_code"

echo "Running trtllm-eval"
echo "  model : $MODEL"
echo "  task  : $TASK"
echo "  config: ${CONFIG:-<none>}"
echo "  extra : $*"

# `--config` must precede the subcommand on trtllm-eval; everything else after
# the subcommand is treated as a subcommand option.
CONFIG_ARGS=()
if [[ -n "$CONFIG" ]]; then
    CONFIG_ARGS+=(--config "$CONFIG")
fi

set -x
exec trtllm-eval \
    --model "$MODEL" \
    --backend pytorch \
    $TRUST_REMOTE_CODE_FLAG \
    "${CONFIG_ARGS[@]}" \
    "$TASK" \
    "$@"

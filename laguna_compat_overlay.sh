#!/usr/bin/env bash
# Build a transformers-4.57-compatible overlay of a Laguna checkpoint directory.
#
# The newer Laguna checkpoints (e.g. /poolside/models/april25/) ship a
# `configuration_laguna.py` that imports `PreTrainedConfig` and `RopeParameters`
# — symbols only present in transformers >= 4.58. To run them on the installed
# transformers 4.57.3, we create an overlay directory of symlinks to all the
# checkpoint files, with two replacements:
#   1. `configuration_laguna.py` is replaced by a patched copy that adds shims
#      for the missing transformers symbols.
#   2. `config.json` is rewritten:
#        - `gating: true` is normalized to the legacy string `"per-head"` that
#          TRT-LLM's LagunaAttention reads to size the per-head gate weight.
#        - The v5-style nested `rope_parameters`
#          (`{"full_attention": {...}, "sliding_attention": {...}}`) is
#          flattened to legacy top-level keys (`rope_theta`, `rope_scaling`,
#          `partial_rotary_factor`, `swa_rope_parameters`) that TRT-LLM's
#          `RopeParams.from_config` and `_build_rope_params` already handle.
#
# Usage:
#   laguna_compat_overlay.sh <src_checkpoint_dir> [overlay_dir]
# Echoes the overlay directory path on stdout.
set -euo pipefail

if [[ $# -lt 1 ]]; then
    echo "Usage: $(basename "$0") <src_checkpoint_dir> [overlay_dir]" >&2
    exit 1
fi

SRC="$(realpath "$1")"
OVERLAY="${2:-/tmp/laguna-compat-$(basename "$SRC")}"

if [[ ! -d "$SRC" ]]; then
    echo "ERROR: source dir does not exist: $SRC" >&2
    exit 1
fi
if [[ ! -f "$SRC/configuration_laguna.py" ]]; then
    echo "ERROR: $SRC has no configuration_laguna.py" >&2
    exit 1
fi

rm -rf "$OVERLAY"
mkdir -p "$OVERLAY"

# Symlink every entry except configuration_laguna.py and config.json (those are
# replaced by patched copies below).
for entry in "$SRC"/*; do
    name="$(basename "$entry")"
    case "$name" in
        configuration_laguna.py|config.json)
            continue ;;
    esac
    ln -s "$entry" "$OVERLAY/$name"
done

# Rewrite config.json: normalize `gating`, flatten v5 `rope_parameters`,
# and translate compressed-tensors `re:`-prefixed regex `ignore` patterns to
# fnmatch glob patterns (TRT-LLM's `is_module_excluded_from_quantization`
# uses fnmatch.fnmatchcase, which doesn't understand the `re:` prefix nor
# regex syntax — without this fix, `re:.*\.self_attn\.q_proj$`-style
# entries match nothing and the modules get unintentionally quantized).
python3 - "$SRC/config.json" "$OVERLAY/config.json" <<'PY'
import json, re, sys, copy
src_path, dst_path = sys.argv[1], sys.argv[2]
with open(src_path) as f:
    cfg = json.load(f)

# Normalize `gating` (newer Laguna configs use boolean `true`; TRT-LLM's
# LagunaAttention reads the legacy string "per-head" to size the per-head
# gate projection).
if cfg.get("gating") is True:
    cfg["gating"] = "per-head"

# Flatten v5 nested `rope_parameters` to legacy top-level keys.
rp = cfg.get("rope_parameters")
if isinstance(rp, dict) and {"full_attention", "sliding_attention"} & set(rp.keys()):
    full = copy.deepcopy(rp.get("full_attention", {}))
    swa = copy.deepcopy(rp.get("sliding_attention", {}))

    if full:
        if "rope_theta" in full:
            cfg["rope_theta"] = full.pop("rope_theta")
        if "partial_rotary_factor" in full:
            cfg["partial_rotary_factor"] = full.pop("partial_rotary_factor")
        if full:
            cfg["rope_scaling"] = full
    if swa:
        cfg["swa_rope_parameters"] = swa

    cfg.pop("rope_parameters", None)


def _re_to_fnmatch(pattern: str) -> str:
    """Best-effort translate a compressed-tensors `re:<regex>` pattern to a
    glob (fnmatch) pattern that TRT-LLM understands. Conservative: handles the
    common shapes seen in compressed-tensors `ignore` lists, namely
    `.*\.foo\.bar$` style anchored regex. Uses no special regex behavior.
    """
    if not pattern.startswith("re:"):
        return pattern
    body = pattern[3:]
    # Strip an optional trailing `$` (end-anchor) since fnmatch always
    # anchors the full string.
    if body.endswith("$"):
        body = body[:-1]
    # Strip a leading `^` if present.
    if body.startswith("^"):
        body = body[1:]
    # Translate `.*` → `*` (regex any-chars → glob any-chars).
    body = body.replace(".*", "*")
    # Translate escaped dot `\.` → literal `.`.
    body = body.replace("\\.", ".")
    # If the result still contains regex-only constructs (e.g. `[^.]`,
    # alternation, character classes), fall back to a permissive `*`-wildcard
    # form: keep only the literal-looking suffix after the last unescaped
    # special char. Conservative; better to over-include than fail silently.
    if any(c in body for c in r".+?()[]{}|^"):
        # Keep tail after last unsupported char, prefixed with `*`.
        # In practice this branch rarely fires for compressed-tensors patterns.
        m = re.search(r"[^.+?()\[\]{}|^]+$", body)
        body = "*" + (m.group(0) if m else body)
    return body


# Translate `ignore` and `targets` lists in any quant config blocks.
for cfg_key in ("quantization_config", "compression_config"):
    qc = cfg.get(cfg_key)
    if not isinstance(qc, dict):
        continue
    if isinstance(qc.get("ignore"), list):
        qc["ignore"] = [_re_to_fnmatch(p) for p in qc["ignore"]]
    # Some quant config groups also use `targets` regexes; translate those too
    # (harmless if TRT-LLM ignores them).
    cgs = qc.get("config_groups", {})
    if isinstance(cgs, dict):
        for gname, g in cgs.items():
            if isinstance(g, dict) and isinstance(g.get("targets"), list):
                g["targets"] = [_re_to_fnmatch(p) for p in g["targets"]]
    tc = qc.get("transform_config", {}) or {}
    cgs = tc.get("config_groups", {}) if isinstance(tc, dict) else {}
    if isinstance(cgs, dict):
        for gname, g in cgs.items():
            if not isinstance(g, dict):
                continue
            for entry in g.get("apply", []) or []:
                if isinstance(entry, dict) and isinstance(entry.get("targets"), list):
                    entry["targets"] = [_re_to_fnmatch(p) for p in entry["targets"]]

with open(dst_path, "w") as f:
    json.dump(cfg, f, indent=2)
PY

# Build the patched configuration_laguna.py: prepend a compat shim that aliases
# PreTrainedConfig/RopeParameters when the local transformers version is too
# old, then drop the original imports for those symbols.
SHIM_HEADER='# --- compat shim (added by laguna_compat_overlay.sh) ---
try:
    from transformers.configuration_utils import PreTrainedConfig  # transformers >= 4.58
except ImportError:
    from transformers.configuration_utils import PretrainedConfig as PreTrainedConfig  # transformers 4.57
try:
    from transformers.modeling_rope_utils import RopeParameters  # transformers >= 4.58
except ImportError:
    RopeParameters = dict  # type: ignore[misc,assignment]
# --- end compat shim ---
'

{
    printf '%s' "$SHIM_HEADER"
    grep -vE '^from transformers\.configuration_utils import PreTrainedConfig$|^from transformers\.modeling_rope_utils import RopeParameters$' \
        "$SRC/configuration_laguna.py"
} > "$OVERLAY/configuration_laguna.py"

echo "$OVERLAY"

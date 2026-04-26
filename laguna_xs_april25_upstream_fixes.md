# Laguna april25 — overlay fixes & upstream actions

This is the short, actionable companion to `laguna_xs_april25_eval_report.md`.
It lists every fix the workspace overlay applies to make the
`/poolside/models/april25/laguna-xs-{bf16,fp8}` checkpoints run on
TensorRT-LLM `1.3.0rc13` (transformers `4.57.3`), and what needs to land
upstream so the overlay can be deleted.

(`laguna-xs-nvfp4` is a client-side checkpoint export bug — the layer-0
dense MLP weights are missing — not a TRT-LLM problem; not covered here.)

## Problems we fixed (all in the overlay; tracked source unchanged)

| # | Symptom | Where the overlay fixes it |
|---|---|---|
| 1 | `ImportError: cannot import name 'PreTrainedConfig' from 'transformers.configuration_utils'` (and same for `RopeParameters`) | `configuration_laguna.py` gets a try/except compat shim aliasing `PreTrainedConfig → PretrainedConfig` and `RopeParameters → dict` |
| 2 | `AssertionError: Per-layer-type RoPE configuration is not supported yet.` | `config.json` rewrite flattens the v5 nested `rope_parameters: {full_attention: {...}, sliding_attention: {...}}` into legacy top-level `rope_theta`, `rope_scaling`, `partial_rotary_factor`, `swa_rope_parameters` |
| 3 | `RuntimeError: tensor a (8192) vs tensor b (64)` during weight load | `config.json` rewrite normalizes `gating: true` back to the legacy string `gating: "per-head"` so `LagunaAttention` sizes the per-head `g_proj` correctly |
| 4 | FP8: model runs but every layer's attention output is **exactly zero** → GSM8K = 0 % | `config.json` rewrite translates the `re:`-prefixed regex patterns in `quantization_config.ignore` (and `config_groups[*].targets`, `transform_config.config_groups[*].apply.targets`) to fnmatch glob patterns |

After all four overlay fixes (and zero source-code changes), GSM8K
strict-match on the overlaid april25 checkpoints:

| ckpt | strict-match | flex-extract | avg |
|---|---:|---:|---:|
| BF16 | 86.96 % | 85.60 % | 86.28 |
| FP8 | 87.19 % | 86.13 % | 86.66 |

(FP8 is essentially BF16-equivalent on this benchmark — full lossless
QuaRot+FP8 once the upstream bugs are worked around.)

## What needs to land upstream so we can drop the overlay

In priority / impact order:

### (a) Honor `re:`-prefixed regex patterns in `quant_config.exclude_modules`

**File:** `tensorrt_llm/models/modeling_utils.py`,
`QuantConfig.is_module_excluded_from_quantization` (currently line 253).

Currently uses `fnmatch.fnmatchcase`, which silently does not match
the `re:.*\.foo$` patterns that compressed-tensors writes. Result on
this checkpoint: q/k/v/o/g_proj and the MoE router get FP8-quantized
despite being in `ignore`, the attention output goes FP8 and gets
clipped to zero by an uncalibrated `o_proj.input_scale = 1.0`, and
**every compressed-tensors checkpoint with regex `ignore` patterns
silently mis-quantizes**. Laguna just happened to have a per-head
attention gate that exposed this as a hard error / 0 % accuracy
instead of "subtly wrong logits".

Suggested patch:

```python
import re
def is_module_excluded_from_quantization(self, name: str) -> bool:
    if self.exclude_modules is None:
        return False
    for pattern in self.exclude_modules:
        if pattern.startswith("re:"):
            if re.fullmatch(pattern[3:], name):
                return True
        elif fnmatch.fnmatchcase(name, pattern):
            return True
    return False
```

The same `re:`-prefix dispatch should also be applied wherever
`config_groups[*].targets` and `transform_config.*.apply.targets`
strings are matched against module names.

**Drops overlay step #4.** Highest impact — this is a TRT-LLM-wide
correctness bug, not Laguna-specific.

### (b) Accept the v5 nested `rope_parameters` (per-layer-type) format

**File:** `tensorrt_llm/_torch/attention_backend/interface.py`,
`RopeParams.from_config` (around line 487):

```python
assert not set(hf_rope_parameters.keys()).issubset(
    ALLOWED_ATTENTION_LAYER_TYPES), (
        "Per-layer-type RoPE configuration is not supported yet.")
config.update(hf_rope_parameters)
```

This rejects the new HuggingFace v5 schema:

```jsonc
"rope_parameters": {
  "full_attention":   { "rope_type": "yarn",    "rope_theta": 500000.0, ... },
  "sliding_attention":{ "rope_type": "default", "rope_theta": 10000.0,  ... }
}
```

Suggested behavior: when `rope_parameters` keys are layer-type names,
flatten the `full_attention` entry onto `config` and store the
`sliding_attention` entry as `config.swa_rope_parameters`. Then call
the existing flat-form path. (This is exactly what the overlay does
today.)

**Drops overlay step #2.** Likely also needed by future Laguna and
other layer-type-aware models from compressed-tensors / vLLM exports.

### (c) Accept `gating: true` as equivalent to `gating: "per-head"`

**File:** `tensorrt_llm/_torch/models/modeling_laguna.py`, line ~348:

```python
self._gate_per_head = gating == "per-head"
```

Two-character fix:

```python
self._gate_per_head = (gating == "per-head") or gating is True
```

**Drops overlay step #3.** Trivial; safe (the boolean form is what
newer Laguna configs ship, and the per-head per-layer `g_proj` weights
in the checkpoint are the same shape either way).

### (d) Pre-supply transformers ≥ 4.58 symbols (or bump the dep)

**Cause:** `configuration_laguna.py` (custom modeling code shipped
inside the checkpoint) imports `PreTrainedConfig` (PascalCase) and
`RopeParameters`, both introduced in transformers ≥ 4.58. The TRT-LLM
container ships transformers 4.57.3.

Two equally good options upstream:

1. Bump TRT-LLM's `requirements.txt` to transformers ≥ 4.58 (the
   intended path; `auto_map`-loaded custom modeling code is allowed to
   use new transformers symbols).
2. Or, if the dep bump is risky, ship a tiny shim module under
   `tensorrt_llm/_torch/_transformers_compat.py` that re-exports
   `PretrainedConfig as PreTrainedConfig` and a stub `RopeParameters`
   — and register it on `transformers.configuration_utils` /
   `transformers.modeling_rope_utils` via `sys.modules` patches at
   import time, so any checkpoint's `auto_map` code can find them.

**Drops overlay step #1.** Lowest priority because it's not a TRT-LLM
correctness bug, just an environment mismatch.

### (e) Defensive upcast in `LagunaAttention.forward` (optional, defense-in-depth)

Once (a) lands, the FP8 attention output path no longer fires for this
checkpoint and the existing code is correct. But the PyTorch failure
mode is loud and ugly when it does fire (`Promotion for Float8 Types
is not supported`) and the fix is one line. Two-line patch in
`tensorrt_llm/_torch/models/modeling_laguna.py` (around line 455):

```python
if self._use_gating and gate_input is not None:
    gate = self.g_proj(gate_input)
    work_dtype = self.g_proj.weight.dtype
    if attn_output.dtype != work_dtype:
        attn_output = attn_output.to(work_dtype)
    gate = F.softplus(gate.float()).to(work_dtype)
    ...
```

No-op on the BF16 path; protects future FP8 Laguna variants whose
attention output legitimately *is* FP8 (e.g. an end-to-end FP8 export
where `o_proj` is also quantized).

## Verification path once upstream lands

For each upstream fix landed, the corresponding line in
`laguna_compat_overlay.sh` can be removed and we should still get the
same accuracy:

| Upstream fix | Overlay step to delete | Expected ckpt-load behavior |
|---|---|---|
| (a) `re:` exclude patterns | Step #4 (`_re_to_fnmatch` translator) | FP8 GSM8K stays at 87 % strict; without (a) it was 0 % |
| (b) v5 `rope_parameters` | Step #2 (rope flatten) | `RopeParams.from_config` no longer asserts |
| (c) `gating: true` | Step #3 (gating normalize) | Layer-0 weight load no longer shape-mismatches |
| (d) transformers ≥ 4.58 | Step #1 (PreTrainedConfig shim) + `configuration_laguna.py` patch | `AutoConfig.from_pretrained` no longer ImportErrors |
| (e) defensive upcast | n/a (the overlay never patched this) | No-op once (a) lands |

When all of (a)–(d) land upstream, `laguna_compat_overlay.sh` can be
deleted in its entirety and both BF16 and FP8 april25 checkpoints will
load directly from `/poolside/models/april25/` with no preprocessing.

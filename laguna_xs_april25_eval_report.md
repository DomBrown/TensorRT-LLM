# Laguna-xs `april25` checkpoints — TRT-LLM evaluation report

Evaluation of the three updated `laguna-xs` checkpoints in
`/poolside/models/april25/` (BF16 / FP8 / NVFP4) with TensorRT-LLM
`1.3.0rc13` and `transformers==4.57.3`, running GSM8K (5-shot, full 1,319
samples) via `trtllm-eval`.

## TL;DR

| Checkpoint | Status | Headline |
|---|---|---|
| `laguna-xs-bf16` | ✅ Runs (with a small config-only overlay) | GSM8K strict-match **86.96 %** / flexible-extract 85.60 % / avg 86.28 |
| `laguna-xs-fp8` | ✅ Runs (with the same overlay, after a TRT-LLM-side bug worked around in the overlay) | GSM8K strict-match **87.19 %** / flexible-extract 86.13 % / avg 86.66 — essentially BF16-equivalent (FP8 lossless on this benchmark). Root cause was a TRT-LLM bug in `is_module_excluded_from_quantization` (uses `fnmatch`; can't parse the `re:`-prefixed regex patterns that compressed-tensors writes into `quantization_config.ignore`). The overlay translates those patterns to fnmatch globs. |
| `laguna-xs-nvfp4` | ❌ Cannot load — checkpoint export bug | Layer 0 dense MLP weights (`gate_proj`, `up_proj`, `down_proj`) are missing `weight_packed` / `weight_global_scale` / `input_global_scale`; only the per-block `weight_scale` was saved |

**Action items:**
1. **NVFP4 (client):** re-export `laguna-xs-nvfp4` — layer 0 dense MLP is
   incomplete (only `weight_scale` is saved; `weight_packed` /
   `weight_global_scale` / `input_global_scale` are missing).
2. **FP8 (TRT-LLM upstream — recommended):** patch
   `QuantConfig.is_module_excluded_from_quantization` to recognize the
   `re:`-prefixed regex patterns that compressed-tensors writes into
   `quantization_config.ignore`. Without this, `re:`-style patterns in
   the `ignore` list silently match nothing and modules the client
   intended to keep BF16 (here: q/k/v/o/g_proj and the MoE router) get
   FP8-quantized anyway. See "FP8 — root cause and fix" below.
3. **FP8 (client, optional):** until a TRT-LLM upstream fix is available,
   re-export `laguna-xs-fp8` with `quantization_config.ignore` written
   in fnmatch glob form (e.g. `"*self_attn.q_proj"`) instead of regex
   form (`"re:.*\\.self_attn\\.q_proj$"`). The current overlay already
   does this translation automatically, so this is purely a
   nice-to-have for clients who want their checkpoint to load without
   an overlay.

All three of BF16 / FP8 (with overlay) / NVFP4 (after re-export) should
then run cleanly on TRT-LLM with no source-code changes.

## Reproduction (in this container)

Two helper scripts at the workspace root:

- `run_trtllm_eval.sh` — wraps `trtllm-eval` with `--model`/`--task`,
  forces `--backend pytorch --trust_remote_code`, and forwards extra args
  to the subcommand.
- `laguna_compat_overlay.sh` — builds a per-checkpoint overlay directory
  of symlinks to the original checkpoint, replacing only
  `configuration_laguna.py` (compat shim, see below) and `config.json`
  (label/format normalization, see below).

```bash
# Build overlays (one per checkpoint; outputs go under /tmp/)
/code/tensorrt_llm/laguna_compat_overlay.sh /poolside/models/april25/laguna-xs-bf16
/code/tensorrt_llm/laguna_compat_overlay.sh /poolside/models/april25/laguna-xs-fp8
/code/tensorrt_llm/laguna_compat_overlay.sh /poolside/models/april25/laguna-xs-nvfp4

# Run GSM8K (5-shot, full 1,319 samples)
/code/tensorrt_llm/run_trtllm_eval.sh --model /tmp/laguna-april25-laguna-xs-bf16   --task gsm8k
/code/tensorrt_llm/run_trtllm_eval.sh --model /tmp/laguna-april25-laguna-xs-fp8    --task gsm8k
/code/tensorrt_llm/run_trtllm_eval.sh --model /tmp/laguna-april25-laguna-xs-nvfp4  --task gsm8k
```

Hardware used for the BF16 run: 1× NVIDIA RTX PRO 6000 Blackwell Server
Edition (≈ 95 GiB).

## Why the overlay is needed

The overlay is a *non-destructive* workaround that leaves the originals on
`/poolside/models/april25/` untouched. It makes four independent fixes:

### 1. `configuration_laguna.py` imports symbols only present in transformers ≥ 4.58

```python
from transformers.configuration_utils import PreTrainedConfig
from transformers.modeling_rope_utils import RopeParameters
```

Both are PascalCase aliases introduced in transformers ≥ 4.58. The
container ships transformers 4.57.3, where these are `PretrainedConfig`
and "doesn't exist" respectively. The overlay prepends a compat shim:

```python
try:
    from transformers.configuration_utils import PreTrainedConfig
except ImportError:
    from transformers.configuration_utils import PretrainedConfig as PreTrainedConfig
try:
    from transformers.modeling_rope_utils import RopeParameters
except ImportError:
    RopeParameters = dict  # only used in type annotations
```

### 2. `rope_parameters` switched to the v5 nested-by-layer-type format

```jsonc
"rope_parameters": {
  "full_attention":   { "rope_type": "yarn", "rope_theta": 500000.0, "factor": 32.0, ... },
  "sliding_attention":{ "rope_type": "default", "rope_theta": 10000.0, "partial_rotary_factor": 1.0 }
}
```

TRT-LLM's `RopeParams.from_config` explicitly rejects this layout
(`AssertionError: Per-layer-type RoPE configuration is not supported yet.`).
The overlay flattens it to the legacy keys TRT-LLM already understands:

- `full_attention` → top-level `rope_theta` + `rope_scaling` + `partial_rotary_factor`
- `sliding_attention` → top-level `swa_rope_parameters`

### 4. `re:`-prefixed regex patterns in `quantization_config.ignore`

The compressed-tensors export writes `ignore` patterns as
`re:<regex>` strings (e.g. `re:.*\.self_attn\.q_proj$`). TRT-LLM's
`QuantConfig.is_module_excluded_from_quantization` matches these with
`fnmatch.fnmatchcase`, which doesn't know the `re:` prefix. The overlay
translates them to glob patterns (`*self_attn.q_proj`,
`*mlp.gate`, etc.). See "FP8 — root cause and fix" below for the full
chain of consequences when this isn't done — short version: TRT-LLM
silently quantizes modules the client intended to keep BF16, and on
Laguna that breaks GSM8K to 0 % accuracy. This is the highest-impact
fix in the overlay.

### 3. `gating: "per-head"` was renamed to `gating: true`

TRT-LLM's `LagunaAttention` checks `gating == "per-head"` to size the
`g_proj` weight per attention head (`num_heads`-wide). With `gating: true`
the check fails and it falls back to `num_heads × head_dim` (8192 for
sliding layers), which doesn't match the actual checkpoint weights —
they're still per-head (size 48 for full layers, 64 for sliding):

```
RuntimeError: The size of tensor a (8192) must match the size of tensor b (64) at non-singleton dimension 0
```

The overlay rewrites `gating: true` back to `"per-head"`. The weights in
the checkpoint are unchanged; this is purely a label change between the
old and new configs.

### Suggested upstream fix

In `tensorrt_llm/_torch/models/modeling_laguna.py`, replacing

```python
self._gate_per_head = gating == "per-head"
```

with something like

```python
self._gate_per_head = (gating == "per-head") or gating is True
```

would let new-format Laguna configs load without the overlay's `gating`
rewrite. The other two overlay fixes (transformers symbol shim, rope
flattening) are independent of TRT-LLM and would still need the overlay
on transformers < 4.58 — or a TRT-LLM-side update of `RopeParams.from_config`
to accept the v5 per-layer-type rope dict.

## BF16 — full result

Hyperparameters (defaults from `trtllm-eval gsm8k`):

- 5-shot prompting
- `max_input_length = 4096`, `max_output_length = 256`
- pytorch backend, single GPU, `trust_remote_code=True`
- Full GSM8K test split (1,319 problems)

Result:

| Filter | Metric | Value | Stderr |
|---|---|---:|---:|
| flexible-extract | exact_match | 85.60 % | ±0.97 |
| strict-match | exact_match | 86.96 % | ±0.93 |

Average reported by `trtllm-eval`: **86.28** (mean of the two filter
scores; not a standard GSM8K headline metric — prefer **strict-match**).

The strict-match score is bit-identical to a fresh-container re-run
(86.96 % both times), confirming greedy-decoding determinism. The
flexible-extract score drifts within stderr across runs (85.60 vs 85.82).

For context (same eval pipeline, prior `/poolside/models/old/` BF16
checkpoint): strict-match 90.52 %. The april25 BF16 is ~3.6 points lower
on strict-match — outside ±2.6 statistical noise — though substantially
*higher* on the lenient flexible-extract (85.60 vs 63.61 on old),
suggesting output-formatting changes in the new checkpoint (slightly less
consistent emission of the `#### <answer>` terminator) more than a real
reasoning regression.

For sanity-checking the rotation invariance, a BF16 checkpoint built by
dequantizing the FP8 ckpt's weights (preserving the QuaRot rotation and
the LayerNorm-gain absorption baked into them) was also evaluated:
50-sample GSM8K strict-match 88 % / flex 84 %, again within sample
noise of the unrotated BF16. Code: `dequantize_fp8_to_bf16.py` at the
workspace root.

## FP8 — full result

| Filter | Metric | Value | Stderr |
|---|---|---:|---:|
| flexible-extract | exact_match | 86.13 % | ±0.95 |
| strict-match | exact_match | 87.19 % | ±0.92 |

Average reported by `trtllm-eval`: **86.66**. Strict-match is within ±0.5
points of the BF16 reference (86.96 %) — well inside the combined ±1.3
stderr — i.e. **FP8 quantization is essentially lossless on GSM8K once
the upstream bug is worked around**. The fix lives entirely in the
overlay's `config.json` rewrite (translate `re:` patterns to globs); no
TRT-LLM source-code changes were needed.

## FP8 — failure analysis

### Quant scheme

`/poolside/models/april25/laguna-xs-fp8/config.json` declares:

```jsonc
"quantization_config": {
  "quant_method": "compressed-tensors",
  "format": "float-quantized",
  "config_groups": { "group_0": {
    "weights":           { "num_bits": 8, "type": "float", "strategy": "block",
                           "block_structure": [128, 128] },
    "input_activations": { "num_bits": 8, "type": "float", "strategy": "group",
                           "group_size": 128, "dynamic": true }
  } },
  "transform_config": { "config_groups": { "R1": {
    "type": "hadamard",
    "head_dim": 128,
    "apply": [
      { "location": "weight_output", "targets": [".*embed_tokens$", ".*o_proj$", ".*down_proj$"] },
      { "location": "weight_input",  "inverse": true,
        "targets": [".*q_proj$", ".*k_proj$", ".*v_proj$",
                    ".*gate_proj$", ".*up_proj$", ".*mlp.gate$",
                    ".*g_proj$", ".*lm_head$"] }
    ]
  } } },
  "kv_cache_scheme": { "num_bits": 8, "type": "float", "strategy": "tensor" },
  "ignore": ["lm_head", ".*self_attn.[qkvog]_proj$", ".*mlp.gate$"]
}
```

### What the rotation actually is

This is a standard QuaRot/SpinQuant transform with a deterministic
**block-128 Walsh–Hadamard rotation** along the 2048-d residual stream
(=`16 × 128` block-diagonal). Every `R[i*128:(i+1)*128, i*128:(i+1)*128]`
block has entries `±1/√128 ≈ ±0.0884` — confirmed empirically:

```
R[0:128, 0:128] unique rounded values: [-0.088, +0.088]
R[0:128, 0:128] absmax = 0.0885,  std = 0.0883
||block @ block.T - I|| / √128 = 0.011   (orthogonal)
on-diagonal-block fraction of total ||R||² = 99.95 %
```

The rotation is **fully baked into the stored weights**. With `R` recovered
by least-squares from `embed_bf16 @ R = embed_fp8`, every other rotated
weight in the FP8 checkpoint matches the predicted rotated form to within
the lstsq numerical noise (cosine similarity ≈ 0.9998):

| Module | Predicted form | rel. err vs FP8 ckpt | cos-sim |
|---|---|---:|---:|
| `embed_tokens` | `embed_bf16 @ R` | 0.0023 | 1.0000 |
| `q_proj.weight` (layer 0) | `(W_bf16 ⊙ ln_input) @ R` | 0.0218 | 0.99976 |
| `k_proj.weight` | `(W_bf16 ⊙ ln_input) @ R` | 0.0221 | 0.99976 |
| `v_proj.weight` | `(W_bf16 ⊙ ln_input) @ R` | 0.0222 | 0.99975 |
| `g_proj.weight` | `(W_bf16 ⊙ ln_input) @ R` | 0.0243 | 0.99970 |
| `mlp.gate_proj.weight` (dequant) | `(W_bf16 ⊙ ln_post) @ R` | 0.0342 | — |
| `o_proj.weight` | `R^T @ W_bf16` | 0.0221 | — |

(`⊙ ln_input` = elementwise scaling by the corresponding `*_layernorm.weight`,
i.e. LayerNorm gain absorption into the next linear.)

In other words: the FP8 checkpoint has both
1. **LayerNorm gains absorbed** — `input_layernorm.weight` and
   `post_attention_layernorm.weight` and the final `model.norm.weight`
   are exactly `1.0` everywhere (vs. trained values in the BF16 ckpt
   ranging 0–0.4).
2. **Block-128 Hadamard rotation** baked into `embed_tokens`, `lm_head`,
   `o_proj`, `down_proj` (output side) and `q/k/v/gate/up/mlp.gate/g/lm_head`
   (input side).

Mathematically this is supposed to be a runtime no-op: the residual lives
in the rotated basis, every read from the residual sees `R^T R = I` cancel
out, every write to the residual lands in the same rotated basis. The
model *should* produce identical logits to an unrotated BF16 model.

### Root cause and fix

The QuaRot rotation is, as expected, a runtime no-op when TRT-LLM's
forward path is correct: the BF16-with-rotation-baked-in checkpoint
(produced by `dequantize_fp8_to_bf16.py`, which dequantizes every FP8
weight in the FP8 checkpoint to BF16 and strips `quantization_config`)
runs at GSM8K strict-match 88 % / flex 84 % on a 50-sample smoke test —
matching the unrotated BF16 ckpt within sample noise (90 % / 88 %). So
the rotation math itself is fine and TRT-LLM preserves it.

The bug is upstream of all the rotation math: the compressed-tensors
quant config's `ignore` list is **silently dropped on the floor by
TRT-LLM**.

Compressed-tensors writes `quantization_config.ignore` as
*regex* patterns prefixed with `re:`, e.g.

```jsonc
"ignore": [
  "lm_head",
  "re:.*\\.self_attn\\.q_proj$",
  "re:.*\\.self_attn\\.o_proj$",
  "re:.*\\.self_attn\\.g_proj$",
  ...
]
```

TRT-LLM's `QuantConfig.is_module_excluded_from_quantization`
(`tensorrt_llm/models/modeling_utils.py`) implements the check with
`fnmatch.fnmatchcase(name, pattern)`. fnmatch doesn't understand the
`re:` prefix or regex meta-characters — the literal string
`re:.*\.self_attn\.q_proj$` matches no module name. Result: **none of
the `re:` exclusions take effect**, and Laguna's `q/k/v/o/g_proj` and
`mlp.gate` (the MoE router) get FP8-quantized despite being listed in
`ignore`. That cascades into the symptom we saw:

1. `o_proj.has_fp8_block_scales == True` (because the global quant config
   says FP8_BLOCK_SCALES and `o_proj` wasn't excluded from it).
2. `Attention._use_quantize_output()` returns True.
3. The TRTLLM attention backend allocates the attention output buffer as
   `torch.float8_e4m3fn` and quantizes the output using
   `o_proj.inv_input_scale` as the FP8 output scale.
4. `o_proj` is *also* loaded with BF16 weights from the checkpoint
   (because the checkpoint stores them BF16), so `o_proj.input_scale`
   is left at the default `1.0` and never calibrated.
5. The FP8 quantization clamps to `[−448, 448]` divided by
   `inv_input_scale = 1.0 / 1.0 = 1.0`, which is fine for typical
   attention output magnitudes (~0.3 in the BF16 reference). But Laguna's
   per-head `g_proj * attn_output` multiply happens *before* `o_proj`,
   on raw FP8 (no dequant). PyTorch refuses to promote FP8↔BF16
   (`mul_cuda not implemented for Float8_e4m3fn`).
6. Even if the FP8 mul is patched (we tried; the patch upcasts
   `attn_output` to BF16 before the gate), the cast is bit-for-bit
   correct but the attention path is now passing FP8-quantized
   activations through a chain of Linears that don't expect quantized
   inputs (because they were silently turned into FP8 modules with
   default `input_scale = 1.0`). The diagnostic instrumentation in
   `LagunaDecoderLayer.forward` shows **every layer's `post_attn`
   output equals exactly zero** — i.e. the attention block is producing
   garbage end-to-end.

The instrumentation that pinpointed the failure (per-layer activation
absmax via the `LAGUNA_DIAG=1 LAGUNA_DIAG_ALL=1` env vars):

```text
=== BF16 (works) ===                      === FP8 (broken) ===
L0 post_attn: absmax=3.36e-01             L0 post_attn: absmax=0.00e+00
L1 post_attn: absmax=3.28e-01             L1 post_attn: absmax=0.00e+00
L2 post_attn: absmax=1.87e-01             L2 post_attn: absmax=0.00e+00
…                                         …
```

### Two fixes (the overlay applies #1; upstream should also do #2)

**Fix 1 (overlay, applied here)**: in
`laguna_compat_overlay.sh`, translate every `re:`-prefixed
regex pattern in `quantization_config.ignore` (and `targets` lists in
`config_groups` / `transform_config`) to a glob pattern that fnmatch
understands. Conservative translation:

```text
re:.*\.self_attn\.q_proj$  →  *self_attn.q_proj
re:.*\.mlp\.gate$          →  *mlp.gate
```

With this translation in the overlay's `config.json`,
`is_module_excluded_from_quantization` correctly excludes the attention
projections and MoE router, those modules get
`UnquantizedLinearMethod` (BF16), `_use_quantize_output()` returns
False, the attention output stays BF16, and **GSM8K runs cleanly with
no source-code changes**. Result above: strict-match 87.19 %.

**Fix 2 (TRT-LLM upstream — recommended)**: teach
`QuantConfig.is_module_excluded_from_quantization` (and the analogous
`config_groups.targets` matching) to recognize the `re:` prefix and
dispatch to `re.fullmatch` instead of `fnmatch`:

```python
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

This is also what `compressed-tensors`-on-vLLM does and what every
client exporting via `compressed-tensors` will expect. Without it,
*every* compressed-tensors checkpoint that uses regex `ignore`
patterns silently quantizes modules the user intended to keep
unquantized — Laguna is just a particularly visible casualty because
its per-head attention gate exposes the FP8 dtype mismatch as a hard
error.

### BF16-with-rotation-baked-in cross-check

To verify the rotation invariance experimentally, the
`dequantize_fp8_to_bf16.py` helper at the workspace root rewrites the
FP8 checkpoint into a BF16 checkpoint that preserves the rotation
(`weight_bf16 = weight_fp8.to(fp32) * weight_scale_full`, then the
quantization configs are stripped). Result:

- April25 FP8 → BF16-rotated, GSM8K (50 samples): strict-match 88 % /
  flex 84 %. Within sample noise of the unrotated BF16 ckpt (90 % /
  88 %).

This was the experiment that proved the rotation was correctly applied
in the weights *and* preserved by TRT-LLM's BF16 forward path,
isolating the regression to the FP8 forward path — and ultimately
pointing to the `re:` exclude-pattern bug.

## NVFP4 — failure analysis

`/poolside/models/april25/laguna-xs-nvfp4/config.json` declares:

```jsonc
"quantization_config": {
  "quant_method": "compressed-tensors",
  "format": "nvfp4-pack-quantized",
  "config_groups": { "group_0": {
    "weights":           { "num_bits": 4, "type": "float",
                           "strategy": "tensor_group", "group_size": 16 },
    "input_activations": { "num_bits": 4, "type": "float",
                           "strategy": "tensor_group", "group_size": 16 }
  } },
  "ignore": ["lm_head",
             ".*self_attn.q_proj$", ".*self_attn.k_proj$",
             ".*self_attn.v_proj$", ".*self_attn.o_proj$",
             ".*self_attn.g_proj$", ".*mlp.gate$"]
  // NOTE: NO transform_config — clean NVFP4
}
```

This is a clean compressed-tensors NVFP4 (no Hadamard transform) and
maps cleanly to TRT-LLM's `QuantAlgo.NVFP4`. It should "just work".

It doesn't, because the **checkpoint itself is incomplete**:

```
=== ALL `model.layers.0.mlp.*` keys in the safetensors index ===
  model.layers.0.mlp.down_proj.weight_scale
  model.layers.0.mlp.gate_proj.weight_scale
  model.layers.0.mlp.up_proj.weight_scale
```

For comparison, layer 1's MoE expert 0 has the full healthy NVFP4 set:

```
model.layers.1.mlp.experts.0.gate_proj.input_global_scale
model.layers.1.mlp.experts.0.gate_proj.weight_global_scale
model.layers.1.mlp.experts.0.gate_proj.weight_packed
model.layers.1.mlp.experts.0.gate_proj.weight_scale
```

Layer 0's dense MLP is missing **`weight_packed`**, **`weight_global_scale`**,
and **`input_global_scale`** — only the per-block `weight_scale` was
exported. Without `weight_packed` (the actual U8-packed FP4 weights) or
a fallback BF16 `weight`, the linear layer can't be initialized.

The `ignore` list does **not** exclude these modules — so they're
expected to be NVFP4-quantized, not BF16. This isn't a TRT-LLM
compatibility gap; the checkpoint is malformed and would fail in any
inference engine.

The actual error surfaces as an `AssertionError` in
`load_weights_fused_gate_up_helper`:

```
File ".../tensorrt_llm/_torch/modules/linear.py", line 272,
    in load_weights_fused_gate_up_helper
    assert all('weight' in weights[i] for i in range(2))
```

…because TRT-LLM tries to fuse `gate_proj`/`up_proj` and finds neither a
plain `weight` (BF16) nor a `weight_packed` (NVFP4) tensor for them.

Only 3 modules in the entire checkpoint are affected — all three are
inside layer 0:

```
layer  0  mlp.down_proj : ['weight_scale']
layer  0  mlp.gate_proj : ['weight_scale']
layer  0  mlp.up_proj   : ['weight_scale']
```

**Action:** ask the client to re-export `laguna-xs-nvfp4`, ensuring
layer 0's dense MLP weights are included.

## Files

- `run_trtllm_eval.sh`, `laguna_compat_overlay.sh` — at workspace root
- BF16 run log: `/tmp/laguna_xs_bf16_apr25_gsm8k.log`
- FP8 run log: `/tmp/laguna_xs_fp8_apr25_gsm8k.log`
- NVFP4 run log: `/tmp/laguna_xs_nvfp4_apr25_gsm8k.log`
- Overlay directories: `/tmp/laguna-april25-laguna-xs-{bf16,fp8,nvfp4}/`

## Environment

- TensorRT-LLM `1.3.0rc13`
- `transformers==4.57.3`
- 1× NVIDIA RTX PRO 6000 Blackwell Server Edition
- Ubuntu / Python 3.12

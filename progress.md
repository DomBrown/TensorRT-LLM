# Laguna-XS TRT-LLM Progress

## Build

Fixed the build command: `--ccache` is not a valid flag, the correct one is `--use_ccache`.
Also added `--configure_cmake` because the build directory existed but had no Makefile.

Working build command:
```bash
python ./scripts/build_wheel.py \
  --trt_root /usr/local/tensorrt \
  -a native -s -i -b RelWithDebInfo \
  --nvtx --use_ccache --configure_cmake
```

Build completed and installed into `.venv-3.12`.

## Model

- HF checkpoint: `/home/scratch.dcampora_gpu/laguna-xs-hf`
- Out-of-tree TRT-LLM implementation: `/home/scratch.dcampora_gpu/projects/poolside/dan_bench_serving/modeling_laguna_trtllm.py`
- The checkpoint has `quantization_config` / `compression_config` fields in `config.json` that TRT-LLM doesn't understand. Both test scripts strip these via a shadow dir (symlinks + cleaned config).

## Code review of modeling_laguna_trtllm.py

All imports verified to exist in the current codebase. Architecture is correctly implemented. Key findings:

- **`fuse_qk_norm_rope=False`** is correctly hardcoded: global and sliding layers have different RoPE configs (yarn/partial=0.5 vs linear/full), so the fused kernel — which reads these globally — cannot be used.
- **`LagunaGate.load_weights` drops `e_score_correction_bias`**: the weight mapper renames `mlp.experts.e_score_correction_bias` → `mlp.gate.e_score_correction_bias`, but `load_weights` only copies `weight`. Fix:
  ```python
  def load_weights(self, weights, allow_partial_loading=False):
      assert len(weights) == 1
      w = weights[0].get("weight")
      if w is not None:
          self.weight.copy_(w[:])
      b = weights[0].get("e_score_correction_bias")
      if b is not None:
          self.e_score_correction_bias.copy_(b[:])
  ```
  **No current impact** — all 256 values of `e_score_correction_bias` are exactly 0.0 in this checkpoint. Will silently break once a checkpoint with trained bias values is used.
- **Expert weight loading** (VANILLA mode): the mapper renames `gate_proj`→`w1`, `up_proj`→`w3`, `down_proj`→`w2` in `handle_special_instance_module`, which matches what `FusedMoEVanilla.load_weights` expects (`{expert_idx}.w1`, etc.). Correct.
- **`LagunaModel.forward`** correctly overrides `DecoderModel.forward` to handle the `(hidden_states, residual)` 2-tuple its layers return.
- **`forward_impl` positional args** in `LagunaAttention.forward` are in the right order.

## Test scripts

Written to `/code/tensorrt_llm/` (the NFS mount for the scratch dir is `ro`):

- `run_laguna_smoke_test.py` — 4 prompts, reports tok/s, exits 0/1
- `run_laguna_gsm8k.py` — 8-shot CoT on all 1319 GSM8K test examples, extracts numerical answer, reports accuracy; accepts `--limit N` and `--output path.jsonl`

## TODO

- [ ] Run smoke test: `cd /code/tensorrt_llm && .venv-3.12/bin/python run_laguna_smoke_test.py`
- [ ] Run GSM8K: `.venv-3.12/bin/python run_laguna_gsm8k.py --output /tmp/gsm8k_results.jsonl`
- [ ] Fix `LagunaGate.load_weights` to also load `e_score_correction_bias` (low priority until a checkpoint with non-zero bias is trained)

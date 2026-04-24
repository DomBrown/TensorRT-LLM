# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Laguna / Laguna-XS model for TensorRT-LLM PyTorch backend."""

import os
from typing import Dict, List, Optional, Type

import torch
import torch.nn.functional as F
from torch import nn
from transformers import PretrainedConfig

from tensorrt_llm.functional import PositionEmbeddingType, RotaryScalingType

from ..attention_backend import AttentionMetadata
from ..attention_backend.interface import (
    PositionalEmbeddingParams,
    PredefinedAttentionMask,
    RopeParams,
)
from ..distributed import AllReduce, AllReduceParams
from ..modules.attention import _helix_cp_allgather_input, _helix_cp_output_projection
from ..modules.decoder_layer import DecoderLayer
from ..modules.embedding import Embedding
from ..modules.fused_moe import MiniMaxM2MoeRoutingMethod, create_moe, get_moe_cls
from ..modules.fused_moe.interface import MoE, MoEWeightLoadingMode
from ..modules.fused_moe.interface import MoE as MoEInterface
from ..modules.gated_mlp import GatedMLP
from ..modules.linear import Linear, TensorParallelMode
from ..modules.qk_norm_attention import QKNormRoPEAttention
from ..modules.rms_norm import RMSNorm
from ..speculative import SpecMetadata
from ..utils import AuxStreamType
from .checkpoints.hf.weight_mapper import HfWeightMapper
from .modeling_speculative import SpecDecOneEngineForCausalLM
from .modeling_utils import DecoderModel, register_auto_model, register_mapper


def _diag_enabled():
    return os.environ.get("LAGUNA_DIAG", "") == "1"


def _diag_all_enabled():
    """Per-layer compact summary (1 line per tensor) for all 40 layers."""
    return os.environ.get("LAGUNA_DIAG_ALL", "") == "1"


def _diag_tensor(label: str, t: torch.Tensor, rank: int = 0):
    """Print tensor statistics for debugging."""
    if not _diag_enabled():
        return
    with torch.no_grad():
        ft = t.float()
        print(
            f"[DIAG rank={rank}] {label}: "
            f"shape={tuple(t.shape)} dtype={t.dtype} "
            f"mean={ft.mean().item():.6e} std={ft.std().item():.6e} "
            f"min={ft.min().item():.6e} max={ft.max().item():.6e} "
            f"absmax={ft.abs().max().item():.6e}",
            flush=True,
        )


def _diag_check_nan(label: str, t: torch.Tensor, rank: int = 0):
    """Emit a loud marker the first time a NaN/Inf appears in a tensor.

    Runs whenever LAGUNA_DIAG=1 regardless of which layer, so we can find the
    earliest point a numerical blow-up happens. Cheap: two reductions on the
    flattened tensor, no device sync beyond the reduction itself.
    """
    if not _diag_enabled():
        return
    with torch.no_grad():
        has_nan = bool(torch.isnan(t).any().item())
        has_inf = bool(torch.isinf(t).any().item())
        if has_nan or has_inf:
            ft = t.float()
            finite_mask = torch.isfinite(ft)
            finite_absmax = (
                ft[finite_mask].abs().max().item() if finite_mask.any() else float("nan")
            )
            print(
                f"[DIAG-NAN rank={rank}] {label}: "
                f"shape={tuple(t.shape)} dtype={t.dtype} "
                f"has_nan={has_nan} has_inf={has_inf} "
                f"finite_absmax={finite_absmax:.6e}",
                flush=True,
            )


def _diag_compact(label: str, t: torch.Tensor, rank: int = 0):
    """One-line absmax + NaN/Inf flag for per-layer scanning (LAGUNA_DIAG_ALL=1)."""
    if not (_diag_enabled() and _diag_all_enabled()):
        return
    with torch.no_grad():
        has_nan = bool(torch.isnan(t).any().item())
        has_inf = bool(torch.isinf(t).any().item())
        if has_nan or has_inf:
            ft = t.float()
            finite_mask = torch.isfinite(ft)
            if finite_mask.any():
                absmax = ft[finite_mask].abs().max().item()
            else:
                absmax = float("nan")
            tag = "NaN" if has_nan else "Inf"
            print(
                f"[DIAG-ALL rank={rank}] {label}: {tag}! finite_absmax={absmax:.4e}",
                flush=True,
            )
        else:
            print(
                f"[DIAG-ALL rank={rank}] {label}: absmax={t.float().abs().max().item():.4e}",
                flush=True,
            )


# ---------------------------------------------------------------------------
#  Gate (MoE router)
# ---------------------------------------------------------------------------


class LagunaGate(nn.Module):
    """Sigmoid router with e_score_correction_bias (MiniMaxM2 routing)."""

    def __init__(
        self,
        hidden_size: int,
        num_experts: int,
        top_k: int,
        dtype: Optional[torch.dtype] = None,
        moe_backend_cls: Type[MoE] = None,
    ):
        super().__init__()
        self.top_k = top_k
        self.moe_backend_cls = moe_backend_cls
        self.weight = nn.Parameter(
            torch.empty((num_experts, hidden_size), dtype=dtype),
            requires_grad=False,
        )
        self.e_score_correction_bias = nn.Parameter(
            torch.zeros(num_experts, dtype=torch.float32),
            requires_grad=False,
        )
        self.out_dtype = dtype

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        logits = torch.ops.trtllm.cublas_mm(
            hidden_states, self.weight.t(), bias=None, out_dtype=self.out_dtype
        )
        return logits

    def load_weights(self, weights: List[Dict], allow_partial_loading: bool = False):
        assert len(weights) == 1
        w = weights[0].get("weight")
        if w is not None:
            self.weight.copy_(w[:])
        b = weights[0].get("e_score_correction_bias")
        if b is not None:
            self.e_score_correction_bias.copy_(b[:])

    @property
    def routing_method(self):
        return MiniMaxM2MoeRoutingMethod(
            top_k=self.top_k,
            num_experts=self.weight.shape[0],
            callable_e_score_correction_bias=lambda: self.e_score_correction_bias,
            output_dtype=torch.float32,
        )


# ---------------------------------------------------------------------------
#  MoE
# ---------------------------------------------------------------------------


class LagunaMoE(nn.Module):
    def __init__(self, model_config, layer_idx, aux_stream_dict):
        super().__init__()
        config = model_config.pretrained_config
        self.hidden_dim = config.hidden_size
        self.num_experts = config.num_experts
        self.top_k = config.num_experts_per_tok
        self.mapping = model_config.mapping

        self.allreduce = None
        if self.mapping.tp_size > 1:
            self.allreduce = AllReduce(
                mapping=model_config.mapping,
                strategy=model_config.allreduce_strategy,
            )

        self.gate = LagunaGate(
            hidden_size=self.hidden_dim,
            num_experts=self.num_experts,
            top_k=self.top_k,
            dtype=config.torch_dtype,
            moe_backend_cls=get_moe_cls(model_config),
        )

        self.experts = create_moe(
            num_experts=self.num_experts,
            routing_method=self.gate.routing_method,
            hidden_size=self.hidden_dim,
            intermediate_size=config.moe_intermediate_size,
            aux_stream_dict=aux_stream_dict,
            dtype=config.torch_dtype,
            reduce_results=False,
            model_config=model_config,
            layer_idx=layer_idx,
            weight_loading_mode=MoEWeightLoadingMode.VANILLA,
        )

        shared_size = getattr(config, "shared_expert_intermediate_size", 0)
        if shared_size > 0:
            # In DEP mode each rank processes different tokens, so TP-sharding the
            # shared expert across ranks would produce partial outputs that cannot
            # be combined with AllReduce (ranks hold different tokens).  Use
            # overridden_tp_size=1 so every rank holds full weights and computes
            # the shared expert independently.
            shared_expert_tp_size = 1 if self.mapping.enable_attention_dp else None
            self.shared_expert = GatedMLP(
                hidden_size=self.hidden_dim,
                intermediate_size=shared_size,
                bias=False,
                dtype=config.torch_dtype,
                config=model_config,
                overridden_tp_size=shared_expert_tp_size,
                is_shared_expert=True,
                reduce_output=False,
            )
        else:
            self.shared_expert = None

        self.routed_scaling_factor = float(getattr(config, "moe_routed_scaling_factor", 1.0))

    def forward(
        self,
        hidden_states: torch.Tensor,
        attn_metadata: AttentionMetadata,
        all_reduce_params: Optional[AllReduceParams] = None,
        **kwargs,
    ) -> torch.Tensor:
        orig_shape = hidden_states.shape
        hidden_states = hidden_states.view(-1, self.hidden_dim)
        all_rank_num_tokens = attn_metadata.all_rank_num_tokens
        rank = self.mapping.rank if hasattr(self.mapping, "rank") else 0
        do_diag = (
            _diag_enabled()
            and hasattr(self, "_layer_idx_for_diag")
            and self._layer_idx_for_diag < 3
        )

        router_logits = self.gate(hidden_states)

        if do_diag:
            _diag_tensor(f"MoE_L{self._layer_idx_for_diag} router_logits", router_logits, rank)

        routed_output = self.experts(
            hidden_states,
            router_logits,
            all_rank_num_tokens=all_rank_num_tokens,
        )

        if do_diag:
            _diag_tensor(f"MoE_L{self._layer_idx_for_diag} routed_pre_scale", routed_output, rank)

        if self.routed_scaling_factor != 1.0:
            routed_output = routed_output * self.routed_scaling_factor

        if self.shared_expert is not None:
            shared_output = self.shared_expert(hidden_states)
            if do_diag:
                _diag_tensor(f"MoE_L{self._layer_idx_for_diag} shared_output", shared_output, rank)
            final_output = routed_output + shared_output
        else:
            final_output = routed_output

        if do_diag:
            _diag_tensor(f"MoE_L{self._layer_idx_for_diag} pre_allreduce", final_output, rank)

        # In DEP mode (enable_attention_dp) AllGatherReduceScatter already
        # produced complete per-rank outputs via ReduceScatter; an AllReduce
        # here would be semantically wrong (ranks hold different tokens) and
        # causes a NCCL deadlock.  Skip it in DEP mode.
        if (
            self.mapping.tp_size > 1
            and self.allreduce is not None
            and not self.mapping.enable_attention_dp
        ):
            final_output = self.allreduce(final_output, all_reduce_params=all_reduce_params)

        if do_diag:
            _diag_tensor(f"MoE_L{self._layer_idx_for_diag} post_allreduce", final_output, rank)

        return final_output.view(orig_shape)


# ---------------------------------------------------------------------------
#  Attention
# ---------------------------------------------------------------------------


class LagunaAttention(QKNormRoPEAttention):
    """Laguna attention with per-head softplus gating, per-layer heads,
    dual RoPE, and sliding window."""

    def __init__(
        self,
        model_config,
        layer_idx: Optional[int] = None,
        reduce_output: bool = True,
    ):
        config = model_config.pretrained_config

        per_layer_heads = getattr(config, "num_attention_heads_per_layer", None)
        num_heads = (
            per_layer_heads[layer_idx]
            if per_layer_heads is not None
            else config.num_attention_heads
        )

        layer_types = getattr(config, "layer_types", None)
        is_sliding = layer_types is not None and layer_types[layer_idx] == "sliding_attention"

        rope_params = self._build_rope_params(config, is_sliding)

        self.attention_window_size = config.sliding_window if is_sliding else None

        pos_embd_params = PositionalEmbeddingParams(
            type=PositionEmbeddingType.rope_gpt_neox,
            rope=rope_params,
        )

        gating = getattr(config, "gating", False)
        self._use_gating = bool(gating)
        self._gate_per_head = gating == "per-head"
        self._num_heads_local = None  # set after super().__init__

        # fuse_qk_norm_rope=False is required: the fused kernel reads
        # partial_rotary_factor and yarn params from pretrained_config
        # globally, ignoring per-layer RopeParams. Laguna has different
        # RoPE per layer type (global=yarn/partial=0.5, SWA=linear/full).
        super().__init__(
            hidden_size=config.hidden_size,
            num_attention_heads=num_heads,
            num_key_value_heads=config.num_key_value_heads,
            max_position_embeddings=config.max_position_embeddings,
            bias=getattr(config, "qkv_bias", False) or getattr(config, "attention_bias", False),
            pos_embd_params=pos_embd_params,
            fuse_qk_norm_rope=False,
            layer_idx=layer_idx,
            dtype=config.torch_dtype,
            dense_bias=False,
            config=model_config,
            reduce_output=reduce_output,
        )

        self._num_heads_local = self.num_heads

        if self._use_gating:
            g_out = num_heads if self._gate_per_head else num_heads * self.head_dim
            # In DEP mode (enable_attention_dp=True) the attention base class uses
            # a local mapping with tp_size=1, so self.num_heads == num_heads (full).
            # g_proj must match: no COLUMN sharding, each GPU holds the full weight.
            g_tp_mode = (
                None if model_config.mapping.enable_attention_dp else TensorParallelMode.COLUMN
            )
            self.g_proj = Linear(
                config.hidden_size,
                g_out,
                bias=False,
                dtype=config.torch_dtype,
                mapping=model_config.mapping,
                tensor_parallel_mode=g_tp_mode,
            )

    @staticmethod
    def _build_rope_params(config, is_sliding: bool) -> RopeParams:
        if is_sliding:
            swa_rp = getattr(config, "swa_rope_parameters", None)
            if swa_rp is not None:
                rp = RopeParams()
                rp.theta = float(swa_rp.get("rope_theta", 10000.0))
                rp.dim = int(config.head_dim * float(swa_rp.get("partial_rotary_factor", 1.0)))
                rp.max_positions = config.max_position_embeddings
                rt = swa_rp.get("rope_type", "linear")
                if rt == "linear":
                    rp.scale_type = RotaryScalingType.linear
                    rp.scale = float(swa_rp.get("factor", 1.0))
                else:
                    rp.scale_type = RotaryScalingType.none
                return rp
            return RopeParams.from_config(config)
        else:
            rp = RopeParams.from_config(config)
            prf = getattr(config, "partial_rotary_factor", None)
            if prf is not None:
                rp.dim = int(config.head_dim * prf)
            return rp

    def forward(
        self,
        position_ids,
        hidden_states,
        attn_metadata,
        attention_mask=PredefinedAttentionMask.CAUSAL,
        all_reduce_params=None,
        lora_params=None,
        attention_window_size=None,
        attention_mask_data=None,
        **kwargs,
    ):
        # Override the full Attention.forward so we can insert the g_proj
        # gate between the attention kernel output and o_proj (the correct
        # position per the HF reference implementation).
        gate_input = hidden_states if self._use_gating else None

        hidden_states = _helix_cp_allgather_input(
            hidden_states, attn_metadata, self.mapping, self.layer_idx
        )

        qkv = self.qkv_proj(hidden_states)
        q, k, v = qkv, None, None

        # QK-norm + RoPE (inherited from QKNormRoPEAttention.apply_rope)
        q, k, v = self.apply_rope(q, k, v, position_ids)
        q, k, v = self.convert_qkv(q, k, v)

        window = attention_window_size or self.attention_window_size
        attn_output = self.forward_impl(
            q,
            k,
            v,
            attn_metadata,
            attention_mask,
            window,
            attention_mask_data,
            mrope_config=None,
            has_lora=bool(lora_params),
        )

        # Per-head softplus gating (pre-o_proj, matching HF reference)
        if self._use_gating and gate_input is not None:
            gate = self.g_proj(gate_input)
            gate = F.softplus(gate.float()).to(attn_output.dtype)
            if self._gate_per_head:
                shape = attn_output.shape
                attn_output = (
                    attn_output.view(*shape[:-1], self._num_heads_local, self.head_dim)
                    * gate.unsqueeze(-1)
                ).view(shape)
            else:
                attn_output = attn_output * gate

        attn_output = _helix_cp_output_projection(
            self.o_proj,
            attn_output,
            attn_metadata,
            all_reduce_params,
            self.mapping,
            self.mapping_o,
            self.layer_idx,
            lora_params,
        )
        return attn_output


# ---------------------------------------------------------------------------
#  Decoder layer
# ---------------------------------------------------------------------------


class LagunaDecoderLayer(DecoderLayer):
    def __init__(self, model_config, layer_idx: int, aux_stream_dict):
        super().__init__()
        config = model_config.pretrained_config
        self.layer_idx = layer_idx
        self.mapping = model_config.mapping

        self.self_attn = LagunaAttention(
            model_config,
            layer_idx=layer_idx,
            reduce_output=self.mapping.tp_size > 1,
        )

        mlp_only_layers = getattr(config, "mlp_only_layers", []) or []
        if (layer_idx not in mlp_only_layers) and (
            config.num_experts > 0 and (layer_idx + 1) % config.decoder_sparse_step == 0
        ):
            self.mlp = LagunaMoE(model_config, layer_idx, aux_stream_dict)
            self.mlp._layer_idx_for_diag = layer_idx
        else:
            # In DEP mode each rank holds different tokens; TP-sharding the
            # dense MLP would require an AllReduce that deadlocks across ranks
            # (ranks are at different positions in their token streams).
            # Use overridden_tp_size=1 so every rank computes full MLP locally.
            dense_mlp_tp_size = 1 if model_config.mapping.enable_attention_dp else None
            self.mlp = GatedMLP(
                hidden_size=config.hidden_size,
                intermediate_size=config.intermediate_size,
                bias=False,
                dtype=config.torch_dtype,
                config=model_config,
                overridden_tp_size=dense_mlp_tp_size,
            )

        self.input_layernorm = RMSNorm(
            hidden_size=config.hidden_size,
            eps=config.rms_norm_eps,
            dtype=config.torch_dtype,
        )
        self.post_attention_layernorm = RMSNorm(
            hidden_size=config.hidden_size,
            eps=config.rms_norm_eps,
            dtype=config.torch_dtype,
        )

    def forward(
        self,
        position_ids: torch.IntTensor,
        hidden_states: torch.Tensor,
        attn_metadata: AttentionMetadata,
        residual: Optional[torch.Tensor] = None,
        spec_metadata: Optional[SpecMetadata] = None,
        **kwargs,
    ):
        rank = self.mapping.rank if hasattr(self.mapping, "rank") else 0
        do_diag = _diag_enabled() and self.layer_idx < 3

        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)

        if do_diag:
            _diag_tensor(f"L{self.layer_idx} post_ln_input", hidden_states, rank)
        _diag_check_nan(f"L{self.layer_idx} post_ln_input", hidden_states, rank)
        _diag_compact(f"L{self.layer_idx} post_ln_input", hidden_states, rank)

        hidden_states = self.self_attn(
            position_ids=position_ids,
            hidden_states=hidden_states,
            attn_metadata=attn_metadata,
            **kwargs,
        )

        if do_diag:
            _diag_tensor(f"L{self.layer_idx} post_attn", hidden_states, rank)
        _diag_check_nan(f"L{self.layer_idx} post_attn", hidden_states, rank)
        _diag_compact(f"L{self.layer_idx} post_attn", hidden_states, rank)

        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)

        if do_diag:
            _diag_tensor(f"L{self.layer_idx} post_ln_mlp", hidden_states, rank)
        _diag_check_nan(f"L{self.layer_idx} post_ln_mlp", hidden_states, rank)
        _diag_compact(f"L{self.layer_idx} post_ln_mlp", hidden_states, rank)

        if isinstance(self.mlp, LagunaMoE):
            hidden_states = self.mlp(hidden_states, attn_metadata)
        else:
            hidden_states = self.mlp(hidden_states)

        if do_diag:
            _diag_tensor(f"L{self.layer_idx} post_mlp", hidden_states, rank)
            _diag_tensor(f"L{self.layer_idx} residual", residual, rank)
        _diag_check_nan(f"L{self.layer_idx} post_mlp", hidden_states, rank)
        _diag_check_nan(f"L{self.layer_idx} residual", residual, rank)
        _diag_compact(f"L{self.layer_idx} post_mlp", hidden_states, rank)
        _diag_compact(f"L{self.layer_idx} residual", residual, rank)

        if spec_metadata is not None:
            spec_metadata.maybe_capture_hidden_states(self.layer_idx, hidden_states, residual)

        return hidden_states, residual


# ---------------------------------------------------------------------------
#  Model + CausalLM
# ---------------------------------------------------------------------------


class LagunaModel(DecoderModel):
    def __init__(self, model_config):
        super().__init__(model_config)
        config = model_config.pretrained_config

        self.aux_stream_dict = {
            AuxStreamType.MoeChunkingOverlap: torch.cuda.Stream(),
            AuxStreamType.MoeBalancer: torch.cuda.Stream(),
            AuxStreamType.MoeOutputMemset: torch.cuda.Stream(),
        }

        self.embed_tokens = Embedding(
            config.vocab_size,
            config.hidden_size,
            dtype=config.torch_dtype,
            mapping=model_config.mapping,
            tensor_parallel_mode=TensorParallelMode.COLUMN,
            gather_output=True,
        )

        self.layers = nn.ModuleList(
            [
                LagunaDecoderLayer(model_config, layer_idx, self.aux_stream_dict)
                for layer_idx in range(config.num_hidden_layers)
            ]
        )

        self.norm = RMSNorm(
            hidden_size=config.hidden_size,
            eps=config.rms_norm_eps,
            dtype=config.torch_dtype,
        )

    def forward(
        self,
        attn_metadata: AttentionMetadata,
        input_ids: Optional[torch.IntTensor] = None,
        position_ids: Optional[torch.IntTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        spec_metadata: Optional[SpecMetadata] = None,
        **kwargs,
    ) -> torch.Tensor:
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        rank = self.layers[0].mapping.rank if self.layers else 0
        if _diag_enabled():
            _diag_tensor("embed_out", inputs_embeds, rank)

        hidden_states = inputs_embeds
        residual = None
        for layer in self.layers:
            hidden_states, residual = layer(
                position_ids=position_ids,
                hidden_states=hidden_states,
                attn_metadata=attn_metadata,
                residual=residual,
                spec_metadata=spec_metadata,
                **kwargs,
            )
        hidden_states, _ = self.norm(hidden_states, residual)

        if _diag_enabled():
            _diag_tensor("final_norm_out", hidden_states, rank)

        return hidden_states


@register_auto_model("LagunaForCausalLM")
class LagunaForCausalLM(SpecDecOneEngineForCausalLM[LagunaModel, PretrainedConfig]):
    def __init__(self, model_config):
        super().__init__(LagunaModel(model_config), model_config)

    def load_weights(
        self,
        weights: Dict,
        weight_mapper: Optional["LagunaHfWeightMapper"] = None,
        params_map: Optional[Dict[str, str]] = None,
        allow_partial_loading: bool = False,
    ):
        if weight_mapper is not None:
            weights = weight_mapper.preprocess_weights(weights)

        super().load_weights(
            weights=weights,
            weight_mapper=weight_mapper,
            params_map=params_map,
            allow_partial_loading=allow_partial_loading,
        )


# ---------------------------------------------------------------------------
#  Weight mapper (HF checkpoint -> TRT-LLM module names)
# ---------------------------------------------------------------------------


@register_mapper("HF", "LagunaForCausalLM")
class LagunaHfWeightMapper(HfWeightMapper):
    def preprocess_weights(self, weights: dict) -> dict:
        weights = self.rename_by_params_map(
            {
                r"(.*)mlp\.experts\.e_score_correction_bias(.*)": r"\1mlp.gate.e_score_correction_bias\2",
            },
            weights,
        )

        # Translate llm-compressor "compressed-tensors" NVFP4 naming into what
        # TRT-LLM's NVFP4 linear / MoE loaders expect. llm-compressor stores
        #   weight_packed        (uint8, packed FP4)
        #   weight_scale         (FP8 per-group)
        #   weight_global_scale  (FP32 scalar = (FP8_MAX*FP4_MAX)/amax_weight)
        #   input_global_scale   (FP32 scalar = (FP8_MAX*FP4_MAX)/amax_input)
        # TRT-LLM's loaders expect:
        #   weight               (uint8 view of float4_e2m1x2)
        #   weight_scale         (FP8, same)
        #   weight_scale_2       (FP32 scalar = amax_weight/(FP8_MAX*FP4_MAX))
        #   input_scale          (FP32 scalar = amax_input/(FP8_MAX*FP4_MAX))
        # which is the reciprocal of the *_global_scale values.
        if self._config is not None and self._config.quant_config.quant_mode.has_nvfp4():

            def _invert_global_scale(t: torch.Tensor) -> torch.Tensor:
                """Convert compressed-tensors multiplier form to TRT-LLM
                divisor form, protecting against uncalibrated experts.

                llm-compressor stores global_scale=(FP8_MAX*FP4_MAX)/amax;
                TRT-LLM expects {weight_scale_2, input_scale}=amax/(FP8_MAX*FP4_MAX),
                i.e. the reciprocal. Some checkpoints store 0 for experts with
                no calibration data (amax unavailable); a naive 1/x would
                produce Inf which then contaminates the MoE per-layer
                max-reduction over experts and zeroes out *every* expert's
                input_scale. We preserve 0 -> 0 so those entries are harmless
                in the max reduction and the corresponding expert's alpha
                evaluates to 0 (silencing the dead expert).
                """
                t = t.to(torch.float32)
                out = torch.where(
                    t > 0,
                    1.0 / torch.clamp(t, min=torch.finfo(torch.float32).tiny),
                    torch.zeros_like(t),
                )
                # Belt-and-braces: any residual Inf/NaN becomes 0.
                return torch.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0).contiguous()

            renamed: dict = {}
            for name, value in weights.items():
                if name.endswith(".weight_packed"):
                    new_name = name[: -len(".weight_packed")] + ".weight"
                    renamed[new_name] = value
                elif name.endswith(".weight_global_scale"):
                    new_name = name[: -len(".weight_global_scale")] + ".weight_scale_2"
                    t = value[...] if not isinstance(value, torch.Tensor) else value
                    renamed[new_name] = _invert_global_scale(t)
                elif name.endswith(".input_global_scale"):
                    new_name = name[: -len(".input_global_scale")] + ".input_scale"
                    t = value[...] if not isinstance(value, torch.Tensor) else value
                    renamed[new_name] = _invert_global_scale(t)
                else:
                    renamed[name] = value
            weights = renamed

        return weights

    def is_special_instance_module(self, module: nn.Module) -> bool:
        return isinstance(module, MoEInterface)

    def handle_special_instance_module(
        self,
        module: nn.Module,
        module_name: str,
        module_weights: dict,
        allow_partial_loading: bool = False,
    ) -> None:
        if isinstance(module, MoEInterface):
            # DeepSeekFP8 block-scales MoE expects "weight_scale_inv" but NVFP4
            # MoE expects the original "weight_scale" (plus "weight_scale_2").
            use_fp8_block_rename = (
                self._config is not None
                and self._config.quant_config.quant_mode.has_fp8_block_scales()
            )
            updated = {}
            for wn, wv in module_weights.items():
                new_wn = (
                    wn.replace("gate_proj", "w1")
                    .replace("up_proj", "w3")
                    .replace("down_proj", "w2")
                )
                if use_fp8_block_rename:
                    # ModelOpt uses "weight_scale"; TRT-LLM DeepSeekFP8
                    # quantization expects "weight_scale_inv" (same values).
                    new_wn = new_wn.replace("weight_scale", "weight_scale_inv")
                # filter_weights strips module prefix leaving "experts.X.w1.weight";
                # VANILLA mode expects "X.w1.weight" (integer prefix only).
                if new_wn.startswith("experts."):
                    new_wn = new_wn[len("experts.") :]
                updated[new_wn] = wv
            if os.environ.get("LAGUNA_DEBUG_LOAD"):
                sample = sorted(updated.keys())[:6]
                print(
                    f"[DEBUG handle_special] module_name={module_name} sample keys: {sample}",
                    flush=True,
                )
            del module_weights
            module.load_weights(weights=[updated], allow_partial_loading=allow_partial_loading)

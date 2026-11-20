# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Runtime-owned recurrent state for the Qwen3.6 FP8 MTP verifier."""

from dataclasses import dataclass

import torch
import triton as tr
import triton.language as tl

from tensorrt_llm._torch.pyexecutor.kv_cache.mamba_cache_manager.common import (
    MambaAcceptanceBatch,
    MambaLayerCache,
    MambaStateLayout,
    ReplayLayout,
    _promote_mamba_state_triton,
    _stack_state_views,
)
from tensorrt_llm._torch.pyexecutor.kv_cache.mamba_cache_manager.mamba_cache_manager_v2 import (
    MambaHybridCacheManagerV2,
)
from tensorrt_llm.runtime.kv_cache_manager_v2 import DataRole

_FACTORS = DataRole("qwen36_fp8_factors")
_PENDING = DataRole("qwen36_fp8_pending")


@dataclass(frozen=True, kw_only=True)
class VerifierLayerCache(MambaLayerCache):
    """One recurrent layer's convolution scratch and factored replay history."""

    windows: torch.Tensor
    factors: torch.Tensor
    pending: torch.Tensor


@tr.jit
def _set_pending(
    Pending,
    Slots,
    Accepted,
    Dummy,
    COUNT: tl.constexpr,
    PENDING_STRIDE: tl.constexpr,
    B: tl.constexpr,
    HAS_ACCEPTANCE: tl.constexpr,
    HAS_DUMMY: tl.constexpr,
):
    index = tl.program_id(0) * B + tl.arange(0, B)
    active = index < COUNT
    if HAS_DUMMY:
        active &= ~tl.load(Dummy + index, active, False)
    slot = tl.load(Slots + index, active, 0)
    accepted = tl.load(Accepted + index, active, 1) - 1 if HAS_ACCEPTANCE else 0
    tl.store(Pending + slot * PENDING_STRIDE, accepted, active)


def clear_pending(pending: torch.Tensor, slots: torch.Tensor) -> None:
    """Clear accepted factors after a non-speculative generation step."""
    count = slots.shape[0]
    _set_pending[(tr.cdiv(count, 32),)](
        pending, slots, slots, slots, count, pending.stride(0), 32, False, False
    )


class FactoredVerifierState:
    """Keep the first GDN state and FP32 rank-one updates for later rows."""

    replay_shares_state_slot = False
    intermediate_ssm = None

    def __init__(self, width: int) -> None:
        self.width = width
        self.intermediate_indices = None
        self.intermediate_conv = None
        self._conv_states = ()
        self._stacked_conv = None
        self._factors = []
        self._pending = None

    def replay_layout(self, context: MambaStateLayout, layer_id: int) -> ReplayLayout:
        """Reserve rank-one factors and an accepted-draft count per replay slot."""
        heads, value_dim, key_dim = context.ssm_state_shape
        layout = {
            _FACTORS: (torch.float32, [self.width - 1, heads, key_dim + value_dim + 1]),
        }
        if layer_id == context.mamba_pp_layers[0]:
            layout[_PENDING] = (torch.int32, [1])
        return layout

    def bind_replay_pool(self, layers: list[dict[DataRole, torch.Tensor]]) -> None:
        self._factors = [layer[_FACTORS] for layer in layers]
        self._pending = layers[0][_PENDING][:, 0]

    def bind(
        self,
        context: MambaStateLayout,
        ssm_states: list[torch.Tensor],
        conv_states: list[torch.Tensor],
    ) -> None:
        if not context.mamba_pp_layers:
            return
        self._conv_states = tuple(conv_states)
        self._stacked_conv = _stack_state_views(conv_states)
        device = conv_states[0].device
        shape = (len(conv_states), context.max_batch_size, self.width, *context.conv_state_shape)
        self.intermediate_conv = torch.empty(shape, dtype=context.conv_state_dtype, device=device)
        self.intermediate_indices = torch.arange(
            context.max_batch_size, dtype=torch.int32, device=device
        )

    def reset_slots(
        self, slots: torch.Tensor, host_slots: list[int], replay_slots: torch.Tensor | None = None
    ) -> None:
        """Reset replay bookkeeping for a request entering prefill."""
        self.reset_replay_slots(slots if replay_slots is None else replay_slots)

    def reset_replay_slots(self, replay_slots: torch.Tensor) -> None:
        self._pending.index_fill_(0, replay_slots, 0)

    def make_layer_cache(
        self, layer_offset: int, conv: torch.Tensor, temporal: torch.Tensor
    ) -> VerifierLayerCache:
        return VerifierLayerCache(
            conv=conv,
            temporal=temporal,
            windows=self.intermediate_conv[layer_offset],
            factors=self._factors[layer_offset],
            pending=self._pending,
        )

    def update(self, batch: MambaAcceptanceBatch) -> None:
        """Commit the accepted convolution window and the factor replay length."""
        if self._stacked_conv is not None:
            _promote_mamba_state_triton(
                self._stacked_conv,
                self.intermediate_conv,
                batch.source_state_indices,
                batch.accepted_positions,
                batch.destination_state_indices,
            )
        else:
            for layer, state in enumerate(self._conv_states):
                _promote_mamba_state_triton(
                    state.unsqueeze(0),
                    self.intermediate_conv[layer : layer + 1],
                    batch.source_state_indices,
                    batch.accepted_positions,
                    batch.destination_state_indices,
                )
        has_dummy = batch.is_dummy_request is not None
        dummy = batch.is_dummy_request if has_dummy else batch.replay_indices
        _set_pending[(tr.cdiv(batch.num_generations, 32),)](
            self._pending,
            batch.replay_indices,
            batch.num_accepted_tokens,
            dummy,
            batch.num_generations,
            self._pending.stride(0),
            32,
            True,
            has_dummy,
        )

    def shutdown(self) -> None:
        self.intermediate_indices = None
        self.intermediate_conv = None
        self._conv_states = ()
        self._stacked_conv = None
        self._factors = []
        self._pending = None


class Qwen36HybridCacheManager(MambaHybridCacheManagerV2):
    """Use the normal hybrid manager with a target-specific speculative state policy."""

    _skips_steady_state_index_upload = True

    def __init__(self, *args, use_replay_state_update: bool | None = None, **kwargs) -> None:
        kwargs.setdefault("conv_state_layout", "q_k_v")
        super().__init__(*args, **kwargs)

    def _initialize_spec_state(self) -> FactoredVerifierState | None:
        if self.spec_config is None or self.local_num_mamba_layers == 0:
            return None
        return FactoredVerifierState(self.spec_config.tokens_per_gen_step)

    @classmethod
    def _estimate_private_replay_bytes_per_slot(cls, model_config, mapping, **kwargs) -> int:
        spec_config = kwargs.get("spec_config")
        if spec_config is None:
            return 0
        return (
            model_config.get_num_mamba_layers()
            * (spec_config.tokens_per_gen_step - 1)
            * 32
            * 257
            * 4
            + 4
        )

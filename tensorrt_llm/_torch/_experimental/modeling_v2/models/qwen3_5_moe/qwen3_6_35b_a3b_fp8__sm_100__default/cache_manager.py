# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Request-private FP32 Gated DeltaNet records for speculative verification."""

from __future__ import annotations

import math
from dataclasses import replace
from typing import TYPE_CHECKING

import torch

from tensorrt_llm._torch.modules.fla.cache_manager import (
    GDNIntermediateState,
    GDNReplayState,
    Qwen35HybridCacheManager,
    _estimated_gdn_replay_state,
)
from tensorrt_llm._torch.utils import is_gdn_replay_enabled

from .ops.recurrent import promote_replay_conv

if TYPE_CHECKING:
    from tensorrt_llm._torch.pyexecutor.kv_cache.mamba_cache_manager.common import (
        MambaAcceptanceBatch,
        MambaStateLayout,
    )


class _Fp32GdnReplayState(GDNReplayState):
    def __init__(self, width: int) -> None:
        super().__init__(width)
        self.replay_history_size = 2 * width
        self.fused_verify = True

    def bind(
        self,
        context: MambaStateLayout,
        ssm_states: list[torch.Tensor],
        conv_states: list[torch.Tensor],
    ) -> None:
        # Candidate storage holds one raw input per rank. The bound windows keep all three taps.
        scratch = replace(context, conv_state_shape=context.conv_state_shape[:1])
        super().bind(scratch, ssm_states, conv_states)

    def replay_layout(self, context, layer_id):
        layout = super().replay_layout(context, layer_id)
        return {
            role: (torch.float32 if dtype == context.conv_state_dtype else dtype, shape)
            for role, (dtype, shape) in layout.items()
        }

    def _bind_state_addressing(self, context) -> None:
        """The verifier folds replay records and writes the checkpoints in-layer."""

    def update(self, batch: MambaAcceptanceBatch) -> None:
        metadata = self.get_replay_metadata()
        staged = self.intermediate_conv
        if metadata is None or staged is None:
            raise RuntimeError("Qwen replay buffers are not bound")
        replay_slots = (
            batch.destination_state_indices
            if batch.replay_indices is None
            else batch.replay_indices
        )
        args = (
            batch.source_state_indices,
            batch.destination_state_indices,
            batch.num_accepted_tokens,
            batch.is_dummy_request,
            replay_slots,
            metadata.prev_num_accepted_tokens,
            metadata.cache_buf_idx,
            metadata.replay_step_width,
            metadata.replay_history_size,
        )
        if self._stacked_conv is not None:
            promote_replay_conv(self._stacked_conv, staged, *args)
        else:
            for index, window in enumerate(self._conv_states):
                promote_replay_conv(
                    window.unsqueeze(0), staged[index : index + 1], *args, advance=index == 0
                )


class Qwen36HybridCacheManager(Qwen35HybridCacheManager):
    """Preserve full FP32 records while reusing the manager's slot lifecycle."""

    def _initialize_spec_state(self) -> GDNIntermediateState | GDNReplayState:
        if (
            self.spec_config is None
            or self._requested_replay is False
            or not is_gdn_replay_enabled()
        ):
            return super()._initialize_spec_state()
        self.gdn_fused_verify = True
        return _Fp32GdnReplayState(self.spec_config.tokens_per_gen_step)

    @classmethod
    def _estimate_private_replay_bytes_per_slot(cls, model_config, mapping, **kwargs) -> int:
        estimate = _estimated_gdn_replay_state(model_config, mapping, **kwargs)
        if estimate is None:
            return 0
        _, layout = estimate
        replay = _Fp32GdnReplayState(layout.spec_config.tokens_per_gen_step)
        return sum(
            dtype.itemsize * math.prod(shape)
            for layer in layout.mamba_pp_layers
            for dtype, shape in replay.replay_layout(layout, layer).values()
        )

    @classmethod
    def _extra_state_bytes_per_rank(cls, model_config, mapping, **kwargs) -> int:
        return 0

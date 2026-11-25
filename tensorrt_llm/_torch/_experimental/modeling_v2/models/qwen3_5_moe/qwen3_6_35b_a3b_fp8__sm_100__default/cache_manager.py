# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Request-private FP32 Gated DeltaNet records for speculative verification."""

from __future__ import annotations

import math

import torch

from tensorrt_llm._torch.modules.fla.cache_manager import (
    GDNIntermediateState,
    GDNReplayState,
    Qwen35HybridCacheManager,
    _estimated_gdn_replay_state,
)
from tensorrt_llm._torch.utils import is_gdn_replay_enabled

# Keep a few accepted prefixes between full FP32 state checkpoints.
_HISTORY = 8


class _Fp32GdnReplayState(GDNReplayState):
    def __init__(self, width: int) -> None:
        super().__init__(width)
        self.replay_history_size = _HISTORY
        self.fused_verify = True

    def replay_layout(self, context, layer_id):
        layout = super().replay_layout(context, layer_id)
        return {
            role: (torch.float32 if dtype == context.conv_state_dtype else dtype, shape)
            for role, (dtype, shape) in layout.items()
        }

    def _bind_state_addressing(self, context) -> None:
        """The verifier folds replay records and writes the checkpoints in-layer."""


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

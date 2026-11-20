# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Input normalization for the Qwen MTP block."""

import torch

from .....catalog._op import Arch, Cell, OpWrapper
from ..kernels import mtp


class DraftInput(OpWrapper):
    """Gather token embeddings and normalize both inputs to the BF16 FC."""

    ARCHS = frozenset({Arch.SM_100})
    CELLS = (
        Cell("one recycled draft token", {"rows": 1}),
        Cell("four verifier rows with repeated token IDs", {"rows": 4}),
    )

    def __call__(
        self,
        ids: torch.Tensor,
        embed: torch.Tensor,
        hidden: torch.Tensor,
        embedding_norm: torch.Tensor,
        hidden_norm: torch.Tensor,
    ) -> torch.Tensor:
        rows = ids.shape[0]
        stack = hidden.new_empty((rows, 4096))
        mtp.draft_input[(rows,)](ids, embed, hidden, embedding_norm, hidden_norm, stack)
        return stack

    def reference(
        self,
        ids: torch.Tensor,
        embed: torch.Tensor,
        hidden: torch.Tensor,
        embedding_norm: torch.Tensor,
        hidden_norm: torch.Tensor,
    ) -> torch.Tensor:
        """Evaluate Gemma-style RMSNorm on the embedding and hidden state."""
        embedded = embed[ids].float()
        states = hidden.float()
        results = []
        for x, weight in ((embedded, embedding_norm), (states, hidden_norm)):
            x = x * torch.rsqrt(x.square().mean(-1, keepdim=True) + 1e-6)
            results.append(x * (1.0 + weight.float()))
        return torch.cat(results, dim=-1).to(hidden.dtype)


draft_input = DraftInput()

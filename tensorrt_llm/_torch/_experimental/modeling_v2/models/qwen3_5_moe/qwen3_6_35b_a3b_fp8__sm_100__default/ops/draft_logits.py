# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Select a draft token from the active vocabulary prefix."""

import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, Cell, OpWrapper

from ..kernels.draft_logits import shortlist_argmax


class DraftArgmax(OpWrapper):
    """Return the mapped token of each first maximum in a BF16 shortlist."""

    ARCHS = frozenset({Arch.SM_100})
    CELLS = (
        Cell("singleton shortlist with masked columns and tied scores", {"rows": 1}),
        Cell("batched shortlist with padded output storage", {"rows": 32, "inplace": True}),
    )

    def __call__(
        self,
        logits: torch.Tensor,
        ids: torch.Tensor,
        length: torch.Tensor,
        out: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return shortlist_argmax(logits, ids, length, out)

    def reference(
        self,
        logits: torch.Tensor,
        ids: torch.Tensor,
        length: torch.Tensor,
        out: torch.Tensor | None = None,
    ) -> torch.Tensor:
        result = ids[logits[:, : length.item()].argmax(dim=1)]
        return result if out is None else out.copy_(result)


draft_argmax = DraftArgmax()

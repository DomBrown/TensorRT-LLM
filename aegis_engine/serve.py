# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Start trtllm-serve with the same BF16 projection policy as engine.sh."""

import torch

from tensorrt_llm.commands.serve import main
from tensorrt_llm.serve import openai_server

if __name__ == "__main__":
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    # Let clients retire idle pooled connections before the server closes them.
    openai_server.TIMEOUT_KEEP_ALIVE = 30
    main()

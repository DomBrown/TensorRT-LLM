# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import os
import sys
from pathlib import Path
from typing import Literal, TextIO


def _token_ids(value: object, vocab_size: int, *, allow_empty: bool = False) -> list[int]:
    if not isinstance(value, list) or (not allow_empty and not value):
        raise ValueError("expected a list of token IDs")
    if any(type(token) is not int or not 0 <= token < vocab_size for token in value):
        raise ValueError("token ID is outside the vocabulary")
    return value


def run(
    directory: Path,
    protocol: TextIO,
    modeling_v2: Literal["off", "require"] = "require",
) -> None:
    """Serve token-ID generation and teacher-forced scoring through the LLM API."""
    import torch

    from tensorrt_llm import LLM, SamplingParams

    text_config = json.loads((directory / "config.json").read_text())["text_config"]
    generation_config = json.loads((directory / "generation_config.json").read_text())
    eos_ids = generation_config["eos_token_id"]
    if isinstance(eos_ids, int):
        eos_ids = [eos_ids]
    pad_id = generation_config["pad_token_id"]
    vocab_size = text_config["vocab_size"]

    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    llm = LLM(
        directory,
        modeling_v2=modeling_v2,
        skip_tokenizer_init=True,
        disable_mm_encoder=True,
        max_seq_len=8192,
        max_batch_size=1,
        max_num_tokens=8192,
        cuda_graph_config={"batch_sizes": [1]},
        kv_cache_config={
            "enable_block_reuse": False,
            "max_tokens": 16384,
            **({"mamba_ssm_cache_dtype": "float32"} if modeling_v2 == "require" else {}),
        },
    )

    def reply(result: dict) -> None:
        protocol.write(json.dumps(result, separators=(",", ":"), allow_nan=False) + "\n")

    def generate(prompt: list[int], length: int, ignore_eos: bool) -> list[int]:
        if length == 0:
            return []
        params = SamplingParams(
            max_tokens=length,
            end_id=-1 if ignore_eos else eos_ids[0],
            pad_id=pad_id,
            stop_token_ids=None if ignore_eos else eos_ids[1:],
            include_stop_str_in_output=True,
            temperature=0,
            detokenize=False,
        )
        output = llm.generate_async(prompt, params, streaming=False).result()
        return output.outputs[0].token_ids

    def score(request: dict, ids: list[int]) -> dict:
        position_count = len(ids) - 1
        top_k = request.get("top_k", 0)
        if type(top_k) is not int or not 0 <= top_k <= vocab_size:
            raise ValueError("top_k must be an integer from 0 to the vocabulary size")
        requested = request.get("logprob_ids")
        if "logprob_ids" in request:
            if not isinstance(requested, list) or len(requested) != position_count:
                raise ValueError("logprob_ids must have one list per position")
            requested = [_token_ids(row, vocab_size, allow_empty=True) for row in requested]

        result = {"top1": []}
        if "top_k" in request:
            result["top_k"] = []
        if requested is not None:
            result["logprobs"] = []
        if not position_count:
            return result

        params = SamplingParams(
            max_tokens=1,
            end_id=-1,
            pad_id=pad_id,
            temperature=0,
            detokenize=False,
            return_context_logits=True,
        )
        output = llm.generate_async(ids[:-1], params, streaming=False).result()
        logits = output.context_logits
        if logits is None or logits.shape[0] != position_count:
            raise RuntimeError("missing teacher-forced context logits")
        for offset in range(0, position_count, 128):
            rows = logits[offset : offset + 128, :vocab_size].float()
            normalizer = rows.logsumexp(dim=-1, keepdim=True)
            result["top1"].extend(rows.argmax(dim=-1).tolist())
            if "top_k" in result:
                values, indices = rows.topk(top_k)
                result["top_k"].extend(
                    list(zip(token_row, prob_row, strict=True))
                    for token_row, prob_row in zip(
                        indices.tolist(), (values - normalizer).tolist(), strict=True
                    )
                )
            if requested is not None:
                result["logprobs"].extend(
                    (row[tokens] - norm).tolist()
                    for row, norm, tokens in zip(
                        rows, normalizer, requested[offset : offset + len(rows)], strict=True
                    )
                )
        return result

    try:
        generate([text_config["bos_token_id"]], 2, True)
        score({"top_k": 1}, [text_config["bos_token_id"]] * 2)
        reply({"ready": True})
        for line in sys.stdin:
            try:
                request = json.loads(line)
                if not isinstance(request, dict):
                    raise ValueError("request must be a JSON object")
                op = request.get("op")
                if op == "shutdown":
                    reply({"ok": True})
                    break
                if op not in ("score", "generate"):
                    raise ValueError(f"unknown operation {op!r}")
                ids = _token_ids(request.get("ids"), vocab_size)
                if op == "score":
                    reply(score(request, ids))
                else:
                    length = request.get("max_new_tokens")
                    if type(length) is not int or length < 0:
                        raise ValueError("max_new_tokens must be a nonnegative integer")
                    ignore_eos = request.get("ignore_eos", False)
                    if type(ignore_eos) is not bool:
                        raise ValueError("ignore_eos must be a boolean")
                    reply({"ids": generate(ids, length, ignore_eos)})
            except (ValueError, TypeError, RuntimeError) as error:
                reply({"error": str(error)})
    finally:
        llm.shutdown()


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("usage: engine.sh MODEL_DIRECTORY")
    with os.fdopen(os.dup(sys.stdout.fileno()), "w", buffering=1) as protocol:
        os.dup2(sys.stderr.fileno(), sys.stdout.fileno())
        run(Path(sys.argv[1]), protocol)


if __name__ == "__main__":
    main()

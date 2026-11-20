# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""JSONL interface for token-based generation and teacher-forced scoring."""

import json
import os
import sys
from contextlib import ExitStack
from pathlib import Path
from typing import TextIO


def run(
    model_dir: Path, output: TextIO, *, modeling_v2: str = "require", draft_len: int = 3
) -> None:
    """Serve one request at a time on a TensorRT-LLM executor."""
    import torch
    import yaml

    from tensorrt_llm import LLM, SamplingParams
    from tensorrt_llm.llmapi import MTPDecodingConfig

    torch.set_num_threads(min(torch.get_num_threads(), 8))
    config = yaml.safe_load((Path(__file__).parent / "serve.yaml").read_text())
    config.pop("speculative_config", None)
    config.update(modeling_v2=modeling_v2, max_batch_size=1)
    config["cuda_graph_config"]["batch_sizes"] = [1]
    if modeling_v2 == "require" and draft_len:
        config.update(
            enable_chunked_prefill=False,
            prefill_cuda_graph_backend="breakable",
            prefill_capture_num_tokens=[128, 256, 512, 1024, 2048, 4096],
        )
    page_size = config["kv_cache_config"].get("tokens_per_block", 32)
    config["kv_cache_config"]["max_tokens"] = 2 * config["max_seq_len"] + 2 * page_size
    generation = json.loads((model_dir / "generation_config.json").read_text())
    eos = generation["eos_token_id"]
    if isinstance(eos, int):
        eos = [eos]
    text_config = json.loads((model_dir / "config.json").read_text())["text_config"]
    vocab_size = text_config["vocab_size"]

    def send(obj: dict) -> None:
        print(json.dumps(obj, separators=(",", ":"), allow_nan=False), file=output, flush=True)

    def check_ids(ids: list[int], *, allow_empty: bool = False) -> None:
        if not isinstance(ids, list) or any(
            type(i) is not int or not 0 <= i < vocab_size for i in ids
        ):
            raise ValueError("expected a list of valid token IDs")
        if not ids and not allow_empty:
            raise ValueError("ids must contain at least one token")

    with ExitStack() as sessions:
        scorer = sessions.enter_context(LLM(str(model_dir), **config))
        generator = scorer
        if modeling_v2 == "require" and draft_len:
            # The runtime speculative sampler does not return teacher-forced logits.
            gen_config = {
                **config,
                "speculative_config": MTPDecodingConfig(max_draft_len=draft_len),
            }
            generator = sessions.enter_context(LLM(str(model_dir), **gen_config))
        send({"ready": True})
        for line in sys.stdin:
            try:
                request = json.loads(line)
                if not isinstance(request, dict):
                    raise ValueError("expected a JSON object")
                op = request["op"]
                if op == "shutdown":
                    send({"ok": True})
                    return
                if op not in {"generate", "score"}:
                    raise ValueError(f"unknown operation {op!r}")
                ids = request["ids"]
                check_ids(ids)
                if op == "generate":
                    count = request["max_new_tokens"]
                    ignore_eos = request.get("ignore_eos", False)
                    if type(count) is not int or count < 0 or type(ignore_eos) is not bool:
                        raise ValueError(
                            "expected nonnegative max_new_tokens and boolean ignore_eos"
                        )
                    if count == 0:
                        send({"ids": []})
                        continue
                    params = SamplingParams(
                        max_tokens=count,
                        temperature=0,
                        ignore_eos=ignore_eos,
                        end_id=eos[0],
                        stop_token_ids=None if ignore_eos else eos[1:],
                        include_stop_str_in_output=True,
                        detokenize=False,
                    )
                    executor = scorer if count == 1 else generator
                    result = executor.generate(ids, params, use_tqdm=False)
                    tokens = result.outputs[0].token_ids
                    if ignore_eos and len(tokens) != count:
                        raise RuntimeError(
                            f"expected {count} generated tokens, received {len(tokens)}"
                        )
                    send({"ids": tokens})
                    continue

                top_k = request.get("top_k", 0)
                requested = request.get("logprob_ids")
                positions = len(ids) - 1
                if type(top_k) is not int or not 0 <= top_k <= vocab_size:
                    raise ValueError("top_k is outside the vocabulary")
                if "logprob_ids" in request:
                    if not isinstance(requested, list) or len(requested) != positions:
                        raise ValueError("logprob_ids must have one list per scored position")
                    for row in requested:
                        check_ids(row, allow_empty=True)
                response = {"top1": []}
                if "top_k" in request:
                    response["top_k"] = [[] for _ in range(positions)]
                if requested is not None:
                    response["logprobs"] = [[] for _ in range(positions)]
                if positions:
                    params = SamplingParams(
                        max_tokens=1, temperature=0, detokenize=False, return_context_logits=True
                    )
                    result = scorer.generate(ids[:-1], params, use_tqdm=False)
                    logits = result.context_logits
                    if logits is None or logits.shape[0] != positions:
                        raise RuntimeError(
                            "executor did not return every requested context-logit row"
                        )
                    for start in range(0, positions, 64):
                        scores = logits[start : start + 64, :vocab_size].float()
                        response["top1"].extend(scores.argmax(dim=-1).tolist())
                        if top_k or requested is not None:
                            scores = scores.log_softmax(dim=-1)
                        if top_k:
                            values, indices = scores.topk(top_k, dim=-1)
                            for offset, (tokens, probs) in enumerate(
                                zip(indices.tolist(), values.tolist())
                            ):
                                response["top_k"][start + offset] = list(zip(tokens, probs))
                        if requested is not None:
                            for offset, row in enumerate(
                                requested[start : start + scores.shape[0]]
                            ):
                                response["logprobs"][start + offset] = scores[offset, row].tolist()
                send(response)
            except (KeyError, ValueError, TypeError, RuntimeError) as exc:
                send({"error": str(exc)})


def main() -> None:
    # Preserve the protocol descriptor before Python and native libraries write logs.
    with os.fdopen(os.dup(sys.stdout.fileno()), "w") as output:
        os.dup2(sys.stderr.fileno(), sys.stdout.fileno())
        run(Path(sys.argv[1]), output)


if __name__ == "__main__":
    main()

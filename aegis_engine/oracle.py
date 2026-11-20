# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Drive the JSONL engine against the Qwen3.6 development oracle."""

import argparse
import json
import math
import subprocess
import time
from pathlib import Path


class Client:
    """Client for one long-lived JSONL engine process."""

    def __init__(self, model: str, command: str) -> None:
        start = time.perf_counter()
        self.process = subprocess.Popen(
            [command, model], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True
        )
        if self.read() != {"ready": True}:
            raise RuntimeError("engine did not send its ready message")
        self.startup_seconds = time.perf_counter() - start

    def read(self) -> dict:
        assert self.process.stdout is not None
        line = self.process.stdout.readline()
        if not line:
            raise RuntimeError(f"engine closed stdout (status {self.process.poll()})")
        result = json.loads(line)
        if "error" in result:
            raise RuntimeError(result["error"])
        return result

    def request(self, **request) -> dict:
        assert self.process.stdin is not None
        print(json.dumps(request, separators=(",", ":")), file=self.process.stdin, flush=True)
        return self.read()

    def close(self) -> None:
        if self.process.poll() is None:
            self.request(op="shutdown")
        self.process.wait()


def kl_at_position(reference: list[list], engine: list[float]) -> float:
    """Compute KL with one additional bin for the mass outside the oracle's top-k."""
    ref_rest = max(0.0, 1.0 - sum(math.exp(v) for _, v in reference))
    engine_rest = max(1e-12, 1.0 - sum(math.exp(v) for v in engine))
    kl = sum(math.exp(v) * (v - e) for (_, v), e in zip(reference, engine, strict=True))
    if ref_rest:
        kl += ref_rest * math.log(ref_rest / engine_rest)
    return kl


def validate(engine: Client, oracle: dict, selected: list[int]) -> bool:
    """Check every position and report the first greedy difference of each item."""
    fragile = {(entry["prompt"], entry["token"]) for entry in oracle.get("fragile", [])}
    kl_sum = 0.0
    positions = strong = hits = ties = forgiven = 0
    failures = []
    for index in selected:
        item = oracle["items"][index]
        sequence, prompt, expected = (item[k] for k in ("seq_ids", "prompt_ids", "gen_ids"))
        reference = item["top_k"]
        wanted = [[token for token, _ in row] for row in reference]
        score = engine.request(op="score", ids=sequence, logprob_ids=wanted)
        item_kl = 0.0
        wrong = []
        for pos, (row, top, probs) in enumerate(
            zip(reference, score["top1"], score["logprobs"], strict=True)
        ):
            item_kl += kl_at_position(row, probs)
            if row[0][1] - row[1][1] >= 0.5:
                strong += 1
                if row[0][0] == top:
                    hits += 1
                else:
                    wrong.append(pos)
        kl_sum += item_kl
        positions += len(reference)
        generated = engine.request(
            op="generate", ids=prompt, max_new_tokens=len(expected), ignore_eos=True
        )["ids"]
        first = next(
            (i for i, (a, b) in enumerate(zip(generated, expected, strict=True)) if a != b),
            None,
        )
        verdict = "exact"
        if first is not None:
            pos = len(prompt) - 1 + first
            row = reference[pos]
            gap = row[0][1] - row[1][1]
            if gap < 0.5:
                ties += 1
                verdict = "near tie"
            elif (index, first) in fragile and (
                generated[first] == score["top1"][pos]
                or [generated[first]]
                == engine.request(
                    op="generate", ids=sequence[: pos + 1], max_new_tokens=1, ignore_eos=True
                )["ids"]
            ):
                forgiven += 1
                verdict = "fragile"
            else:
                failures.append((index, first))
                verdict = f"FAIL gap={gap:.4f}, expected={expected[first]}, got={generated[first]}"
        print(
            f"{index:2d} KL {item_kl / len(reference):.6f}, non-tie errors {wrong}, "
            f"first greedy difference {first} {verdict}",
            flush=True,
        )
    mean_kl = kl_sum / positions
    agreement = hits / strong
    print(
        f"mean KL {mean_kl:.6f}, non-tie top-1 {hits}/{strong} ({agreement:.6%}), "
        f"near ties {ties}, fragile {forgiven}, failures {failures}",
        flush=True,
    )
    return mean_kl <= 0.01 and agreement >= 0.99 and not failures


def main() -> int:
    """Run the full oracle by default and return a failing exit status on mismatch."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="/model")
    parser.add_argument("--command", default="./engine.sh")
    parser.add_argument("--oracle", type=Path, default=Path("/task/oracle_dev.json"))
    parser.add_argument("--items", type=int, nargs="+")
    args = parser.parse_args()
    oracle = json.loads(args.oracle.read_text())
    engine = Client(args.model, args.command)
    print(f"Ready in {engine.startup_seconds:.2f}s", flush=True)
    try:
        return int(not validate(engine, oracle, args.items or list(range(len(oracle["items"])))))
    finally:
        engine.close()


if __name__ == "__main__":
    raise SystemExit(main())

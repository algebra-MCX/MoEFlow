"""Greedy text-only completions against a local vLLM server; optional comparison of two runs."""

from __future__ import annotations

import argparse
import json
import math
import sys
import urllib.request

PROMPTS = [
    "The capital of France is",
    "def fibonacci(n):",
    "Mixture-of-experts models route each token to",
    "1, 2, 3, 4, 5,",
]


def validate_prompts(prompts: object) -> list[str | list[int]]:
    """Accept individual text or token-ID prompts, never an empty or ambiguous batch."""
    if not isinstance(prompts, list) or not prompts:
        raise ValueError("Prompts must be a nonempty JSON list")
    for prompt in prompts:
        if isinstance(prompt, str) and prompt:
            continue
        if isinstance(prompt, list) and prompt and all(type(token) is int and token >= 0 for token in prompt):
            continue
        raise ValueError("Each prompt must be nonempty text or a nonempty list of nonnegative integer token IDs")
    return prompts


def has_complete_logprobs(record: dict) -> bool:
    """A successful correctness probe needs actual output and one finite score per token."""
    tokens, scores = record["tokens"], record["logprobs"]
    return (
        bool(tokens)
        and len(scores) == len(tokens)
        and all(type(score) in (int, float) and math.isfinite(score) for score in scores)
    )


def run(
    port: int,
    out: str,
    model: str = "smoke",
    max_tokens: int = 16,
    data_parallel_rank: int | None = None,
    requests: int | None = None,
    prompts: list[str | list[int]] | None = None,
    *,
    data_parallel_ranks: list[int] | None = None,
) -> None:
    if data_parallel_ranks is not None and (
        data_parallel_rank is not None
        or not data_parallel_ranks
        or any(type(rank) is not int or rank < 0 for rank in data_parallel_ranks)
    ):
        raise ValueError("Specify one nonnegative DP rank or a nonempty list of nonnegative DP ranks")
    ranks = [data_parallel_rank] if data_parallel_ranks is None else data_parallel_ranks
    prompts = validate_prompts(PROMPTS if prompts is None else prompts)
    count = len(prompts) if requests is None else requests
    if count < 1:
        raise ValueError("Request count must be positive")
    results = []
    for index in range(count):
        p = prompts[index % len(prompts)]
        rank = ranks[index % len(ranks)]
        body = json.dumps(
            {"model": model, "prompt": p, "max_tokens": max_tokens, "temperature": 0.0, "logprobs": 1}
        ).encode()
        headers = {"Content-Type": "application/json"}
        if rank is not None:
            headers["X-data-parallel-rank"] = str(rank)
        req = urllib.request.Request(f"http://localhost:{port}/v1/completions", body, headers)
        with urllib.request.urlopen(req, timeout=300) as r:
            response = json.load(r)
        c = response["choices"][0]
        results.append(
            {
                "prompt": p,
                "text": c["text"],
                "tokens": c["logprobs"]["tokens"],
                "logprobs": c["logprobs"]["token_logprobs"],
                "data_parallel_rank": rank,
                "prompt_tokens": response.get("usage", {}).get("prompt_tokens"),
                "finish_reason": c.get("finish_reason"),
            }
        )
        if not has_complete_logprobs(results[-1]):
            raise ValueError(f"Request {index} returned empty output or incomplete/nonfinite token logprobs")
    with open(out, "w") as f:
        json.dump(results, f, indent=1)
    print(json.dumps([r["text"] for r in results]))


def compare_records(a: list[dict], b: list[dict], *, exact_logprobs: bool = False) -> list[dict]:
    """Require matching requests and complete greedy outputs, not just a common prefix."""
    if not a or len(a) != len(b):
        raise ValueError(f"Request counts must be nonzero and equal: {len(a)} vs {len(b)}")
    rows = []
    for x, y in zip(a, b):
        valid = has_complete_logprobs(x) and has_complete_logprobs(y)
        same = 0
        for s, t in zip(x["tokens"], y["tokens"]):
            if s != t:
                break
            same += 1
        lp = [abs(p - q) for p, q in zip(x["logprobs"][:same], y["logprobs"][:same])] if valid else []
        rows.append(
            {
                "prompt": x["prompt"],
                "common_prefix_tokens": same,
                "of": len(x["tokens"]),
                "candidate_tokens": len(y["tokens"]),
                "data_parallel_rank": x.get("data_parallel_rank"),
                "candidate_data_parallel_rank": y.get("data_parallel_rank"),
                "passed": valid
                and x["prompt"] == y["prompt"]
                and x["tokens"] == y["tokens"]
                and x["text"] == y["text"]
                and x.get("data_parallel_rank") == y.get("data_parallel_rank")
                and (not exact_logprobs or (bool(lp) and max(lp) == 0)),
                "max_abs_logprob_diff_on_prefix": max(lp) if lp else None,
            }
        )
    return rows


def compare(a_path: str, b_path: str, *, exact_logprobs: bool = False) -> bool:
    with open(a_path) as a_file, open(b_path) as b_file:
        rows = compare_records(json.load(a_file), json.load(b_file), exact_logprobs=exact_logprobs)
    print(json.dumps(rows, indent=1))
    return all(row["passed"] for row in rows)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int)
    ap.add_argument("--out")
    ap.add_argument("--compare", nargs=2)
    ap.add_argument("--require-match", action="store_true", help="fail if complete greedy outputs differ")
    ap.add_argument("--require-exact-logprobs", action="store_true", help="also fail on any token logprob difference")
    ap.add_argument("--model", default="smoke")
    ap.add_argument("--max-tokens", type=int, default=16)
    ap.add_argument("--requests", type=int, help="serial requests, cycling the prompt set (default: one per prompt)")
    ap.add_argument("--prompts-file", help="JSON list of text or token-ID prompts; preserves the supplied order")
    placement = ap.add_mutually_exclusive_group()
    placement.add_argument("--data-parallel-rank", type=int, help="pin requests using vLLM's routing header")
    placement.add_argument("--data-parallel-ranks", nargs="+", type=int, help="cycle these DP ranks in request order")
    args = ap.parse_args()
    if args.data_parallel_rank is not None and args.data_parallel_rank < 0:
        ap.error("--data-parallel-rank must be nonnegative")
    if args.data_parallel_ranks is not None and min(args.data_parallel_ranks) < 0:
        ap.error("--data-parallel-ranks must be nonnegative")
    if args.requests is not None and args.requests < 1:
        ap.error("--requests must be positive")
    if args.compare:
        matched = compare(*args.compare, exact_logprobs=args.require_exact_logprobs)
        if (args.require_match or args.require_exact_logprobs) and not matched:
            sys.exit(1)
    else:
        prompts = None
        if args.prompts_file:
            with open(args.prompts_file) as prompt_file:
                prompts = validate_prompts(json.load(prompt_file))
        run(
            args.port,
            args.out,
            args.model,
            args.max_tokens,
            args.data_parallel_rank,
            args.requests,
            prompts,
            data_parallel_ranks=args.data_parallel_ranks,
        )
    sys.exit(0)

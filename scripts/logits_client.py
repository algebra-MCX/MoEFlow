"""Collect and strictly compare bounded full-vocabulary raw logits from local vLLM servers.

The server must use --logprobs-mode raw_logits --max-logprobs -1. No runtime patches.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np
from bench.logits import compare_logits, parse_logits, require_raw_logits
from bench.record import make_record, write_record
from smoke_client import PROMPTS


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def server_config(url: str) -> dict:
    try:
        with urllib.request.urlopen(f"{url}/server_info?config_format=json", timeout=120) as response:
            return json.load(response)["vllm_config"]
    except urllib.error.HTTPError as error:
        if error.code == 404:
            raise ValueError(
                "The diagnostic endpoint is disabled; start the local server with "
                "VLLM_SERVER_DEV_MODE=1 and --host 127.0.0.1"
            ) from error
        raise


def collect(args) -> None:
    url = f"http://localhost:{args.port}"
    config = server_config(url)
    require_raw_logits(config, args.vocab_size)
    rows = []
    record = make_record(
        "full_vocabulary_raw_logits",
        {**vars(args), "server_logprobs_mode": "raw_logits", "boundary": "pre_sampling_logits"},
        {"complete": False, "requests": rows},
        seed=0,
    )
    manifest = Path(args.out)
    write_record(manifest, record)
    for index, prompt in enumerate(PROMPTS):
        request = urllib.request.Request(
            f"{url}/v1/completions",
            json.dumps(
                {
                    "model": args.model,
                    "prompt": prompt,
                    "max_tokens": args.steps,
                    "min_tokens": args.steps,
                    "temperature": 0,
                    "seed": 0,
                    "logprobs": -1,
                    "return_tokens_as_token_ids": True,
                }
            ).encode(),
            {"Content-Type": "application/json", "X-data-parallel-rank": str(args.data_parallel_rank)},
        )
        with urllib.request.urlopen(request, timeout=300) as response:
            choice = json.load(response)["choices"][0]
        logits = parse_logits(choice, args.vocab_size, args.steps)
        artifact = manifest.with_name(f"{manifest.stem}_prompt{index}.npz")
        np.savez_compressed(artifact, logits=logits)
        rows.append(
            {
                "prompt": prompt,
                "tokens": choice["logprobs"]["tokens"],
                "data_parallel_rank": args.data_parallel_rank,
                "artifact": artifact.name,
                "sha256": digest(artifact),
                "shape": list(logits.shape),
            }
        )
        write_record(manifest, record)
        print(f"prompt={index}: {logits.shape[0]} steps x {logits.shape[1]} raw logits", flush=True)
    record["results"]["complete"] = True
    write_record(manifest, record)


def compare(a_path: str, b_path: str) -> dict:
    paths = [Path(a_path), Path(b_path)]
    records = [json.loads(path.read_text()) for path in paths]
    for record in records:
        if record["kind"] != "full_vocabulary_raw_logits" or not record["results"]["complete"]:
            raise ValueError("Both raw-logits collections must be complete")
        if record["config"]["server_logprobs_mode"] != "raw_logits":
            raise ValueError("Cannot compare logprobs as raw logits")
    requests = [record["results"]["requests"] for record in records]
    if not requests[0] or len(requests[0]) != len(requests[1]):
        raise ValueError("Request counts must be nonzero and equal")
    results = []
    for a, b in zip(*requests):
        if any(a[key] != b[key] for key in ("prompt", "tokens", "data_parallel_rank")):
            raise ValueError("Prompt, generated context and pinned DP rank must match")
        logits = []
        for path, row in zip(paths, (a, b)):
            artifact = path.with_name(row["artifact"])
            if digest(artifact) != row["sha256"]:
                raise ValueError("Raw logits artifact SHA256 mismatch")
            with np.load(artifact, allow_pickle=False) as archive:
                logits.append(archive["logits"])
        results.append({"prompt": a["prompt"], **compare_logits(*logits)})
    return {
        "passed": all(row["bitwise_equal"] for row in results),
        "manifests": [{"path": str(path), "sha256": digest(path)} for path in paths],
        "requests": results,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int)
    parser.add_argument("--model", default="glm")
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--vocab-size", type=int)
    parser.add_argument("--data-parallel-rank", type=int, default=0)
    parser.add_argument("--compare", nargs=2)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    if args.compare:
        result = compare(*args.compare)
        write_record(args.out, result)
        print(json.dumps(result, indent=1))
        return 0 if result["passed"] else 1
    if not args.port or not args.vocab_size or args.vocab_size < 1 or args.steps < 1 or args.data_parallel_rank < 0:
        parser.error("Collection requires a port, positive vocabulary/step counts and a nonnegative DP rank")
    collect(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())

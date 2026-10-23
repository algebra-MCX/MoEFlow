"""Reject incomplete vLLM serving measurements before accepting performance results."""

import argparse
import json
import math
from pathlib import Path

METRICS = [
    f"{stat}_{metric}_ms"
    for metric in ("ttft", "tpot", "itl", "e2el")
    for stat in ("mean", "median", "std", "p50", "p90", "p95", "p99")
]


def finite_nonnegative(value: object) -> bool:
    return type(value) in (int, float) and math.isfinite(value) and value >= 0


def validate_result(record: dict, requests: int, input_len: int, output_len: int, concurrency: int) -> dict:
    if min(requests, input_len, concurrency) < 1 or output_len < 2:
        raise ValueError("Requests, input length and concurrency must be positive; output length must exceed one")
    expected = {
        "num_prompts": requests,
        "completed": requests,
        "failed": 0,
        "max_concurrency": concurrency,
        "total_input_tokens": requests * input_len,
        "total_output_tokens": requests * output_len,
    }
    for field, value in expected.items():
        if type(record.get(field)) is not int or record[field] != value:
            raise ValueError(f"{field}: expected {value}, got {record.get(field)!r}")
    for field, length in (("input_lens", input_len), ("output_lens", output_len)):
        if record.get(field) != [length] * requests:
            raise ValueError(f"{field}: every request must contain exactly {length} tokens")
    if record.get("errors") != [""] * requests:
        raise ValueError("Missing or failed per-request error records; use --save-detailed")
    for field in ("ttfts", "latencies"):
        values = record.get(field)
        if not isinstance(values, list) or len(values) != requests or not all(map(finite_nonnegative, values)):
            raise ValueError(f"Missing or invalid per-request {field}")
    itls = record.get("itls")
    if (
        not isinstance(itls, list)
        or len(itls) != requests
        or any(
            not isinstance(values, list) or not values or not all(map(finite_nonnegative, values)) for values in itls
        )
    ):
        raise ValueError("Missing or invalid inter-token latencies")
    for field in (*METRICS, "duration", "request_throughput", "output_throughput"):
        if not finite_nonnegative(record.get(field)):
            raise ValueError(f"Missing or nonfinite metric: {field}")
    if record["duration"] <= 0 or record["output_throughput"] <= 0:
        raise ValueError("Duration and output throughput must be positive")
    for field, total in (("request_throughput", requests), ("output_throughput", requests * output_len)):
        if not math.isclose(record[field], total / record["duration"], rel_tol=1e-6):
            raise ValueError(f"{field} does not match completed work / duration")
    return {
        "passed": True,
        "requests": requests,
        "total_output_tokens": requests * output_len,
        "metrics": {field: record[field] for field in METRICS},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path)
    for name in ("requests", "input-len", "output-len", "concurrency"):
        parser.add_argument(f"--{name}", required=True, type=int)
    args = parser.parse_args()
    try:
        result = validate_result(
            json.loads(args.path.read_text()), args.requests, args.input_len, args.output_len, args.concurrency
        )
    except (OSError, ValueError) as error:
        print(json.dumps({"passed": False, "error": str(error)}))
        raise SystemExit(1) from error
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

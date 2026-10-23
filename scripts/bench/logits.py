"""Strict full-vocabulary checks for vLLM's raw-logits completion response."""

from __future__ import annotations

import numpy as np


def require_raw_logits(config: dict, vocab_size: int) -> None:
    """Read the server's reported mode; a client-side label cannot establish raw logits."""
    model = config["model_config"]
    if model.get("logprobs_mode") != "raw_logits":
        raise ValueError("Server must use --logprobs-mode raw_logits")
    cap = model.get("max_logprobs", 0)
    if cap != -1 and cap < vocab_size:
        raise ValueError("Server max_logprobs cannot return the full vocabulary")


def parse_logits(choice: dict, vocab_size: int, steps: int) -> np.ndarray:
    """Require every token ID, every decode step, finite values and no API sentinel clipping."""
    scores = choice["logprobs"]
    rows = scores["top_logprobs"]
    if len(rows) != steps or len(scores["tokens"]) != steps:
        raise ValueError("Missing generated steps in logits response")
    expected = {f"token_id:{token}" for token in range(vocab_size)}
    values = []
    for row in rows:
        if row is None or row.keys() != expected:
            raise ValueError("Response must contain every vocabulary token ID exactly once")
        values.append([row[f"token_id:{token}"] for token in range(vocab_size)])
    logits = np.asarray(values, dtype=np.float32)
    # The completion serializer clamps scores below -9999; such values are not usable raw evidence.
    if not np.isfinite(logits).all() or (logits <= -9999).any():
        raise ValueError("Nonfinite or potentially clipped raw logits")
    return logits


def compare_logits(a: np.ndarray, b: np.ndarray) -> dict:
    if a.dtype != np.float32 or b.dtype != np.float32 or a.shape != b.shape or a.ndim != 2:
        raise ValueError("Logits must be matching two-dimensional FP32 arrays")
    if not a.size or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("Logits must be nonempty and finite")
    delta = a.astype(np.float64) - b.astype(np.float64)
    return {
        "steps": len(a),
        "vocab_size": a.shape[1],
        "values_compared": a.size,
        "bitwise_equal": bool(np.array_equal(a.view(np.uint32), b.view(np.uint32))),
        "max_abs_diff": float(np.abs(delta).max()),
        "normwise_rel_err": float(np.linalg.norm(delta) / max(np.linalg.norm(a.astype(np.float64)), 1e-12)),
    }

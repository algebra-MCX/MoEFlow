"""Numerical acceptance criteria.

* dispatch / restore (pure data movement): bitwise equality.
* expert GEMM path vs fp32 reference: normwise relative error
      err = ||out - ref||_2 / ||ref||_2
  must satisfy err <= max(K_FACTOR * err_vllm, FLOOR_ULPS * BF16_U), where err_vllm is the
  error of vLLM's native BF16 fused-MoE on the same inputs/routing, measured with the
  same metric, and BF16_U = 2^-8 is the bf16 unit roundoff.
* combine: deterministic accumulation -> repeated runs on identical input are bitwise equal.
"""

from __future__ import annotations

import torch


def bitwise_equal(a: torch.Tensor, b: torch.Tensor) -> bool:
    """Compare canonical tensor bytes, including signed zero; finiteness is a separate gate."""
    if a.shape != b.shape or a.dtype != b.dtype or a.device != b.device:
        return False
    return torch.equal(a.contiguous().reshape(-1).view(torch.uint8), b.contiguous().reshape(-1).view(torch.uint8))


BF16_U = 2.0**-8
FLOOR_ULPS = 4
K_FACTOR = 2.0


def normwise_rel_err(out: torch.Tensor, ref: torch.Tensor) -> float:
    ref64 = ref.double()
    denom = ref64.norm().item()
    num = (out.double() - ref64).norm().item()
    return num / denom if denom > 0 else num


def gemm_tolerance(err_vllm: float | None = None) -> float:
    floor = FLOOR_ULPS * BF16_U
    return floor if err_vllm is None else max(K_FACTOR * err_vllm, floor)


def calc_diff(x: torch.Tensor, y: torch.Tensor) -> float:
    """DeepEP's similarity metric (DeepEP tests/utils.py): 1 - 2<x+1, y+1> / (|x+1|^2 + |y+1|^2)."""
    x, y = x.double() + 1, y.double() + 1
    return float(1 - 2 * (x * y).sum() / (x * x + y * y).sum())

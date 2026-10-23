"""Expert compute backends over the expert-major layout (the "GEMM" axis of the comm x GEMM matrix)."""

from __future__ import annotations

from typing import Protocol

import torch


def swiglu_clamp(h: torch.Tensor, limit: float | None) -> torch.Tensor:
    """h [N, 2I] = [gate | up] -> [N, I], computed in fp32, returned in h.dtype."""
    inter = h.shape[-1] // 2
    gate, up = h[:, :inter].float(), h[:, inter:].float()
    if limit is not None:
        gate = gate.clamp(max=limit)
        up = up.clamp(min=-limit, max=limit)
    return (torch.nn.functional.silu(gate) * up).to(h.dtype)


class ExpertCompute(Protocol):
    name: str

    def __call__(
        self,
        a: torch.Tensor,
        expert_offsets: torch.Tensor,
        w13: torch.Tensor,
        w2: torch.Tensor,
        swiglu_limit: float | None,
        row_weight: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """a [N, H] expert-major; w13 [L, 2I, H]; w2 [L, H, I]. Returns [N, H] in a.dtype.

        row_weight [N] fp32, optional: routing weight applied to the fp32 down-projection
        accumulator before the single cast (the branch then is bf16(w * acc)).
        """
        ...


class TorchLoopExperts:
    """One cuBLAS GEMM pair per non-empty local expert. Needs the offsets on host (D2H sync)."""

    name = "torch_loop"

    def __call__(self, a, expert_offsets, w13, w2, swiglu_limit, row_weight=None):
        out = torch.empty_like(a)
        offs = expert_offsets.tolist()
        for e in range(len(offs) - 1):
            s, t = offs[e], offs[e + 1]
            if s == t:
                continue
            h = a[s:t] @ w13[e].t()
            act = swiglu_clamp(h, swiglu_limit)
            if row_weight is None:
                out[s:t] = act @ w2[e].t()
            else:  # fp32 accumulate, weight in fp32, one cast
                y = torch.matmul(act.float(), w2[e].float().t())
                out[s:t] = (y * row_weight[s:t, None]).to(a.dtype)
        return out


class TritonGroupedExperts:
    """Two Triton grouped GEMMs with an unfused activation in between. No host sync."""

    name = "triton_grouped"

    def __init__(self, **gemm_kwargs):
        self.gemm_kwargs = gemm_kwargs

    def __call__(self, a, expert_offsets, w13, w2, swiglu_limit, row_weight=None):
        from .kernels.grouped_gemm import grouped_gemm

        h = grouped_gemm(a, w13, expert_offsets, **self.gemm_kwargs)
        act = swiglu_clamp(h, swiglu_limit)
        return grouped_gemm(act, w2, expert_offsets, row_scale=row_weight, **self.gemm_kwargs)


EXPERT_BACKENDS = {"torch_loop": TorchLoopExperts, "triton_grouped": TritonGroupedExperts}

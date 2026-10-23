"""Deterministic synthetic inputs: weights, hidden states and routing distributions."""

from __future__ import annotations

import math

import torch

from .config import MoEConfig

DISTRIBUTIONS = ("uniform", "zipf", "hotspot")


def make_weights(cfg: MoEConfig, seed: int, device="cpu", dtype=torch.bfloat16):
    """w13 [E, 2I, H], w2 [E, H, I], scaled so activations stay O(1) and the swiglu clamp is exercised."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    E, H, I = cfg.n_routed_experts, cfg.hidden_size, cfg.moe_intermediate_size
    w13 = torch.randn(E, 2 * I, H, generator=g) * (2.0 / math.sqrt(H))
    w2 = torch.randn(E, H, I, generator=g) * (1.0 / math.sqrt(I))
    return w13.to(device=device, dtype=dtype), w2.to(device=device, dtype=dtype)


def make_hidden(num_tokens: int, cfg: MoEConfig, seed: int, device="cpu", dtype=torch.bfloat16):
    g = torch.Generator(device="cpu").manual_seed(seed)
    return torch.randn(num_tokens, cfg.hidden_size, generator=g).to(device=device, dtype=dtype)


def make_logits(
    num_tokens: int,
    cfg: MoEConfig,
    seed: int,
    dist: str = "uniform",
    device="cpu",
    zipf_s: float = 1.2,
    hot_experts: int | None = None,
):
    """Router logits [T, E] fp32 whose top-k selection follows the requested popularity profile.

    uniform: i.i.d. logits. zipf: expert e gets a log-popularity offset -s*log(rank_e).
    hotspot: the first `hot_experts` (default top_k) experts dominate every token.
    """
    g = torch.Generator(device="cpu").manual_seed(seed)
    E = cfg.n_routed_experts
    logits = torch.randn(num_tokens, E, generator=g)
    if dist == "zipf":
        perm = torch.randperm(E, generator=g)
        ranks = torch.empty(E)
        ranks[perm] = torch.arange(1, E + 1, dtype=torch.float32)
        logits = logits + (-zipf_s * torch.log(ranks)) * 2.0
    elif dist == "hotspot":
        n = hot_experts or cfg.top_k
        logits[:, :n] += 20.0
        logits[:, n:] -= 20.0  # keep cold scores below hot scores after the synthetic correction bias
    elif dist != "uniform":
        raise ValueError(dist)
    return logits.to(device=device, dtype=torch.float32)


def make_bias(cfg: MoEConfig, seed: int, device="cpu"):
    g = torch.Generator(device="cpu").manual_seed(seed)
    return (torch.randn(cfg.n_routed_experts, generator=g) * 0.05).to(device=device, dtype=torch.float32)


class ScaleExperts:
    """Hand-checkable expert backend: global expert g maps x -> (g + 1) * x (bit-exact for small dyadic inputs)."""

    name = "scale"

    def __init__(self, rank_expert_offset: int):
        self.offset = rank_expert_offset

    def __call__(self, a, expert_offsets, w13, w2, swiglu_limit, row_weight=None):
        out = torch.empty_like(a)
        offs = expert_offsets.tolist()
        for e in range(len(offs) - 1):
            y = a[offs[e] : offs[e + 1]].float() * (self.offset + e + 1)
            if row_weight is not None:
                y = y * row_weight[offs[e] : offs[e + 1], None]
            out[offs[e] : offs[e + 1]] = y.to(a.dtype)
        return out

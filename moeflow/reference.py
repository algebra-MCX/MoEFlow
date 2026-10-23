"""Naive reference for the MoE layer boundary defined in docs/architecture.md.

Boundary: inputs are hidden states x [T, H] (bf16), router logits [T, E] (fp32) and the
noaux_tc correction bias [E] (fp32); output is the weighted sum of routed experts,
including routed_scaling_factor, cast to bf16. Gate GEMM, shared expert, mHC and norms
are outside the boundary.

Deliberately written token-by-token and expert-by-expert with plain Python loops. It must
not import or share index logic with the code under test (moeflow.router / moeflow.dispatch).
"""

from __future__ import annotations

import math

import torch

from .config import MoEConfig


def route_token(logits: torch.Tensor, bias: torch.Tensor | None, cfg: MoEConfig) -> list[tuple[int, float]]:
    """Route one token. Returns [(expert_id, weight)] in selection order (highest biased score first).

    Semantics (DeepSeek-V3 noaux_tc, as implemented by vLLM grouped_topk):
      score_e  = sigmoid(logit_e)
      biased_e = score_e + bias_e                       (selection only)
      groups ranked by sum of their top-2 biased scores; keep topk_group groups
      experts ranked by biased score inside kept groups; keep top_k
      weight_e = score_e (unbiased); renormalized (sum + 1e-20) if norm_topk_prob; times routed_scaling_factor
    Ties are broken by the lower expert id (vLLM's kernels do not specify tie order).
    """
    E = logits.numel()
    vals = [float(v) for v in logits.double().cpu()]
    b = [0.0] * E if bias is None else [float(v) for v in bias.double().cpu()]
    if cfg.scoring_func == "sigmoid":
        scores = [1.0 / (1.0 + math.exp(-v)) for v in vals]
    elif cfg.scoring_func == "softmax":
        m = max(vals)
        ex = [math.exp(v - m) for v in vals]
        s = sum(ex)
        scores = [v / s for v in ex]
    else:
        raise ValueError(cfg.scoring_func)
    biased = [scores[e] + b[e] for e in range(E)]

    group_size = E // cfg.n_group
    group_rank = []
    for g in range(cfg.n_group):
        members = sorted(biased[g * group_size : (g + 1) * group_size], reverse=True)
        gscore = members[0] + members[1] if bias is not None else members[0]
        group_rank.append((-gscore, g))
    kept_groups = {g for _, g in sorted(group_rank)[: cfg.topk_group]}

    candidates = [(-biased[e], e) for e in range(E) if e // group_size in kept_groups]
    chosen = [e for _, e in sorted(candidates)[: cfg.top_k]]

    weights = [scores[e] for e in chosen]
    if cfg.norm_topk_prob:
        s = sum(weights) + 1e-20  # epsilon as in the HF reference (Glm5NextTextTopkRouter)
        weights = [w / s for w in weights]
    weights = [w * cfg.routed_scaling_factor for w in weights]
    return list(zip(chosen, weights))


def swiglu_clamp(gate: torch.Tensor, up: torch.Tensor, limit: float | None) -> torch.Tensor:
    if limit is not None:
        gate = torch.clamp(gate, max=limit)
        up = torch.clamp(up, min=-limit, max=limit)
    return gate * torch.sigmoid(gate) * up


def expert_forward(
    x: torch.Tensor, w13: torch.Tensor, w2: torch.Tensor, limit: float | None, acc_dtype: torch.dtype = torch.float32
) -> torch.Tensor:
    """One token through one expert. x [H]; w13 [2I, H] = [gate; up]; w2 [H, I]. Returns [H] in acc_dtype."""
    inter = w2.shape[1]
    h = w13.to(acc_dtype) @ x.to(acc_dtype)
    a = swiglu_clamp(h[:inter], h[inter:], limit)
    return w2.to(acc_dtype) @ a


def moe_reference(
    x: torch.Tensor,
    logits: torch.Tensor,
    bias: torch.Tensor | None,
    w13: torch.Tensor,
    w2: torch.Tensor,
    cfg: MoEConfig,
    acc_dtype: torch.dtype = torch.float32,
    routes: list[list[tuple[int, float]]] | None = None,
) -> torch.Tensor:
    """Full layer reference. w13 [E, 2I, H], w2 [E, H, I]. Returns [T, H] in x.dtype.

    If `routes` is given it overrides routing (used to evaluate expert/combine paths on
    exactly the same expert choice as the implementation under test).
    """
    T = x.shape[0]
    out = torch.zeros(T, x.shape[1], dtype=acc_dtype, device=x.device)
    for t in range(T):
        r = routes[t] if routes is not None else route_token(logits[t], bias, cfg)
        acc = torch.zeros(x.shape[1], dtype=acc_dtype, device=x.device)
        for e, w in r:
            acc = acc + w * expert_forward(x[t], w13[e], w2[e], cfg.swiglu_limit, acc_dtype)
        out[t] = acc
    return out.to(x.dtype)


def route_reference(logits: torch.Tensor, bias: torch.Tensor | None, cfg: MoEConfig) -> list[list[tuple[int, float]]]:
    return [route_token(logits[t], bias, cfg) for t in range(logits.shape[0])]


def ep_dispatch_reference(routes_per_rank: list[list[list[tuple[int, float]]]], experts_per_rank: int):
    """What each destination rank must receive under the one-copy-per-(token, rank) contract.

    routes_per_rank[src][t] = [(expert, weight)] for token t owned by rank `src`.
    Returns recv[dst] = list of (src, t, [(local_expert, weight) in top-k slot order]),
    ordered by src ascending, then t ascending.
    """
    world = len(routes_per_rank)
    recv: list[list] = [[] for _ in range(world)]
    for src in range(world):
        for t, route in enumerate(routes_per_rank[src]):
            for dst in range(world):
                hits = [(e - dst * experts_per_rank, w) for e, w in route if e // experts_per_rank == dst]
                if hits:
                    recv[dst].append((src, t, hits))
    return recv

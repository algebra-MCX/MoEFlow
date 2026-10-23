"""Vectorized noaux_tc router (implementation under test; semantics fixed by moeflow.reference.route_token)."""

from __future__ import annotations

import torch

from .config import MoEConfig


def route(logits: torch.Tensor, bias: torch.Tensor | None, cfg: MoEConfig) -> tuple[torch.Tensor, torch.Tensor]:
    """logits [T, E] -> (topk_weights [T, K] fp32, topk_ids [T, K] int32).

    Slot order is descending biased score. Ties resolve to the lower expert id because
    the stable sort keeps index order among equal keys.
    """
    T, E = logits.shape
    logits = logits.float()
    if cfg.scoring_func == "sigmoid":
        scores = torch.sigmoid(logits)
    elif cfg.scoring_func == "softmax":
        scores = torch.softmax(logits, dim=-1)
    else:
        raise ValueError(cfg.scoring_func)
    biased = scores if bias is None else scores + bias.float().unsqueeze(0)

    if cfg.n_group > 1:
        grouped = biased.view(T, cfg.n_group, E // cfg.n_group)
        if bias is not None:
            gscore = grouped.topk(2, dim=-1).values.sum(-1)
        else:
            gscore = grouped.max(-1).values
        gorder = torch.sort(gscore, dim=-1, descending=True, stable=True).indices[:, : cfg.topk_group]
        gmask = torch.zeros_like(gscore, dtype=torch.bool).scatter_(1, gorder, True)
        emask = gmask.unsqueeze(-1).expand_as(grouped).reshape(T, E)
        biased = biased.masked_fill(~emask, float("-inf"))

    ids = torch.sort(biased, dim=-1, descending=True, stable=True).indices[:, : cfg.top_k]
    weights = scores.gather(1, ids)
    if cfg.norm_topk_prob:
        weights = weights / (weights.sum(-1, keepdim=True) + 1e-20)  # HF reference epsilon
    weights = weights * cfg.routed_scaling_factor
    return weights.float(), ids.to(torch.int32)

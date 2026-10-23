"""Shared BF16 weight loading for layer measurements and paired comparisons."""

from __future__ import annotations

from moeflow.config import MoEConfig
from moeflow.testing import make_bias, make_weights

SMALL = MoEConfig(hidden_size=512, moe_intermediate_size=256, n_routed_experts=32, top_k=4)


def load_weights(spec: str, cfg: MoEConfig, device):
    if spec == "synthetic":
        w13, w2 = make_weights(cfg, 0, device=device)
        return w13, w2, make_bias(cfg, 1, device=device)
    if not spec.startswith("real:"):
        raise ValueError(f"Expected synthetic or real:<layer>, got {spec!r}")
    from moeflow.weights import CheckpointReader

    layer = int(spec.split(":")[1])
    reader = CheckpointReader()
    w13, w2 = reader.expert_weights(layer, range(cfg.n_routed_experts))
    _, bias = reader.router(layer)
    return w13.to(device), w2.to(device), bias.to(device)

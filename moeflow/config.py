"""MoE layer configuration for GLM-5.3-Flash (values verified against config.json, see docs/env.md)."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class MoEConfig:
    hidden_size: int = 4096
    moe_intermediate_size: int = 2048
    n_routed_experts: int = 288
    n_shared_experts: int = 1
    top_k: int = 8
    n_group: int = 1
    topk_group: int = 1
    norm_topk_prob: bool = True
    routed_scaling_factor: float = 2.5
    scoring_func: str = "sigmoid"
    topk_method: str = "noaux_tc"
    # SwiGLU clamp: gate clamped from above only, up clamped to [-limit, limit].
    swiglu_limit: float | None = 10.0

    def experts_per_rank(self, ep_size: int) -> int:
        if self.n_routed_experts % ep_size:
            raise ValueError(f"{self.n_routed_experts} experts not divisible by ep_size={ep_size}")
        return self.n_routed_experts // ep_size

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_hf_config(cls, path: str | Path) -> MoEConfig:
        """Build from a HuggingFace config.json (reads text_config when present)."""
        cfg = json.loads(Path(path).read_text())
        t = cfg.get("text_config", cfg)
        return cls(
            hidden_size=t["hidden_size"],
            moe_intermediate_size=t["moe_intermediate_size"],
            n_routed_experts=t["n_routed_experts"],
            n_shared_experts=t["n_shared_experts"],
            top_k=t["num_experts_per_tok"],
            n_group=t["n_group"],
            topk_group=t["topk_group"],
            norm_topk_prob=t["norm_topk_prob"],
            routed_scaling_factor=float(t["routed_scaling_factor"]),
            scoring_func=t["scoring_func"],
            topk_method=t["topk_method"],
            swiglu_limit=t.get("swiglu_limit"),
        )


GLM53_FLASH = MoEConfig()

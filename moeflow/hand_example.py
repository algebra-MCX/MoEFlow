"""The 2-rank / 4-expert / non-uniform routing example of docs/architecture.md §7, executed by the real stages."""

from __future__ import annotations

import torch

from . import dispatch as D
from .comm import simulated_exchange, unpack_meta
from .config import MoEConfig
from .ep_moe import EPMoELayer
from .testing import ScaleExperts

CFG = MoEConfig(hidden_size=2, moe_intermediate_size=1, n_routed_experts=4, top_k=2)
WORLD = 2
X = [torch.tensor([[1.0, 2.0], [3.0, -1.0], [2.0, 0.5]]), torch.tensor([[0.5, 4.0]])]
TOPK_IDS = [torch.tensor([[0, 1], [1, 3], [2, 3]], dtype=torch.int32), torch.tensor([[3, 0]], dtype=torch.int32)]
TOPK_W = [torch.tensor([[1.5, 1.0], [2.0, 0.5], [1.25, 1.25]]), torch.tensor([[2.0, 0.5]])]


def run(dtype=torch.bfloat16) -> dict:
    L = CFG.experts_per_rank(WORLD)
    dummy = torch.zeros(L, 1, 1)
    layers = [
        EPMoELayer(CFG, r, WORLD, dummy, dummy, experts=ScaleExperts(r * L), combine_dtype=dtype) for r in range(WORLD)
    ]
    xs = [x.to(dtype) for x in X]
    states = [ly.stage_dispatch(x, w, i) for ly, x, w, i in zip(layers, xs, TOPK_W, TOPK_IDS)]
    counts = [s.plan.send_counts for s in states]
    recv_x, recv_counts = simulated_exchange([s.send_x for s in states], counts)
    recv_meta, _ = simulated_exchange([s.send_meta for s in states], counts)
    layouts = [D.build_expert_layout(unpack_meta(m)[0], L) for m in recv_meta]
    partials = [ly.stage_experts(rx, rm) for ly, rx, rm in zip(layers, recv_x, recv_meta)]
    back, _ = simulated_exchange(partials, recv_counts)
    outs = [ly.stage_combine(b, s, dtype) for ly, b, s in zip(layers, back, states)]
    return {
        "send_counts": [c.tolist() for c in counts],
        "send_token": [s.plan.send_token.tolist() for s in states],
        "send_slots": [unpack_meta(s.send_meta)[0].tolist() for s in states],
        "recv_counts": [c.tolist() for c in recv_counts],
        "recv_x": [r.float().tolist() for r in recv_x],
        "recv_slots": [unpack_meta(m)[0].tolist() for m in recv_meta],
        "recv_weights": [unpack_meta(m)[1].tolist() for m in recv_meta],
        "expert_offsets": [ly.expert_offsets.tolist() for ly in layouts],
        "expert_entries": [list(zip(ly.entry_row.tolist(), ly.entry_slot.tolist())) for ly in layouts],
        "partials": [p.float().tolist() for p in partials],
        "returned": [b.float().tolist() for b in back],
        "out": [o.float().tolist() for o in outs],
    }


if __name__ == "__main__":
    import json

    for k, v in run().items():
        print(f"{k}: {json.dumps(v)}")

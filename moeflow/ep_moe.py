"""EP MoE pipeline: route -> rank-packed dispatch -> all-to-all -> expert-major layout ->
grouped GEMM -> local weighted reduce -> return all-to-all -> combine.

Every stage except the two exchanges is local, so the same stage code is driven either
by a real transport (`EPMoELayer.forward`) or by the single-process simulator
(`simulate_ep`), which is how the dispatch/combine logic is tested without GPUs.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from . import dispatch as D
from .comm import RowExchange, pack_meta, simulated_exchange, unpack_meta
from .config import MoEConfig
from .experts import ExpertCompute, TorchLoopExperts
from .router import route


@dataclass
class DispatchState:
    plan: D.DispatchPlan
    send_x: torch.Tensor
    send_meta: torch.Tensor
    num_tokens: int


class EPMoELayer:
    """Holds the local expert shard of one EP rank. w13 [L, 2I, H], w2 [L, H, I]."""

    def __init__(
        self,
        cfg: MoEConfig,
        rank: int,
        world: int,
        w13_local: torch.Tensor,
        w2_local: torch.Tensor,
        experts: ExpertCompute | None = None,
        combine_dtype: torch.dtype = torch.bfloat16,
    ):
        self.cfg, self.rank, self.world = cfg, rank, world
        self.num_local = cfg.experts_per_rank(world)
        assert w13_local.shape[0] == self.num_local and w2_local.shape[0] == self.num_local
        self.w13, self.w2 = w13_local, w2_local
        self.experts = experts or TorchLoopExperts()
        # dtype of the partial rows on the return path (bf16 halves bytes; fp32 is exact transport).
        self.combine_dtype = combine_dtype

    # -- local stages ---------------------------------------------------------------------------
    def stage_dispatch(self, x: torch.Tensor, topk_weights: torch.Tensor, topk_ids: torch.Tensor) -> DispatchState:
        plan = D.plan_dispatch(topk_ids, topk_weights, self.world, self.num_local)
        return DispatchState(plan, D.pack_send(x, plan), pack_meta(plan.slot_local_id, plan.slot_weight), x.shape[0])

    def stage_experts(self, recv_x: torch.Tensor, recv_meta: torch.Tensor) -> torch.Tensor:
        local_id, weight = unpack_meta(recv_meta)
        layout = D.build_expert_layout(local_id, self.num_local)
        a = D.gather_expert_input(recv_x, layout)
        y = self.experts(
            a, layout.expert_offsets, self.w13, self.w2, self.cfg.swiglu_limit, D.entry_weights(layout, weight)
        )
        return D.reduce_local(y, layout, weight, self.combine_dtype)

    def stage_combine(self, partials: torch.Tensor, state: DispatchState, out_dtype: torch.dtype) -> torch.Tensor:
        return D.combine(partials, state.plan, state.num_tokens, self.world, out_dtype)

    # -- SPMD entry point -------------------------------------------------------------------------
    def forward(
        self, x: torch.Tensor, router_logits: torch.Tensor, bias: torch.Tensor | None, comm: RowExchange
    ) -> torch.Tensor:
        topk_weights, topk_ids = route(router_logits, bias, self.cfg)
        return self.forward_routed(x, topk_weights, topk_ids, comm)

    def forward_routed(self, x, topk_weights, topk_ids, comm: RowExchange) -> torch.Tensor:
        st = self.stage_dispatch(x, topk_weights, topk_ids)
        recv_counts = comm.exchange_counts(st.plan.send_counts)
        recv_x = comm.exchange(st.send_x, st.plan.send_counts, recv_counts)
        recv_meta = comm.exchange(st.send_meta, st.plan.send_counts, recv_counts)
        partials = self.stage_experts(recv_x, recv_meta)
        back = comm.exchange(partials, recv_counts, st.plan.send_counts)
        return self.stage_combine(back, st, x.dtype)


def simulate_ep(
    layers: list[EPMoELayer], xs: list[torch.Tensor], logits: list[torch.Tensor], bias: torch.Tensor | None
) -> list[torch.Tensor]:
    """Run all ranks' stages in lockstep inside one process."""
    routed = [route(lg, bias, layers[0].cfg) for lg in logits]
    states = [ly.stage_dispatch(x, w, i) for ly, x, (w, i) in zip(layers, xs, routed)]
    send_counts = [s.plan.send_counts for s in states]
    recv_x, recv_counts = simulated_exchange([s.send_x for s in states], send_counts)
    recv_meta, _ = simulated_exchange([s.send_meta for s in states], send_counts)
    partials = [ly.stage_experts(rx, rm) for ly, rx, rm in zip(layers, recv_x, recv_meta)]
    back, _ = simulated_exchange(partials, recv_counts)
    return [ly.stage_combine(b, s, x.dtype) for ly, b, s, x in zip(layers, back, states, xs)]


def shard_experts(w13: torch.Tensor, w2: torch.Tensor, rank: int, world: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Contiguous expert placement: rank r owns experts [r*L, (r+1)*L)."""
    L = w13.shape[0] // world
    return w13[rank * L : (rank + 1) * L], w2[rank * L : (rank + 1) * L]

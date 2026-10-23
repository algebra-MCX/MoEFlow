"""Local (per-rank) stages of the EP pipeline. All functions are communication-free.

Contract (docs/architecture.md §3):
  * Token ownership: each rank owns the tokens of its DP attention shard.
  * Expert placement: contiguous blocks, rank r owns experts [r*L, (r+1)*L), L = E / ep.
  * One copy per (token, destination rank): a token hitting several experts on the same
    rank is sent once, carrying K metadata slots (local expert id or -1, weight).
  * Send order: destination-rank major, then source token index ascending.
  * The destination applies routing weights and reduces its local hits per row in slot
    order (k = 0..K-1) with an fp32 accumulator, then returns one partial row per received
    row (rounding of the weighted branch: see `reduce_local`).
  * The source sums partials over destination ranks in ascending rank order, in fp32.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

INVALID = -1


@dataclass
class DispatchPlan:
    """Source-side plan. Rows of the send buffer are (dst rank, token) pairs."""

    send_token: torch.Tensor  # [S] int64, owning-token index of each send row
    send_rank: torch.Tensor  # [S] int64, destination rank of each send row (non-decreasing)
    send_counts: torch.Tensor  # [world] int64, rows per destination rank
    slot_local_id: torch.Tensor  # [S, K] int32, local expert id on dst, or INVALID
    slot_weight: torch.Tensor  # [S, K] fp32, routing weight (0 where INVALID)


def plan_dispatch(
    topk_ids: torch.Tensor, topk_weights: torch.Tensor, world: int, experts_per_rank: int
) -> DispatchPlan:
    T, K = topk_ids.shape
    ids = topk_ids.long()
    dst = ids // experts_per_rank  # [T, K]
    hit = torch.zeros(T, world, dtype=torch.bool, device=ids.device)
    hit.scatter_(1, dst, True)  # [T, world]
    # nonzero() on the transposed mask yields (rank, token) pairs in rank-major, token-ascending order.
    pairs = hit.t().nonzero()
    send_rank, send_token = pairs[:, 0], pairs[:, 1]
    send_counts = hit.sum(0)

    on_rank = dst[send_token] == send_rank.unsqueeze(1)  # [S, K]
    local = (ids[send_token] - send_rank.unsqueeze(1) * experts_per_rank).to(torch.int32)
    slot_local_id = torch.where(on_rank, local, torch.full_like(local, INVALID))
    slot_weight = torch.where(on_rank, topk_weights[send_token].float(), torch.zeros((), device=ids.device))
    return DispatchPlan(send_token, send_rank, send_counts, slot_local_id, slot_weight)


def pack_send(x: torch.Tensor, plan: DispatchPlan) -> torch.Tensor:
    return x.index_select(0, plan.send_token)


@dataclass
class ExpertLayout:
    """Destination-side expert-major layout over (received row, slot) entries."""

    entry_row: torch.Tensor  # [N] int64, received row of each entry, expert-major order
    entry_slot: torch.Tensor  # [N] int64, top-k slot of each entry
    expert_offsets: torch.Tensor  # [L + 1] int64, entries of local expert e are [off[e], off[e+1])


def build_expert_layout(recv_local_id: torch.Tensor, num_local_experts: int) -> ExpertLayout:
    R, K = recv_local_id.shape
    flat = recv_local_id.reshape(-1).long()
    valid = (flat != INVALID).nonzero().squeeze(1)  # ascending (row, slot)
    experts = flat[valid]
    # Stable sort keeps (row, slot) ascending within each expert -> deterministic layout.
    order = torch.sort(experts, stable=True).indices
    entries = valid[order]
    counts = torch.bincount(experts, minlength=num_local_experts)
    offsets = torch.zeros(num_local_experts + 1, dtype=torch.int64, device=flat.device)
    offsets[1:] = torch.cumsum(counts, 0)
    return ExpertLayout(entries // K, entries % K, offsets)


def gather_expert_input(recv_x: torch.Tensor, layout: ExpertLayout) -> torch.Tensor:
    return recv_x.index_select(0, layout.entry_row)


def entry_weights(layout: ExpertLayout, recv_weight: torch.Tensor) -> torch.Tensor:
    """Routing weight of every expert-major entry, [N] fp32."""
    return recv_weight.float()[layout.entry_row, layout.entry_slot]


def reduce_local(
    expert_out: torch.Tensor, layout: ExpertLayout, recv_weight: torch.Tensor, out_dtype: torch.dtype
) -> torch.Tensor:
    """Per-row reduction over local hits in slot order k = 0..K-1 with an fp32 accumulator.

    expert_out already holds bf16(w * acc_fp32), the routing weight applied in the down-GEMM
    epilogue as vLLM's Triton experts do; only the sum happens here. Invalid slots are skipped.
    Accumulators start at +0.0 like vLLM's moe_sum and DeepEP.
    """
    R, K = recv_weight.shape
    H = expert_out.shape[1]
    slots = torch.zeros(R, K, H, dtype=expert_out.dtype, device=expert_out.device)
    slots[layout.entry_row, layout.entry_slot] = expert_out
    present = torch.zeros(R, K, dtype=torch.bool, device=expert_out.device)
    present[layout.entry_row, layout.entry_slot] = True
    acc = torch.zeros(R, H, dtype=torch.float32, device=expert_out.device)
    for k in range(K):  # fixed order: deterministic, no atomics
        acc = torch.where(present[:, k : k + 1], acc + slots[:, k].float(), acc)
    return acc.to(out_dtype)


def combine(
    partials: torch.Tensor, plan: DispatchPlan, num_tokens: int, world: int, out_dtype: torch.dtype
) -> torch.Tensor:
    """Sum returned partial rows per owning token in ascending destination-rank order."""
    H = partials.shape[1]
    per_rank = torch.zeros(num_tokens, world, H, dtype=torch.float32, device=partials.device)
    per_rank[plan.send_token, plan.send_rank] = partials.float()
    present = torch.zeros(num_tokens, world, dtype=torch.bool, device=partials.device)
    present[plan.send_token, plan.send_rank] = True
    out = torch.zeros(num_tokens, H, dtype=torch.float32, device=partials.device)
    for r in range(world):  # ascending rank order, like DeepEP's intranode combine
        out = torch.where(present[:, r : r + 1], out + per_rank[:, r], out)
    return out.to(out_dtype)

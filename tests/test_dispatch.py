"""Dispatch/restore are pure data movement: bitwise checks against the naive reference."""

import pytest
import torch

from moeflow import dispatch as D
from moeflow.comm import pack_meta, simulated_exchange, unpack_meta
from moeflow.reference import ep_dispatch_reference


def _routes(ids, w):
    return [[(int(e), float(x)) for e, x in zip(ids[t].tolist(), w[t].tolist())] for t in range(ids.shape[0])]


def _random_topk(T, E, K, gen):
    ids = (
        torch.stack([torch.randperm(E, generator=gen)[:K] for _ in range(T)])
        if T
        else torch.empty(0, K, dtype=torch.long)
    )
    return ids.to(torch.int32), torch.rand(T, K, generator=gen)


CASES = {
    "uniform": lambda T, E, K, g: _random_topk(T, E, K, g),
    "same_experts": lambda T, E, K, g: (torch.arange(K, dtype=torch.int32).repeat(T, 1), torch.rand(T, K, generator=g)),
    "one_rank_only": lambda T, E, K, g: (
        torch.stack([torch.randperm(E // 4, generator=g)[:K] for _ in range(T)]).to(torch.int32)
        if T
        else torch.empty(0, K, dtype=torch.int32),
        torch.rand(T, K, generator=g),
    ),
}


@pytest.mark.parametrize("world", [1, 2, 4, 8])
@pytest.mark.parametrize("case", list(CASES))
@pytest.mark.parametrize("tokens_per_rank", [[0, 3, 1, 5, 2, 0, 7, 4], [1] * 8, [16] * 8])
def test_dispatch_matches_reference(world, case, tokens_per_rank):
    E, K, H = 32, 4, 8
    L = E // world
    gen = torch.Generator().manual_seed(world * 100 + len(case))
    per_rank = [CASES[case](tokens_per_rank[r], E, K, gen) for r in range(world)]
    xs = [torch.randn(tokens_per_rank[r], H, generator=gen).to(torch.bfloat16) for r in range(world)]

    plans = [D.plan_dispatch(ids, w, world, L) for ids, w in per_rank]
    sends = [D.pack_send(x, p) for x, p in zip(xs, plans)]
    metas = [pack_meta(p.slot_local_id, p.slot_weight) for p in plans]
    recv_x, recv_counts = simulated_exchange(sends, [p.send_counts for p in plans])
    recv_meta, _ = simulated_exchange(metas, [p.send_counts for p in plans])

    ref = ep_dispatch_reference([_routes(ids, w) for ids, w in per_rank], L)
    for dst in range(world):
        assert recv_x[dst].shape[0] == len(ref[dst])
        local_id, weight = unpack_meta(recv_meta[dst])
        for row, (src, t, hits) in enumerate(ref[dst]):
            assert torch.equal(recv_x[dst][row], xs[src][t])
            got = [(int(i), float(wt)) for i, wt in zip(local_id[row].tolist(), weight[row].tolist()) if i != D.INVALID]
            assert got == [(e, float(torch.tensor(w, dtype=torch.float32))) for e, w in hits]
        assert recv_counts[dst].tolist() == [sum(1 for s, _, _ in ref[dst] if s == src) for src in range(world)]


def test_expert_layout_is_expert_major_and_complete():
    local_id = torch.tensor([[2, -1, 0], [0, 0, -1], [-1, -1, -1], [1, 2, 2]], dtype=torch.int32)
    lay = D.build_expert_layout(local_id, num_local_experts=3)
    assert lay.expert_offsets.tolist() == [0, 3, 4, 7]
    pairs = list(zip(lay.entry_row.tolist(), lay.entry_slot.tolist()))
    assert pairs == [(0, 2), (1, 0), (1, 1), (3, 0), (0, 0), (3, 1), (3, 2)]


def test_restore_roundtrip_bitwise():
    """Identity experts with weight 1 on a single hit: combine must return the input bit-exactly."""
    world, L, H, T = 4, 2, 16, 9
    gen = torch.Generator().manual_seed(7)
    ids = torch.randint(0, world * L, (T, 1), generator=gen, dtype=torch.int32)
    x = torch.randn(T, H, generator=gen).to(torch.bfloat16)
    plan = D.plan_dispatch(ids, torch.ones(T, 1), world, L)
    back = D.pack_send(x, plan)  # what the identity expert would return, in send order
    out = D.combine(back, plan, T, world, torch.bfloat16)
    assert torch.equal(out, x)


def _layout_for(local_id):
    return D.build_expert_layout(local_id, num_local_experts=int(local_id.max()) + 1)


def test_reduce_local_ignores_garbage_weight_in_invalid_slot():
    local_id = torch.tensor([[0, -1]], dtype=torch.int32)
    lay = _layout_for(local_id)
    y = torch.tensor([[3.0]], dtype=torch.bfloat16)
    out = D.reduce_local(y, lay, torch.tensor([[0.5, float("nan")]]), torch.bfloat16)
    assert out.item() == 3.0


def test_accumulators_start_at_positive_zero_like_deepep():
    """DeepEP/moe_sum start from +0.0, so an all-(-0.0) sum yields +0.0; we must match."""
    local_id = torch.tensor([[0]], dtype=torch.int32)
    out = D.reduce_local(
        torch.tensor([[-0.0]], dtype=torch.bfloat16),
        _layout_for(local_id),
        torch.tensor([[1.0]]),
        torch.bfloat16,
    )
    assert not torch.signbit(out).item()
    plan = D.plan_dispatch(torch.tensor([[1]], dtype=torch.int32), torch.ones(1, 1), world=2, experts_per_rank=1)
    out = D.combine(torch.tensor([[-0.0]], dtype=torch.bfloat16), plan, 1, 2, torch.bfloat16)
    assert not torch.signbit(out).item()


def test_plan_dispatch_empty_input():
    plan = D.plan_dispatch(torch.empty(0, 4, dtype=torch.int32), torch.empty(0, 4), world=4, experts_per_rank=2)
    assert plan.send_counts.tolist() == [0, 0, 0, 0] and plan.send_token.numel() == 0


def test_reduce_local_only_sums_preweighted_branches():
    local_id = torch.tensor([[1, -1, 0]], dtype=torch.int32)
    lay = _layout_for(local_id)  # expert-major: e0 <- slot 2, e1 <- slot 0
    y = torch.tensor([[0.25], [3.0]], dtype=torch.bfloat16)  # already weighted
    w = torch.tensor([[7.0, float("nan"), 9.0]])  # must be ignored
    assert D.reduce_local(y, lay, w, torch.float32).item() == 3.25
    assert D.entry_weights(lay, w).tolist() == [9.0, 7.0]

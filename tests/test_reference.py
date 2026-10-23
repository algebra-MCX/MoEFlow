"""Self-checks of the naive reference against closed-form cases."""

import math

import torch

from moeflow.config import MoEConfig
from moeflow.reference import ep_dispatch_reference, expert_forward, moe_reference, route_token, swiglu_clamp


def test_route_token_sigmoid_bias_renorm_scale():
    cfg = MoEConfig(n_routed_experts=4, top_k=2, routed_scaling_factor=2.5)
    logits = torch.tensor([0.0, 1.0, -1.0, 2.0])
    bias = torch.tensor([0.0, 0.0, 0.9, 0.0])  # bias lifts expert 2 into the selection, not its weight
    r = route_token(logits, bias, cfg)
    s = [1 / (1 + math.exp(-v)) for v in (0.0, 1.0, -1.0, 2.0)]
    # biased: e0 .5, e1 .731, e2 .269+.9=1.169, e3 .881 -> select e2, e3
    assert [e for e, _ in r] == [2, 3]
    tot = s[2] + s[3]
    assert math.isclose(r[0][1], 2.5 * s[2] / tot, rel_tol=1e-12)
    assert math.isclose(r[1][1], 2.5 * s[3] / tot, rel_tol=1e-12)
    assert math.isclose(sum(w for _, w in r), 2.5, rel_tol=1e-12)


def test_route_token_tie_breaks_to_lower_id():
    cfg = MoEConfig(n_routed_experts=4, top_k=2)
    r = route_token(torch.zeros(4), None, cfg)
    assert [e for e, _ in r] == [0, 1]


def test_route_token_group_limited():
    cfg = MoEConfig(n_routed_experts=8, top_k=2, n_group=4, topk_group=1)
    logits = torch.tensor([3.0, -9.0, 2.5, 2.5, 0.0, 0.0, 0.0, 0.0])
    # group scores (top-2 sum of biased=score): g0 = s(3)+s(-9), g1 = 2*s(2.5) -> g1 wins
    r = route_token(logits, torch.zeros(8), cfg)
    assert sorted(e for e, _ in r) == [2, 3]


def test_swiglu_clamp_is_asymmetric():
    gate = torch.tensor([-20.0, 20.0])
    up = torch.tensor([-20.0, 20.0])
    out = swiglu_clamp(gate, up, 10.0)
    # gate is clamped only from above, up on both sides
    assert torch.allclose(
        out,
        torch.tensor(
            [-20.0 * torch.sigmoid(torch.tensor(-20.0)) * -10.0, 10.0 * torch.sigmoid(torch.tensor(10.0)) * 10.0]
        ),
    )


def test_expert_forward_matches_closed_form():
    H, I = 3, 2
    x = torch.tensor([1.0, 2.0, -1.0])
    w13 = torch.tensor([[1.0, 0, 0], [0, 1.0, 0], [0, 0, 1.0], [1.0, 1.0, 1.0]])
    w2 = torch.tensor([[1.0, 0], [0, 1.0], [1.0, 1.0]])
    g = torch.tensor([1.0, 2.0])
    u = torch.tensor([-1.0, 2.0])
    a = g * torch.sigmoid(g) * u
    expect = w2 @ a
    assert torch.allclose(expert_forward(x, w13, w2, None), expect)
    assert H == 3 and I == 2


def test_moe_reference_routes_override_and_linearity():
    cfg = MoEConfig(hidden_size=4, moe_intermediate_size=2, n_routed_experts=2, top_k=1, swiglu_limit=None)
    torch.manual_seed(0)
    x = torch.randn(2, 4)
    w13, w2 = torch.randn(2, 4, 4), torch.randn(2, 4, 2)
    routes = [[(1, 0.5)], [(0, 2.0)]]
    out = moe_reference(x, torch.zeros(2, 2), None, w13, w2, cfg, routes=routes)
    assert torch.allclose(out[0], 0.5 * expert_forward(x[0], w13[1], w2[1], None))
    assert torch.allclose(out[1], 2.0 * expert_forward(x[1], w13[0], w2[0], None))


def test_ep_dispatch_reference_one_copy_per_rank():
    routes = [[[(0, 1.0), (1, 2.0)], [(3, 1.0), (0, 0.5)]], [[(2, 1.0), (3, 1.0)]]]
    recv = ep_dispatch_reference(routes, experts_per_rank=2)
    assert recv[0] == [(0, 0, [(0, 1.0), (1, 2.0)]), (0, 1, [(0, 0.5)])]
    assert recv[1] == [(0, 1, [(1, 1.0)]), (1, 0, [(0, 1.0), (1, 1.0)])]

import pytest
import torch

from moeflow.config import GLM53_FLASH, MoEConfig
from moeflow.reference import route_reference
from moeflow.router import route
from moeflow.testing import DISTRIBUTIONS, make_bias, make_logits


def _check(cfg, logits, bias):
    w, ids = route(logits, bias, cfg)
    ref = route_reference(logits, bias, cfg)
    for t, r in enumerate(ref):
        assert ids[t].tolist() == [e for e, _ in r]
        assert torch.allclose(w[t].double(), torch.tensor([x for _, x in r], dtype=torch.float64), rtol=1e-6, atol=1e-7)


@pytest.mark.parametrize("dist", DISTRIBUTIONS)
def test_router_matches_reference_full_shape(dist):
    cfg = GLM53_FLASH
    _check(cfg, make_logits(16, cfg, seed=1, dist=dist), make_bias(cfg, seed=2))


def test_router_group_limited():
    cfg = MoEConfig(n_routed_experts=32, top_k=4, n_group=4, topk_group=2)
    _check(cfg, make_logits(32, cfg, seed=3), make_bias(cfg, seed=4))


def test_router_ties():
    cfg = MoEConfig(n_routed_experts=16, top_k=4)
    _check(cfg, torch.zeros(3, 16), None)


def test_hotspot_really_selects_only_the_hot_experts_after_bias():
    cfg = GLM53_FLASH
    logits = make_logits(128, cfg, seed=1, dist="hotspot")
    _, ids = route(logits, make_bias(cfg, seed=2), cfg)
    expected = torch.arange(cfg.top_k).expand(128, -1)
    assert torch.equal(ids.sort(dim=1).values, expected)

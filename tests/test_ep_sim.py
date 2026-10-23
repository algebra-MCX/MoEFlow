"""End-to-end EP pipeline in the single-process simulator vs the naive reference."""

import pytest
import torch

from moeflow.ep_moe import EPMoELayer, shard_experts, simulate_ep
from moeflow.reference import moe_reference
from moeflow.testing import DISTRIBUTIONS, make_bias, make_hidden, make_logits, make_weights
from moeflow.tolerances import gemm_tolerance, normwise_rel_err


def _run(cfg, world, tokens_per_rank, dist, seed, combine_dtype=torch.bfloat16):
    w13, w2 = make_weights(cfg, seed)
    bias = make_bias(cfg, seed + 1)
    xs = [make_hidden(n, cfg, seed + 10 + r) for r, n in enumerate(tokens_per_rank)]
    lgs = [make_logits(n, cfg, seed + 20 + r, dist=dist) for r, n in enumerate(tokens_per_rank)]
    layers = [
        EPMoELayer(cfg, r, world, *shard_experts(w13, w2, r, world), combine_dtype=combine_dtype) for r in range(world)
    ]
    outs = simulate_ep(layers, xs, lgs, bias)
    refs = [moe_reference(x, lg, bias, w13, w2, cfg) for x, lg in zip(xs, lgs)]
    return outs, refs


@pytest.mark.parametrize("world", [1, 2, 4, 8])
@pytest.mark.parametrize("dist", DISTRIBUTIONS)
def test_ep_sim_matches_reference(small_cfg, world, dist):
    tokens = [3, 0, 5, 1, 8, 2, 0, 4][:world]
    outs, refs = _run(small_cfg, world, tokens, dist, seed=world)
    for o, r in zip(outs, refs):
        assert o.shape == r.shape and o.dtype == torch.bfloat16
        if r.numel():
            assert normwise_rel_err(o, r) <= gemm_tolerance()


def test_ep_sim_deterministic(small_cfg):
    a, _ = _run(small_cfg, 4, [6, 6, 6, 6], "zipf", seed=5)
    b, _ = _run(small_cfg, 4, [6, 6, 6, 6], "zipf", seed=5)
    assert all(torch.equal(x, y) for x, y in zip(a, b))


def test_ep_sim_all_empty(small_cfg):
    outs, _ = _run(small_cfg, 4, [0, 0, 0, 0], "uniform", seed=9)
    assert all(o.shape == (0, small_cfg.hidden_size) for o in outs)


def test_ep_world_invariance_fp32_combine(small_cfg):
    """With fp32 return transport, EP=1/2/4/8 agree to fp32 summation-order noise."""
    base, _ = _run(small_cfg, 1, [12], "uniform", seed=3, combine_dtype=torch.float32)
    for world in (2, 4, 8):
        cfg = small_cfg
        w13, w2 = make_weights(cfg, 3)
        bias = make_bias(cfg, 4)
        x = make_hidden(12, cfg, 13)
        lg = make_logits(12, cfg, 23)
        layers = [
            EPMoELayer(cfg, r, world, *shard_experts(w13, w2, r, world), combine_dtype=torch.float32)
            for r in range(world)
        ]
        xs = [x] + [x[:0]] * (world - 1)
        lgs = [lg] + [lg[:0]] * (world - 1)
        out = simulate_ep(layers, xs, lgs, bias)[0]
        diff = (out.float() - base[0].float()).abs().max().item()
        assert diff <= 2**-7 * base[0].float().abs().max().item()


def test_ep_sim_within_tolerance(small_cfg):
    outs, refs = _run(small_cfg, 4, [5, 3, 0, 6], "zipf", seed=11)
    for o, r in zip(outs, refs):
        if r.numel():
            assert normwise_rel_err(o, r) <= gemm_tolerance()

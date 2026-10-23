"""Fixed-capacity (graph-capturable) path: bitwise vs the naive pipeline, tolerance vs reference."""

import pytest
import torch

from moeflow.ep_moe import EPMoELayer, shard_experts, simulate_ep
from moeflow.experts import TritonGroupedExperts
from moeflow.fixed import FixedEPLayer, meta_lanes, simulate_fixed
from moeflow.reference import moe_reference
from moeflow.testing import DISTRIBUTIONS, make_bias, make_hidden, make_logits, make_weights
from moeflow.tolerances import gemm_tolerance, normwise_rel_err

pytestmark = pytest.mark.gpu


def _setup(cfg, world, C, valid, dist, seed):
    w13, w2 = make_weights(cfg, seed, device="cuda")
    bias = make_bias(cfg, seed + 1, device="cuda")
    xs = [make_hidden(C, cfg, seed + 10 + r, device="cuda") for r in range(world)]
    lgs = [make_logits(C, cfg, seed + 20 + r, dist=dist, device="cuda") for r in range(world)]
    fixed = [FixedEPLayer(cfg, r, world, *shard_experts(w13, w2, r, world), capacity=C) for r in range(world)]
    naive = [
        EPMoELayer(cfg, r, world, *shard_experts(w13, w2, r, world), experts=TritonGroupedExperts())
        for r in range(world)
    ]
    nv = [torch.tensor(v, device="cuda") for v in valid]
    return w13, w2, bias, xs, lgs, fixed, naive, nv


@pytest.mark.parametrize("world", [1, 2, 4])
@pytest.mark.parametrize("dist", DISTRIBUTIONS)
def test_fixed_bitwise_equals_naive_on_valid_rows(small_cfg, world, dist):
    C = 6
    valid = [6, 0, 3, 1][:world]
    w13, w2, bias, xs, lgs, fixed, naive, nv = _setup(small_cfg, world, C, valid, dist, seed=world)
    outs = simulate_fixed(fixed, xs, lgs, bias, num_valid=nv)
    ref = simulate_ep(naive, [x[:v] for x, v in zip(xs, valid)], [lg[:v] for lg, v in zip(lgs, valid)], bias)
    for o, r, v in zip(outs, ref, valid):
        assert torch.equal(o[:v], r)
        assert torch.all(o[v:] == 0)  # padding rows receive nothing


def test_fixed_matches_reference(small_cfg):
    w13, w2, bias, xs, lgs, fixed, _, nv = _setup(small_cfg, 4, 5, [5, 5, 5, 5], "zipf", seed=3)
    outs = simulate_fixed(fixed, xs, lgs, bias, num_valid=nv)
    for o, x, lg in zip(outs, xs, lgs):
        assert normwise_rel_err(o, moe_reference(x, lg, bias, w13, w2, small_cfg)) <= gemm_tolerance()


def test_fixed_all_tokens_same_experts(small_cfg):
    """Every token on every rank picks experts 0..K-1 (all land on rank 0)."""
    w13, w2, bias, xs, _, fixed, naive, nv = _setup(small_cfg, 4, 4, [4, 4, 4, 4], "uniform", seed=5)
    lgs = [torch.full((4, small_cfg.n_routed_experts), -10.0, device="cuda") for _ in range(4)]
    for lg in lgs:
        lg[:, : small_cfg.top_k] = 10.0
    outs = simulate_fixed(fixed, xs, lgs, bias, num_valid=nv)
    ref = simulate_ep(naive, xs, lgs, bias)
    assert all(torch.equal(o, r) for o, r in zip(outs, ref))


def test_fixed_buffers_are_static(small_cfg):
    _, _, bias, xs, lgs, fixed, _, nv = _setup(small_cfg, 2, 3, [3, 2], "uniform", seed=7)
    ptrs = [(ly.send.data_ptr(), ly.recv.data_ptr(), ly.ret_send.data_ptr(), ly.ret_recv.data_ptr()) for ly in fixed]
    simulate_fixed(fixed, xs, lgs, bias, num_valid=nv)
    simulate_fixed(fixed, xs, lgs, bias, num_valid=nv)
    assert ptrs == [
        (ly.send.data_ptr(), ly.recv.data_ptr(), ly.ret_send.data_ptr(), ly.ret_recv.data_ptr()) for ly in fixed
    ]


def test_meta_lanes():
    assert meta_lanes(8) == 34


def test_stop_after_prefixes_world1(small_cfg):
    """stop_after returns the documented intermediate; the full path equals forward() output."""
    import torch.distributed as dist

    from moeflow.fixed import STAGES

    if not dist.is_initialized():
        import tempfile

        dist.init_process_group(
            "nccl",
            init_method=f"file://{tempfile.mkdtemp()}/s",
            rank=0,
            world_size=1,
            device_id=torch.device("cuda", 0),
        )
    try:
        w13, w2, bias, xs, lgs, fixed, naive, nv = _setup(small_cfg, 1, 4, [4], "zipf", seed=13)
        ly = fixed[0]
        outs = {s: ly.forward(xs[0], lgs[0], bias, nv[0], stop_after=None if s == "COMBINE" else s) for s in STAGES}
        assert outs["ROUTER"].shape == (4, small_cfg.top_k)
        assert outs["DISPATCH"].shape == (4, 1) and outs["DISPATCH"].dtype == torch.bool
        assert outs["A2A_FORWARD"].data_ptr() == ly.recv.data_ptr()
        assert torch.equal(outs["COMBINE"], simulate_fixed(fixed, xs, lgs, bias, num_valid=nv)[0])
    finally:
        dist.destroy_process_group()


FLAG_SETS = [
    dict(metadata_kernel=True),
    dict(metadata_kernel=True, fused_reduce=True),
    dict(fused_combine=True),
    dict(metadata_kernel=True, fused_reduce=True, fused_combine=True),
]


@pytest.mark.parametrize("flags", FLAG_SETS, ids=lambda f: "+".join(f))
@pytest.mark.parametrize("dist", DISTRIBUTIONS)
def test_fused_variants_bitwise_equal_baseline_fixed(small_cfg, flags, dist):
    C, world, valid = 5, 4, [5, 0, 2, 4]
    w13, w2, bias, xs, lgs, base, _, nv = _setup(small_cfg, world, C, valid, dist, seed=17)
    fused = [
        FixedEPLayer(small_cfg, r, world, *shard_experts(w13, w2, r, world), capacity=C, **flags) for r in range(world)
    ]
    a = simulate_fixed(base, xs, lgs, bias, num_valid=nv)
    b = simulate_fixed(fused, xs, lgs, bias, num_valid=nv)
    assert all(torch.equal(x, y) for x, y in zip(a, b))


def test_fused_reduce_requires_metadata_kernel(small_cfg):
    w13, w2 = make_weights(small_cfg, 0, device="cuda")
    with pytest.raises(ValueError):
        FixedEPLayer(small_cfg, 0, 1, w13, w2, capacity=2, fused_reduce=True)


def test_forward_routed_external_experts_and_partial_capacity(small_cfg):
    """expert_fn built from the same ops as the built-in path gives bitwise-identical results,
    and T < C tokens work (rows beyond T are padding)."""
    import tempfile

    import torch.distributed as dist

    from moeflow import dispatch as D
    from moeflow.router import route

    if not dist.is_initialized():
        dist.init_process_group(
            "nccl",
            init_method=f"file://{tempfile.mkdtemp()}/s",
            rank=0,
            world_size=1,
            device_id=torch.device("cuda", 0),
        )
    try:
        cfg = small_cfg
        w13, w2 = make_weights(cfg, 2, device="cuda")
        bias = make_bias(cfg, 3, device="cuda")
        x = make_hidden(3, cfg, 4, device="cuda")
        lg = make_logits(3, cfg, 5, device="cuda")
        experts = TritonGroupedExperts()

        def expert_fn(x_rows, local_ids, wts):
            lay = D.build_expert_layout(local_ids.to(torch.int32), cfg.n_routed_experts)
            y = experts(
                D.gather_expert_input(x_rows, lay),
                lay.expert_offsets,
                w13,
                w2,
                cfg.swiglu_limit,
                D.entry_weights(lay, wts),
            )
            return D.reduce_local(y, lay, wts, torch.bfloat16)

        builtin = FixedEPLayer(cfg, 0, 1, w13, w2, capacity=8)
        external = FixedEPLayer(cfg, 0, 1, None, None, capacity=8, device=torch.device("cuda"))
        w, ids = route(lg, bias, cfg)
        a = builtin.forward_routed(x, w, ids)
        b = external.forward_routed(x, w, ids, expert_fn=expert_fn)
        assert a.shape == (3, cfg.hidden_size) and torch.equal(a, b)
        assert normwise_rel_err(a, moe_reference(x, lg, bias, w13, w2, cfg)) <= gemm_tolerance()
    finally:
        dist.destroy_process_group()

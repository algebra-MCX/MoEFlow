import pytest
import torch

from moeflow.config import GLM53_FLASH
from moeflow.ep_moe import EPMoELayer, shard_experts, simulate_ep
from moeflow.experts import TorchLoopExperts, TritonGroupedExperts
from moeflow.kernels.grouped_gemm import build_tile_schedule, grouped_gemm
from moeflow.reference import moe_reference
from moeflow.testing import make_bias, make_hidden, make_logits, make_weights
from moeflow.tolerances import gemm_tolerance, normwise_rel_err

pytestmark = pytest.mark.gpu


def _offsets(counts, device="cuda"):
    off = torch.zeros(len(counts) + 1, dtype=torch.int64, device=device)
    off[1:] = torch.cumsum(torch.tensor(counts, device=device), 0)
    return off


@pytest.mark.parametrize("counts", [[0, 0, 0], [1, 0, 3], [17, 0, 33, 1], [0] * 35 + [64], [5] * 72])
@pytest.mark.parametrize("nk", [(256, 128), (4096, 4096), (4096, 2048)])
def test_grouped_gemm_matches_per_expert_matmul(counts, nk):
    N, K = nk
    E = len(counts)
    g = torch.Generator(device="cpu").manual_seed(sum(counts) + N)
    w = (torch.randn(E, N, K, generator=g) / K**0.5).to("cuda", torch.bfloat16)
    a = torch.randn(sum(counts), K, generator=g).to("cuda", torch.bfloat16)
    off = _offsets(counts)
    out = grouped_gemm(a, w, off)
    o = off.tolist()
    for e in range(E):
        if o[e + 1] > o[e]:
            ref = a[o[e] : o[e + 1]].float() @ w[e].float().t()
            assert normwise_rel_err(out[o[e] : o[e + 1]], ref) <= 2 * 2**-8


def test_tile_schedule_covers_every_row_once():
    off = _offsets([3, 0, 40, 16, 1])
    e, s, t, nmax = build_tile_schedule(off, 60, 16)
    covered = torch.zeros(60, dtype=torch.int32)
    for be, bs, bt in zip(e.tolist(), s.tolist(), t.tolist()):
        if be >= 0:
            assert off[be] <= bs < bt <= off[be + 1]
            covered[bs:bt] += 1
    assert covered.eq(1).all()


@pytest.mark.slow
@pytest.mark.parametrize("world", [1, 4])
@pytest.mark.parametrize("backend", [TorchLoopExperts, TritonGroupedExperts])
def test_full_shape_layer_sim_vs_reference(world, backend):
    cfg = GLM53_FLASH
    w13, w2 = make_weights(cfg, 0, device="cuda")
    bias = make_bias(cfg, 1, device="cuda")
    tokens = [4, 1, 0, 3][:world] if world > 1 else [8]
    xs = [make_hidden(n, cfg, 10 + r, device="cuda") for r, n in enumerate(tokens)]
    lgs = [make_logits(n, cfg, 20 + r, dist="zipf", device="cuda") for r, n in enumerate(tokens)]
    layers = [EPMoELayer(cfg, r, world, *shard_experts(w13, w2, r, world), experts=backend()) for r in range(world)]
    outs = simulate_ep(layers, xs, lgs, bias)
    for x, lg, o in zip(xs, lgs, outs):
        if x.shape[0]:
            ref = moe_reference(x, lg, bias, w13, w2, cfg)
            assert normwise_rel_err(o, ref) <= gemm_tolerance()


def test_grouped_gemm_row_scale_applied_before_cast():
    counts = [3, 0, 5]
    g = torch.Generator(device="cpu").manual_seed(5)
    w = (torch.randn(3, 128, 64, generator=g) / 8).to("cuda", torch.bfloat16)
    a = torch.randn(8, 64, generator=g).to("cuda", torch.bfloat16)
    scale = torch.rand(8, generator=g).to("cuda") * 3
    off = _offsets(counts)
    out = grouped_gemm(a, w, off, row_scale=scale)
    o = off.tolist()
    for e in range(3):
        if o[e + 1] > o[e]:
            ref = (a[o[e] : o[e + 1]].float() @ w[e].float().t()) * scale[o[e] : o[e + 1], None]
            assert normwise_rel_err(out[o[e] : o[e + 1]], ref) <= 2 * 2**-8


def test_grouped_gemm_a_index_gathers_inside_kernel():
    g = torch.Generator(device="cpu").manual_seed(9)
    w = (torch.randn(2, 128, 64, generator=g) / 8).to("cuda", torch.bfloat16)
    src = torch.randn(5, 64, generator=g).to("cuda", torch.bfloat16)
    idx = torch.tensor([4, 0, 0, 3, 1, 2], device="cuda")  # 6 output rows: e0 <- 2 rows, e1 <- 4 rows
    off = _offsets([2, 4])
    out = grouped_gemm(src, w, off, a_index=idx)
    ref = grouped_gemm(src.index_select(0, idx), w, off)
    assert torch.equal(out, ref)


def test_grouped_gemm_rows_past_last_offset_untouched():
    w = torch.ones(1, 128, 64, device="cuda", dtype=torch.bfloat16)
    a = torch.ones(4, 64, device="cuda", dtype=torch.bfloat16)
    out = torch.full((4, 128), -7.0, device="cuda", dtype=torch.bfloat16)
    grouped_gemm(a, w, _offsets([2]), out=out)  # only rows 0..1 belong to an expert
    assert torch.all(out[:2] == 64) and torch.all(out[2:] == -7)

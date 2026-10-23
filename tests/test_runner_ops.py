import pytest
import torch

from moeflow.config import MoEConfig
from moeflow.integrations.vllm_runner_ops import (
    combine_bf16_ring,
    local_to_global_ids,
    make_combine_fn,
    make_prepare_fn,
)


def test_local_to_global_ids_rank0_and_rank2():
    local = torch.tensor([[0, -1, 3], [-1, -1, 1]])
    w = torch.tensor([[0.5, 9.0, 0.25], [7.0, 7.0, 1.0]])
    g0, w0 = local_to_global_ids(local, w, rank_offset=0, num_experts=16)
    assert g0.tolist() == [[0, 15, 3], [15, 15, 1]]  # filler = last expert, owned by another rank
    assert w0.tolist() == [[0.5, 0.0, 0.25], [0.0, 0.0, 1.0]]
    g2, _ = local_to_global_ids(local, w, rank_offset=8, num_experts=16)
    assert g2.tolist() == [[8, 0, 11], [0, 0, 9]]  # filler = expert 0, owned by rank 0


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.gpu)])
def test_ring_combine_preserves_intermediate_rounding_and_masks_padding(device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("no CUDA device")
    # Exact FP32 sum is 2; intermediate BF16 rounding depends on the source's Ring order.
    partials = torch.tensor([1, 256, 1, -256], dtype=torch.bfloat16, device=device).reshape(4, 1, 1)
    hit = torch.ones(1, 4, dtype=torch.bool, device=device)
    assert [combine_bf16_ring(partials, hit, rank).item() for rank in range(4)] == [1, 2, 2, 0]
    partials = torch.cat([partials, torch.full_like(partials, float("nan"))], dim=1)
    hit = torch.cat([hit, torch.zeros_like(hit)])
    assert combine_bf16_ring(partials, hit, 0).tolist() == [[1], [0]]


@pytest.mark.gpu
def test_ring_combine_graph_replays_changing_data():
    if not torch.cuda.is_available():
        pytest.skip("no CUDA device")
    partials = torch.zeros(4, 1, 1, dtype=torch.bfloat16, device="cuda")
    hit = torch.ones(1, 4, dtype=torch.bool, device="cuda")
    combine_bf16_ring(partials, hit, 0)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = combine_bf16_ring(partials, hit, 0)
    for _ in range(8):
        for values, expected in [([1, 256, 1, -256], 1), ([2, 256, 2, -256], 4)]:
            partials.copy_(torch.tensor(values, dtype=torch.bfloat16, device="cuda").reshape(4, 1, 1))
            graph.replay()
            assert out.item() == expected


def test_make_combine_fn_selects_mode_and_binds_rank():
    assert make_combine_fn("fp32", 0) is None
    ring = make_combine_fn("bf16_ring", 2)
    assert ring.func is combine_bf16_ring and ring.keywords == {"rank": 2}
    assert make_combine_fn("bf16_ring_fused", 1).keywords == {"rank": 1}
    with pytest.raises(ValueError):
        make_combine_fn("unknown", 0)


def test_make_prepare_fn_binds_rank_range():
    cfg = MoEConfig()
    fn = make_prepare_fn(cfg, rank_offset=72, num_experts=288, ids_dtype=torch.int32)
    assert fn.keywords == {
        "hidden_size": cfg.hidden_size,
        "top_k": cfg.top_k,
        "rank_offset": 72,
        "num_experts": 288,
        "ids_dtype": torch.int32,
    }

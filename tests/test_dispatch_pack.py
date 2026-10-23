"""Compare communication bytes, including metadata and stale padding, before any performance run."""

import pytest
import torch

from moeflow.config import MoEConfig
from moeflow.fixed import FixedEPLayer

pytestmark = pytest.mark.gpu


def layers(world, capacity, hidden):
    cfg = MoEConfig(hidden_size=hidden, n_routed_experts=16, top_k=4)
    return [
        FixedEPLayer(cfg, 0, world, None, None, capacity=capacity, device=torch.device("cuda"), fused_dispatch=fused)
        for fused in (False, True)
    ]


@pytest.mark.parametrize(
    "world,tokens,capacity,hidden,valid",
    [(2, 0, 4, 130, None), (2, 1, 1, 130, 1), (4, 3, 8, 4096, 1), (4, 8, 8, 130, 0)],
)
@pytest.mark.parametrize("id_dtype", [torch.int32, torch.int64])
def test_dispatch_pack_preserves_hidden_and_metadata_bytes(world, tokens, capacity, hidden, valid, id_dtype):
    if not torch.cuda.is_available():
        pytest.skip("no CUDA device")
    torch.manual_seed(42)
    plain, fused = layers(world, capacity, hidden)
    plain.send.view(torch.uint8).random_(0, 256)
    fused.send.copy_(plain.send)
    x = torch.randn(tokens, hidden, dtype=torch.bfloat16, device="cuda")
    ids = torch.randint(16, (tokens, 4), dtype=id_dtype, device="cuda")
    weights = torch.randn(tokens, 4, device="cuda")
    if tokens:
        ids[0].fill_(0)  # repeated hits on one owner, leaving the other ranks empty
        weights[0, 0] = -0.0
    num_valid = None if valid is None else torch.tensor(valid, device="cuda")
    a = plain.stage_dispatch_pack(x, weights, ids, num_valid)
    b = fused.stage_dispatch_pack(x, weights, ids, num_valid)
    assert torch.equal(a, b)
    # Negative metadata IDs can look like NaNs when interpreted as BF16. Compare their bytes.
    assert torch.equal(plain.send.view(torch.uint8), fused.send.view(torch.uint8))


def test_dispatch_pack_graph_changes_routes_and_valid_prefix_1000_times():
    if not torch.cuda.is_available():
        pytest.skip("no CUDA device")
    plain, fused = layers(4, 8, 130)
    x = torch.zeros(8, 130, dtype=torch.bfloat16, device="cuda")
    ids = torch.zeros(8, 4, dtype=torch.int32, device="cuda")
    weights = torch.ones(8, 4, device="cuda")
    valid = torch.tensor(8, device="cuda")
    fused.stage_dispatch_pack(x, weights, ids, valid)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        hit = fused.stage_dispatch_pack(x, weights, ids, valid)
    for step in range(1000):
        x.fill_(step % 7)
        ids.fill_(step % 16)
        weights.fill_(step / 1000)
        valid.fill_(step % 9)
        graph.replay()
        expected = plain.stage_dispatch_pack(x, weights, ids, valid)
        assert torch.equal(hit, expected)
        assert torch.equal(plain.send.view(torch.uint8), fused.send.view(torch.uint8))

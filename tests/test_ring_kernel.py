import pytest
import torch

from moeflow.integrations.vllm_runner_ops import combine_bf16_ring as torch_combine
from moeflow.kernels.reduce import combine_bf16_ring

pytestmark = pytest.mark.gpu


@pytest.mark.parametrize("world,tokens,hidden", [(2, 5, 130), (4, 1, 1), (4, 3, 4096)])
def test_fused_ring_matches_torch_rounding_and_padding(world, tokens, hidden):
    if not torch.cuda.is_available():
        pytest.skip("no CUDA device")
    torch.manual_seed(42)
    ret = torch.randn(world, tokens, hidden, dtype=torch.bfloat16, device="cuda")
    hit = torch.rand(tokens, world, device="cuda") > 0.25
    hit[-1] = False
    ret[:, -1] = float("nan")
    for rank in range(world):
        expected = torch_combine(ret, hit, rank)
        assert torch.equal(combine_bf16_ring(ret, hit, rank), expected)
    ret = torch.tensor([1, 256, 1, -256], dtype=torch.bfloat16, device="cuda").reshape(4, 1, 1)
    hit = torch.ones(1, 4, dtype=torch.bool, device="cuda")
    assert [combine_bf16_ring(ret, hit, rank).item() for rank in range(4)] == [1, 2, 2, 0]


def test_fused_ring_graph_replays_changing_inputs_1000_times():
    if not torch.cuda.is_available():
        pytest.skip("no CUDA device")
    ret = torch.zeros(4, 1, 130, dtype=torch.bfloat16, device="cuda")
    hit = torch.ones(1, 4, dtype=torch.bool, device="cuda")
    out = torch.empty_like(ret[0])
    combine_bf16_ring(ret, hit, 0, out=out)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        combine_bf16_ring(ret, hit, 0, out=out)
    for step in range(1000):
        value, expected = (1, 1) if step % 2 == 0 else (2, 4)
        ret.copy_(torch.tensor([value, 256, value, -256], dtype=torch.bfloat16, device="cuda").reshape(4, 1, 1))
        hit.fill_(step % 3 != 0)
        graph.replay()
        assert torch.equal(out, torch.full_like(out, expected if step % 3 != 0 else 0))

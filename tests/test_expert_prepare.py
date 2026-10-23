"""Check the fused adapter input format against values chosen before metadata serialization."""

import pytest
import torch

from moeflow.kernels.expert_prepare import prepare_expert_inputs

pytestmark = pytest.mark.gpu


def pack(hidden, local, weights, valid):
    count, width = hidden.shape
    top_k = local.shape[1]
    rows = torch.empty(count, width + 2 * (2 * top_k + 1), dtype=torch.bfloat16, device=hidden.device)
    rows[:, :width] = hidden
    meta = rows[:, width:].view(torch.int32)
    meta[:, :top_k] = local
    meta[:, top_k : 2 * top_k] = weights.view(torch.int32)
    meta[:, 2 * top_k] = valid
    return rows


@pytest.mark.parametrize("ids_dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize(
    "count,hidden_size,top_k,offset,experts,local_count",
    [(0, 130, 4, 0, 16, 8), (1, 130, 4, 8, 16, 8), (5, 4096, 8, 0, 288, 72), (8, 130, 8, 216, 288, 72)],
)
def test_expert_preparation_preserves_hidden_and_weight_bits(
    count, hidden_size, top_k, offset, experts, local_count, ids_dtype
):
    if not torch.cuda.is_available():
        pytest.skip("no CUDA device")
    torch.manual_seed(42)
    hidden = torch.randn(count, hidden_size, dtype=torch.bfloat16, device="cuda")
    local = torch.randint(-1, local_count, (count, top_k), dtype=torch.int32, device="cuda")
    weights = torch.randn(count, top_k, device="cuda")
    valid = torch.randint(0, 2, (count,), dtype=torch.int32, device="cuda")
    if count:
        valid[0] = 0
        hidden[0] = float("nan")
        weights[0] = float("nan")
    present = (valid[:, None] != 0) & (local >= 0)
    expected_ids = torch.where(present, local + offset, experts - 1 if offset == 0 else 0).to(ids_dtype)
    expected_weights = torch.where(present, weights, 0)
    result = prepare_expert_inputs(pack(hidden, local, weights, valid), hidden_size, top_k, offset, experts, ids_dtype)
    assert result[0].is_contiguous() and torch.equal(result[0].view(torch.uint8), hidden.view(torch.uint8))
    assert torch.equal(result[1], expected_ids)
    assert torch.equal(result[2].view(torch.uint8), expected_weights.view(torch.uint8))


def test_expert_preparation_graph_replays_new_metadata_1000_times():
    if not torch.cuda.is_available():
        pytest.skip("no CUDA device")
    hidden = torch.zeros(8, 130, dtype=torch.bfloat16, device="cuda")
    local = torch.zeros(8, 4, dtype=torch.int32, device="cuda")
    weights = torch.ones(8, 4, device="cuda")
    valid = torch.ones(8, dtype=torch.int32, device="cuda")
    rows = pack(hidden, local, weights, valid)
    prepare_expert_inputs(rows, 130, 4, 8, 16)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out_hidden, out_ids, out_weights = prepare_expert_inputs(rows, 130, 4, 8, 16)
    for step in range(1000):
        hidden.fill_(step % 7)
        local.fill_(step % 8 if step % 3 else -1)
        weights.fill_(step / 1000)
        valid.fill_(step % 2)
        rows.copy_(pack(hidden, local, weights, valid))
        graph.replay()
        present = (valid[:, None] != 0) & (local >= 0)
        assert torch.equal(out_hidden, hidden)
        assert torch.equal(out_ids, torch.where(present, local + 8, 0))
        assert torch.equal(out_weights, torch.where(present, weights, 0))

import torch

from moeflow.comm import pack_meta, simulated_exchange, unpack_meta


def test_pack_meta_roundtrip_is_bitwise():
    ids = torch.tensor([[3, -1], [0, 7]], dtype=torch.int32)
    w = torch.tensor([[0.1, -0.0], [float("inf"), 1e-40]], dtype=torch.float32)
    i2, w2 = unpack_meta(pack_meta(ids, w))
    assert torch.equal(i2, ids)
    assert torch.equal(w2.view(torch.int32), w.view(torch.int32))


def test_simulated_exchange_orders_by_source_then_send_order():
    # rank0 sends [a0, a1] to rank1 and [] to itself; rank1 sends [b0] to rank0 and [b1] to itself
    s0 = torch.tensor([[10.0], [11.0]])
    s1 = torch.tensor([[20.0], [21.0]])
    recv, counts = simulated_exchange([s0, s1], [torch.tensor([0, 2]), torch.tensor([1, 1])])
    assert recv[0].flatten().tolist() == [20.0]
    assert recv[1].flatten().tolist() == [10.0, 11.0, 21.0]
    assert [c.tolist() for c in counts] == [[0, 1], [2, 1]]

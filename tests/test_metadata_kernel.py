"""Single-kernel GPU metadata vs the torch reference construction."""

import pytest
import torch

from moeflow.kernels.grouped_gemm import build_tile_schedule, grouped_gemm
from moeflow.kernels.metadata import ExpertMetadata

pytestmark = pytest.mark.gpu


def _meta(rows, K, L, gen, valid_frac=0.8, invalid_slot_frac=0.3):
    ids = torch.randint(0, L, (rows, K), generator=gen, dtype=torch.int32)
    ids[torch.rand(rows, K, generator=gen) < invalid_slot_frac] = -1
    w = torch.rand(rows, K, generator=gen)
    valid = (torch.rand(rows, generator=gen) < valid_frac).to(torch.int32)
    meta = torch.cat([ids, w.view(torch.int32), valid[:, None], torch.zeros(rows, 1, dtype=torch.int32)], 1)
    return meta.cuda(), ids, w, valid


def _torch_reference(ids, w, valid, L):
    K = ids.shape[1]
    local = torch.where(valid[:, None] != 0, ids.long(), -1)
    keys = torch.where(local >= 0, local, L).reshape(-1)
    order = torch.sort(keys, stable=True).indices
    counts = torch.bincount(keys, minlength=L + 1)[:L]
    offsets = torch.zeros(L + 1, dtype=torch.int64)
    offsets[1:] = torch.cumsum(counts, 0)
    return order // K, order % K, w.reshape(-1)[order], offsets


@pytest.mark.parametrize("rows,K,L", [(1, 8, 72), (4, 8, 72), (32, 8, 72), (512, 8, 72), (1024, 8, 36), (7, 4, 8)])
@pytest.mark.parametrize("valid_frac", [1.0, 0.5, 0.0])
def test_metadata_matches_torch(rows, K, L, valid_frac):
    gen = torch.Generator().manual_seed(rows * 31 + K + L)
    meta, ids, w, valid = _meta(rows, K, L, gen, valid_frac=valid_frac)
    md = ExpertMetadata(rows, K, L, block_m=16, device="cuda")
    md.build(meta)
    er, es, ew, off = _torch_reference(ids, w, valid, L)
    n = int(off[-1])
    assert md.offsets.cpu().tolist() == off.tolist()
    assert torch.equal(md.entry_row[:n].cpu().long(), er[:n]) and torch.equal(md.entry_slot[:n].cpu().long(), es[:n])
    assert torch.equal(md.entry_w[:n].cpu(), ew[:n])
    te, ts, tt, mt = build_tile_schedule(off.cuda(), rows * K, 16)
    assert mt == md.max_tiles
    ok = te >= 0
    assert torch.equal(md.tile_e.cpu(), te.cpu())
    assert torch.equal(md.tile_s[ok].cpu(), ts[ok].cpu()) and torch.equal(md.tile_t[ok].cpu(), tt[ok].cpu())


def test_metadata_schedule_drives_grouped_gemm():
    gen = torch.Generator().manual_seed(3)
    rows, K, L = 16, 4, 6
    meta, ids, w, valid = _meta(rows, K, L, gen)
    md = ExpertMetadata(rows, K, L, block_m=16, device="cuda")
    md.build(meta)
    a = torch.randn(rows, 64, generator=gen).to("cuda", torch.bfloat16)
    wt = (torch.randn(L, 128, 64, generator=gen) / 8).to("cuda", torch.bfloat16)
    out = grouped_gemm(a, wt, md.offsets, a_index=md.entry_row, schedule=md.schedule)
    ref = grouped_gemm(a, wt, md.offsets.long(), a_index=md.entry_row)
    n = int(md.offsets[-1])
    assert torch.equal(out[:n], ref[:n])


def test_metadata_capturable_and_replays():
    gen = torch.Generator().manual_seed(4)
    meta, *_ = _meta(64, 8, 72, gen)
    md = ExpertMetadata(64, 8, 72, block_m=16, device="cuda")
    md.build(meta)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        md.build(meta)
    meta2, ids2, w2, valid2 = _meta(64, 8, 72, gen)
    meta.copy_(meta2)
    g.replay()
    torch.cuda.synchronize()
    _, _, _, off = _torch_reference(ids2, w2, valid2, 72)
    assert md.offsets.cpu().tolist() == off.tolist()

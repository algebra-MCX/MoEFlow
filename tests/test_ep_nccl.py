"""Multi-GPU NCCL EP pipeline as a unit test: wraps tests/multigpu/ep_nccl_check.py."""

import json

import pytest

from test_probes import _need_gpus, _torchrun


@pytest.mark.multigpu
@pytest.mark.parametrize("nproc", [2, 4])
def test_ep_nccl_bitwise_vs_sim_and_within_tolerance(nproc, tmp_path):
    _need_gpus(nproc)
    out = tmp_path / "res.json"
    r = _torchrun(nproc, "tests/multigpu/ep_nccl_check.py", "--iters", "20", "--out", str(out), timeout=600)
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
    d = json.loads(out.read_text())
    rows = [x for per_rank in d["per_rank"] for x in per_rank]
    assert d["passed"] and all(x["bitwise_vs_sim"] for x in rows)


def _check_fixed_graph_gate(nproc, tmp_path, *flags):
    _need_gpus(nproc)
    out = tmp_path / "graph.json"
    r = _torchrun(
        nproc, "tests/multigpu/fixed_graph_check.py", "--replays", "1000", "--out", str(out), *flags, timeout=1200
    )
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
    d = json.loads(out.read_text())
    assert d["passed"] and all(x["G3_G4_mismatches"] == 0 for x in d["per_rank"])
    assert d["G5_rank0_process_group_destroyed"]
    assert all(f"[rank{rank}] process_group_destroyed" in r.stdout for rank in range(nproc))
    assert all(n == 2 for n in d["per_rank"][0]["collectives_per_forward"].values())


@pytest.mark.multigpu
@pytest.mark.parametrize("nproc", [2, 4])
def test_fixed_path_cuda_graph_gate(nproc, tmp_path):
    """1000 interleaved replays, including strict hotspots and idle ranks."""
    _check_fixed_graph_gate(nproc, tmp_path)


@pytest.mark.multigpu
def test_fixed_fused_path_cuda_graph_gate(tmp_path):
    _check_fixed_graph_gate(4, tmp_path, "--metadata-kernel", "--fused-reduce", "--fused-combine")

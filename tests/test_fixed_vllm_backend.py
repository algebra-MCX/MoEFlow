import json
from types import SimpleNamespace

import pytest
import torch

from test_probes import _need_gpus, _torchrun


def test_fixed_expert_adapter_compacts_packed_hidden_rows(monkeypatch):
    from bench.backends import MoEFlowFixedVllmBackend, VllmModularBackend

    def apply_local(self, hidden, weights, ids):
        assert hidden.is_contiguous()
        return hidden

    monkeypatch.setattr(VllmModularBackend, "_apply", apply_local)
    backend = MoEFlowFixedVllmBackend.__new__(MoEFlowFixedVllmBackend)
    backend.fused_expert_prep = False
    backend.info, backend.fixed = SimpleNamespace(rank=0), SimpleNamespace(L=4)
    backend.cfg, backend.topk_dtype = SimpleNamespace(n_routed_experts=8), torch.int32
    hidden = torch.arange(16).reshape(2, 8)[:, :4]
    assert not hidden.is_contiguous()
    out = backend._local_experts(hidden, torch.zeros(2, 2, dtype=torch.int32), torch.ones(2, 2))
    assert torch.equal(out, hidden)


@pytest.mark.multigpu
@pytest.mark.parametrize("nproc", [2, 4])
def test_fixed_vllm_experts_reference_and_graph(nproc, tmp_path):
    _need_gpus(nproc)
    out = tmp_path / "fixed_vllm.json"
    result = _torchrun(
        nproc,
        "scripts/bench_layer.py",
        "--backend",
        "vllm:moeflow_fixed",
        "--shape",
        "small",
        "--tokens",
        "1,8,64",
        "--inner",
        "5",
        "--out",
        str(out),
        timeout=180,
    )
    assert result.returncode == 0, result.stdout[-3000:] + result.stderr[-3000:]
    points = json.loads(out.read_text())["results"]["points"]
    assert len(points) == 3 and all(point["graph_capturable"] for point in points)
    assert all(check["passed"] for point in points for check in point["check_per_rank"])


@pytest.mark.multigpu
@pytest.mark.parametrize("variant", ["fp32", "ring", "dispatch", "expert_prep", "direct_output"])
def test_native_fixed_paired_comparison_checks_both_arms(tmp_path, variant):
    _need_gpus(2)
    out = tmp_path / "paired.json"
    flags = {
        "fp32": ["--b-fused-combine"],
        "ring": ["--a-fixed-ring", "--b-combine-mode", "bf16_ring_fused"],
        "dispatch": [
            "--a-fixed-combine-mode",
            "bf16_ring_fused",
            "--b-combine-mode",
            "bf16_ring_fused",
            "--b-fused-dispatch",
        ],
        "expert_prep": [
            "--a-fixed-combine-mode",
            "bf16_ring_fused",
            "--b-combine-mode",
            "bf16_ring_fused",
            "--a-fused-dispatch",
            "--b-fused-dispatch",
            "--b-fused-expert-prep",
            "--check-graph-replays",
            "1000",
        ],
    }
    flags["direct_output"] = [*flags["expert_prep"], "--a-fused-expert-prep", "--b-direct-expert-output"]
    result = _torchrun(
        2,
        "scripts/ab_native_fixed.py",
        "--shape",
        "small",
        "--weights",
        "synthetic",
        "--tokens",
        "1,8",
        *flags[variant],
        "--reps",
        "10",
        "--inner",
        "5",
        "--out",
        str(out),
        timeout=180,
    )
    assert result.returncode == 0, result.stdout[-3000:] + result.stderr[-3000:]
    record = json.loads(out.read_text())
    assert record["results"]["complete"] and record["seed"] == 0
    for point in record["results"]["points"]:
        assert point["paired"]["n"] == 10
        assert all(c["a"]["passed"] and c["b"]["passed"] for c in point["checks_per_rank"])
        if variant != "fp32":
            assert all(c["bitwise_equal"] for c in point["checks_per_rank"])
        if variant in ("expert_prep", "direct_output"):
            assert point["graph_replays_checked"] == 1000


@pytest.mark.multigpu
def test_layer_cli_records_fused_modes_and_reference_checks(tmp_path):
    _need_gpus(2)
    out = tmp_path / "modes.json"
    result = _torchrun(
        2,
        "scripts/bench_layer.py",
        "--backend",
        "vllm:moeflow_fixed",
        "--shape",
        "small",
        "--tokens",
        "1,8",
        "--fixed-combine-mode",
        "bf16_ring_fused",
        "--fixed-fused-dispatch",
        "--reps",
        "10",
        "--inner",
        "5",
        "--out",
        str(out),
        timeout=120,
    )
    assert result.returncode == 0, result.stdout[-3000:] + result.stderr[-3000:]
    record = json.loads(out.read_text())
    assert record["results"]["complete"] and record["config"]["fixed_fused_dispatch"]
    assert record["config"]["fixed_combine_mode"] == "bf16_ring_fused"
    for point in record["results"]["points"]:
        assert point["graph_capturable"]
        assert all(c["passed"] and c["tokens_checked"] == point["tokens_per_rank"] for c in point["check_per_rank"])

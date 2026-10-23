import json

import pytest
import torch
from bench.record import make_record, write_record
from bench.timing import time_eager, time_graph


def test_record_has_provenance_fields(tmp_path):
    rec = make_record("unit", {"a": 1}, {"x": [1.0]}, seed=7)
    p = write_record(tmp_path / "r.json", rec)
    d = json.loads(p.read_text())
    assert set(d) >= {"kind", "created", "git", "env", "seed", "command", "config", "results"}
    assert "sha" in d["git"] and d["seed"] == 7 and "python" in d["env"]


@pytest.mark.gpu
def test_time_eager_and_graph_single_rank():
    x = torch.randn(1024, 1024, device="cuda")
    out = torch.empty_like(x)
    fn = lambda: torch.matmul(x, x, out=out)  # noqa: E731
    e = time_eager(fn, reps=10, inner=5)
    g = time_graph(fn, reps=10, inner=5)
    assert len(e.per_call_us) == 10 and all(t > 0 for t in e.per_call_us)
    assert g.capturable and len(g.per_call_us) == 10 and all(t > 0 for t in g.per_call_us)


@pytest.mark.gpu
def test_time_graph_reports_uncapturable():
    x = torch.randn(16, device="cuda")
    g = time_graph(lambda: x.sum().item(), reps=2, inner=1)  # .item() syncs -> capture fails
    assert not g.capturable and g.per_call_us == [] and g.error


@pytest.mark.gpu
def test_alternating_ab_detects_slower_arm():
    from bench.stats import paired_ab
    from bench.timing import time_graphs_alternating

    x = torch.randn(2048, 2048, device="cuda")
    o = torch.empty_like(x)
    fast = lambda: torch.matmul(x[:256], x, out=o[:256])  # noqa: E731
    slow = lambda: torch.matmul(x, x, out=o)  # noqa: E731
    a, b = time_graphs_alternating(fast, slow, reps=10, inner=5)
    r = paired_ab(a, b)
    assert len(a) == len(b) == 10 and r.mean_diff > 0 and r.significant

"""MoEFlowRunner integration smoke test as a unit test (needs a Slurm allocation with >=2 GPUs).

Runs scripts/smoke_vllm.sh (native vs MoEFlowRunner, trimmed checkpoint) and asserts that the
runner was injected, intercepted every logged MoE call, agreed with vLLM's router, and that the
greedy outputs match the native run.
"""

import json
import os
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
MODEL = os.environ.get("MOEFLOW_SMOKE_MODEL", "checkpoints/GLM-5.3-Flash-4L")

pytestmark = [pytest.mark.vllm, pytest.mark.multigpu, pytest.mark.slow]


@pytest.mark.parametrize("runner_mode", ["route_check", "fixed"])
def test_moeflow_runner_smoke(tmp_path, runner_mode):
    if not os.environ.get("MOEFLOW_JOBID"):
        pytest.skip("set MOEFLOW_JOBID to a Slurm allocation with >= 2 GPUs")
    if not Path(MODEL).exists():
        pytest.skip(f"trimmed checkpoint missing: {MODEL}")
    out = Path(os.environ.get("MOEFLOW_SMOKE_ARTIFACT_DIR", tmp_path)) / runner_mode
    dp_size = os.environ.get("MOEFLOW_SMOKE_DP", "2")
    r = subprocess.run(
        ["scripts/smoke_vllm.sh", MODEL, dp_size, str(out), runner_mode],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=3600,
    )
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
    assert "mode=native server_ready_and_client_ok=1" in r.stdout
    assert "mode=moeflow server_ready_and_client_ok=1" in r.stdout
    summary = json.loads(r.stdout.strip().splitlines()[-1])
    assert summary["intercepted_with_tokens"] > 0 and summary["not_intercepted"] == 0
    if runner_mode == "route_check":
        assert summary["min_token_expert_set_match"] == 1.0
    else:
        assert summary["fixed_calls_logged"] > 0
    for row in json.loads((out / "compare.json").read_text()):
        assert row["common_prefix_tokens"] == row["of"], row

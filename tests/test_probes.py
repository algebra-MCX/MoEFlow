"""Software-stack probes as unit tests (single GPU, and multi-GPU via torchrun)."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable


def _torchrun(nproc: int, script: str, *args: str, env: dict | None = None, timeout: int = 300):
    port = 29500 + (os.getpid() % 2000)
    cmd = [PY, "-m", "torch.distributed.run", f"--nproc-per-node={nproc}", f"--master-port={port}", script, *args]
    env = {**os.environ, "PYTHONPATH": f"{ROOT / 'scripts'}:{os.environ.get('PYTHONPATH', '')}", **(env or {})}
    return subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, timeout=timeout, env=env)


def _need_gpus(n: int):
    if torch.cuda.device_count() < n:
        pytest.skip(f"needs {n} GPUs, have {torch.cuda.device_count()}")


@pytest.mark.gpu
def test_triton_min_kernel():
    r = subprocess.run([PY, "scripts/probes/triton_min.py"], cwd=ROOT, capture_output=True, text=True, timeout=300)
    assert r.returncode == 0, r.stderr[-2000:]
    info = json.loads(r.stdout.strip().splitlines()[-1])
    assert info["bitwise_equal_to_torch"] and tuple(info["capability"]) >= (10, 0)


@pytest.mark.multigpu
@pytest.mark.parametrize("nproc", [2, 4])
def test_nccl_all_to_all_eager(nproc):
    _need_gpus(nproc)
    r = _torchrun(nproc, "scripts/probes/nccl_a2a.py", env={"PROBE_SKIP_GRAPH": "1"}, timeout=180)
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
    assert r.stdout.count("destroyed") == nproc


@pytest.mark.multigpu
def test_nccl_all_to_all_graph_capture_and_teardown():
    """Graph is freed before destroy_process_group (see limitations L9)."""
    _need_gpus(4)
    try:
        r = _torchrun(4, "scripts/probes/nccl_a2a.py", timeout=180)
    except subprocess.TimeoutExpired:
        pytest.fail("watchdog: probe did not exit within 180 s")
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]

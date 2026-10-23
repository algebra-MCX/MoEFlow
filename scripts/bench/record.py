"""Self-describing result records: every JSON carries git sha, environment, seed and command."""

from __future__ import annotations

import datetime
import json
import os
import platform
import socket
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


def git_info(repo: Path = REPO) -> dict:
    def run(*args):
        try:
            return subprocess.run(["git", *args], cwd=repo, capture_output=True, text=True, timeout=10).stdout.strip()
        except (OSError, subprocess.SubprocessError):
            return ""

    return {"sha": run("rev-parse", "HEAD"), "dirty": bool(run("status", "--porcelain", "--untracked-files=no"))}


def env_info() -> dict:
    info = {
        "hostname": socket.gethostname(),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "is_root": os.geteuid() == 0,
    }
    try:
        import torch

        info["torch"] = torch.__version__
        info["cuda"] = torch.version.cuda
        if torch.cuda.is_available():
            info["nccl"] = ".".join(map(str, torch.cuda.nccl.version()))
            info["gpu"] = torch.cuda.get_device_name()
            info["gpu_count_visible"] = torch.cuda.device_count()
    except ImportError:
        pass
    for mod in ("triton", "vllm", "deep_ep", "flashinfer"):
        try:
            info[mod] = __import__(mod).__version__
        except (ImportError, AttributeError):
            info[mod] = None
    return info


def make_record(kind: str, config: dict, results: dict, seed: int | None = None) -> dict:
    return {
        "kind": kind,
        "created": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
        "git": git_info(),
        "env": env_info(),
        "seed": seed,
        "command": [sys.executable, *sys.argv],
        "config": config,
        "results": results,
    }


def write_record(path: str | Path, record: dict) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=1, sort_keys=False, default=str))
    return path

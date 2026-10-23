"""Bound profiler collection to validated layer execution, excluding checkpoint loading."""

from __future__ import annotations

from contextlib import contextmanager

import torch


@contextmanager
def profile_layer(enabled: bool, label: str):
    if not enabled:
        yield
        return
    torch.cuda.synchronize()
    torch.cuda.profiler.start()
    try:
        with torch.cuda.nvtx.range(label):
            yield
    finally:
        torch.cuda.synchronize()
        torch.cuda.profiler.stop()

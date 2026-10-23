"""Multi-rank latency measurement (barrier-aligned start, slowest rank wins).

Each repetition: barrier on all ranks -> CUDA events around `inner` back-to-back calls ->
per-rank mean per call -> all_reduce(MAX) so the reported sample is the slowest rank.
Eager mode calls the function; graph mode captures `inner` calls into one CUDA graph once
and replays it per repetition.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import torch
import torch.distributed as dist


@dataclass
class TimingResult:
    mode: str
    per_call_us: list[float]  # one sample per repetition, max over ranks
    inner: int
    capturable: bool = True
    error: str | None = None


def _max_over_ranks(x: float, group) -> float:
    if not (dist.is_available() and dist.is_initialized()):
        return x
    t = torch.tensor([x], dtype=torch.float64, device="cuda")
    dist.all_reduce(t, op=dist.ReduceOp.MAX, group=group)
    return float(t.item())


def _barrier(group) -> None:
    if dist.is_available() and dist.is_initialized():
        dist.barrier(group=group, device_ids=[torch.cuda.current_device()])
    torch.cuda.synchronize()


def time_eager(fn: Callable[[], object], reps: int = 10, inner: int = 20, warmup: int = 5, group=None) -> TimingResult:
    for _ in range(warmup):
        fn()
    samples = []
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    for _ in range(reps):
        _barrier(group)
        start.record()
        for _ in range(inner):
            fn()
        end.record()
        end.synchronize()
        samples.append(_max_over_ranks(start.elapsed_time(end) * 1e3 / inner, group))
    return TimingResult("eager", samples, inner)


def time_graph(fn: Callable[[], object], reps: int = 10, inner: int = 20, warmup: int = 3, group=None) -> TimingResult:
    """Capture `inner` calls in one graph. If capture fails, report it instead of falling back."""
    try:
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for _ in range(warmup):
                fn()
        torch.cuda.current_stream().wait_stream(side)
        _barrier(group)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            for _ in range(inner):
                fn()
        _barrier(group)
    except Exception as e:  # "not capturable" is itself a result
        torch.cuda.synchronize()
        return TimingResult("graph", [], inner, capturable=False, error=f"{type(e).__name__}: {e}"[:500])
    graph.replay()  # warm replay
    samples = []
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    for _ in range(reps):
        _barrier(group)
        start.record()
        graph.replay()
        end.record()
        end.synchronize()
        samples.append(_max_over_ranks(start.elapsed_time(end) * 1e3 / inner, group))
    del graph
    torch.cuda.synchronize()
    return TimingResult("graph", samples, inner)


def capture(fn: Callable[[], object], warmup: int = 3, inner: int = 1, group=None) -> torch.cuda.CUDAGraph:
    """Capture `inner` back-to-back calls of fn into one graph (raises if not capturable)."""
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(warmup):
            fn()
    torch.cuda.current_stream().wait_stream(side)
    _barrier(group)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(inner):
            fn()
    _barrier(group)
    return g


def time_graphs_alternating(fn_a, fn_b, reps: int = 10, inner: int = 20, group=None) -> tuple[list, list]:
    """A/B ablation timing: both variants captured, then replayed A, B, A, B, ...

    Returns per-call microseconds (max over ranks) for each arm, paired by repetition index.
    """
    ga, gb = capture(fn_a, inner=inner, group=group), capture(fn_b, inner=inner, group=group)
    ga.replay()
    gb.replay()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    out = ([], [])
    for _ in range(reps):
        for g, sink in ((ga, out[0]), (gb, out[1])):
            _barrier(group)
            start.record()
            g.replay()
            end.record()
            end.synchronize()
            sink.append(_max_over_ranks(start.elapsed_time(end) * 1e3 / inner, group))
    del ga, gb
    torch.cuda.synchronize()
    return out

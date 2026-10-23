"""Reference checks performed before a layer is eligible for performance measurement."""

from __future__ import annotations

import torch
import torch.distributed as dist

from moeflow.tolerances import bitwise_equal, calc_diff, gemm_tolerance, normwise_rel_err


def check_reference(out: torch.Tensor, ref: torch.Tensor) -> dict:
    """Apply the tolerance floor; callers gather these results across all ranks."""
    error = normwise_rel_err(out, ref) if ref.numel() else 0.0
    finite = bool(torch.isfinite(out).all()) and bool(torch.isfinite(ref).all())
    tolerance = gemm_tolerance()
    return {
        "tokens_checked": len(ref),
        "normwise_rel_err": error,
        "max_abs_err": float((out.float() - ref.float()).abs().max()) if ref.numel() else 0.0,
        "calc_diff": calc_diff(out, ref) if ref.numel() else 0.0,
        "finite": finite,
        "tolerance": tolerance,
        "passed": finite and error <= tolerance,
    }


def require_matching_variants(a: torch.Tensor, b: torch.Tensor) -> None:
    """Every rank must agree before either A/B variant is timed."""
    equal = bitwise_equal(a, b) and bool(torch.isfinite(a).all())
    if dist.is_initialized():
        flag = torch.tensor(int(equal), device=a.device)
        dist.all_reduce(flag, op=dist.ReduceOp.MIN)
        equal = bool(flag.item())
    if not equal:
        raise RuntimeError("A/B outputs differ on at least one rank; performance measurement rejected")


def require_matching_graphs(fn_a, fn_b, update, replays: int) -> None:
    """Check captured outputs after each input update, before timing either implementation."""
    from .timing import capture

    def capture_output(fn):
        output = None

        def run():
            nonlocal output
            output = fn()

        graph = capture(run)
        return graph, output

    graph_a, output_a = capture_output(fn_a)
    graph_b, output_b = capture_output(fn_b)
    try:
        for step in range(replays):
            update(step)
            graph_a.replay()
            graph_b.replay()
            require_matching_variants(output_a, output_b)
    finally:
        del graph_a, graph_b, output_a, output_b
        torch.cuda.synchronize()


def check_backend_pair(a: torch.Tensor, b: torch.Tensor, ref: torch.Tensor) -> dict:
    """Different reduction backends may round differently; check each against the frozen reference."""
    return {
        "a": check_reference(a, ref),
        "b": check_reference(b, ref),
        "bitwise_equal": bitwise_equal(a, b),
        "equality_check": "bytes",
    }

"""Opt-in eager diagnostics; these synchronous comparisons are never performance samples."""

import json
import os
from itertools import permutations
from pathlib import Path

import torch
import torch.distributed as dist

from ..tolerances import bitwise_equal, normwise_rel_err


def compare_outputs(fixed, native, is_padding=None) -> dict:
    """Compare identical-input routed outputs, excluding known padding from error metrics."""
    if fixed.shape != native.shape:
        raise ValueError("Layer output shapes differ")
    if is_padding is not None:
        if is_padding.shape != (fixed.shape[0],):
            raise ValueError("Padding mask does not match local rows")
        fixed, native = fixed[~is_padding], native[~is_padding]
    finite = bool(torch.isfinite(fixed).all() and torch.isfinite(native).all())
    return {
        "padding_known": is_padding is not None,
        "rows_compared": fixed.shape[0],
        "finite": finite,
        "bitwise_equal": bitwise_equal(fixed, native),
        "equality_check": "bytes",
        "normwise_vs_native": normwise_rel_err(fixed, native) if finite and fixed.numel() else None,
        "max_abs_diff": (fixed.float() - native.float()).abs().max().item() if finite and fixed.numel() else None,
    }


def compare_reduction_orders(partials, native) -> dict:
    """Test small-owner rounding hypotheses on CPU; an exact match is diagnostic evidence."""
    partials, native = partials.detach().cpu(), native.detach().cpu()
    owners = partials.shape[0]
    if owners > 4 or not native.numel() or not torch.isfinite(partials).all() or not torch.isfinite(native).all():
        return {"tested": 0, "exact_matches": []}
    candidates = {"fp32_then_bf16": partials.float().sum(0).to(native.dtype)}
    for order in permutations(range(owners)):
        acc = torch.zeros_like(native)
        for owner in order:
            acc = (acc.float() + partials[owner].float()).to(native.dtype)
        candidates["bf16_order_" + "".join(map(str, order))] = acc
    if owners == 4:
        for a, b, c, d in ((0, 1, 2, 3), (0, 2, 1, 3), (0, 3, 1, 2)):
            candidates[f"bf16_pairs_{a}{b}_{c}{d}"] = (partials[a] + partials[b]) + (partials[c] + partials[d])
    errors = {name: normwise_rel_err(value, native) for name, value in candidates.items()}
    best = min(errors, key=errors.get)
    return {
        "tested": len(candidates),
        "exact_matches": [name for name, value in candidates.items() if bitwise_equal(value, native)],
        "best_candidate": best,
        "best_normwise_vs_native": errors[best],
    }


class FixedLayerProbe:
    """Compare two request forwards per layer, with a collective decision on all EP ranks."""

    def __init__(self, directory: Path):
        self.directory = directory
        self.calls = 0

    def compare(
        self,
        experts,
        hidden,
        weights,
        ids,
        fixed_output,
        layer_name,
        capacity,
        owner_partials=None,
        experts_per_rank=None,
    ):
        if self.calls >= 2:
            return
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("Fixed layer diagnostics require eager execution")
        from vllm.distributed import get_ep_group
        from vllm.forward_context import get_forward_context

        group = get_ep_group()
        enabled = torch.tensor(int((self.directory / "enabled").is_file()), device=hidden.device)
        dist.all_reduce(enabled, op=dist.ReduceOp.MIN, group=group.device_group)
        if not enabled.item():
            return
        self.calls += 1  # Include empty ranks so every rank exhausts the same collective budget.
        fixed = fixed_output.clone()
        native = experts.forward_modular(x=hidden.clone(), topk_weights=weights.clone(), topk_ids=ids.clone())
        padding = get_forward_context().is_padding
        row = {
            "layer": layer_name,
            "ep_rank": group.rank_in_group,
            "sample": self.calls,
            "execution_rows": hidden.shape[0],
            "capacity": capacity,
            **compare_outputs(fixed, native, padding),
        }
        if owner_partials is not None:
            partials = owner_partials[:, : hidden.shape[0]].detach().cpu().clone()
            destinations = ids.detach().cpu() // experts_per_rank
            for owner in range(partials.shape[0]):
                partials[owner].masked_fill_(~(destinations == owner).any(-1, keepdim=True), 0)
            native_cpu = native.detach().cpu()
            if padding is not None:
                valid = ~padding.detach().cpu()
                partials, native_cpu = partials[:, valid], native_cpu[valid]
            row["reduction_candidates"] = compare_reduction_orders(partials, native_cpu)
        with (self.directory / f"layer_comparison_pid{os.getpid()}.jsonl").open("a") as output:
            output.write(json.dumps(row, allow_nan=False) + "\n")

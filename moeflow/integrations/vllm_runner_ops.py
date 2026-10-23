"""Small tensor helpers for the vLLM runner (no vLLM imports, unit-testable on CPU)."""

from __future__ import annotations

from functools import partial

import torch


def local_to_global_ids(local_ids: torch.Tensor, weights: torch.Tensor, rank_offset: int, num_experts: int):
    """Map per-rank local expert ids (-1 = no hit) to vLLM's global ids for a local fused-experts call.

    Invalid slots point at an expert this rank does not own, so vLLM's expert_map turns them into
    -1 and the kernel skips them (DeepEP-HT convention, see PROVENANCE.md); their weight is zeroed.
    """
    filler = num_experts - 1 if rank_offset == 0 else 0
    invalid = local_ids < 0
    gids = torch.where(invalid, torch.full_like(local_ids, filler), local_ids + rank_offset)
    return gids, torch.where(invalid, torch.zeros_like(weights), weights)


def combine_bf16_ring(partials: torch.Tensor, hit: torch.Tensor, rank: int) -> torch.Tensor:
    """Experimental four-rank Ring-compatible rounding; validate against the active NCCL setup.

    Start at the owner's successor and round after each addition, as observed in the native
    Ring trace. This is opt-in because other NCCL algorithms/topologies can use other orders.
    """
    if partials.dtype != torch.bfloat16:
        raise ValueError("BF16 Ring combine requires BF16 partials")
    world = partials.shape[0]
    if not 0 <= rank < world:
        raise ValueError("Source rank is outside the partials")
    acc = torch.zeros_like(partials[0])
    for step in range(1, world + 1):
        owner = (rank + step) % world
        value = torch.where(hit[:, owner : owner + 1], partials[owner], 0)
        acc = (acc.float() + value.float()).to(torch.bfloat16)
    return acc


def make_combine_fn(mode: str, rank: int):
    """Adapter combine for the BF16 Ring modes; None keeps the layer's FP32 combine."""
    if mode == "fp32":
        return None
    if mode == "bf16_ring":
        return partial(combine_bf16_ring, rank=rank)
    if mode == "bf16_ring_fused":
        from ..kernels.reduce import combine_bf16_ring as fused

        return partial(fused, rank=rank)
    raise ValueError("combine mode must be fp32, bf16_ring or bf16_ring_fused")


def make_prepare_fn(cfg, rank_offset: int, num_experts: int, ids_dtype: torch.dtype):
    """Fused expert-input preparation bound to this rank's expert range."""
    from ..kernels.expert_prepare import prepare_expert_inputs

    return partial(
        prepare_expert_inputs,
        hidden_size=cfg.hidden_size,
        top_k=cfg.top_k,
        rank_offset=rank_offset,
        num_experts=num_experts,
        ids_dtype=ids_dtype,
    )

"""Unpack received rows into contiguous hidden, global expert IDs and FP32 weights."""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _prepare(
    rows,
    meta,
    hidden,
    ids,
    weights,
    H: tl.constexpr,
    K: tl.constexpr,
    WIDTH: tl.constexpr,
    OFFSET: tl.constexpr,
    FILLER: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    row = tl.program_id(0)
    columns = tl.program_id(1) * BLOCK_H + tl.arange(0, BLOCK_H)
    value = tl.load(rows + row * WIDTH + columns, mask=columns < H, other=0)
    tl.store(hidden + row * H + columns, value, mask=columns < H)
    if tl.program_id(1) == 0:
        start = row * (WIDTH // 2) + H // 2
        slots = tl.arange(0, BLOCK_K)
        local = tl.load(meta + start + slots, mask=slots < K, other=-1)
        valid = tl.load(meta + start + 2 * K) != 0
        present = valid & (local >= 0)
        global_id = tl.where(present, local + OFFSET, FILLER)
        weight = tl.load(meta + start + K + slots, mask=slots < K, other=0).to(tl.float32, bitcast=True)
        tl.store(ids + row * K + slots, global_id, mask=slots < K)
        tl.store(weights + row * K + slots, tl.where(present, weight, 0), mask=slots < K)


def prepare_expert_inputs(
    rows: torch.Tensor,
    hidden_size: int,
    top_k: int,
    rank_offset: int,
    num_experts: int,
    ids_dtype: torch.dtype = torch.int32,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return hidden[N,H], global IDs[N,K] and weights[N,K] from fixed communication rows."""
    if rows.dtype != torch.bfloat16 or hidden_size % 2 or not rows.is_contiguous():
        raise ValueError("Expert preparation requires contiguous BF16 rows and even hidden width")
    count, width = rows.shape
    if hidden_size < 1 or top_k < 1 or width % 2 or width < hidden_size + 2 * (2 * top_k + 1):
        raise ValueError("Received rows do not contain the configured hidden and routing metadata")
    if ids_dtype not in (torch.int32, torch.int64):
        raise ValueError("Expert IDs must be INT32 or INT64")
    hidden = torch.empty((count, hidden_size), dtype=torch.bfloat16, device=rows.device)
    ids = torch.empty((count, top_k), dtype=ids_dtype, device=rows.device)
    weights = torch.empty((count, top_k), dtype=torch.float32, device=rows.device)
    if count:
        _prepare[(count, triton.cdiv(hidden_size, 1024))](
            rows,
            rows.view(torch.int32),
            hidden,
            ids,
            weights,
            H=hidden_size,
            K=top_k,
            WIDTH=width,
            OFFSET=rank_offset,
            FILLER=num_experts - 1 if rank_offset == 0 else 0,
            BLOCK_H=1024,
            BLOCK_K=triton.next_power_of_2(top_k),
        )
    return hidden, ids, weights

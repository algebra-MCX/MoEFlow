"""Pack fixed-capacity hidden rows and routing metadata without Torch intermediates."""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _pack(
    x,
    x_stride,
    ids,
    ids_stride,
    weights,
    weights_stride,
    send,
    meta,
    hit,
    num_valid,
    T: tl.constexpr,
    C: tl.constexpr,
    H: tl.constexpr,
    K: tl.constexpr,
    R: tl.constexpr,
    L: tl.constexpr,
    ROW: tl.constexpr,
    HAS_VALID: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    row = tl.program_id(0)
    token, owner = row % C, row // C
    columns = tl.program_id(1) * BLOCK_H + tl.arange(0, BLOCK_H)
    hidden = tl.load(x + token * x_stride + columns, mask=(token < T) & (columns < H), other=0)
    # Match the Torch path: padding keeps its previous payload, but its valid flag is cleared.
    tl.store(send + row * ROW + columns, hidden, mask=(token < T) & (columns < H))
    if tl.program_id(1) == 0:
        slots = tl.arange(0, BLOCK_K)
        expert = tl.load(ids + token * ids_stride + slots, mask=(token < T) & (slots < K), other=0)
        on = (expert // L == owner) & (slots < K) & (token < T)
        valid = token < T
        if HAS_VALID:
            valid = valid & (token < tl.load(num_valid))
        present = (tl.sum(on.to(tl.int32), 0) > 0) & valid
        local = tl.where(on & present, expert % L, -1).to(tl.int32)
        weight = tl.load(weights + token * weights_stride + slots, mask=(token < T) & (slots < K), other=0)
        weight = tl.where(on & present, weight, 0).to(tl.float32)
        start = row * (ROW // 2) + H // 2
        tl.store(meta + start + slots, local, mask=(token < T) & (slots < K))
        tl.store(meta + start + K + slots, weight.to(tl.int32, bitcast=True), mask=(token < T) & (slots < K))
        tl.store(meta + start + 2 * K, present.to(tl.int32))
        tl.store(hit + token * R + owner, present)


def dispatch_pack(x, weights, ids, send, experts_per_rank: int, num_valid=None) -> torch.Tensor:
    """Fill BF16 send[R,C,H+2(2K+1)] and return hit[C,R], preserving metadata bytes."""
    world, capacity, width = send.shape
    tokens, hidden = x.shape
    top_k = ids.shape[1]
    if x.dtype != torch.bfloat16 or weights.dtype != torch.float32 or hidden % 2:
        raise ValueError("Fused dispatch requires BF16 hidden with even width and FP32 routing weights")
    if tokens > capacity or width != hidden + 2 * (2 * top_k + 1):
        raise ValueError("Input dimensions do not match the fixed-capacity send buffer")
    if ids.shape != weights.shape or ids.shape[0] != tokens or top_k < 1:
        raise ValueError("Routing IDs and weights must match all input rows")
    if ids.dtype not in (torch.int32, torch.int64) or send.dtype != torch.bfloat16:
        raise ValueError("Fused dispatch requires integer IDs and a BF16 send buffer")
    if any(value.stride(-1) != 1 for value in (x, ids, weights)) or not send.is_contiguous():
        raise ValueError("Fused dispatch requires contiguous inner dimensions and send storage")
    hit = torch.empty((capacity, world), dtype=torch.bool, device=x.device)
    _pack[(world * capacity, triton.cdiv(hidden, 1024))](
        x,
        x.stride(0),
        ids,
        ids.stride(0),
        weights,
        weights.stride(0),
        send,
        send.view(torch.int32),
        hit,
        num_valid,
        T=tokens,
        C=capacity,
        H=hidden,
        K=top_k,
        R=world,
        L=experts_per_rank,
        ROW=width,
        HAS_VALID=num_valid is not None,
        BLOCK_H=1024,
        BLOCK_K=triton.next_power_of_2(top_k),
    )
    return hit

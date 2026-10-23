"""Fused owner-side reduce and source-side combine (same order and rounding as the torch versions).

owner_reduce:  out[r] = bf16( sum_{k=0..K-1, inv[r*K+k] >= 0} fp32(y[inv[r*K+k]]) )
               y holds already-weighted branches bf16(w * acc) (GEMM2 epilogue).
combine:       out[t] = bf16( sum_{d=0..R-1, hit[t, d]} fp32(ret[d, t]) )   (ascending d, like DeepEP)
Accumulators start at +0.0 and skipped terms are not added, matching vLLM moe_sum and DeepEP.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _owner_reduce_kernel(y_ptr, stride_y, inv_ptr, out_ptr, stride_out, H, K: tl.constexpr, BLOCK_H: tl.constexpr):
    r = tl.program_id(0)
    cols = tl.program_id(1) * BLOCK_H + tl.arange(0, BLOCK_H)
    cm = cols < H
    acc = tl.zeros((BLOCK_H,), dtype=tl.float32)
    for k in tl.static_range(K):
        p = tl.load(inv_ptr + r * K + k)
        if p >= 0:
            acc += tl.load(y_ptr + p.to(tl.int64) * stride_y + cols, mask=cm, other=0.0).to(tl.float32)
    tl.store(out_ptr + r.to(tl.int64) * stride_out + cols, acc.to(out_ptr.dtype.element_ty), mask=cm)


@triton.jit
def _combine_kernel(
    ret_ptr,
    stride_rank,
    stride_tok,
    hit_ptr,
    stride_hit,
    out_ptr,
    stride_out,
    H,
    R: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    t = tl.program_id(0)
    cols = tl.program_id(1) * BLOCK_H + tl.arange(0, BLOCK_H)
    cm = cols < H
    acc = tl.zeros((BLOCK_H,), dtype=tl.float32)
    for d in tl.static_range(R):
        if tl.load(hit_ptr + t * stride_hit + d) != 0:
            acc += tl.load(ret_ptr + d * stride_rank + t.to(tl.int64) * stride_tok + cols, mask=cm, other=0.0).to(
                tl.float32
            )
    tl.store(out_ptr + t.to(tl.int64) * stride_out + cols, acc.to(out_ptr.dtype.element_ty), mask=cm)


@triton.jit
def _combine_bf16_ring_kernel(
    ret_ptr,
    stride_rank,
    stride_tok,
    hit_ptr,
    stride_hit,
    out_ptr,
    stride_out,
    H: tl.constexpr,
    R: tl.constexpr,
    SOURCE_RANK: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    token = tl.program_id(0)
    columns = tl.program_id(1) * BLOCK_H + tl.arange(0, BLOCK_H)
    acc = tl.zeros((BLOCK_H,), dtype=tl.float32)
    for step in tl.static_range(1, R + 1):
        owner = (SOURCE_RANK + step) % R
        present = tl.load(hit_ptr + token * stride_hit + owner) != 0
        value = tl.load(
            ret_ptr + owner * stride_rank + token.to(tl.int64) * stride_tok + columns,
            mask=(columns < H) & present,
            other=0,
        ).to(tl.float32)
        # Keep each BF16 rounding point explicit; one final cast changes the native sum.
        acc = (acc + value).to(tl.bfloat16).to(tl.float32)
    tl.store(out_ptr + token.to(tl.int64) * stride_out + columns, acc.to(tl.bfloat16), mask=columns < H)


def owner_reduce(y: torch.Tensor, inv: torch.Tensor, out: torch.Tensor, top_k: int, block_h: int = 1024) -> None:
    """y [N, H] expert-major; inv [rows*K] int32; out [rows, H]."""
    rows, H = out.shape
    if rows == 0:
        return
    _owner_reduce_kernel[(rows, triton.cdiv(H, block_h))](
        y, y.stride(0), inv, out, out.stride(0), H, K=top_k, BLOCK_H=block_h
    )


def combine(ret: torch.Tensor, hit: torch.Tensor, out: torch.Tensor, block_h: int = 1024) -> None:
    """ret [R, C, H]; hit [C, R] bool/int; out [C, H]."""
    R, C, H = ret.shape
    if C == 0:
        return
    hit_u8 = hit if hit.dtype == torch.uint8 else hit.to(torch.uint8)
    _combine_kernel[(C, triton.cdiv(H, block_h))](
        ret, ret.stride(0), ret.stride(1), hit_u8, hit_u8.stride(0), out, out.stride(0), H, R=R, BLOCK_H=block_h
    )


def combine_bf16_ring(ret: torch.Tensor, hit: torch.Tensor, rank: int, out=None, block_h: int = 1024) -> torch.Tensor:
    """Fuse the opt-in Ring-compatible sum, preserving each intermediate BF16 rounding."""
    if ret.dtype != torch.bfloat16:
        raise ValueError("BF16 Ring combine requires BF16 partials")
    world, tokens, hidden = ret.shape
    if not 0 <= rank < world:
        raise ValueError("Source rank is outside the partials")
    out = torch.empty_like(ret[0]) if out is None else out
    if out.dtype != torch.bfloat16 or out.shape != ret.shape[1:]:
        raise ValueError("BF16 Ring output must match the token and hidden dimensions")
    if tokens:
        _combine_bf16_ring_kernel[(tokens, triton.cdiv(hidden, block_h))](
            ret,
            ret.stride(0),
            ret.stride(1),
            hit,
            hit.stride(0),
            out,
            out.stride(0),
            H=hidden,
            R=world,
            SOURCE_RANK=rank,
            BLOCK_H=block_h,
        )
    return out

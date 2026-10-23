"""Triton BF16 grouped GEMM over an expert-major layout: C[rows of e] = A[rows of e] @ W[e]^T.

The tile schedule (which expert and row range each M-block covers) is computed on device,
so launching does not require a device-to-host copy of per-expert counts. The grid size
is an upper bound derived from the host-known row count: cdiv(N, BLOCK_M) + num_experts.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _grouped_gemm_kernel(
    a_ptr,
    w_ptr,
    c_ptr,
    blk_expert_ptr,
    blk_row_start_ptr,
    blk_row_end_ptr,
    row_scale_ptr,
    a_index_ptr,
    N,
    K,
    stride_am,
    stride_ak,
    stride_we,
    stride_wn,
    stride_wk,
    stride_cm,
    stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    HAS_ROW_SCALE: tl.constexpr,
    HAS_A_INDEX: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    expert = tl.load(blk_expert_ptr + pid_m)
    if expert < 0:
        return
    row_start = tl.load(blk_row_start_ptr + pid_m)
    row_end = tl.load(blk_row_end_ptr + pid_m)

    rows = row_start + tl.arange(0, BLOCK_M)
    cols = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    row_mask = rows < row_end
    col_mask = cols < N

    if HAS_A_INDEX:  # gather fused into the load: A row of output row r is a[a_index[r]]
        a_rows = tl.load(a_index_ptr + rows, mask=row_mask, other=0).to(tl.int64)
    else:
        a_rows = rows.to(tl.int64)
    w_base = w_ptr + expert.to(tl.int64) * stride_we
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k0 in range(0, K, BLOCK_K):
        ks = k0 + tl.arange(0, BLOCK_K)
        k_mask = ks < K
        a = tl.load(
            a_ptr + a_rows[:, None] * stride_am + ks[None, :] * stride_ak,
            mask=row_mask[:, None] & k_mask[None, :],
            other=0.0,
        )
        w = tl.load(
            w_base + cols[None, :].to(tl.int64) * stride_wn + ks[:, None] * stride_wk,
            mask=k_mask[:, None] & col_mask[None, :],
            other=0.0,
        )
        acc = tl.dot(a, w, acc)
    if HAS_ROW_SCALE:
        # Routing weight applied to the fp32 accumulator before the single cast (vLLM MUL_ROUTED_WEIGHT).
        acc = acc * tl.load(row_scale_ptr + rows, mask=row_mask, other=0.0)[:, None]
    tl.store(
        c_ptr + rows[:, None].to(tl.int64) * stride_cm + cols[None, :] * stride_cn,
        acc.to(c_ptr.dtype.element_ty),
        mask=row_mask[:, None] & col_mask[None, :],
    )


def build_tile_schedule(expert_offsets: torch.Tensor, num_rows: int, block_m: int):
    """Map each M-block id to (expert, row_start, row_end) on device. Unused blocks get expert -1."""
    num_experts = expert_offsets.numel() - 1
    max_blocks = triton.cdiv(num_rows, block_m) + num_experts
    counts = expert_offsets[1:] - expert_offsets[:-1]
    nblk = (counts + block_m - 1) // block_m
    cum = torch.cumsum(nblk, 0)
    blk = torch.arange(max_blocks, device=expert_offsets.device)
    expert = torch.searchsorted(cum, blk, right=True)
    valid = expert < num_experts
    expert_c = expert.clamp(max=num_experts - 1)
    first_blk = cum[expert_c] - nblk[expert_c]
    row_start = expert_offsets[expert_c] + (blk - first_blk) * block_m
    row_end = torch.minimum(row_start + block_m, expert_offsets[expert_c + 1])
    expert = torch.where(valid, expert_c, torch.full_like(expert_c, -1))
    return expert.to(torch.int32), row_start.to(torch.int32), row_end.to(torch.int32), max_blocks


def grouped_gemm(
    a: torch.Tensor,
    w: torch.Tensor,
    expert_offsets: torch.Tensor,
    out: torch.Tensor | None = None,
    row_scale: torch.Tensor | None = None,
    a_index: torch.Tensor | None = None,
    schedule: tuple | None = None,
    block_m: int = 16,
    block_n: int = 128,
    block_k: int = 64,
    num_warps: int = 4,
    num_stages: int = 4,
) -> torch.Tensor:
    """a [M, K] expert-major rows; w [E, N, K]; expert_offsets [E + 1]. Returns [M, N] in a.dtype.

    row_scale [M] fp32, optional: multiplies each output row in fp32 before the cast to a.dtype.
    a_index [M] int, optional: output row r reads A row a_index[r] (a may then have any row count);
        the output has M = a_index.numel() rows. Rows outside the expert segments are not written.
    schedule, optional: precomputed (tile_expert, tile_row_start, tile_row_end, max_tiles) for
        block_m, e.g. from kernels.metadata.ExpertMetadata; expert_offsets is then unused.
    """
    M, K = (a_index.numel(), a.shape[1]) if a_index is not None else a.shape
    E, N, Kw = w.shape
    assert Kw == K and (schedule is not None or expert_offsets.numel() == E + 1)
    if out is None:
        out = torch.empty(M, N, dtype=a.dtype, device=a.device)
    if M == 0:
        return out
    if schedule is not None:
        blk_e, blk_s, blk_t, max_blocks = schedule
    else:
        blk_e, blk_s, blk_t, max_blocks = build_tile_schedule(expert_offsets, M, block_m)
    grid = (max_blocks, triton.cdiv(N, block_n))
    _grouped_gemm_kernel[grid](
        a,
        w,
        out,
        blk_e,
        blk_s,
        blk_t,
        row_scale if row_scale is not None else blk_e,
        a_index if a_index is not None else blk_e,
        N,
        K,
        a.stride(0),
        a.stride(1),
        w.stride(0),
        w.stride(1),
        w.stride(2),
        out.stride(0),
        out.stride(1),
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
        HAS_ROW_SCALE=row_scale is not None,
        HAS_A_INDEX=a_index is not None,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return out

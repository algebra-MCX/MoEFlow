"""Single-kernel GPU metadata for the expert-major layout.

From the received routing metadata (local expert id per (row, slot), valid flag per row) one
Triton program computes, without host involvement:
  * entry_row / entry_slot: (row, slot) pairs in expert-major order, stable in (row, slot);
  * entry_w: the routing weight of every entry (for the GEMM2 epilogue);
  * inv: (row, slot) -> expert-major position, -1 if the slot is not valid;
  * expert_offsets [L + 1];
  * a bounded tile map (tile_expert, tile_row_start, tile_row_end) of `max_tiles` entries in
    the format `grouped_gemm(schedule=...)` consumes, unused tiles marked with expert -1.
It replaces the sort / scatter-add / cumsum / searchsorted chain of `FixedEPLayer.stage_experts`.

Stable expert-major order comes from sorting packed keys `expert * BLOCK + flat_index`
inside one CTA (invalid slots get the sentinel expert L, padding lanes L + 1).
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _metadata_kernel(
    meta_ptr,
    stride_row,
    n_entries,
    entry_row_ptr,
    entry_slot_ptr,
    entry_w_ptr,
    inv_ptr,
    offsets_ptr,
    tile_e_ptr,
    tile_s_ptr,
    tile_t_ptr,
    max_tiles,
    K: tl.constexpr,
    L: tl.constexpr,
    LB: tl.constexpr,
    BLOCK: tl.constexpr,
    BLOCK_M: tl.constexpr,
    TCHUNK: tl.constexpr,
):
    idx = tl.arange(0, BLOCK)
    inb = idx < n_entries
    row = idx // K
    slot = idx % K
    lid = tl.load(meta_ptr + row * stride_row + slot, mask=inb, other=-1)
    valid = tl.load(meta_ptr + row * stride_row + 2 * K, mask=inb, other=0)
    key = tl.where(inb, tl.where((lid >= 0) & (valid != 0), lid, L), L + 1)

    packed = tl.sort(key * BLOCK + idx)
    src = packed % BLOCK
    out_ok = idx < n_entries
    tl.store(entry_row_ptr + idx, src // K, mask=out_ok)
    tl.store(entry_slot_ptr + idx, src % K, mask=out_ok)
    w = tl.load(meta_ptr + (src // K) * stride_row + K + (src % K), mask=out_ok, other=0).to(tl.float32, bitcast=True)
    tl.store(entry_w_ptr + idx, w, mask=out_ok)
    # Inverse map (row, slot) -> expert-major position, -1 for invalid slots.
    sorted_key = packed // BLOCK
    tl.store(inv_ptr + src, tl.where(sorted_key < L, idx, -1), mask=out_ok)

    # Per-expert counts and exclusive offsets (bins >= L hold sentinel / padding lanes).
    counts = tl.histogram(key, LB)  # [LB]
    e_ids = tl.arange(0, LB)
    counts = tl.where(e_ids < L, counts, 0)
    incl = tl.cumsum(counts, 0)
    tl.store(offsets_ptr + 1 + e_ids, incl, mask=e_ids < L)
    tl.store(offsets_ptr + e_ids, incl - counts, mask=e_ids == 0)

    # Tile map: expert e owns tiles [cum_tiles[e] - ntile[e], cum_tiles[e]).
    excl = incl - counts
    ntile = (counts + BLOCK_M - 1) // BLOCK_M
    cum_tiles = tl.cumsum(ntile, 0)
    for t0 in range(0, max_tiles, TCHUNK):
        t = t0 + tl.arange(0, TCHUNK)
        # expert(t) = number of experts whose tile range ends at or before t
        e_of_t = tl.sum((cum_tiles[None, :] <= t[:, None]).to(tl.int32), axis=1)
        sel = (e_of_t[:, None] == e_ids[None, :]).to(tl.int32)  # one-hot over experts
        first = tl.sum(sel * (cum_tiles - ntile)[None, :], axis=1)
        base = tl.sum(sel * excl[None, :], axis=1)
        end = tl.sum(sel * incl[None, :], axis=1)
        start = base + (t - first) * BLOCK_M
        ok = (e_of_t < L) & (t < max_tiles)
        tl.store(tile_e_ptr + t, tl.where(ok, e_of_t, -1), mask=t < max_tiles)
        tl.store(tile_s_ptr + t, start, mask=t < max_tiles)
        tl.store(tile_t_ptr + t, tl.minimum(start + BLOCK_M, end), mask=t < max_tiles)


class ExpertMetadata:
    """Static output buffers (graph-friendly) + launcher."""

    def __init__(self, n_rows: int, top_k: int, num_local_experts: int, block_m: int, device):
        self.N = n_rows * top_k
        self.K, self.L, self.block_m = top_k, num_local_experts, block_m
        self.BLOCK = triton.next_power_of_2(max(self.N, 16))
        self.LB = triton.next_power_of_2(num_local_experts + 2)
        self.max_tiles = triton.cdiv(self.N, block_m) + num_local_experts
        i32 = dict(dtype=torch.int32, device=device)
        self.entry_row = torch.zeros(self.N, **i32)
        self.entry_slot = torch.zeros(self.N, **i32)
        self.entry_w = torch.zeros(self.N, dtype=torch.float32, device=device)
        self.inv = torch.zeros(self.N, **i32)
        self.offsets = torch.zeros(num_local_experts + 1, **i32)
        self.tile_e = torch.zeros(self.max_tiles, **i32)
        self.tile_s = torch.zeros(self.max_tiles, **i32)
        self.tile_t = torch.zeros(self.max_tiles, **i32)

    def build(self, meta_i32: torch.Tensor) -> None:
        """meta_i32 [rows, >= 2K+1] int32 view: K local ids, K fp32-bit weights, valid flag."""
        assert meta_i32.shape[0] * self.K == self.N
        _metadata_kernel[(1,)](
            meta_i32,
            meta_i32.stride(0),
            self.N,
            self.entry_row,
            self.entry_slot,
            self.entry_w,
            self.inv,
            self.offsets,
            self.tile_e,
            self.tile_s,
            self.tile_t,
            self.max_tiles,
            K=self.K,
            L=self.L,
            LB=self.LB,
            BLOCK=self.BLOCK,
            BLOCK_M=self.block_m,
            TCHUNK=64,
            num_warps=8,
        )

    @property
    def schedule(self):
        return self.tile_e, self.tile_s, self.tile_t, self.max_tiles

"""Fixed-capacity, CUDA-graph-capturable EP MoE path for decode.

Same contract as the naive pipeline (docs/architecture.md §4) — one copy per (token, rank),
slot-order fp32 owner reduce, ascending-rank fp32 combine — but with static shapes and no
device-to-host traffic:

* Positions are fixed: token t of rank s bound for rank d always occupies row `send[d, t]`;
  a valid flag says whether t actually hits d. No count exchange is needed.
* Hidden state and routing metadata travel in one row: `[H bf16 | K int32 ids | K fp32 weights
  | int32 valid]` bit-packed into bf16 lanes, so a forward pass issues exactly two equal-split
  all-to-alls (dispatch, return).
* The expert-major layout is a fixed-length stable sort with a sentinel for invalid slots;
  the grouped GEMM computes its tile schedule on device and gathers A rows inside the kernel.

Cost of the static shapes ("padding"): every token is sent to every rank, i.e. the dispatch
moves ep x C rows instead of the number of (token, rank) hits; the expert GEMMs only touch
valid rows. Both are measured by scripts/bench_stages.py.
"""

from __future__ import annotations

import torch
import torch.distributed as dist

from .config import MoEConfig
from .experts import swiglu_clamp
from .kernels.dispatch_pack import dispatch_pack
from .kernels.grouped_gemm import grouped_gemm
from .kernels.metadata import ExpertMetadata
from .kernels.reduce import combine as fused_combine
from .kernels.reduce import owner_reduce
from .router import route


def meta_lanes(top_k: int) -> int:
    """bf16 lanes needed for K int32 ids + K fp32 weights + one int32 valid flag."""
    return 2 * (2 * top_k + 1)


class FixedEPLayer:
    def __init__(
        self,
        cfg: MoEConfig,
        rank: int,
        world: int,
        w13_local: torch.Tensor | None,
        w2_local: torch.Tensor | None,
        capacity: int,
        group: dist.ProcessGroup | None = None,
        gemm_kwargs: dict | None = None,
        metadata_kernel: bool = False,
        fused_reduce: bool = False,
        fused_combine: bool = False,
        device: torch.device | None = None,
        fused_dispatch: bool = False,
    ):
        """Optimisation switches (each A/B-tested on its own):
        metadata_kernel: one Triton kernel builds the expert-major layout + tile map;
        fused_reduce:    one Triton kernel for the owner-side slot-order reduce (needs metadata_kernel);
        fused_combine:   one Triton kernel for the source-side ascending-rank combine.
        fused_dispatch: one Triton kernel fills hidden rows, routing metadata and the hit mask.
        """
        if fused_reduce and not metadata_kernel:
            raise ValueError("fused_reduce consumes the inverse map produced by metadata_kernel")
        self.cfg, self.rank, self.world, self.C = cfg, rank, world, capacity
        self.L = cfg.experts_per_rank(world)
        self.K, self.H = cfg.top_k, cfg.hidden_size
        self.w13, self.w2 = w13_local, w2_local
        self.group = group
        self.gemm_kwargs = gemm_kwargs or {}
        dev = device if device is not None else w13_local.device
        R, C, K, H = world, capacity, self.K, self.H
        self.row_width = H + meta_lanes(K)
        # Static buffers: their addresses are what a captured graph replays against.
        self.send = torch.zeros(R, C, self.row_width, dtype=torch.bfloat16, device=dev)
        self.recv = torch.zeros_like(self.send)
        self.ret_send = torch.zeros(R, C, H, dtype=torch.bfloat16, device=dev)
        self.ret_recv = torch.zeros_like(self.ret_send)
        self.n_entries = R * C * K
        self._ones = torch.ones(self.n_entries, dtype=torch.int64, device=dev)
        self._dst_ids = torch.arange(R, device=dev)
        self._tok_ids = torch.arange(C, device=dev)
        self.fused_reduce, self.fused_combine = fused_reduce, fused_combine
        self.fused_dispatch = fused_dispatch
        self.block_m = self.gemm_kwargs.get("block_m", 16)
        self.md = ExpertMetadata(R * C, K, self.L, self.block_m, dev) if metadata_kernel else None
        self.out = torch.zeros(C, H, dtype=torch.bfloat16, device=dev)

    # -- stage 1 (source): route and pack ---------------------------------------------------------
    def stage_route(self, logits: torch.Tensor, bias: torch.Tensor | None) -> tuple[torch.Tensor, torch.Tensor]:
        """ROUTER: logits [C, E] -> (weights [C, K] fp32, ids [C, K] int64)."""
        weights, ids = route(logits, bias, self.cfg)
        return weights, ids.long()

    def stage_dispatch_pack(
        self, x: torch.Tensor, weights: torch.Tensor, ids: torch.Tensor, num_valid: torch.Tensor | None = None
    ) -> torch.Tensor:
        """DISPATCH: fill the static send buffer. Rows t >= num_valid (device scalar) are padding.

        Returns hit [C, R] (token t has at least one expert on rank r).
        """
        R, K, H, L = self.world, self.K, self.H, self.L
        T = x.shape[0]
        assert T <= self.C, f"{T} tokens exceed capacity {self.C}"
        if self.fused_dispatch:
            return dispatch_pack(x, weights, ids, self.send, L, num_valid)
        dst = ids // L  # [T, K]
        on = dst.unsqueeze(1) == self._dst_ids.view(1, R, 1)  # [C, R, K]
        hit = on.any(-1)  # [C, R]
        if num_valid is not None:
            hit &= (self._tok_ids[:T] < num_valid).unsqueeze(1)
        local = torch.where(on, (ids - dst * L).unsqueeze(1), torch.full_like(on, -1, dtype=torch.long))
        local = torch.where(hit.unsqueeze(-1), local, -1).to(torch.int32)  # [C, R, K]
        wts = torch.where(on & hit.unsqueeze(-1), weights.unsqueeze(1), 0.0).float()

        send = self.send
        send[:, :T, :H] = x.unsqueeze(0)
        meta = send[:, :, H:].view(torch.int32)  # [R, C, 2K+1]
        meta[:, :T, :K] = local.transpose(0, 1)
        meta[:, :T, K : 2 * K] = wts.transpose(0, 1).view(torch.int32)
        meta[:, :T, 2 * K] = hit.t().to(torch.int32)
        if T < self.C:  # rows beyond T carry no tokens this call
            meta[:, T:, 2 * K] = 0
            hit = torch.cat([hit, hit.new_zeros(self.C - T, R)])
        return hit

    def stage_pack(
        self, x: torch.Tensor, logits: torch.Tensor, bias: torch.Tensor | None, num_valid: torch.Tensor | None = None
    ) -> torch.Tensor:
        """ROUTER + DISPATCH. Returns hit [C, R]."""
        weights, ids = self.stage_route(logits, bias)
        return self.stage_dispatch_pack(x, weights, ids, num_valid)

    # -- stage 2 (destination): experts + local reduce ------------------------------------------
    def stage_experts(self, expert_fn=None, prepare_fn=None, *, direct_output=False) -> torch.Tensor | None:
        """Consumes self.recv [R(src), C, row], fills self.ret_send [R(src), C, H] with partials.

        expert_fn(x_rows [RC, H], local_ids [RC, K] (-1 = no hit), weights [RC, K] fp32) -> [RC, H]
        replaces the built-in Triton experts + owner reduce (e.g. vLLM's FP8 fused experts).
        Optional prepare_fn owns the received-row format and returns the arguments for expert_fn.
        """
        R, C, K, H = self.world, self.C, self.K, self.H
        rows = self.recv.view(R * C, self.row_width)
        ret = self.ret_send.view(R * C, H)
        if direct_output and prepare_fn is None:
            raise ValueError("Direct expert output requires custom input preparation")
        if prepare_fn is not None:
            if expert_fn is None:
                raise ValueError("Custom input preparation requires an external expert function")
            output = expert_fn(*prepare_fn(rows))
            if not direct_output:
                ret.copy_(output)
                return None
            if (
                output.shape != (R * C, H)
                or output.dtype != self.ret_send.dtype
                or output.device != self.ret_send.device
                or not output.is_contiguous()
            ):
                raise ValueError("Direct expert output must be contiguous and match return shape, dtype and device")
            return output.view_as(self.ret_send)
        meta = rows[:, H:].view(torch.int32)
        local = torch.where((meta[:, 2 * K] != 0).unsqueeze(1), meta[:, :K].long(), -1)  # [RC, K]
        wts = meta[:, K : 2 * K].view(torch.float32)
        if expert_fn is not None:
            ret.copy_(expert_fn(rows[:, :H], local, wts))
        else:
            self._builtin_experts(rows[:, :H], meta, local, wts)
        return None

    def _builtin_experts(self, x, meta, local, wts) -> None:
        """Triton grouped GEMM experts in expert-major order, then the slot-ordered owner reduce."""
        R, C, K, H, L = self.world, self.C, self.K, self.H, self.L
        if self.md is not None:
            self.md.build(meta)
            offsets, entry_row, entry_w = self.md.offsets, self.md.entry_row, self.md.entry_w
            sched = {"schedule": self.md.schedule}
        else:
            keys = torch.where(local >= 0, local, L).reshape(-1)  # sentinel L sorts last
            order = torch.sort(keys, stable=True).indices  # (row, slot) ascending per expert
            counts = torch.zeros(L + 1, dtype=torch.int64, device=keys.device).scatter_add_(0, keys, self._ones)
            offsets = torch.zeros(L + 1, dtype=torch.int64, device=keys.device)
            offsets[1:] = torch.cumsum(counts[:L], 0)
            entry_row, entry_slot = order // K, order % K
            entry_w = wts[entry_row, entry_slot]
            sched = {}

        h = grouped_gemm(x, self.w13, offsets, a_index=entry_row, **sched, **self.gemm_kwargs)
        act = swiglu_clamp(h, self.cfg.swiglu_limit)
        y = grouped_gemm(act, self.w2, offsets, row_scale=entry_w, **sched, **self.gemm_kwargs)  # bf16(w * acc)

        ret = self.ret_send.view(R * C, H)
        if self.fused_reduce:
            owner_reduce(y, self.md.inv, ret, K)
            return
        if self.md is not None:
            entry_row, entry_slot = self.md.entry_row.long(), self.md.entry_slot.long()
        # Owner reduce in slot order; entries past offsets[L] are routed to a dummy row RC.
        valid_entry = torch.arange(self.n_entries, device=y.device) < offsets[L]
        tgt_row = torch.where(valid_entry, entry_row, R * C)
        slots = torch.zeros(R * C + 1, K, H, dtype=y.dtype, device=y.device)
        slots[tgt_row, entry_slot] = y
        present = local >= 0
        acc = torch.zeros(R * C, H, dtype=torch.float32, device=y.device)
        for k in range(K):
            acc = torch.where(present[:, k : k + 1], acc + slots[: R * C, k].float(), acc)
        ret.copy_(acc)

    # -- stage 3 (source): combine --------------------------------------------------------------
    def stage_combine(self, hit: torch.Tensor, out_dtype: torch.dtype) -> torch.Tensor:
        if self.fused_combine and out_dtype == self.out.dtype:
            fused_combine(self.ret_recv, hit, self.out)
            return self.out
        acc = torch.zeros(self.C, self.H, dtype=torch.float32, device=hit.device)
        for d in range(self.world):  # ascending rank, as DeepEP's combine
            acc = torch.where(hit[:, d : d + 1], acc + self.ret_recv[d].float(), acc)
        return acc.to(out_dtype)

    # -- SPMD entry points (capturable) ---------------------------------------------------------
    def forward_routed(
        self,
        x,
        weights,
        ids,
        num_valid: torch.Tensor | None = None,
        expert_fn=None,
        combine_fn=None,
        prepare_fn=None,
        direct_expert_output=False,
    ):
        """Pre-routed variant; optional adapter combine consumes owner partials and the hit mask."""
        T = x.shape[0]
        hit = self.stage_dispatch_pack(x, weights, ids if self.fused_dispatch else ids.long(), num_valid)
        dist.all_to_all_single(self.recv, self.send, group=self.group)
        partials = self.stage_experts(expert_fn, prepare_fn=prepare_fn, direct_output=direct_expert_output)
        # Keep the returned allocation alive until the collective has consumed it.
        dist.all_to_all_single(self.ret_recv, self.ret_send if partials is None else partials, group=self.group)
        out = self.stage_combine(hit, x.dtype) if combine_fn is None else combine_fn(self.ret_recv, hit)
        return out[:T]

    def forward(self, x, logits, bias, num_valid: torch.Tensor | None = None, stop_after: str | None = None):
        """Full layer; `stop_after` (one of STAGES) truncates it for prefix-graph stage timing."""
        weights, ids = self.stage_route(logits, bias)
        if stop_after == "ROUTER":
            return weights
        hit = self.stage_dispatch_pack(x, weights, ids, num_valid)
        if stop_after == "DISPATCH":
            return hit
        dist.all_to_all_single(self.recv, self.send, group=self.group)
        if stop_after == "A2A_FORWARD":
            return self.recv
        self.stage_experts()
        if stop_after == "EXPERT":
            return self.ret_send
        dist.all_to_all_single(self.ret_recv, self.ret_send, group=self.group)
        if stop_after == "A2A_RETURN":
            return self.ret_recv
        return self.stage_combine(hit, x.dtype)


STAGES = ("ROUTER", "DISPATCH", "A2A_FORWARD", "EXPERT", "A2A_RETURN", "COMBINE")


def simulate_fixed(layers: list[FixedEPLayer], xs, logits, bias, num_valid=None) -> list[torch.Tensor]:
    """Single-process lockstep driver: equal-split all-to-all is recv[d][s] = send[s][d]."""
    R = len(layers)
    hits = [
        ly.stage_pack(x, lg, bias, None if num_valid is None else num_valid[i])
        for i, (ly, x, lg) in enumerate(zip(layers, xs, logits))
    ]
    for d in range(R):
        for s in range(R):
            layers[d].recv[s].copy_(layers[s].send[d])
    for ly in layers:
        ly.stage_experts()
    for d in range(R):
        for s in range(R):
            layers[d].ret_recv[s].copy_(layers[s].ret_send[d])
    return [ly.stage_combine(h, x.dtype) for ly, h, x in zip(layers, hits, xs)]

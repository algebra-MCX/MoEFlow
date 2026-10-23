"""All-to-all transports with a common row-exchange contract.

exchange(send, send_counts) semantics, for every transport:
  * `send` rows are grouped by destination rank in ascending order; `send_counts[d]` rows go to d.
  * The receiver gets rows grouped by source rank in ascending order, each source's rows in
    the order the source sent them; `recv_counts[s]` rows came from s.
"""

from __future__ import annotations

from typing import Protocol

import torch
import torch.distributed as dist


class RowExchange(Protocol):
    rank: int
    world: int

    def exchange_counts(self, send_counts: torch.Tensor) -> torch.Tensor: ...

    def exchange(self, send: torch.Tensor, send_counts: torch.Tensor, recv_counts: torch.Tensor) -> torch.Tensor: ...


class NcclRowExchange:
    """Naive NCCL baseline: counts are copied to host to size the variable-split all_to_all."""

    def __init__(self, group: dist.ProcessGroup | None = None):
        self.group = group
        self.rank = dist.get_rank(group)
        self.world = dist.get_world_size(group)

    def exchange_counts(self, send_counts: torch.Tensor) -> torch.Tensor:
        recv_counts = torch.empty_like(send_counts)
        dist.all_to_all_single(recv_counts, send_counts.contiguous(), group=self.group)
        return recv_counts

    def exchange(self, send, send_counts, recv_counts):
        in_splits = send_counts.tolist()  # D2H sync (by design in the naive path)
        out_splits = recv_counts.tolist()
        recv = torch.empty((sum(out_splits),) + tuple(send.shape[1:]), dtype=send.dtype, device=send.device)
        dist.all_to_all_single(recv, send.contiguous(), out_splits, in_splits, group=self.group)
        return recv


def simulated_exchange(
    sends: list[torch.Tensor], send_counts: list[torch.Tensor]
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """Single-process stand-in for one collective across all ranks (used by tests and the hand example)."""
    world = len(sends)
    chunks = [list(torch.split(s, c.tolist(), dim=0)) for s, c in zip(sends, send_counts)]
    recvs = [torch.cat([chunks[src][dst] for src in range(world)], 0) for dst in range(world)]
    recv_counts = [torch.stack([send_counts[src][dst] for src in range(world)]) for dst in range(world)]
    return recvs, recv_counts


def pack_meta(local_id: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """[S, K] int32 + [S, K] fp32 -> [S, 2K] int32 (weights bit-cast) so metadata moves in one collective."""
    return torch.cat([local_id.to(torch.int32), weight.float().view(torch.int32)], dim=1)


def unpack_meta(meta: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    k = meta.shape[1] // 2
    return meta[:, :k].contiguous(), meta[:, k:].contiguous().view(torch.float32)

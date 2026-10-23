"""NCCL all_to_all_single probe: one process per GPU, launched with torchrun.

Rank r sends value (r * 1000 + dst) to every dst; each rank verifies what it received, for an
even and an uneven split. Unless PROBE_SKIP_GRAPH=1, one all_to_all_single is also captured in
a CUDA Graph and replayed. Each teardown step is logged so a hang can be located.
"""

import datetime
import json
import os

import torch
import torch.distributed as dist


def check_even(rank: int, world: int, chunk: int = 4096) -> bool:
    send = torch.empty(world * chunk, device="cuda", dtype=torch.float32)
    for dst in range(world):
        send[dst * chunk : (dst + 1) * chunk] = rank * 1000 + dst
    recv = torch.empty_like(send)
    dist.all_to_all_single(recv, send)
    torch.cuda.synchronize()
    expect = torch.cat([torch.full((chunk,), src * 1000 + rank, device="cuda") for src in range(world)])
    return torch.equal(recv, expect)


def check_uneven(rank: int, world: int) -> bool:
    """Rank r sends (dst + 1) * (r + 1) elements to dst."""
    in_splits = [(d + 1) * (rank + 1) for d in range(world)]
    out_splits = [(rank + 1) * (s + 1) for s in range(world)]
    send = torch.cat([torch.full((n,), rank * 1000.0 + d, device="cuda") for d, n in enumerate(in_splits)])
    recv = torch.empty(sum(out_splits), device="cuda", dtype=torch.float32)
    dist.all_to_all_single(recv, send, out_splits, in_splits)
    torch.cuda.synchronize()
    expect = torch.cat([torch.full((n,), s * 1000.0 + rank, device="cuda") for s, n in enumerate(out_splits)])
    return torch.equal(recv, expect)


def check_graph(rank: int, world: int, chunk: int = 4096) -> str:
    send = torch.cat([torch.full((chunk,), rank * 1000.0 + d, device="cuda") for d in range(world)])
    recv = torch.empty_like(send)
    expect = torch.cat([torch.full((chunk,), s * 1000.0 + rank, device="cuda") for s in range(world)])
    try:
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            dist.all_to_all_single(recv, send)  # warmup outside capture
        torch.cuda.current_stream().wait_stream(side)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            dist.all_to_all_single(recv, send)
        recv.zero_()
        torch.cuda.synchronize()
        graph.replay()
        torch.cuda.synchronize()
        return "replay_correct" if torch.equal(recv, expect) else "replay_WRONG"
    except Exception as e:  # reported, not hidden
        return f"error: {type(e).__name__}: {e}"[:300]


def main():
    rank, world = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
    local = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local)
    dist.init_process_group("nccl", timeout=datetime.timedelta(seconds=120), device_id=torch.device("cuda", local))
    res = {
        "rank": rank,
        "world": world,
        "nccl": ".".join(map(str, torch.cuda.nccl.version())),
        "eager_even_ok": check_even(rank, world),
        "eager_uneven_ok": check_uneven(rank, world),
        "cuda_graph": "skipped" if os.environ.get("PROBE_SKIP_GRAPH") == "1" else check_graph(rank, world),
    }
    out = [None] * world
    dist.all_gather_object(out, res)
    if rank == 0:
        print(json.dumps(out), flush=True)
    print(f"[rank{rank}] before barrier", flush=True)
    dist.barrier()
    print(f"[rank{rank}] before destroy", flush=True)
    dist.destroy_process_group()
    print(f"[rank{rank}] destroyed", flush=True)
    ok = all(
        r["eager_even_ok"] and r["eager_uneven_ok"] and r["cuda_graph"] in ("skipped", "replay_correct") for r in out
    )
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()

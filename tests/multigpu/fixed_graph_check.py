"""CUDA-graph gate for the fixed-capacity decode path, one process per GPU.

Checks, in order:
  G1 the sequence of collectives recorded while capturing is identical on every rank;
  G2 static buffer addresses do not change across replays;
  G3 replaying one graph with new token counts (num_valid) and routing distributions gives
     outputs bitwise equal to the eager naive-NCCL pipeline, and zeros on padding rows;
  G4 graphs of several capacities replayed in a random interleaving for `--replays` steps
     show no contamination and no hang (watchdog);
  G5 graphs are destroyed before the process group (limitations L9).
"""

from __future__ import annotations

import argparse
import datetime
import faulthandler
import gc
import json
import os
import random
import sys

import torch
import torch.distributed as dist
from bench.record import env_info, git_info

from moeflow.comm import NcclRowExchange
from moeflow.config import GLM53_FLASH, MoEConfig
from moeflow.ep_moe import EPMoELayer, shard_experts
from moeflow.experts import TritonGroupedExperts
from moeflow.fixed import FixedEPLayer
from moeflow.testing import DISTRIBUTIONS, make_bias, make_hidden, make_logits, make_weights

SMALL = MoEConfig(hidden_size=512, moe_intermediate_size=256, n_routed_experts=32, top_k=4, swiglu_limit=1.0)


class CollectiveLog:
    """Records (op, numel, dtype) of every all_to_all_single issued through dist during capture."""

    def __init__(self):
        self.calls = []
        self._orig = dist.all_to_all_single

    def __enter__(self):
        def wrapped(out, inp, *a, **kw):
            self.calls.append(("all_to_all_single", out.numel(), str(out.dtype)))
            return self._orig(out, inp, *a, **kw)

        dist.all_to_all_single = wrapped
        return self

    def __exit__(self, *exc):
        dist.all_to_all_single = self._orig


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shape", choices=["small", "full"], default="small")
    ap.add_argument("--capacities", default="1,8,64")
    ap.add_argument("--replays", type=int, default=1000)
    ap.add_argument("--check-every", type=int, default=1)
    ap.add_argument("--watchdog", type=int, default=120)
    ap.add_argument("--metadata-kernel", action="store_true")
    ap.add_argument("--fused-reduce", action="store_true")
    ap.add_argument("--fused-combine", action="store_true")
    ap.add_argument("--fused-dispatch", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    rank, world, local = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"]), int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local)
    dist.init_process_group(
        "nccl", timeout=datetime.timedelta(seconds=args.watchdog), device_id=torch.device("cuda", local)
    )
    faulthandler.dump_traceback_later(args.watchdog * 3, exit=True)
    cfg = GLM53_FLASH if args.shape == "full" else SMALL
    dev = torch.device("cuda", local)
    w13, w2 = make_weights(cfg, 0, device=dev)
    bias = make_bias(cfg, 1, device=dev)
    l13, l2 = (t.contiguous() for t in shard_experts(w13, w2, rank, world))
    naive = EPMoELayer(cfg, rank, world, l13, l2, experts=TritonGroupedExperts())
    comm = NcclRowExchange()
    caps = [int(c) for c in args.capacities.split(",")]

    graphs, static, layers, logs = {}, {}, {}, {}
    for C in caps:
        faulthandler.dump_traceback_later(args.watchdog, exit=True)
        layer = FixedEPLayer(
            cfg,
            rank,
            world,
            l13,
            l2,
            capacity=C,
            metadata_kernel=args.metadata_kernel,
            fused_reduce=args.fused_reduce,
            fused_combine=args.fused_combine,
            fused_dispatch=args.fused_dispatch,
        )
        xb = torch.zeros(C, cfg.hidden_size, dtype=torch.bfloat16, device=dev)
        lb = torch.zeros(C, cfg.n_routed_experts, dtype=torch.float32, device=dev)
        nvb = torch.tensor(C, device=dev)
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            layer.forward(xb, lb, bias, nvb)  # warm-up (Triton compile, NCCL setup) outside capture
        torch.cuda.current_stream().wait_stream(side)
        torch.cuda.synchronize()
        dist.barrier(device_ids=[local])
        g = torch.cuda.CUDAGraph()
        with CollectiveLog() as log, torch.cuda.graph(g):
            ob = layer.forward(xb, lb, bias, nvb)
        torch.cuda.synchronize()
        graphs[C], static[C], layers[C], logs[C] = g, (xb, lb, nvb, ob), layer, log.calls
        # The dictionary must own the only graph reference. A surviving loop variable keeps
        # the final NCCL graph alive across destroy_process_group and can hang teardown.
        del g
    ptrs = {
        C: (
            layers[C].send.data_ptr(),
            layers[C].recv.data_ptr(),
            layers[C].ret_send.data_ptr(),
            layers[C].ret_recv.data_ptr(),
            static[C][3].data_ptr(),
        )
        for C in caps
    }

    # G1: identical collective sequences across ranks.
    all_logs = [None] * world
    dist.all_gather_object(all_logs, logs)
    g1 = all(lg == all_logs[0] for lg in all_logs)

    # G3/G4: random interleaving of capacities; schedule shared by all ranks, inputs per rank.
    sched = random.Random(1234)
    mismatches, pad_dirty, checked = 0, 0, 0
    for step in range(args.replays):
        faulthandler.dump_traceback_later(args.watchdog, exit=True)
        C = sched.choice(caps)
        dist_name = DISTRIBUTIONS[sched.randrange(3)]
        mine = random.Random(step * 1000 + rank)
        v = mine.randint(0, C)
        xb, lb, nvb, ob = static[C]
        x = make_hidden(C, cfg, step * 7 + rank, device=dev)
        lg = make_logits(C, cfg, step * 7 + rank + 3, dist=dist_name, device=dev)
        xb.copy_(x)
        lb.copy_(lg)
        nvb.fill_(v)
        graphs[C].replay()
        if step % args.check_every == 0:
            torch.cuda.synchronize()
            ref = naive.forward(x[:v], lg[:v], bias, comm)  # collective: every rank checks the same steps
            torch.cuda.synchronize()
            mismatches += int(not torch.equal(ob[:v], ref))
            pad_dirty += int(not bool(torch.all(ob[v:] == 0)))
            checked += 1
    torch.cuda.synchronize()

    g2 = ptrs == {
        C: (
            layers[C].send.data_ptr(),
            layers[C].recv.data_ptr(),
            layers[C].ret_send.data_ptr(),
            layers[C].ret_recv.data_ptr(),
            static[C][3].data_ptr(),
        )
        for C in caps
    }
    res = {
        "rank": rank,
        "G1_collective_order_identical": g1,
        "G2_static_addresses": g2,
        "G3_G4_checked_replays": checked,
        "G3_G4_mismatches": mismatches,
        "padding_rows_dirty": pad_dirty,
        "collectives_per_forward": {str(C): len(logs[C]) for C in caps},
    }
    out = [None] * world
    dist.all_gather_object(out, res)
    ok = all(
        r["G1_collective_order_identical"]
        and r["G2_static_addresses"]
        and r["G3_G4_mismatches"] == 0
        and r["padding_rows_dirty"] == 0
        for r in out
    )

    # G5: release graphs before tearing down NCCL.
    graphs.clear()
    gc.collect()
    torch.cuda.synchronize()
    dist.barrier(device_ids=[local])
    print(f"[rank{rank}] before destroy_process_group", flush=True)
    dist.destroy_process_group()
    print(f"[rank{rank}] process_group_destroyed", flush=True)
    if rank == 0:
        summary = {
            "world": world,
            "shape": args.shape,
            "capacities": caps,
            "replays": args.replays,
            "passed": ok,
            "G5_rank0_process_group_destroyed": True,
            "flags": {
                "metadata_kernel": args.metadata_kernel,
                "fused_reduce": args.fused_reduce,
                "fused_combine": args.fused_combine,
                "fused_dispatch": args.fused_dispatch,
            },
            "git": git_info(),
            "env": env_info(),
            "seed": 1234,
            "command": [sys.executable, *sys.argv],
            "per_rank": out,
        }
        print(json.dumps(summary, indent=1), flush=True)
        if args.out:
            with open(args.out, "w") as f:
                json.dump(summary, f, indent=1)
    faulthandler.cancel_dump_traceback_later()
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()

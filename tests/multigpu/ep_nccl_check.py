"""Multi-GPU check of the NCCL EP pipeline (one process per GPU, launched by torchrun).

For every case, each rank verifies that
  1. its output is bitwise identical to the single-process simulator on the same inputs
     (the transport must be pure data movement), and
  2. its output matches the naive reference within the frozen tolerance.
Then it runs `--iters` back-to-back forwards to check for hangs. A watchdog aborts the
process if any step stalls for longer than `--watchdog` seconds.
"""

from __future__ import annotations

import argparse
import datetime
import faulthandler
import json
import os
import sys

import torch
import torch.distributed as dist

from moeflow.comm import NcclRowExchange
from moeflow.config import GLM53_FLASH, MoEConfig
from moeflow.ep_moe import EPMoELayer, shard_experts, simulate_ep
from moeflow.experts import EXPERT_BACKENDS
from moeflow.reference import moe_reference
from moeflow.testing import DISTRIBUTIONS, make_bias, make_hidden, make_logits, make_weights
from moeflow.tolerances import gemm_tolerance, normwise_rel_err

SMALL = MoEConfig(hidden_size=256, moe_intermediate_size=128, n_routed_experts=32, top_k=4, swiglu_limit=1.0)


def token_counts(world: int, case: str, seed: int) -> list[int]:
    g = torch.Generator().manual_seed(seed)
    if case == "empty_rank":
        return [0 if r == world - 1 else int(torch.randint(1, 9, (1,), generator=g)) for r in range(world)]
    if case == "skewed":
        return [64 if r == 0 else 1 for r in range(world)]
    return [int(torch.randint(1, 17, (1,), generator=g)) for _ in range(world)]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--shape", choices=["small", "full"], default="small")
    p.add_argument("--experts", choices=list(EXPERT_BACKENDS), default="triton_grouped")
    p.add_argument("--iters", type=int, default=50)
    p.add_argument("--watchdog", type=int, default=120)
    p.add_argument("--out", default=None)
    args = p.parse_args()

    faulthandler.dump_traceback_later(args.watchdog, exit=True)
    rank, world = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
    local = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local)
    dist.init_process_group(
        "nccl", timeout=datetime.timedelta(seconds=args.watchdog), device_id=torch.device("cuda", local)
    )
    comm = NcclRowExchange()
    cfg = GLM53_FLASH if args.shape == "full" else SMALL
    dev = torch.device("cuda", local)

    w13, w2 = make_weights(cfg, 0, device=dev)
    bias = make_bias(cfg, 1, device=dev)
    layers = [
        EPMoELayer(cfg, r, world, *shard_experts(w13, w2, r, world), experts=EXPERT_BACKENDS[args.experts]())
        for r in range(world)
    ]
    me = layers[rank]

    results = []
    for ci, (case, dist_name) in enumerate([(c, d) for c in ("random", "empty_rank", "skewed") for d in DISTRIBUTIONS]):
        faulthandler.dump_traceback_later(args.watchdog, exit=True)
        counts = token_counts(world, case, ci)
        if case == "skewed":
            dist_name = "hotspot"
        xs = [make_hidden(n, cfg, 100 * ci + r, device=dev) for r, n in enumerate(counts)]
        lgs = [make_logits(n, cfg, 100 * ci + 50 + r, dist=dist_name, device=dev) for r, n in enumerate(counts)]
        out = me.forward(xs[rank], lgs[rank], bias, comm)
        torch.cuda.synchronize()
        sim = simulate_ep(layers, xs, lgs, bias)[rank]
        bitwise = bool(torch.equal(out, sim))
        err = normwise_rel_err(out, moe_reference(xs[rank], lgs[rank], bias, w13, w2, cfg)) if counts[rank] else 0.0
        results.append(
            {
                "case": case,
                "dist": dist_name,
                "tokens": counts,
                "bitwise_vs_sim": bitwise,
                "rel_err": err,
                "tol": gemm_tolerance(),
                "ok": bitwise and err <= gemm_tolerance(),
            }
        )

    # Hang check: back-to-back forwards with changing token counts and routing.
    for it in range(args.iters):
        faulthandler.dump_traceback_later(args.watchdog, exit=True)
        n = (it * 7 + rank * 3) % 17
        me.forward(
            make_hidden(n, cfg, it, device=dev),
            make_logits(n, cfg, it + 1, dist=DISTRIBUTIONS[it % 3], device=dev),
            bias,
            comm,
        )
    torch.cuda.synchronize()
    faulthandler.cancel_dump_traceback_later()

    gathered = [None] * world
    dist.all_gather_object(gathered, results)
    ok = all(r["ok"] for rr in gathered for r in rr)
    if rank == 0:
        summary = {
            "world": world,
            "shape": args.shape,
            "experts": args.experts,
            "iters": args.iters,
            "passed": ok,
            "per_rank": gathered,
        }
        text = json.dumps(summary, indent=1)
        print(text)
        if args.out:
            with open(args.out, "w") as f:
                f.write(text)
    dist.barrier()
    dist.destroy_process_group()
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()

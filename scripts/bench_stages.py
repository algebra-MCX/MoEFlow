"""Six-stage breakdown of the fixed-capacity path under CUDA-graph replay (prefix-graph differencing).

For each stage S in ROUTER, DISPATCH, A2A_FORWARD, EXPERT, A2A_RETURN, COMBINE a graph of the
layer truncated after S is captured and timed (barrier-aligned, max over ranks);
stage time = t(prefix up to S) - t(prefix up to the previous stage). Also reports the
padding cost: the full-layer replay time at num_valid = 1 vs num_valid = C for the same graph.

    torchrun --nproc-per-node=4 scripts/bench_stages.py --tokens 1,8,64,128 --out results/stage3/stages.json
"""

from __future__ import annotations

import argparse
import faulthandler

import torch
import torch.distributed as dist
from bench.backends import init_dist
from bench.inputs import load_weights
from bench.record import make_record, write_record
from bench.stats import summarize
from bench.timing import time_graph

from moeflow.config import GLM53_FLASH
from moeflow.ep_moe import shard_experts
from moeflow.fixed import STAGES, FixedEPLayer
from moeflow.testing import make_hidden, make_logits


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", default="1,2,4,8,16,32,64,128")
    ap.add_argument("--dist", default="uniform")
    ap.add_argument("--weights", default="real:3")
    ap.add_argument("--reps", type=int, default=10)
    ap.add_argument("--inner", type=int, default=20)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    info, _ = init_dist(None)
    cfg = GLM53_FLASH
    faulthandler.dump_traceback_later(1800, exit=True)
    w13, w2, bias = load_weights(args.weights, cfg, info.device)
    l13, l2 = (t.contiguous() for t in shard_experts(w13, w2, info.rank, info.world))
    del w13, w2
    points = []
    for C in [int(t) for t in args.tokens.split(",")]:
        faulthandler.dump_traceback_later(120, exit=True)
        ly = FixedEPLayer(cfg, info.rank, info.world, l13, l2, capacity=C)
        x = make_hidden(C, cfg, 100 + C + info.rank, device=info.device)
        lg = make_logits(C, cfg, 200 + C + info.rank, dist=args.dist, device=info.device)
        nv = torch.tensor(C, device=info.device)
        prefix = {}
        for s in STAGES:
            t = time_graph(
                lambda s=s, ly=ly, x=x, lg=lg, nv=nv: ly.forward(
                    x, lg, bias, nv, stop_after=None if s == "COMBINE" else s
                ),
                reps=args.reps,
                inner=args.inner,
            )
            assert t.capturable, t.error
            prefix[s] = summarize(t.per_call_us).to_dict()
        stage_us, prev = {}, 0.0
        for s in STAGES:
            stage_us[s] = prefix[s]["median"] - prev
            prev = prefix[s]["median"]
        nv.fill_(1)
        one = time_graph(lambda ly=ly, x=x, lg=lg, nv=nv: ly.forward(x, lg, bias, nv), reps=args.reps, inner=args.inner)
        points.append(
            {
                "capacity": C,
                "prefix_us": prefix,
                "stage_median_us": stage_us,
                "full_valid_C_us": prefix["COMBINE"]["median"],
                "full_valid_1_us": summarize(one.per_call_us).median,
            }
        )
        if info.rank == 0:
            print(
                f"C={C}: "
                + ", ".join(f"{k} {v:.1f}" for k, v in stage_us.items())
                + f" | total {prefix['COMBINE']['median']:.1f} us (valid=1: {points[-1]['full_valid_1_us']:.1f})",
                flush=True,
            )
        del ly
        torch.cuda.empty_cache()
    if info.rank == 0:
        write_record(
            args.out, make_record("stage_breakdown_fixed", {**vars(args), "world": info.world}, {"points": points})
        )
    dist.barrier(device_ids=[info.local_rank])
    dist.destroy_process_group()
    faulthandler.cancel_dump_traceback_later()


if __name__ == "__main__":
    main()

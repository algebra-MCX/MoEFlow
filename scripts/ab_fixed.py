"""A/B ablation of one fixed-path optimisation flag under CUDA-graph replay.

    torchrun --nproc-per-node=4 scripts/ab_fixed.py --a "" --b "metadata_kernel" --tokens 1,8,64,128 \
        --out results/stage3/ab_metadata.json

--a / --b are comma-separated FixedEPLayer flags. Both arms are checked bitwise-equal before timing.
"""

from __future__ import annotations

import argparse
import faulthandler

import torch
import torch.distributed as dist
from bench.backends import init_dist
from bench.inputs import load_weights
from bench.record import make_record, write_record
from bench.stats import paired_ab, summarize
from bench.timing import time_graphs_alternating
from bench.validation import require_matching_variants

from moeflow.config import GLM53_FLASH
from moeflow.ep_moe import shard_experts
from moeflow.fixed import FixedEPLayer
from moeflow.testing import make_hidden, make_logits


def flags(spec: str) -> dict:
    return {f: True for f in spec.split(",") if f}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", default="")
    ap.add_argument("--b", required=True)
    ap.add_argument("--tokens", default="1,2,4,8,16,32,64,128")
    ap.add_argument("--dist", default="uniform")
    ap.add_argument("--weights", default="real:3")
    ap.add_argument("--reps", type=int, default=20)
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
        la = FixedEPLayer(cfg, info.rank, info.world, l13, l2, capacity=C, **flags(args.a))
        lb = FixedEPLayer(cfg, info.rank, info.world, l13, l2, capacity=C, **flags(args.b))
        x = make_hidden(C, cfg, 300 + C + info.rank, device=info.device)
        lg = make_logits(C, cfg, 400 + C + info.rank, dist=args.dist, device=info.device)
        require_matching_variants(la.forward(x, lg, bias).clone(), lb.forward(x, lg, bias).clone())
        a, b = time_graphs_alternating(
            lambda la=la, x=x, lg=lg: la.forward(x, lg, bias),
            lambda lb=lb, x=x, lg=lg: lb.forward(x, lg, bias),
            reps=args.reps,
            inner=args.inner,
        )
        r = paired_ab(a, b)
        points.append(
            {
                "capacity": C,
                "bitwise_equal_all_ranks": True,
                "a_us": a,
                "b_us": b,
                "a": summarize(a).to_dict(),
                "b": summarize(b).to_dict(),
                "paired": r.to_dict(),
            }
        )
        if info.rank == 0:
            print(
                f"C={C}: A {summarize(a).median:.1f} us, B {summarize(b).median:.1f} us, diff {r.mean_diff:+.1f} "
                f"CI [{r.ci95[0]:+.1f}, {r.ci95[1]:+.1f}] "
                f"beneficial={r.beneficial} bitwise=True",
                flush=True,
            )
        del la, lb
        torch.cuda.empty_cache()
    if info.rank == 0:
        write_record(args.out, make_record("ab_fixed", {**vars(args), "world": info.world}, {"points": points}))
    dist.barrier(device_ids=[info.local_rank])
    dist.destroy_process_group()
    faulthandler.cancel_dump_traceback_later()


if __name__ == "__main__":
    main()

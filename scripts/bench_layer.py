"""MoE layer benchmark (one process per GPU, launched with torchrun).

    torchrun --nproc-per-node=4 scripts/bench_layer.py --backend vllm:allgather_reducescatter \
        --tokens 1,2,4,8,16,32,64,128 --dist uniform --weights real:3 --out results/stage1/x.json

Backends: `moeflow:<expert backend>` (MoEFlow pipeline, naive NCCL), `moeflow:fixed` (fixed-capacity,
graph-capturable path) or `vllm:<prepare/finalize>`
with vLLM's Triton experts, where <prepare/finalize> is a vLLM all2all backend or `moeflow`.
For every working point the script (1) checks the output against the naive reference on this
rank's tokens and (2) times eager and CUDA-graph replay with the slowest-rank rule.
"""

from __future__ import annotations

import argparse
import faulthandler
import math

import torch
import torch.distributed as dist
from bench.backends import (
    MoEFlowBackend,
    MoEFlowFixedBackend,
    MoEFlowFixedVllmBackend,
    VllmModularBackend,
    init_dist,
)
from bench.inputs import SMALL, load_weights
from bench.profiling import profile_layer
from bench.record import make_record, write_record
from bench.stats import summarize
from bench.timing import time_eager, time_graph
from bench.validation import check_reference

from moeflow.config import GLM53_FLASH
from moeflow.ep_moe import shard_experts
from moeflow.reference import moe_reference
from moeflow.testing import make_hidden, make_logits


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", required=True)
    ap.add_argument("--tokens", default="1,2,4,8,16,32,64,128")
    ap.add_argument("--dist", default="uniform", choices=["uniform", "zipf", "hotspot"])
    ap.add_argument("--weights", default="synthetic")
    ap.add_argument("--shape", default="full", choices=["full", "small"])
    ap.add_argument("--reps", type=int, default=10)
    ap.add_argument("--inner", type=int, default=20)
    ap.add_argument("--no-graph", action="store_true")
    ap.add_argument("--profile", action="store_true", help="enable cudaProfilerApi collection around validated timing")
    ap.add_argument("--fixed-fused-combine", action="store_true", help="fuse combine for vllm:moeflow_fixed only")
    ap.add_argument("--fixed-combine-mode", choices=["fp32", "bf16_ring", "bf16_ring_fused"], default="fp32")
    ap.add_argument("--fixed-fused-dispatch", action="store_true", help="fuse dispatch for vllm:moeflow_fixed only")
    ap.add_argument("--fixed-fused-expert-prep", action="store_true")
    ap.add_argument("--fixed-direct-expert-output", action="store_true")
    ap.add_argument("--check-tokens", type=int, default=8, help="tokens per rank checked against the reference")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--watchdog", type=int, default=120)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    kind, _, sub = args.backend.partition(":")
    fixed_options = (
        args.fixed_fused_combine
        or args.fixed_fused_dispatch
        or args.fixed_fused_expert_prep
        or args.fixed_direct_expert_output
        or args.fixed_combine_mode != "fp32"
    )
    if fixed_options and args.backend != "vllm:moeflow_fixed":
        ap.error("Fixed-path switches require --backend vllm:moeflow_fixed")
    if args.fixed_fused_combine and args.fixed_combine_mode != "fp32":
        ap.error("FP32 fused combine and BF16 Ring modes select different arithmetic")
    if args.fixed_direct_expert_output and not args.fixed_fused_expert_prep:
        ap.error("Direct expert output requires --fixed-fused-expert-prep")
    info, vllm_config = init_dist(sub if kind == "vllm" else None)
    cfg = GLM53_FLASH if args.shape == "full" else SMALL
    faulthandler.dump_traceback_later(1800, exit=True)  # weight loading budget
    w13, w2, bias = load_weights(args.weights, cfg, info.device)
    l13, l2 = shard_experts(w13, w2, info.rank, info.world)
    l13, l2 = l13.contiguous(), l2.contiguous()
    tokens = [int(t) for t in args.tokens.split(",")]

    points = []
    for T in tokens:
        faulthandler.dump_traceback_later(args.watchdog, exit=True)
        seed = args.seed * 1_000_003 + T * 101 + info.rank
        x = make_hidden(T, cfg, seed, device=info.device)
        logits = make_logits(T, cfg, seed + 7, dist=args.dist, device=info.device)
        if args.backend == "vllm:moeflow_fixed":
            be = MoEFlowFixedVllmBackend(
                cfg,
                info,
                vllm_config,
                l13,
                l2,
                bias,
                capacity=T,
                max_tokens_per_rank=max(tokens),
                fused_combine=args.fixed_fused_combine,
                combine_mode=args.fixed_combine_mode,
                fused_dispatch=args.fixed_fused_dispatch,
                fused_expert_prep=args.fixed_fused_expert_prep,
                direct_expert_output=args.fixed_direct_expert_output,
            )
        elif kind == "moeflow" and sub == "fixed":
            be = MoEFlowFixedBackend(cfg, info, l13, l2, bias, capacity=T)
        elif kind == "moeflow":
            be = MoEFlowBackend(cfg, info, l13, l2, bias, experts=sub)
        else:
            be = VllmModularBackend(
                cfg,
                info,
                vllm_config,
                sub,
                l13,
                l2,
                bias,
                max_tokens_per_rank=max(tokens),
                tokens_per_rank=[T] * info.world,
            )
        out = be.run(x, logits)
        torch.cuda.synchronize()
        n = min(T, args.check_tokens)
        ref = moe_reference(x[:n], logits[:n], bias, w13, w2, cfg)
        o = out[:n]
        check = check_reference(o, ref)
        check["finite"] = check["finite"] and bool(torch.isfinite(out).all())
        check["passed"] = check["passed"] and check["finite"]
        checks = [None] * info.world
        dist.all_gather_object(checks, check)
        if not all(c["passed"] for c in checks):
            if info.rank == 0:
                write_record(
                    args.out,
                    make_record(
                        "layer_bench",
                        {**vars(args), "label": be.label, "world": info.world, "moe_config": cfg.to_dict()},
                        {"points": points, "failed_check": {"tokens_per_rank": T, "check_per_rank": checks}},
                        seed=args.seed,
                    ),
                )
            raise RuntimeError(f"Reference check failed before timing at T={T}: {checks}")
        dist.barrier(device_ids=[info.local_rank])
        with profile_layer(args.profile, f"LAYER:{args.backend}:T={T}"):
            eager = time_eager(lambda be=be, x=x, logits=logits: be.run(x, logits), reps=args.reps, inner=args.inner)
            graph = (
                None
                if args.no_graph
                else time_graph(lambda be=be, x=x, logits=logits: be.run(x, logits), reps=args.reps, inner=args.inner)
            )
        points.append(
            {
                "tokens_per_rank": T,
                "check_per_rank": checks,
                "eager_us": eager.per_call_us,
                "eager": summarize(eager.per_call_us).to_dict(),
                "graph_capturable": None if graph is None else graph.capturable,
                "graph_error": None if graph is None else graph.error,
                "graph_us": None if graph is None else graph.per_call_us,
                "graph": None if graph is None or not graph.capturable else summarize(graph.per_call_us).to_dict(),
            }
        )
        if info.rank == 0:
            # Preserve completed points even if a later point hangs or fails.
            write_record(
                args.out,
                make_record(
                    "layer_bench",
                    {**vars(args), "label": be.label, "world": info.world, "moe_config": cfg.to_dict()},
                    {"points": points, "complete": False},
                    seed=args.seed,
                ),
            )
            g = points[-1]["graph"]
            print(
                f"[{be.label}] T={T}: eager median {points[-1]['eager']['median']:.1f} us; graph "
                f"{'n/a' if g is None else f'{g["median"]:.1f} us'}; "
                f"max rel err {max(c['normwise_rel_err'] for c in checks):.2e}",
                flush=True,
            )
    dist.barrier(device_ids=[info.local_rank])
    dist.destroy_process_group()
    faulthandler.cancel_dump_traceback_later()
    assert all(math.isfinite(p["eager"]["median"]) for p in points)
    if info.rank == 0:
        rec = make_record(
            "layer_bench",
            {**vars(args), "label": be.label, "world": info.world, "moe_config": cfg.to_dict()},
            {"points": points, "complete": True},
            seed=args.seed,
        )
        write_record(args.out, rec)


if __name__ == "__main__":
    main()

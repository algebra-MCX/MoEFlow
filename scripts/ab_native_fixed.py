"""Paired complete-layer comparisons with the same router, BF16 experts and inputs."""

from __future__ import annotations

import argparse
import faulthandler
from functools import partial

import torch.distributed as dist
from bench.backends import MoEFlowFixedVllmBackend, VllmModularBackend, init_dist
from bench.inputs import SMALL, load_weights
from bench.record import make_record, write_record
from bench.stats import paired_ab, summarize
from bench.timing import time_graphs_alternating
from bench.validation import check_backend_pair, require_matching_graphs, require_matching_variants

from moeflow.config import GLM53_FLASH
from moeflow.ep_moe import shard_experts
from moeflow.reference import moe_reference
from moeflow.testing import make_hidden, make_logits


def update_graph_inputs(x, logits, cfg, seed, step):
    x.copy_(make_hidden(len(x), cfg, seed + 10_000 + step, device=x.device))
    logits.copy_(
        make_logits(
            len(x),
            cfg,
            seed + 20_000 + step,
            dist=("uniform", "zipf", "hotspot")[step % 3],
            device=x.device,
        )
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shape", choices=["small", "full"], default="full")
    parser.add_argument("--weights", default="real:3")
    parser.add_argument("--tokens", default="1,2,4,8,16,32,64,128")
    parser.add_argument("--dist", choices=["uniform", "zipf", "hotspot"], default="uniform")
    parser.add_argument("--b-fused-combine", action="store_true")
    parser.add_argument("--b-combine-mode", choices=["fp32", "bf16_ring", "bf16_ring_fused"], default="fp32")
    parser.add_argument("--a-fixed-combine-mode", choices=["bf16_ring", "bf16_ring_fused"])
    parser.add_argument("--a-fixed-ring", dest="a_fixed_combine_mode", action="store_const", const="bf16_ring")
    parser.add_argument("--b-fused-dispatch", action="store_true")
    parser.add_argument("--a-fused-dispatch", action="store_true")
    parser.add_argument("--b-fused-expert-prep", action="store_true")
    parser.add_argument("--a-fused-expert-prep", action="store_true")
    parser.add_argument("--b-direct-expert-output", action="store_true")
    parser.add_argument(
        "--check-graph-replays", type=int, default=0, help="check changing-input fixed A/B Graph outputs before timing"
    )
    parser.add_argument("--reps", type=int, default=20)
    parser.add_argument("--inner", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    if args.b_fused_combine and args.b_combine_mode != "fp32":
        parser.error("--b-fused-combine and BF16 Ring modes select different arithmetic")
    if args.a_fixed_combine_mode and args.b_combine_mode == "fp32":
        parser.error("Fixed Ring A/B requires a BF16 Ring combine in both arms")
    if args.a_fused_dispatch and not args.a_fixed_combine_mode:
        parser.error("--a-fused-dispatch requires a fixed A backend")
    if args.a_fused_expert_prep and not args.a_fixed_combine_mode:
        parser.error("--a-fused-expert-prep requires a fixed A backend")
    if args.b_direct_expert_output and not args.b_fused_expert_prep:
        parser.error("--b-direct-expert-output requires --b-fused-expert-prep")
    if args.check_graph_replays < 0 or (args.check_graph_replays and not args.a_fixed_combine_mode):
        parser.error("Graph output checks require nonnegative replays and a fixed A backend")

    info, vllm_config = init_dist("allgather_reducescatter")
    cfg = GLM53_FLASH if args.shape == "full" else SMALL
    faulthandler.dump_traceback_later(1800, exit=True)
    w13, w2, bias = load_weights(args.weights, cfg, info.device)
    local_w13, local_w2 = (w.contiguous() for w in shard_experts(w13, w2, info.rank, info.world))
    tokens = [int(token) for token in args.tokens.split(",")]
    points = []
    record = None
    if info.rank == 0:
        record = make_record(
            "ab_native_fixed",
            {**vars(args), "world": info.world},
            {"points": points, "complete": False},
            seed=args.seed,
        )

    for token_count in tokens:
        faulthandler.dump_traceback_later(120, exit=True)
        if args.a_fixed_combine_mode:
            backend_a = MoEFlowFixedVllmBackend(
                cfg,
                info,
                vllm_config,
                local_w13,
                local_w2,
                bias,
                capacity=token_count,
                max_tokens_per_rank=max(tokens),
                combine_mode=args.a_fixed_combine_mode,
                fused_dispatch=args.a_fused_dispatch,
                fused_expert_prep=args.a_fused_expert_prep,
            )
        else:
            backend_a = VllmModularBackend(
                cfg,
                info,
                vllm_config,
                "allgather_reducescatter",
                local_w13,
                local_w2,
                bias,
                max(tokens),
                [token_count] * info.world,
            )
        backend_b = MoEFlowFixedVllmBackend(
            cfg,
            info,
            vllm_config,
            local_w13,
            local_w2,
            bias,
            capacity=token_count,
            max_tokens_per_rank=max(tokens),
            fused_combine=args.b_fused_combine,
            combine_mode=args.b_combine_mode,
            fused_dispatch=args.b_fused_dispatch,
            fused_expert_prep=args.b_fused_expert_prep,
            direct_expert_output=args.b_direct_expert_output,
        )
        seed = args.seed * 1_000_003 + token_count * 101 + info.rank
        x = make_hidden(token_count, cfg, seed, device=info.device)
        logits = make_logits(token_count, cfg, seed + 7, dist=args.dist, device=info.device)
        a_out, b_out = backend_a.run(x, logits).clone(), backend_b.run(x, logits).clone()
        if args.a_fixed_combine_mode:
            require_matching_variants(a_out, b_out)
        reference = moe_reference(x, logits, bias, w13, w2, cfg)
        check = check_backend_pair(a_out, b_out, reference)
        checks = [None] * info.world
        dist.all_gather_object(checks, check)
        if not all(c["a"]["passed"] and c["b"]["passed"] for c in checks):
            if record is not None:
                record["results"]["failed_check"] = {"tokens_per_rank": token_count, "checks": checks}
                write_record(args.out, record)
            raise RuntimeError("Reference check failed before paired backend timing")

        if args.check_graph_replays:
            original_x, original_logits = x.clone(), logits.clone()

            require_matching_graphs(
                partial(backend_a.run, x, logits),
                partial(backend_b.run, x, logits),
                partial(update_graph_inputs, x, logits, cfg, seed),
                args.check_graph_replays,
            )
            x.copy_(original_x)
            logits.copy_(original_logits)

        a_us, b_us = time_graphs_alternating(
            partial(backend_a.run, x, logits), partial(backend_b.run, x, logits), reps=args.reps, inner=args.inner
        )
        paired = paired_ab(a_us, b_us)
        points.append(
            {
                "tokens_per_rank": token_count,
                "a_backend": backend_a.label,
                "b_backend": backend_b.label,
                "graph_replays_checked": args.check_graph_replays,
                "checks_per_rank": checks,
                "a_us": a_us,
                "b_us": b_us,
                "a": summarize(a_us).to_dict(),
                "b": summarize(b_us).to_dict(),
                "paired": paired.to_dict(),
            }
        )
        if record is not None:
            write_record(args.out, record)
            print(
                f"T={token_count}: A {summarize(a_us).median:.1f} us; B {summarize(b_us).median:.1f} us; "
                f"CI(B-A)=[{paired.ci95[0]:+.1f},{paired.ci95[1]:+.1f}] beneficial={paired.beneficial}",
                flush=True,
            )

    dist.barrier(device_ids=[info.local_rank])
    dist.destroy_process_group()
    faulthandler.cancel_dump_traceback_later()
    if record is not None:
        record["results"]["complete"] = True
        write_record(args.out, record)


if __name__ == "__main__":
    main()

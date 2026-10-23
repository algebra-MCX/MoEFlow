"""Paired Graph microbenchmark of the BF16 Ring combine only, not the complete MoE layer."""

import argparse
import faulthandler
from functools import partial

import torch
import torch.distributed as dist
from bench.backends import init_dist
from bench.record import make_record, write_record
from bench.stats import paired_ab, summarize
from bench.timing import time_graphs_alternating
from bench.validation import require_matching_variants

from moeflow.integrations.vllm_runner_ops import combine_bf16_ring as torch_combine
from moeflow.kernels.reduce import combine_bf16_ring


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokens", default="1,8,64,128")
    parser.add_argument("--hidden", type=int, default=4096)
    parser.add_argument("--reps", type=int, default=20)
    parser.add_argument("--inner", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    info, _ = init_dist(None)
    faulthandler.dump_traceback_later(120, exit=True)
    points, record = [], None
    if info.rank == 0:
        record = make_record(
            "bf16_ring_combine_ab",
            {**vars(args), "world": info.world, "boundary": "source_combine_only", "inputs": "synthetic_partials"},
            {"points": points, "complete": False},
            seed=args.seed,
        )
        write_record(args.out, record)
    for tokens in map(int, args.tokens.split(",")):
        torch.manual_seed(args.seed + 101 * tokens + info.rank)
        ret = torch.randn(info.world, tokens, args.hidden, dtype=torch.bfloat16, device=info.device)
        hit = torch.rand(tokens, info.world, device=info.device) > 0.25
        fn_a, fn_b = partial(torch_combine, ret, hit, info.rank), partial(combine_bf16_ring, ret, hit, info.rank)
        require_matching_variants(fn_a(), fn_b())
        a, b = time_graphs_alternating(fn_a, fn_b, reps=args.reps, inner=args.inner)
        result = {
            "tokens": tokens,
            "bitwise_equal_all_ranks": True,
            "a_us": a,
            "b_us": b,
            "a": summarize(a).to_dict(),
            "b": summarize(b).to_dict(),
            "paired": paired_ab(a, b).to_dict(),
        }
        points.append(result)
        if record is not None:
            write_record(args.out, record)
            print(
                f"C={tokens}: torch {result['a']['median']:.1f} us, fused {result['b']['median']:.1f} us; "
                f"CI(B-A)={result['paired']['ci95']} beneficial={result['paired']['beneficial']}",
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

"""Generate the initial W1 report only after serving and exact-output gates pass."""

import argparse
import json
from pathlib import Path

from validate_serving_result import validate_result

# Runs recorded before the package rename use the mode name "miniep_fixed".
FIXED_MODES = ("moeflow_fixed", "miniep_fixed")


def fixed_mode(root: Path) -> str:
    return next((mode for mode in FIXED_MODES if (root / f"{mode}_W1.exitcode").exists()), FIXED_MODES[0])


def make_report(root: Path) -> str:
    records = []
    fixed = fixed_mode(root)
    for mode in ("native_agrs", fixed):
        if (root / f"{mode}_W1.exitcode").read_text().strip() != "0":
            raise ValueError(f"Failed serving run: {mode}")
        record = json.loads((root / f"{mode}_W1.json").read_text())
        if record.get("moeflow_mode") != mode or record.get("moeflow_workload") != "W1":
            raise ValueError(f"Wrong mode or workload: {mode}")
        validate_result(record, requests=16, input_len=1024, output_len=1024, concurrency=4)
        records.append(record)
    a, b = records
    for field in ("model_id", "tokenizer_id", "moeflow_git", "moeflow_workload"):
        if not a.get(field) or a[field] != b.get(field):
            raise ValueError(f"Incomparable serving configuration: {field}")
    checks = json.loads((root / f"compare_{fixed}.json").read_text())
    if (
        len(checks) != 4
        or {row["data_parallel_rank"] for row in checks} != {0, 1, 2, 3}
        or not all(
            row["passed"]
            and row["of"] == row["candidate_tokens"] == row["common_prefix_tokens"] == 64
            and row["max_abs_logprob_diff_on_prefix"] == 0
            for row in checks
        )
    ):
        raise ValueError("Missing four-rank exact-output gate")
    lines = [
        "# W1：首轮完整 FP8 模型端到端测量",
        "",
        "EP4，1024-token 输入 / 1024-token 输出，16 个请求，总并发 4。",
        "两条路径均使用 vLLM Triton FP8 experts、FP8 KV cache 和 decode Graph；fixed 启用三项融合。",
        f"顺序：{'native → fixed' if a['date'] < b['date'] else 'fixed → native'}；各预热 2 请求。"
        "以下是单轮结果，没有独立重复实验的置信区间。",
        "",
        "| 指标（ms；median / p95 / p99） | native AG/RS | fixed 三项融合 |",
        "|---|---:|---:|",
    ]
    for metric in ("ttft", "tpot", "itl", "e2el"):
        cells = [" / ".join(f"{r[f'{stat}_{metric}_ms']:.2f}" for stat in ("median", "p95", "p99")) for r in records]
        lines.append(f"| {metric.upper()} | {cells[0]} | {cells[1]} |")
    lines += [
        "",
        "| 完成量与吞吐 | native AG/RS | fixed 三项融合 |",
        "|---|---:|---:|",
        f"| 完成请求 / 失败请求 | {a['completed']} / {a['failed']} | {b['completed']} / {b['failed']} |",
        f"| 生成 token | {a['total_output_tokens']} | {b['total_output_tokens']} |",
        f"| 生成吞吐（token/s） | {a['output_throughput']:.2f} | {b['output_throughput']:.2f} |",
        "",
        "每条路径先收集四个顺序执行的 1024-token / 64-token 请求，分别固定到 DP0/1/2/3；",
        "候选的完整输出与 token logprob 全部精确匹配原生。随后才采候选性能。",
        "逐请求实际长度、完成数、错误字段、有限延迟和吞吐计数均通过校验。",
        "",
        "这些采样不证明全部并发输入数值正确。8192-token 精确分数门禁仍失败，",
        "原生重跑也有分数波动；见[长输入记录](e2e/long-prefill.md)。W2–W6 和 EP8 仍未完成。",
        "GPU 分配预检无残留进程，但节点非独占、时钟未锁定；单轮差值不能作为稳定收益结论。",
        "",
        f"原始数据：`{root}`；代码版本：`{a['moeflow_git']}`。",
    ]
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.write_text(make_report(args.root))

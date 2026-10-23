"""Generate readable baseline tables from completed, comparable layer measurements."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

LABELS = {
    "vllm:allgather_reducescatter": "AG/RS",
    "vllm:deepep_high_throughput": "DeepEP HT",
    "vllm:deepep_low_latency": "DeepEP LL",
    "vllm:moeflow": "MoEFlow P/F",
    "moeflow:triton_grouped": "MoEFlow naive",
    "moeflow:fixed": "MoEFlow fixed",
    "vllm:moeflow_fixed": "MoEFlow fixed / vLLM GEMM",
}
# Runs recorded before the package rename name the backend "miniep"; map them onto current labels.
LEGACY_PREFIX = "miniep"


def backend_of(record: dict) -> str:
    return record["config"]["backend"].replace(LEGACY_PREFIX, "moeflow")


COMPARABLE_FIELDS = ("world", "weights", "shape", "dist", "seed", "reps", "inner", "check_tokens", "moe_config")


def read_results(directory: Path) -> list[tuple[Path, dict]]:
    records = [(path, json.loads(path.read_text())) for path in sorted(directory.glob("*.json"))]
    if not records:
        raise ValueError(f"No layer results in {directory}")
    expected = {key: records[0][1]["config"][key] for key in COMPARABLE_FIELDS}
    for path, record in records:
        if not record["results"].get("complete") or path.with_suffix(".exitcode").read_text().strip() != "0":
            raise ValueError(f"Incomplete or failed run: {path}")
        if any(record["config"][key] != value for key, value in expected.items()):
            raise ValueError(f"Incomparable configuration: {path}")
        if not record["results"]["points"] or any(not p["check_per_rank"] for p in record["results"]["points"]):
            raise ValueError(f"Missing correctness checks: {path}")
        if not all(c["passed"] for p in record["results"]["points"] for c in p["check_per_rank"]):
            raise ValueError(f"Failed correctness check: {path}")
    return sorted(records, key=lambda item: list(LABELS).index(backend_of(item[1])))


def make_tables(records: list[tuple[Path, dict]]) -> str:
    lines = [
        "# 阶段 1 基线原始统计表",
        "",
        "由 `scripts/summarize_stage1.py` 自动生成。单位为 µs；每个样本取最慢 rank。",
        "",
    ]
    for mode, title in (("graph", "CUDA Graph replay"), ("eager", "Eager")):
        lines += [f"## {title}", "", "| 后端 | tokens/rank | median | p95 | 标准差 |", "|---|---:|---:|---:|---:|"]
        for _, record in records:
            label = LABELS[backend_of(record)]
            for point in record["results"]["points"]:
                stats = point[mode]
                if stats:
                    values = f"{stats['median']:.1f} | {stats['p95']:.1f} | {stats['std']:.1f}"
                else:
                    status = "不可捕获" if point["graph_capturable"] is False else "未测量"
                    values = f"{status} | — | —"
                lines.append(f"| {label} | {point['tokens_per_rank']} | {values} |")
        lines.append("")
    lines += ["## 来源", ""]
    for path, record in records:
        label = LABELS[backend_of(record)]
        lines.append(f"- {label}: `{path}`, git `{record['git']['sha']}`, dirty={record['git']['dirty']}。")
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    args.out.write_text(make_tables(read_results(args.directory)))


if __name__ == "__main__":
    main()

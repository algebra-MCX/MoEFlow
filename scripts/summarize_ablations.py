"""Generate A/B tables without presenting unvalidated output as a speed improvement."""

import argparse
import json
from pathlib import Path


def make_table(paths: list[Path]) -> str:
    lines = [
        "# 固定路径 A/B 消融",
        "",
        "单位 µs；每组 20 次交替配对，每个样本取最慢 rank。三元组为 median / p95 / 标准差。",
        "",
        "| A → B | 容量 | A 统计 | B 统计 | median 改善 | B−A 的 95% CI | 有收益 |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for path in paths:
        record = json.loads(path.read_text())
        label = f"{record['config']['a'] or '默认'} → {record['config']['b']}"
        for point in record["results"]["points"]:
            if not point["bitwise_equal_all_ranks"]:
                raise ValueError(f"Unvalidated A/B output: {path}")
            a, b, paired = point["a"], point["b"], point["paired"]
            stats_a = f"{a['median']:.1f} / {a['p95']:.1f} / {a['std']:.1f}"
            stats_b = f"{b['median']:.1f} / {b['p95']:.1f} / {b['std']:.1f}"
            low, high = paired["ci95"]
            gain = 100 * (a["median"] - b["median"]) / a["median"]
            lines.append(
                f"| {label} | {point['capacity']} | {stats_a} | {stats_b} | {gain:.1f}% | "
                f"[{low:.1f}, {high:.1f}] | {paired['beneficial']} |"
            )
        lines += ["", f"原始数据：`{path}`。", ""]
    lines.append("各行只对其 A/B 判定收益；不同开关的百分比不能相加。所有开关仍默认关闭。")
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+", type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    args.out.write_text(make_table(args.paths))


if __name__ == "__main__":
    main()

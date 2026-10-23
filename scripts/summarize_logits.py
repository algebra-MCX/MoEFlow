"""Regenerate the two pinned-rank raw-logits report, rechecking artifacts before reporting success."""

import argparse
import json
from pathlib import Path

from logits_client import compare

# Runs recorded before the package rename name the candidate file "logits_miniep_fixed.json".
FIXED_NAMES = ("logits_moeflow_fixed.json", "logits_miniep_fixed.json")


def make_report(root: Path) -> str:
    lines = [
        "# 完整词表 raw logits 对照",
        "",
        "比较完整模型的原生 AG/RS 与显式 fused BF16 Ring 模式；两个 DP rank 分别固定请求。",
        "本轮服务使用 eager。生成位置只覆盖每个 prompt 的前 8 步，不能推广到全部上下文或服务 Graph。",
        "生成本表时重新验证 NPZ 的 SHA256、生成上下文及全部 FP32 字节；不引入数值容差。",
        "",
        "| DP rank | prompt 数 | 每 prompt 生成步数 | 词表大小 | 比较分数数量 | 全部逐字节一致 | 最大绝对差 |",
        "|---:|---:|---:|---:|---:|---|---:|",
    ]
    total = 0
    for rank in (0, 1):
        directory = root / f"rank{rank}"
        a = directory / "logits_native_agrs.json"
        b = next((directory / name for name in FIXED_NAMES if (directory / name).exists()), directory / FIXED_NAMES[0])
        record_a, record_b = (json.loads(path.read_text()) for path in (a, b))
        result = compare(str(a), str(b))
        if not result["passed"]:
            raise ValueError("Nonexact raw logits cannot be reported as a passing comparison")
        config = record_a["config"]
        if config["data_parallel_rank"] != rank or record_b["config"]["data_parallel_rank"] != rank:
            raise ValueError("Capture directory and actual pinned DP rank differ")
        rows = result["requests"]
        count = sum(row["values_compared"] for row in rows)
        total += count
        lines.append(
            f"| {rank} | {len(rows)} | {config['steps']} | {config['vocab_size']} | {count} | True | "
            f"{max(row['max_abs_diff'] for row in rows):.3g} |"
        )
    lines += [
        "",
        f"共比较 {total} 个分数。每组另保留四条完整 64-token greedy 对照。",
        "本轮不测性能，也没有覆盖 active DP ranks 2/3、长 prefill、其他 NCCL 算法或拓扑。",
        "",
        f"原始 manifest、NPZ、逐请求比较和服务日志：`{root}`；启动命令：`{root / 'commands.log'}`。",
    ]
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.write_text(make_report(args.root))

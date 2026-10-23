"""Report validated native/fixed pairs using the same vLLM router and BF16 experts."""

import argparse
import json
from pathlib import Path


def make_table(path: Path) -> str:
    record = json.loads(path.read_text())
    config, results = record["config"], record["results"]
    if record["kind"] != "ab_native_fixed" or not results["complete"] or not results["points"]:
        raise ValueError("Incomplete paired backend run")
    mode_a = config.get("a_fixed_combine_mode") or ("bf16_ring" if config.get("a_fixed_ring") else None)
    ring_ab = mode_a is not None
    label_a = mode_a or "AG/RS"
    if config.get("a_fused_dispatch"):
        label_a += " + fused dispatch"
    if config.get("a_fused_expert_prep"):
        label_a += " + fused expert preparation"
    label_b = config.get("b_combine_mode", "fp32") + (" + fused dispatch" if config.get("b_fused_dispatch") else "")
    if config.get("b_fused_expert_prep"):
        label_b += " + fused expert preparation"
    if config.get("b_direct_expert_output"):
        label_b += " + direct expert output"
    lines = [
        "# 同 GEMM 的完整 MoE 层配对测量",
        "",
        f"EP={config['world']}，{config['weights']}，{config['dist']} 路由，种子 {config['seed']}。",
        f"每点 {config['reps']} 次交替配对，每次 {config['inner']} 次 Graph replay，样本取最慢 rank。",
        f"A：{label_a}；B：fixed {label_b}；FP32 fused combine：{config['b_fused_combine']}。其他融合开关关闭。",
        "单位 µs；统计为 median / p95 / 标准差。所有 rank 的全部输入 token 先通过独立 reference 对拍。",
        "",
        "| 每 rank token | A 统计 | B 统计 | median 改善 | B−A 的 95% CI | 有收益 |",
        "|---:|---:|---:|---:|---:|---|",
    ]
    for point in results["points"]:
        if point.get("graph_replays_checked", 0) != config.get("check_graph_replays", 0):
            raise ValueError("Missing claimed Graph output checks")
        checks = point["checks_per_rank"]
        if len(checks) != config["world"] or not all(
            check[arm]["passed"] and check[arm]["tokens_checked"] == point["tokens_per_rank"]
            for check in checks
            for arm in ("a", "b")
        ):
            raise ValueError("Unvalidated or partial-token backend output")
        if ring_ab and not all(check["bitwise_equal"] for check in checks):
            raise ValueError("Ring fusion A/B must be bitwise equal on every rank")
        if len(point["a_us"]) != config["reps"] or len(point["b_us"]) != config["reps"]:
            raise ValueError("Incomplete paired timing samples")
        a, b, paired = point["a"], point["b"], point["paired"]
        stats = [f"{s['median']:.1f} / {s['p95']:.1f} / {s['std']:.1f}" for s in (a, b)]
        gain = 100 * (a["median"] - b["median"]) / a["median"]
        low, high = paired["ci95"]
        lines.append(
            f"| {point['tokens_per_rank']} | {stats[0]} | {stats[1]} | {gain:.1f}% | "
            f"[{low:.1f}, {high:.1f}] | {paired['beneficial']} |"
        )
    lines += [
        "",
        "不同通信后端的归约次序可能不同；这里沿用冻结的数值容差，不要求跨后端逐位一致。",
        "同一路径的 kernel 开关消融仍要求逐位一致。负的 median 改善表示 fixed 更慢。",
        "这组测量只包含 routed MoE layer，不代表完整模型的 TTFT、TPOT 或 ITL。",
        "",
        f"原始数据：`{path}`；代码版本：`{record['git']['sha']}`。",
    ]
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    args.out.write_text(make_table(args.path))


if __name__ == "__main__":
    main()

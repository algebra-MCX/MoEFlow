"""Summarize graph replay kernels, excluding eager calls and timing collectives."""

import argparse
import hashlib
import json
import sqlite3
from pathlib import Path


def read_kernels(path: Path) -> dict:
    with sqlite3.connect(f"file:{path.resolve()}?mode=ro", uri=True) as connection:
        rows = connection.execute(
            "SELECT s.value, COUNT(*), SUM(k.end-k.start) "
            "FROM CUPTI_ACTIVITY_KIND_KERNEL k JOIN StringIds s ON s.id=k.demangledName "
            "WHERE k.graphId IS NOT NULL GROUP BY s.value ORDER BY SUM(k.end-k.start) DESC"
        ).fetchall()
        has_copies = connection.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='CUPTI_ACTIVITY_KIND_MEMCPY'"
        ).fetchone()
        copies = (
            connection.execute(
                "SELECT copyKind, bytes, COUNT(*), SUM(end-start) FROM CUPTI_ACTIVITY_KIND_MEMCPY "
                "WHERE graphNodeId IS NOT NULL GROUP BY copyKind, bytes"
            ).fetchall()
            if has_copies
            else None
        )
    forwards = sum(count for name, count, _ in rows if "single_group_topk" in name)
    if not forwards:
        raise ValueError(f"No graph-replayed router calls: {path}")
    return {
        "forwards": forwards,
        "kernels_per_forward": sum(count for _, count, _ in rows) / forwards,
        "framework_kernels_per_forward": sum(count for name, count, _ in rows if "at::native::" in name) / forwards,
        "framework_kernel_sum_us": sum(ns for name, _, ns in rows if "at::native::" in name) / forwards / 1000,
        "gemm_kernel_sum_us": sum(ns for name, _, ns in rows if name == "fused_moe_kernel") / forwards / 1000,
        "copies_per_forward": sum(count for _, _, count, _ in copies) / forwards if copies is not None else None,
        "copy_bytes_per_forward": sum(size * count for _, size, count, _ in copies) / forwards
        if copies is not None
        else None,
        "copy_sum_us": sum(ns for _, _, _, ns in copies) / forwards / 1000 if copies is not None else None,
        "copies": [{"kind": kind, "bytes": size, "count": count, "sum_ns": ns} for kind, size, count, ns in copies]
        if copies is not None
        else None,
        "kernels": [{"name": name, "count": count, "sum_ns": ns} for name, count, ns in rows],
    }


def summarize(root: Path, names=("native", "fixed", "fixed_combine")) -> tuple[dict, str]:
    variants, artifacts = {}, []
    for variant in names:
        record = json.loads((root / f"{variant}.json").read_text())
        if (root / f"{variant}.exitcode").read_text().strip() != "0" or not record["results"]["complete"]:
            raise ValueError(f"Incomplete or failed trace: {variant}")
        points = record["results"]["points"]
        if len(points) != 1 or not all(check["passed"] for check in points[0]["check_per_rank"]):
            raise ValueError(f"Unvalidated or multiple-point trace: {variant}")
        variants[variant] = {"git": record["git"], **read_kernels(root / f"{variant}.sqlite")}
        for suffix in ("nsys-rep", "sqlite"):
            path = root / f"{variant}.{suffix}"
            with path.open("rb") as source:
                digest = hashlib.file_digest(source, "sha256").hexdigest()
            artifacts.append(
                {
                    "path": str(path),
                    "bytes": path.stat().st_size,
                    "sha256": digest,
                }
            )
    lines = [
        "# 同 GEMM 路径的 Graph replay 诊断",
        "",
        "EP=4、T=1、真实第 3 层 BF16 权重，只采 rank 0。只统计 graphId 非空的 GPU kernel，",
        "排除 eager 调用和计时器在 Graph 外执行的 barrier/AllReduce。按每次 forward 的一次 router 调用归一化。",
        "每条路径共有 55 次 forward：包含 1 次 warm replay 和 10 次计时 replay，每次 5 个 forward。",
        "",
        "| 路径 | kernel 数/forward | PyTorch 辅助 kernel 数 | 辅助 kernel 累计 µs | GEMM kernel 累计 µs |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, value in variants.items():
        lines.append(
            f"| {name} | {value['kernels_per_forward']:.0f} | {value['framework_kernels_per_forward']:.0f} | "
            f"{value['framework_kernel_sum_us']:.1f} | {value['gemm_kernel_sum_us']:.1f} |"
        )
    lines += [
        "",
        "| 路径 | Graph memcpy 数/forward | 拷贝 KiB/forward | memcpy 累计 µs |",
        "|---|---:|---:|---:|",
    ]
    for name, value in variants.items():
        if value["copies_per_forward"] is None:
            lines.append(f"| {name} | n/a | n/a | n/a |")
        else:
            lines.append(
                f"| {name} | {value['copies_per_forward']:.0f} | {value['copy_bytes_per_forward'] / 1024:.1f} | "
                f"{value['copy_sum_us']:.2f} |"
            )
    lines += [
        "",
        "辅助 kernel 以完整符号中的 at::native:: 识别，包含 PyTorch 的拷贝、类型转换、mask 和小算子。",
        "Graph memcpy 按 graphNodeId 非空单独统计；kernel 表不包含这些节点。n/a 表示 trace 缺少拷贝事件表。",
        "辅助 kernel 为零不等于零拷贝；memcpy 累计时间也不能直接作为完整层延迟的分摊。",
        "GPU kernel 累计时间不是完整层 wall time，不能直接作为额外延迟的严格分摊。",
        "采集包含 profiler 开销；性能判断继续使用未插桩的交替配对结果。",
        "该 trace 支持优先减少 fixed 打包、ID 转换和合并中的小 kernel；尚未逐一归因或证明全部差值来源。",
        "",
        f"原始统计、命令和退出码：`{root}`。完整符号及大文件 SHA256 见同目录 `replay_summary.json`。",
    ]
    return {"variants": variants, "artifacts": artifacts}, "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--variants", nargs="+", default=["native", "fixed", "fixed_combine"])
    args = parser.parse_args()
    summary, report = summarize(args.root, args.variants)
    (args.root / "replay_summary.json").write_text(json.dumps(summary, indent=1))
    args.out.write_text(report)


if __name__ == "__main__":
    main()

"""
aggregate_results.py — 跨种子汇总实验结果，计算均值与标准差。

读取结构:
  logs/multi_seed_results/seed_<N>/<group>/<problem>.csv

输出:
  logs/multi_seed_results/aggregate_summary.csv   — 完整汇总表
  logs/multi_seed_results/aggregate_report.txt    — 可读报告
"""

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path
import math


METRICS = ["ar_mean", "ar_std", "placed_mean", "viol_rate", "cap_viol_total", "conflict_viol_total"]
ENV_GROUPS   = ["ecu_eq_svc_p", "ecu_gt_svc_p", "ecu_lt_svc_p"]
PROBLEM_DIRS = [
    "problem3_ppo",
    "problem4_ppo_mask",
    "problem5_ppo_lagrangian",
    "problem6_ppo_opt",
    "problem_dqn",
    "problem_ddqn",
]


def mean(vals):
    return sum(vals) / len(vals) if vals else float("nan")


def std(vals):
    if len(vals) < 2:
        return 0.0
    m = mean(vals)
    return math.sqrt(sum((v - m) ** 2 for v in vals) / (len(vals) - 1))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive-dir", required=True, type=Path)
    parser.add_argument("--seeds", nargs="+", required=True)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def main():
    args = parse_args()
    archive_dir = args.archive_dir
    seeds = args.seeds

    # key: (group, problem, method) → {metric: [values across seeds]}
    data: dict[tuple, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))

    missing = []
    for seed in seeds:
        for group in ENV_GROUPS:
            for problem in PROBLEM_DIRS:
                csv_path = archive_dir / f"seed_{seed}" / group / f"{problem}.csv"
                if not csv_path.exists():
                    missing.append(str(csv_path))
                    continue
                for row in read_csv(csv_path):
                    method = row["method"]
                    if "Random" in method:
                        continue
                    key = (group, problem, method)
                    for m in METRICS:
                        try:
                            data[key][m].append(float(row[m]))
                        except (KeyError, ValueError):
                            pass

    if missing:
        print(f"WARNING: {len(missing)} CSV files not found:")
        for p in missing:
            print(f"  {p}")

    # ── 写 aggregate_summary.csv ──────────────────────────────────────────────
    out_csv = archive_dir / "aggregate_summary.csv"
    header = [
        "group", "problem", "method", "n_seeds",
        "ar_mean_avg", "ar_mean_std",
        "placed_mean_avg", "placed_mean_std",
        "viol_rate_avg", "viol_rate_std",
        "cap_viol_avg", "cap_viol_std",
        "conflict_viol_avg", "conflict_viol_std",
    ]

    rows_out = []
    for (group, problem, method), metrics in sorted(data.items()):
        ar_vals  = metrics.get("ar_mean", [])
        plc_vals = metrics.get("placed_mean", [])
        vr_vals  = metrics.get("viol_rate", [])
        cap_vals = metrics.get("cap_viol_total", [])
        con_vals = metrics.get("conflict_viol_total", [])

        rows_out.append({
            "group":   group,
            "problem": problem,
            "method":  method,
            "n_seeds": len(ar_vals),
            "ar_mean_avg":        f"{mean(ar_vals):.6f}",
            "ar_mean_std":        f"{std(ar_vals):.6f}",
            "placed_mean_avg":    f"{mean(plc_vals):.4f}",
            "placed_mean_std":    f"{std(plc_vals):.4f}",
            "viol_rate_avg":      f"{mean(vr_vals):.4f}",
            "viol_rate_std":      f"{std(vr_vals):.4f}",
            "cap_viol_avg":       f"{mean(cap_vals):.2f}",
            "cap_viol_std":       f"{std(cap_vals):.2f}",
            "conflict_viol_avg":  f"{mean(con_vals):.2f}",
            "conflict_viol_std":  f"{std(con_vals):.2f}",
        })

    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        writer.writerows(rows_out)
    print(f"Saved: {out_csv}")

    # ── 写 aggregate_report.txt ───────────────────────────────────────────────
    out_txt = archive_dir / "aggregate_report.txt"
    lines = []
    lines.append(f"Multi-Seed Aggregate Report  (seeds: {', '.join(seeds)})")
    lines.append("=" * 72)

    for group in ENV_GROUPS:
        lines.append(f"\n{'─'*72}")
        lines.append(f"  {group}")
        lines.append(f"{'─'*72}")
        lines.append(f"  {'Problem':<28} {'Method':<22} {'AR mean±std':>14}  {'Placed':>7}  {'ViolRate':>8}  {'CapV':>5}  {'ConV':>5}")
        lines.append(f"  {'-'*28} {'-'*22} {'-'*14}  {'-'*7}  {'-'*8}  {'-'*5}  {'-'*5}")

        # 先打印 ILP 一次（取第一个 problem 的数据，各 problem 值相同）
        ilp_printed = False
        for problem in PROBLEM_DIRS:
            for (g, p, method), metrics in sorted(data.items()):
                if g != group or p != problem or "ILP" not in method:
                    continue
                if not ilp_printed:
                    ar_vals  = metrics.get("ar_mean", [])
                    plc_vals = metrics.get("placed_mean", [])
                    vr_vals  = metrics.get("viol_rate", [])
                    cap_vals = metrics.get("cap_viol_total", [])
                    con_vals = metrics.get("conflict_viol_total", [])
                    ar_str  = f"{mean(ar_vals):.4f}±{std(ar_vals):.4f}"
                    plc_str = f"{mean(plc_vals):.2f}±{std(plc_vals):.2f}"
                    vr_str  = f"{mean(vr_vals):.4f}±{std(vr_vals):.4f}"
                    cap_str = f"{mean(cap_vals):.1f}"
                    con_str = f"{mean(con_vals):.1f}"
                    lines.append(
                        f"  {'ILP (Optimal)':<28} {method:<22} {ar_str:>14}  {plc_str:>7}  {vr_str:>8}  {cap_str:>5}  {con_str:>5}"
                    )
                    lines.append(f"  {'-'*28} {'-'*22} {'-'*14}  {'-'*7}  {'-'*8}  {'-'*5}  {'-'*5}")
                    ilp_printed = True
                break

        # 再打印各 problem 的非 ILP 方法
        for problem in PROBLEM_DIRS:
            for (g, p, method), metrics in sorted(data.items()):
                if g != group or p != problem or "ILP" in method:
                    continue
                ar_vals  = metrics.get("ar_mean", [])
                plc_vals = metrics.get("placed_mean", [])
                vr_vals  = metrics.get("viol_rate", [])
                cap_vals = metrics.get("cap_viol_total", [])
                con_vals = metrics.get("conflict_viol_total", [])

                ar_str  = f"{mean(ar_vals):.4f}±{std(ar_vals):.4f}"
                plc_str = f"{mean(plc_vals):.2f}±{std(plc_vals):.2f}"
                vr_str  = f"{mean(vr_vals):.4f}±{std(vr_vals):.4f}"
                cap_str = f"{mean(cap_vals):.1f}"
                con_str = f"{mean(con_vals):.1f}"

                lines.append(
                    f"  {problem:<28} {method:<22} {ar_str:>14}  {plc_str:>7}  {vr_str:>8}  {cap_str:>5}  {con_str:>5}"
                )

    lines.append("")
    out_txt.write_text("\n".join(lines))
    print(f"Saved: {out_txt}")

    # 终端打印报告
    print("\n" + "\n".join(lines))


if __name__ == "__main__":
    main()

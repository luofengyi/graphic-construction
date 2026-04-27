import argparse
import csv
import os
import re
import statistics

from summarize_runs import collect_metrics, parse_seeds


def mean_std(values):
    if not values:
        return None, None
    mean_v = statistics.mean(values)
    std_v = statistics.stdev(values) if len(values) > 1 else 0.0
    return mean_v, std_v


def mean_std_text(values):
    mean_v, std_v = mean_std(values)
    if mean_v is None:
        return "N/A"
    return f"{mean_v:.4f}±{std_v:.4f}"


def extract_threshold(dir_name):
    m = re.match(r"sim_threshold_(.+)$", dir_name)
    if not m:
        return None
    try:
        return float(m.group(1))
    except ValueError:
        return None


def summarize_one_threshold(threshold_dir, seeds):
    rows = collect_metrics(threshold_dir)
    if seeds:
        rows = [r for r in rows if int(r.get("seed", -1)) in seeds]
    if not rows:
        return None

    dev_acc_values = [float(r["best_dev_acc"]) for r in rows if r.get("best_dev_acc") is not None]
    dev_f1_values = [float(r["best_dev_f1"]) for r in rows if r.get("best_dev_f1") is not None]
    test_acc_values = [float(r["best_test_acc"]) for r in rows if r.get("best_test_acc") is not None]
    test_f1_values = [float(r["best_test_f1"]) for r in rows if r.get("best_test_f1") is not None]

    mean_test_f1, _ = mean_std(test_f1_values)
    return {
        "threshold": extract_threshold(os.path.basename(threshold_dir)),
        "run_count": len(rows),
        "seeds": ",".join(str(int(r.get("seed"))) for r in sorted(rows, key=lambda x: int(x.get("seed", 0)))),
        "best_dev_acc_mean_std": mean_std_text(dev_acc_values),
        "best_dev_f1_mean_std": mean_std_text(dev_f1_values),
        "best_test_acc_mean_std": mean_std_text(test_acc_values),
        "best_test_f1_mean_std": mean_std_text(test_f1_values),
        "best_test_f1_mean": mean_test_f1 if mean_test_f1 is not None else -1.0,
    }


def export_csv(rows, path):
    fieldnames = [
        "threshold",
        "run_count",
        "seeds",
        "best_dev_acc_mean_std",
        "best_dev_f1_mean_std",
        "best_test_acc_mean_std",
        "best_test_f1_mean_std",
    ]
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})


def export_markdown(rows, path):
    lines = []
    lines.append("# Sim Threshold Comparison")
    lines.append("")
    lines.append("| threshold | run_count | seeds | best_dev_acc (mean±std) | best_dev_f1 (mean±std) | best_test_acc (mean±std) | best_test_f1 (mean±std) |")
    lines.append("|---:|---:|---|---:|---:|---:|---:|")
    for r in rows:
        lines.append(
            f"| {r['threshold']:.2f} | {r['run_count']} | {r['seeds']} | "
            f"{r['best_dev_acc_mean_std']} | {r['best_dev_f1_mean_std']} | "
            f"{r['best_test_acc_mean_std']} | {r['best_test_f1_mean_std']} |"
        )
    lines.append("")
    if rows:
        best = rows[0]
        lines.append(
            f"- Best by test mean F1: threshold={best['threshold']:.2f}, best_test_f1={best['best_test_f1_mean_std']}"
        )
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Generate horizontal comparison table across sim_threshold sweeps."
    )
    parser.add_argument(
        "--sweep_root",
        type=str,
        default="./run_outputs",
        help="Root directory containing sim_threshold_* folders.",
    )
    parser.add_argument(
        "--seeds",
        type=str,
        default="24,42,77",
        help="Comma separated seeds to include in each threshold summary.",
    )
    parser.add_argument(
        "--csv_name",
        type=str,
        default="sim_threshold_comparison.csv",
        help="Output CSV filename under sweep_root.",
    )
    parser.add_argument(
        "--md_name",
        type=str,
        default="sim_threshold_comparison.md",
        help="Output Markdown filename under sweep_root.",
    )
    args = parser.parse_args()

    seeds = parse_seeds(args.seeds)
    if not os.path.isdir(args.sweep_root):
        raise FileNotFoundError(f"Sweep root not found: {args.sweep_root}")

    candidates = []
    for name in sorted(os.listdir(args.sweep_root)):
        threshold = extract_threshold(name)
        if threshold is None:
            continue
        folder = os.path.join(args.sweep_root, name)
        if os.path.isdir(folder):
            candidates.append(folder)

    if not candidates:
        raise RuntimeError(f"No sim_threshold_* folders found under: {args.sweep_root}")

    summary_rows = []
    for folder in candidates:
        row = summarize_one_threshold(folder, seeds)
        if row is not None:
            summary_rows.append(row)

    if not summary_rows:
        raise RuntimeError("No valid runs found for requested thresholds/seeds.")

    summary_rows.sort(key=lambda x: (-x["best_test_f1_mean"], x["threshold"]))

    csv_path = os.path.join(args.sweep_root, args.csv_name)
    md_path = os.path.join(args.sweep_root, args.md_name)
    export_csv(summary_rows, csv_path)
    export_markdown(summary_rows, md_path)

    print("[Threshold Sweep Summary]")
    print(f"sweep_root: {args.sweep_root}")
    print(f"seeds: {','.join(str(s) for s in seeds) if seeds else 'all'}")
    print(f"best threshold by test mean F1: {summary_rows[0]['threshold']:.2f}")
    print(f"CSV: {csv_path}")
    print(f"Markdown: {md_path}")


if __name__ == "__main__":
    main()

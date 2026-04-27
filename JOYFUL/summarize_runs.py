import argparse
import csv
import json
import os
import statistics


def parse_seeds(seed_str):
    if seed_str is None or str(seed_str).strip() == "":
        return []
    return [int(x.strip()) for x in str(seed_str).split(",") if x.strip()]


def mean_std_text(values):
    if not values:
        return "N/A"
    mean_v = statistics.mean(values)
    std_v = statistics.stdev(values) if len(values) > 1 else 0.0
    return f"{mean_v:.4f}±{std_v:.4f}"


def collect_metrics(output_dir):
    rows = []
    if not os.path.isdir(output_dir):
        raise FileNotFoundError(f"Output directory not found: {output_dir}")

    for name in sorted(os.listdir(output_dir)):
        run_dir = os.path.join(output_dir, name)
        if not os.path.isdir(run_dir):
            continue
        metrics_file = os.path.join(run_dir, "best_metrics.json")
        if not os.path.isfile(metrics_file):
            continue
        with open(metrics_file, "r", encoding="utf-8") as f:
            metrics = json.load(f)
        metrics["run_dir"] = run_dir
        rows.append(metrics)
    return rows


def export_csv(rows, export_path):
    fieldnames = [
        "run_name",
        "dataset",
        "modalities",
        "seed",
        "epochs",
        "best_epoch",
        "best_dev_f1",
        "best_test_f1",
        "run_dir",
    ]
    with open(export_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})


def export_markdown(rows, aggregate, export_path):
    lines = []
    lines.append("# Run Summary")
    lines.append("")
    lines.append("## Aggregate")
    lines.append("")
    lines.append(f"- Seeds: {aggregate['seeds_text']}")
    lines.append(f"- best_dev_f1: {aggregate['best_dev_text']}")
    lines.append(f"- best_test_f1: {aggregate['best_test_text']}")
    lines.append("")
    lines.append("## Per-run")
    lines.append("")
    lines.append("| run_name | seed | best_epoch | best_dev_f1 | best_test_f1 |")
    lines.append("|---|---:|---:|---:|---:|")
    for row in rows:
        lines.append(
            f"| {row.get('run_name','')} | {row.get('seed','')} | {row.get('best_epoch','')} | "
            f"{float(row.get('best_dev_f1', 0.0)):.4f} | {float(row.get('best_test_f1', 0.0)):.4f} |"
        )
    with open(export_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Summarize multi-seed run outputs and export table."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./run_outputs",
        help="Directory containing per-run folders with best_metrics.json.",
    )
    parser.add_argument(
        "--seeds",
        type=str,
        default="24,42,77",
        help="Comma separated seeds to include, e.g. 24,42,77.",
    )
    parser.add_argument(
        "--csv_name",
        type=str,
        default="summary_table.csv",
        help="CSV output filename (written under output_dir).",
    )
    parser.add_argument(
        "--md_name",
        type=str,
        default="summary_table.md",
        help="Markdown output filename (written under output_dir).",
    )
    args = parser.parse_args()

    target_seeds = parse_seeds(args.seeds)
    rows = collect_metrics(args.output_dir)
    if target_seeds:
        rows = [r for r in rows if int(r.get("seed", -1)) in target_seeds]

    if not rows:
        raise RuntimeError(
            f"No best_metrics.json found for seeds: {target_seeds} in {args.output_dir}"
        )

    rows = sorted(rows, key=lambda x: int(x.get("seed", 0)))
    dev_values = [float(r["best_dev_f1"]) for r in rows if r.get("best_dev_f1") is not None]
    test_values = [float(r["best_test_f1"]) for r in rows if r.get("best_test_f1") is not None]

    aggregate = {
        "seeds_text": ",".join(str(r.get("seed")) for r in rows),
        "best_dev_text": mean_std_text(dev_values),
        "best_test_text": mean_std_text(test_values),
    }

    csv_path = os.path.join(args.output_dir, args.csv_name)
    md_path = os.path.join(args.output_dir, args.md_name)
    export_csv(rows, csv_path)
    export_markdown(rows, aggregate, md_path)

    print("[Summary]")
    print(f"Seeds: {aggregate['seeds_text']}")
    print(f"best_dev_f1: {aggregate['best_dev_text']}")
    print(f"best_test_f1: {aggregate['best_test_text']}")
    print(f"CSV: {csv_path}")
    print(f"Markdown: {md_path}")


if __name__ == "__main__":
    main()

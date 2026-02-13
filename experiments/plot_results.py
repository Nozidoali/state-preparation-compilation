"""Generate summary and plots from QSP benchmark CSV.

Usage
-----
    python -m experiments.plot_results <csv_file> [--output-dir <dir>]
"""
from __future__ import annotations

import argparse
import csv
import math
import os
from pathlib import Path
from typing import Any, Dict, List


def load_csv(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            row["n"] = int(row["n"])
            row["m"] = int(row["m"])
            row["precision_bits"] = int(row["precision_bits"])
            row["t_count"] = int(row["t_count"])
            row["cnot_count"] = int(row["cnot_count"])
            row["qubit_count"] = int(row["qubit_count"])
            row["gate_count"] = int(row["gate_count"])
            row["time_ms"] = float(row["time_ms"])
            row["fidelity"] = float(row.get("fidelity", 0))
            row["success"] = row["success"] == "true"
            rows.append(row)
    return rows


def _mean(vals):
    return sum(vals) / len(vals) if vals else 0


def _std(vals):
    if len(vals) < 2:
        return 0
    m = _mean(vals)
    return math.sqrt(sum((v - m) ** 2 for v in vals) / (len(vals) - 1))


def geomean(values):
    positive = [v for v in values if v > 0]
    if not positive:
        return 0
    return math.exp(sum(math.log(v) for v in positive) / len(positive))


def _safe_import_plt(out_dir: Path):
    mpl_dir = out_dir / ".mplconfig"
    cache_dir = out_dir / ".cache"
    mpl_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_dir))
    os.environ.setdefault("XDG_CACHE_HOME", str(cache_dir))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def generate_summary(rows: List[Dict], out_dir: Path) -> None:
    lines = ["# QSP T-Cost Comparison: Rotation vs QROM\n"]

    approaches = sorted(set(r["approach"] for r in rows))
    families = sorted(set(r["family"] for r in rows))
    ns = sorted(set(r["n"] for r in rows))
    precisions = sorted(set(r["precision_bits"] for r in rows if r["precision_bits"] > 0))
    non_dense = [f for f in families if "dense" not in f]

    lines.append("## Gate Breakdown by Approach and n\n")
    lines.append("| Approach | n | T-count | CNOT | Qubits | Gates |")
    lines.append("|----------|---|--------:|-----:|-------:|------:|")

    for approach in approaches:
        first = True
        for n in ns:
            sub = [r for r in rows if r["n"] == n and r["approach"] == approach and r["success"]]
            if approach.startswith("rotation"):
                sub = [r for r in sub if r["family"] in non_dense]
            if not sub:
                continue
            t = round(_mean([r["t_count"] for r in sub]))
            cx = round(_mean([r["cnot_count"] for r in sub]))
            q = round(_mean([r["qubit_count"] for r in sub]))
            g = round(_mean([r["gate_count"] for r in sub]))
            label = f"**{approach}**" if first else ""
            lines.append(f"| {label} | {n} | {t} | {cx} | {q} | {g} |")
            first = False

    lines.append("\n## Per-Family Geometric Mean T-Count\n")
    header = "| Family | " + " | ".join(approaches) + " |"
    sep = "|--------|" + "|".join(["--------:"] * len(approaches)) + "|"
    lines.extend([header, sep])
    for fam in families:
        cols = [fam]
        for approach in approaches:
            vals = [r["t_count"] for r in rows
                    if r["family"] == fam and r["approach"] == approach and r["success"]]
            cols.append(f"{geomean(vals):.0f}" if vals else "-")
        lines.append("| " + " | ".join(cols) + " |")

    (out_dir / "comparison_summary.md").write_text("\n".join(lines))


def plot_t_cost_scaling(rows: List[Dict], out_dir: Path) -> None:
    plt = _safe_import_plt(out_dir)
    approaches = sorted(set(r["approach"] for r in rows))
    ns = sorted(set(r["n"] for r in rows))

    fig, ax = plt.subplots(figsize=(10, 6))
    markers = {"rotation_sparse": "v", "rotation_dense": "^", "qrom": "s", "selectswap": "D"}
    colors = {"rotation_sparse": "#03A9F4", "rotation_dense": "#0D47A1",
              "qrom": "#FF5722", "selectswap": "#4CAF50"}

    for approach in approaches:
        x_vals, y_vals = [], []
        for n in ns:
            vals = [r["t_count"] for r in rows
                    if r["n"] == n and r["approach"] == approach and r["success"]]
            if vals:
                x_vals.append(n)
                y_vals.append(_mean(vals))
        if x_vals:
            ax.plot(x_vals, y_vals, marker=markers.get(approach, "o"),
                    color=colors.get(approach, "#9C27B0"),
                    label=approach, linewidth=2, markersize=8)

    ax.set_xlabel("Number of qubits (n)")
    ax.set_ylabel("Mean T-count")
    ax.set_title("T-Count Scaling: Rotation vs QROM")
    ax.legend()
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "t_cost_vs_n.png", dpi=150)
    plt.close(fig)


def plot_gate_breakdown(rows: List[Dict], out_dir: Path) -> None:
    plt = _safe_import_plt(out_dir)
    import numpy as np

    approaches = sorted(set(r["approach"] for r in rows))
    ns = sorted(set(r["n"] for r in rows))
    metrics = [("t_count", "T-count"), ("cnot_count", "CNOT"),
               ("qubit_count", "Qubits"), ("gate_count", "Total gates")]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    width = 0.8 / len(approaches)
    x = np.arange(len(ns))

    colors = {"rotation_sparse": "#03A9F4", "rotation_dense": "#0D47A1",
              "qrom": "#FF5722", "selectswap": "#4CAF50"}

    for ax, (key, label) in zip(axes.flat, metrics):
        for i, approach in enumerate(approaches):
            heights = []
            for n in ns:
                vals = [r[key] for r in rows
                        if r["n"] == n and r["approach"] == approach and r["success"]]
                heights.append(_mean(vals) if vals else 0)
            offset = (i - len(approaches) / 2 + 0.5) * width
            ax.bar(x + offset, heights, width, label=approach,
                   color=colors.get(approach, "#9C27B0"))
        ax.set_xlabel("n")
        ax.set_ylabel(f"Mean {label}")
        ax.set_title(label)
        ax.set_xticks(x)
        ax.set_xticklabels(ns)
        ax.legend(fontsize=8)
        ax.set_yscale("log")
        ax.grid(True, alpha=0.3, axis="y")

    fig.tight_layout()
    fig.savefig(out_dir / "gate_count_comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_runtime(rows: List[Dict], out_dir: Path) -> None:
    plt = _safe_import_plt(out_dir)
    approaches = sorted(set(r["approach"] for r in rows))
    ns = sorted(set(r["n"] for r in rows))

    fig, ax = plt.subplots(figsize=(10, 6))
    markers = {"rotation_sparse": "v", "rotation_dense": "^", "qrom": "s", "selectswap": "D"}
    colors = {"rotation_sparse": "#03A9F4", "rotation_dense": "#0D47A1",
              "qrom": "#FF5722", "selectswap": "#4CAF50"}

    for approach in approaches:
        x_vals, y_vals = [], []
        for n in ns:
            vals = [r["time_ms"] for r in rows
                    if r["n"] == n and r["approach"] == approach and r["success"]]
            if vals:
                x_vals.append(n)
                y_vals.append(_mean(vals))
        if x_vals:
            ax.plot(x_vals, y_vals, marker=markers.get(approach, "o"),
                    color=colors.get(approach, "#9C27B0"),
                    label=approach, linewidth=2, markersize=8)

    ax.set_xlabel("Number of qubits (n)")
    ax.set_ylabel("Mean runtime (ms)")
    ax.set_title("Synthesis Runtime Comparison")
    ax.legend()
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "runtime_comparison.png", dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate QSP summary and plots")
    parser.add_argument("csv_file", type=Path, help="Path to results CSV")
    parser.add_argument("-o", "--output-dir", type=Path, default=None)
    args = parser.parse_args()

    out_dir = args.output_dir or args.csv_file.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = load_csv(args.csv_file)
    print(f"Loaded {len(rows)} rows")

    generate_summary(rows, out_dir)
    print("Generated comparison_summary.md")

    try:
        plot_t_cost_scaling(rows, out_dir)
        plot_gate_breakdown(rows, out_dir)
        plot_runtime(rows, out_dir)
        print("Generated plots")
    except ImportError as e:
        print(f"Skipping plots (missing dependency): {e}")


if __name__ == "__main__":
    main()

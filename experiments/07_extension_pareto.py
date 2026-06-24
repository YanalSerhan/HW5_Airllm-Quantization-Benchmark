"""
Experiment 07 — Original Extension / Creative Addition (Phase 7).
Generates an additional comparative graph: Quality vs. Speed Pareto frontier.
"""

import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt


def main():
    csv_file = Path("results/benchmark_metrics.csv")
    if not csv_file.exists():
        print("Error: benchmark_metrics.csv not found.")
        sys.exit(1)

    labels = []
    throughputs = []
    qualities = []

    with open(csv_file, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            labels.append(row["quantization_level"])
            throughputs.append(float(row["throughput_tokens_per_sec"]))
            qualities.append(float(row["quality_score"]))

    # Plot
    fig, ax = plt.subplots(figsize=(8, 5))

    # Scatter points
    colors = ["#4C72B0", "#DD8452", "#55A868"]
    for i in range(len(labels)):
        ax.scatter(
            throughputs[i], qualities[i],
            color=colors[i % len(colors)], s=150, zorder=5, label=labels[i],
        )
        ax.annotate(labels[i], (throughputs[i], qualities[i]),
                    xytext=(5, -15), textcoords='offset points', fontsize=10, fontweight='bold')

    # Connect points to show Pareto frontier (assuming roughly ordered)
    # Sort by throughput ascending
    sorted_indices = sorted(range(len(throughputs)), key=lambda k: throughputs[k])
    ax.plot([throughputs[i] for i in sorted_indices],
            [qualities[i] for i in sorted_indices],
            linestyle='--', color='gray', alpha=0.7, zorder=1)

    ax.set_xlabel("Throughput (Tokens / Second) -> Higher is better")
    ax.set_ylabel("Quality Score -> Higher is better")
    ax.set_title("Original Extension: Quality vs. Speed Pareto Frontier")
    ax.grid(True, linestyle=":", alpha=0.6)

    # Save
    out_dir = Path("figures")
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "pareto_frontier.png"
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"Saved Pareto frontier plot to {out_path}")

if __name__ == "__main__":
    main()

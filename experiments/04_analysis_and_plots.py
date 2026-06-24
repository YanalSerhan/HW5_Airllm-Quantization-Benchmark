"""
Experiment 04 — Analysis & Plot Generation.

Loads results/benchmark_metrics.csv produced by experiment 03 and:
  1. Prints and saves a Markdown comparison table.
  2. Generates four required figures via HW5SDK.run_plots():
       - figures/latency_comparison.png   (TTFT & TPOT side-by-side)
       - figures/throughput_comparison.png (tokens/sec per quant level)
       - figures/memory_usage.png          (peak RAM, CPU-only machine)
       - figures/roofline_diagram.png

Run with: uv run python experiments/04_analysis_and_plots.py
Prerequisites: results/benchmark_metrics.csv must exist (run 03 first).
"""

import csv
import logging
import sys
from pathlib import Path

# Allow running from project root without installing the package
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hw5_airllm_benchmark.sdk.sdk import HW5SDK  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

_CSV_FILE = Path("results/benchmark_metrics.csv")
_TABLE_FILE = Path("results/performance_table.md")

_DISPLAY_COLS = [
    ("quantization_level", "Quant Level", "<10"),
    ("ttft_seconds", "TTFT (s)", ">10"),
    ("tpot_seconds", "TPOT (s)", ">10"),
    ("throughput_tokens_per_sec", "Tok/s", ">8"),
    ("peak_ram_gb", "RAM (GB)", ">9"),
    ("total_time_seconds", "Total (s)", ">10"),
    ("estimated_energy_wh", "Energy (Wh)", ">12"),
    ("quality_score", "Quality", ">9"),
]


def _load_csv(path: Path) -> list[dict]:
    """Load metric rows from the benchmark CSV."""
    if not path.exists():
        logger.error("CSV not found: %s — run 03_airllm_run.py first.", path)
        sys.exit(1)
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _build_markdown_table(rows: list[dict]) -> str:
    """Render a Markdown comparison table from metric rows."""
    headers = [c[1] for c in _DISPLAY_COLS]
    sep = ["-" * max(len(h), 8) for h in headers]
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(sep) + " |"]
    for row in rows:
        cells = []
        for col, _, _ in _DISPLAY_COLS:
            val = row.get(col, "N/A")
            try:
                val = f"{float(val):.4f}" if "." in str(val) else val
            except (ValueError, TypeError):
                pass
            cells.append(str(val))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _print_table(rows: list[dict]) -> None:
    """Print a formatted comparison table to stdout."""
    print("\n=== Performance Comparison Table ===")
    # Header
    header_parts = []
    for _, label, fmt in _DISPLAY_COLS:
        header_parts.append(format(label, fmt))
    print("  ".join(header_parts))
    print("  ".join("-" * max(len(label), 8) for _, label, _ in _DISPLAY_COLS))
    # Rows
    for row in rows:
        parts = []
        for col, _, fmt in _DISPLAY_COLS:
            val = row.get(col, "N/A")
            try:
                val = f"{float(val):.4f}"
            except (ValueError, TypeError):
                pass
            parts.append(format(str(val), fmt))
        print("  ".join(parts))
    print()


def main() -> None:
    """Load CSV, generate table, generate all figures."""
    sdk = HW5SDK()

    rows = _load_csv(_CSV_FILE)
    logger.info("Loaded %d metric rows from %s", len(rows), _CSV_FILE)

    # 1. Print and save comparison table
    _print_table(rows)
    md_table = _build_markdown_table(rows)
    _TABLE_FILE.parent.mkdir(parents=True, exist_ok=True)
    _TABLE_FILE.write_text(md_table, encoding="utf-8")
    logger.info("Markdown table saved to %s", _TABLE_FILE)

    # 2. Generate figures (requires matplotlib)
    logger.info("Generating figures …")
    saved_figures = sdk.run_plots(rows, figures_dir="figures")
    for fig_path in saved_figures:
        logger.info("  Saved: %s", fig_path)

    print(f"\n{len(saved_figures)} figure(s) saved to figures/")
    print("Run complete. Check figures/ and results/ for all outputs.")


if __name__ == "__main__":
    main()

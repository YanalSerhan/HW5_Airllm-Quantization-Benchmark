"""
Plotter service — generates all required figures for Phase 4 analysis.

Building-block contract (GUIDE §16):
  Input:  list[dict] of metric rows, output directory path
  Output: PNG files written to figures/
  Setup:  constructor receives config dict; no hidden global state
"""

import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# Column name constants (mirror constants.py to avoid circular import)
_COL_QUANT = "quantization_level"
_COL_TTFT = "ttft_seconds"
_COL_TPOT = "tpot_seconds"
_COL_THROUGHPUT = "throughput_tokens_per_sec"
_COL_RAM = "peak_ram_gb"
_COL_VRAM = "peak_vram_gb"
_COL_QUALITY = "quality_score"

_QUANT_ORDER = ["4bit", "8bit", "fp16"]
_COLORS = ["#4C72B0", "#DD8452", "#55A868"]


class Plotter:
    """
    Generates comparison figures from benchmark metric rows.

    Why: Separating all matplotlib logic into a dedicated service keeps
    experiment scripts thin and makes it trivial to add new plot types
    without touching the SDK or benchmarking code. (GUIDE §16.2)
    """

    def __init__(self, cfg: dict, figures_dir: str = "figures"):
        self._cfg = cfg
        self._out = Path(figures_dir)
        self._out.mkdir(parents=True, exist_ok=True)

    def plot_all(self, rows: list[dict]) -> list[Path]:
        """
        Generate all Phase 4 required figures from metric rows.

        Returns list of Path objects pointing to saved PNG files.
        """
        if not rows:
            logger.warning("No metric rows provided; skipping plots.")
            return []
        saved: list[Path] = []
        saved.append(self._plot_performance_comparison(rows))
        saved.append(self._plot_memory_usage(rows))
        saved.append(self._plot_roofline(rows))
        logger.info("Saved %d figures to %s", len(saved), self._out)
        return saved

    # ------------------------------------------------------------------
    # Private plotting helpers
    # ------------------------------------------------------------------

    def _plot_performance_comparison(self, rows: list[dict]) -> Path:
        """
        Grouped bar chart: TTFT, TPOT, and throughput per quant level.

        Shows the latency/speed trade-off across quantization levels,
        the primary numeric result required by EX05 §5.4.
        """
        import matplotlib.pyplot as plt
        import numpy as np

        labels, ttfts, tpots, thrpts = self._extract_latency(rows)
        x = np.arange(len(labels))
        w = 0.25

        fig, ax1 = plt.subplots(figsize=(9, 5))
        ax2 = ax1.twinx()

        ax1.bar(x - w, ttfts, w, label="TTFT (s)", color=_COLORS[0])
        ax1.bar(x, tpots, w, label="TPOT (s)", color=_COLORS[1])
        ax2.bar(x + w, thrpts, w, label="Throughput (tok/s)", color=_COLORS[2])

        ax1.set_xlabel("Quantization Level")
        ax1.set_ylabel("Latency (seconds)")
        ax2.set_ylabel("Throughput (tokens/sec)")
        ax1.set_xticks(x)
        ax1.set_xticklabels(labels)
        ax1.set_title("Performance Comparison: Latency & Throughput by Quantization")
        lines1, labs1 = ax1.get_legend_handles_labels()
        lines2, labs2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labs1 + labs2, loc="upper left")
        fig.tight_layout()

        out = self._out / "performance_comparison.png"
        fig.savefig(out, dpi=150)
        plt.close(fig)
        return out

    def _plot_memory_usage(self, rows: list[dict]) -> Path:
        """
        Grouped bar chart: Peak RAM and VRAM per quantization level.

        Visualises the memory-footprint trade-off — the core motivation
        for quantization as described in L08 §5.
        """
        import matplotlib.pyplot as plt
        import numpy as np

        labels = self._sorted_labels(rows)
        rams = [self._first(rows, q, _COL_RAM) for q in labels]
        vrams = [self._first(rows, q, _COL_VRAM) for q in labels]
        x = np.arange(len(labels))
        w = 0.35

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.bar(x - w / 2, rams, w, label="Peak RAM (GB)", color=_COLORS[0])
        ax.bar(x + w / 2, vrams, w, label="Peak VRAM (GB)", color=_COLORS[1])
        ax.set_xlabel("Quantization Level")
        ax.set_ylabel("Memory (GB)")
        ax.set_title("Peak Memory Usage by Quantization Level")
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.legend()
        fig.tight_layout()

        out = self._out / "memory_usage.png"
        fig.savefig(out, dpi=150)
        plt.close(fig)
        return out

    def _plot_roofline(self, rows: list[dict]) -> Path:
        """
        Qualitative roofline diagram: throughput vs memory bandwidth pressure.

        Maps each quantization level onto a roofline chart to illustrate
        whether each config is compute-bound (Prefill) or memory-bandwidth-
        bound (Decode), as discussed in L08 §3.1 and EX05 §3.
        """
        import matplotlib.pyplot as plt
        import numpy as np

        labels = self._sorted_labels(rows)
        thrpts = [self._first(rows, q, _COL_THROUGHPUT) for q in labels]
        rams = [self._first(rows, q, _COL_RAM) for q in labels]

        # Arithmetic intensity proxy: throughput / RAM pressure
        intensities = [t / max(r, 0.01) for t, r in zip(thrpts, rams)]

        # Draw roofline ceiling lines (qualitative — CPU-only system)
        max_x = max(max(intensities) * 1.5, 0.02) if intensities else 0.02
        x_range = np.linspace(0.0, max_x, 200)
        memory_bw_limit = 2.0  # GB/s approx NVMe SSD bandwidth
        compute_limit = max(thrpts) * 1.6 if thrpts else 10.0
        roofline = np.minimum(x_range * memory_bw_limit, compute_limit)

        fig, ax = plt.subplots(figsize=(9, 5))
        ax.plot(x_range, roofline, "k--", linewidth=1.5, label="Hardware Roofline")
        ax.axvline(
            compute_limit / memory_bw_limit,
            color="gray",
            linestyle=":",
            label="Memory-Bound | Compute-Bound",
        )

        for i, (label, intensity, thr) in enumerate(
            zip(labels, intensities, thrpts)
        ):
            ax.scatter(intensity, thr, s=120, color=_COLORS[i % len(_COLORS)], zorder=5)
            ax.annotate(
                label,
                (intensity, thr),
                textcoords="offset points",
                xytext=(6, 4),
                fontsize=9,
            )

        ax.set_xlabel("Arithmetic Intensity Proxy (throughput / RAM GB)")
        ax.set_ylabel("Achieved Throughput (tokens/sec)")
        ax.set_title("Roofline Diagram: Compute-Bound vs Memory-Bound by Quantization")
        ax.legend()
        fig.tight_layout()

        out = self._out / "roofline_diagram.png"
        fig.savefig(out, dpi=150)
        plt.close(fig)
        return out

    # ------------------------------------------------------------------
    # Utility helpers
    # ------------------------------------------------------------------

    def _sorted_labels(self, rows: list[dict]) -> list[str]:
        """Return quantization labels sorted by _QUANT_ORDER preference."""
        found = list({r.get(_COL_QUANT, "unknown") for r in rows})
        return sorted(found, key=lambda q: _QUANT_ORDER.index(q) if q in _QUANT_ORDER else 99)

    def _first(self, rows: list[dict], quant: str, col: str) -> float:
        """Return the first value for a given quantization level and column."""
        for r in rows:
            if r.get(_COL_QUANT) == quant:
                return float(r.get(col, 0.0))
        return 0.0

    def _extract_latency(self, rows: list[dict]):
        """Extract labels and latency metrics in sorted order."""
        labels = self._sorted_labels(rows)
        ttfts = [self._first(rows, q, _COL_TTFT) for q in labels]
        tpots = [self._first(rows, q, _COL_TPOT) for q in labels]
        thrpts = [self._first(rows, q, _COL_THROUGHPUT) for q in labels]
        return labels, ttfts, tpots, thrpts

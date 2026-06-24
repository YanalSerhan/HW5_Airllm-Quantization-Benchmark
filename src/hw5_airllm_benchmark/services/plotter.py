"""
Plotter service — generates all required figures for performance & economic analysis.

Building-block contract (GUIDE §16):
  Input:  list[dict] of metric rows OR economic_result dict, output directory path
  Output: PNG files written to figures/
  Setup:  constructor receives config dict; no hidden global state

Throughput and roofline plots are delegated to _perf_plotter.py.
Economic plots are delegated to _economic_plotter.py.
Both extractions keep this module under 150 lines. (GUIDE §3.2)
"""

import logging
from pathlib import Path

from . import _economic_plotter as _ep
from . import _perf_plotter as _pp

logger = logging.getLogger(__name__)

_COL_QUANT = "quantization_level"
_COL_TTFT = "ttft_seconds"
_COL_TPOT = "tpot_seconds"
_COL_RAM = "peak_ram_gb"
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
        saved: list[Path] = [
            self._plot_latency_comparison(rows),
            _pp.plot_throughput(rows, self._out),
            self._plot_memory_usage(rows),
            _pp.plot_roofline(rows, self._out),
        ]
        logger.info("Saved %d figures to %s", len(saved), self._out)
        return saved

    def plot_economic_analysis(self, economic_result: dict) -> list[Path]:
        """
        Generate Phase 5 economic figures: break-even curve + cost-per-quant.

        Delegates to _economic_plotter helpers.
        """
        saved: list[Path] = [
            _ep.plot_breakeven(economic_result, self._out),
            _ep.plot_cost_per_quant(economic_result, self._out),
        ]
        logger.info("Saved %d economic figures to %s", len(saved), self._out)
        return saved

    # ------------------------------------------------------------------
    # Private performance plotting helpers (latency + RAM only)
    # ------------------------------------------------------------------

    def _plot_latency_comparison(self, rows: list[dict]) -> Path:
        """
        Grouped bar chart: TTFT and TPOT per quantization level.

        Uses a single Y-axis (seconds) for both metrics — no dual-axis
        confusion. Throughput (tok/s) is on a separate chart. (Bug-fix)
        """
        import matplotlib.pyplot as plt
        import numpy as np

        labels = self._sorted_labels(rows)
        ttfts = [self._first(rows, q, _COL_TTFT) for q in labels]
        tpots = [self._first(rows, q, _COL_TPOT) for q in labels]
        x = np.arange(len(labels))
        w = 0.35
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.bar(x - w / 2, ttfts, w, label="TTFT — Time to First Token (s)",
               color=_COLORS[0])
        ax.bar(x + w / 2, tpots, w, label="TPOT — Time per Output Token (s)",
               color=_COLORS[1])
        ax.set_xlabel("Quantization Level")
        ax.set_ylabel("Latency (seconds)")
        ax.set_title("Latency by Quantization Level: TTFT vs TPOT")
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.legend()
        ax.grid(True, axis="y", alpha=0.3)
        fig.tight_layout()
        out = self._out / "latency_comparison.png"
        fig.savefig(out, dpi=150)
        plt.close(fig)
        return out

    def _plot_memory_usage(self, rows: list[dict]) -> Path:
        """Bar chart: Peak process RAM per quantization level (CPU-only system)."""
        import matplotlib.pyplot as plt
        import numpy as np

        labels = self._sorted_labels(rows)
        rams = [self._first(rows, q, _COL_RAM) for q in labels]
        x = np.arange(len(labels))
        fig, ax = plt.subplots(figsize=(8, 5))
        bars = ax.bar(x, rams, 0.5, color=_COLORS[0])
        ax.bar_label(bars, fmt="%.2f GB", padding=3, fontsize=9)
        ax.set_xlabel("Quantization Level")
        ax.set_ylabel("Peak Process RAM (GB)")
        ax.set_title("Peak RAM Usage by Quantization Level\n(CPU-only — no discrete GPU)")
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.grid(True, axis="y", alpha=0.3)
        fig.tight_layout()
        out = self._out / "memory_usage.png"
        fig.savefig(out, dpi=150)
        plt.close(fig)
        return out

    # ------------------------------------------------------------------
    # Utility helpers
    # ------------------------------------------------------------------

    def _sorted_labels(self, rows: list[dict]) -> list[str]:
        """Return quantization labels sorted by _QUANT_ORDER preference."""
        found = list({r.get(_COL_QUANT, "unknown") for r in rows})
        return sorted(
            found, key=lambda q: _QUANT_ORDER.index(q) if q in _QUANT_ORDER else 99
        )

    def _first(self, rows: list[dict], quant: str, col: str) -> float:
        """Return first value for a given quantization level and column."""
        for r in rows:
            if r.get(_COL_QUANT) == quant:
                return float(r.get(col, 0.0))
        return 0.0

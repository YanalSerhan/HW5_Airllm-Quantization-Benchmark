"""
Performance plot helpers — throughput bar chart and roofline diagram.

Extracted from plotter.py to keep that module under 150 lines. (GUIDE §3.2)

Building-block contract (GUIDE §16):
  Input:  list[dict] of metric rows, output directory Path
  Output: PNG files written to provided directory
  Setup:  pure functions; no hidden global state
"""

import logging
from pathlib import Path

logger = logging.getLogger(__name__)

_COL_QUANT = "quantization_level"
_COL_THROUGHPUT = "throughput_tokens_per_sec"
_COL_RAM = "peak_ram_gb"
_QUANT_ORDER = ["4bit", "8bit", "fp16"]
_COLORS = ["#4C72B0", "#DD8452", "#55A868"]


def _sorted_labels(rows: list[dict]) -> list[str]:
    """Return quantization labels sorted by _QUANT_ORDER preference."""
    found = list({r.get(_COL_QUANT, "unknown") for r in rows})
    return sorted(
        found, key=lambda q: _QUANT_ORDER.index(q) if q in _QUANT_ORDER else 99
    )


def _first(rows: list[dict], quant: str, col: str) -> float:
    """Return first value for a given quantization level and column."""
    for r in rows:
        if r.get(_COL_QUANT) == quant:
            return float(r.get(col, 0.0))
    return 0.0


def plot_throughput(rows: list[dict], out_dir: Path) -> Path:
    """
    Bar chart: throughput (tokens/sec) per quantization level.

    Separate from latency chart so both use honest, same-unit Y-axes.
    Corresponds to EX05 §5.4 throughput requirement.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    labels = _sorted_labels(rows)
    thrpts = [_first(rows, q, _COL_THROUGHPUT) for q in labels]
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(x, thrpts, 0.5, color=_COLORS[2])
    ax.bar_label(bars, fmt="%.4f tok/s", padding=3, fontsize=9)
    ax.set_xlabel("Quantization Level")
    ax.set_ylabel("Throughput (tokens / second)")
    ax.set_title("Inference Throughput by Quantization Level")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    out = out_dir / "throughput_comparison.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    logger.info("Saved throughput chart: %s", out)
    return out


def plot_roofline(rows: list[dict], out_dir: Path) -> Path:
    """
    Qualitative roofline diagram: throughput vs arithmetic intensity proxy.

    Maps each quantization config onto a roofline to illustrate whether
    each is compute-bound (Prefill) or memory-bandwidth-bound (Decode),
    as discussed in L08 §3.1 and EX05 §3.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    labels = _sorted_labels(rows)
    thrpts = [_first(rows, q, _COL_THROUGHPUT) for q in labels]
    rams = [_first(rows, q, _COL_RAM) for q in labels]
    intensities = [t / max(r, 0.01) for t, r in zip(thrpts, rams)]
    max_x = max(max(intensities) * 1.5, 0.02) if intensities else 0.02
    x_range = np.linspace(0.0, max_x, 200)
    memory_bw_limit = 2.0
    compute_limit = max(thrpts) * 1.6 if thrpts else 10.0
    roofline = np.minimum(x_range * memory_bw_limit, compute_limit)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(x_range, roofline, "k--", linewidth=1.5, label="Hardware Roofline")
    ax.axvline(
        compute_limit / memory_bw_limit, color="gray", linestyle=":",
        label="Memory-Bound | Compute-Bound boundary",
    )
    for i, (label, intensity, thr) in enumerate(zip(labels, intensities, thrpts)):
        ax.scatter(intensity, thr, s=140, color=_COLORS[i % len(_COLORS)], zorder=5)
        ax.annotate(label, (intensity, thr), textcoords="offset points",
                    xytext=(6, 4), fontsize=9)
    ax.set_xlabel("Arithmetic Intensity Proxy (throughput / peak RAM GB)")
    ax.set_ylabel("Achieved Throughput (tokens/sec)")
    ax.set_title("Roofline Diagram: Compute-Bound vs Memory-Bound")
    ax.legend()
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    out = out_dir / "roofline_diagram.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    logger.info("Saved roofline diagram: %s", out)
    return out

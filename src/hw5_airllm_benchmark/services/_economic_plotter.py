"""
Economic plot helpers — break-even curve and per-quant cost bar chart.

Extracted from plotter.py to keep each module under 150 lines (GUIDE §3.2).

Building-block contract (GUIDE §16):
  Input:  economic_result dict (from EconomicAnalyser.analyse())
  Output: PNG files written to the provided output directory
  Setup:  pure functions; no hidden global state
"""

import logging
from pathlib import Path

logger = logging.getLogger(__name__)

_COLORS_QUANT = ["#4C72B0", "#DD8452", "#55A868"]
_COLOR_API = "#E05252"
_COLOR_ONPREM = "#4C8BBE"
_COLOR_BREAKEVEN = "#2CA02C"


def plot_breakeven(economic_result: dict, out_dir: Path) -> Path:
    """
    Dual line chart: cumulative API cost vs On-Prem cost over request volume.

    Marks the break-even crossover clearly with a vertical dashed line and
    annotation. Illustrates EX05 §5.3 break-even analysis requirement.
    """
    import matplotlib.pyplot as plt

    curve = economic_result.get("cost_curve", {})
    volumes = curve.get("volumes", [])
    api_vals = curve.get("api_cumulative", [])
    onprem_vals = curve.get("onprem_cumulative", [])
    break_even = economic_result.get("break_even_requests")

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(volumes, api_vals, color=_COLOR_API, linewidth=2.0, label="API (OpenAI GPT-4o)")
    ax.plot(
        volumes, onprem_vals, color=_COLOR_ONPREM, linewidth=2.0,
        label="On-Premise (CAPEX + OPEX)",
    )

    if isinstance(break_even, int) and 0 < break_even <= max(volumes, default=0):
        ax.axvline(break_even, color=_COLOR_BREAKEVEN, linestyle="--", linewidth=1.5)
        ax.annotate(
            f"Break-even\n~{break_even:,} requests",
            xy=(break_even, break_even * economic_result.get("api_cost_per_request_usd", 0)),
            xytext=(break_even + max(volumes, default=1) * 0.05,
                    max(api_vals, default=1) * 0.5),
            arrowprops={"arrowstyle": "->", "color": _COLOR_BREAKEVEN},
            color=_COLOR_BREAKEVEN, fontsize=9,
        )

    _add_assumptions_box(ax, economic_result)
    ax.set_xlabel("Cumulative Requests")
    ax.set_ylabel("Cumulative Cost (USD)")
    ax.set_title("Economic Break-Even Analysis: API vs On-Premise Deployment")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    out = out_dir / "economic_breakeven.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    logger.info("Saved break-even chart: %s", out)
    return out


def plot_cost_per_quant(economic_result: dict, out_dir: Path) -> Path:
    """
    Grouped bar chart: per-request cost breakdown by quantization level.

    Shows electricity (OPEX) vs amortised CAPEX vs API cost side-by-side,
    making explicit which quantization level is most cost-efficient locally
    vs API — directly addressing EX05 §5.1 and §5.2.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    per_quant = economic_result.get("per_quant_costs", [])
    if not per_quant:
        logger.warning("No per-quant cost data; skipping cost-per-quant plot.")
        return out_dir / "cost_per_quant_level.png"

    labels = [q["quantization_level"] for q in per_quant]
    elec = [q["onprem_electricity_usd"] for q in per_quant]
    capex = [q["onprem_capex_amortised_usd"] for q in per_quant]
    api = [q["api_cost_usd"] for q in per_quant]
    x = np.arange(len(labels))
    w = 0.25

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x - w, elec, w, label="On-Prem: Electricity (OPEX)", color=_COLOR_ONPREM)
    ax.bar(x, capex, w, label="On-Prem: Amortised CAPEX", color="#7BAFD4", alpha=0.85)
    ax.bar(x + w, api, w, label="API Cost (GPT-4o)", color=_COLOR_API)

    ax.set_xlabel("Quantization Level")
    ax.set_ylabel("Cost per Request (USD)")
    ax.set_title("Per-Request Cost Breakdown by Quantization Level")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()

    out = out_dir / "cost_per_quant_level.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    logger.info("Saved cost-per-quant chart: %s", out)
    return out


def _add_assumptions_box(ax, economic_result: dict) -> None:
    """Overlay a small text box listing key pricing assumptions."""
    a = economic_result.get("assumptions", {})
    lines = [
        f"Electricity: ${a.get('electricity_kwh_price_usd', '?')}/kWh",
        f"HW cost: ${a.get('hardware_cost_usd', '?')} over "
        f"{a.get('hardware_lifetime_years', '?')} yrs",
        f"API: ${a.get('openai_input_price_per_1k_usd', '?')}/1k in, "
        f"${a.get('openai_output_price_per_1k_usd', '?')}/1k out",
    ]
    ax.text(
        0.98, 0.05, "\n".join(lines),
        transform=ax.transAxes,
        fontsize=7, verticalalignment="bottom", horizontalalignment="right",
        bbox={"boxstyle": "round,pad=0.3", "facecolor": "lightyellow", "alpha": 0.7},
    )

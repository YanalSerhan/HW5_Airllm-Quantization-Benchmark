"""
Experiment 05 — Economic / Business Analysis (Phase 5, EX05 §5.5).

Loads results/benchmark_metrics.csv (produced by experiment 03) and:
  1. Runs EconomicAnalyser to compute API vs On-Prem costs, break-even, and
     full cost-curve arrays.
  2. Prints a human-readable assumptions + results table to stdout.
  3. Generates two Phase 5 figures via HW5SDK.run_economic_plots():
       - figures/economic_breakeven.png
       - figures/cost_per_quant_level.png
  4. Saves results/economic_analysis.json with the full structured result.

Run with: uv run python experiments/05_economic_analysis.py
Prerequisites: results/benchmark_metrics.csv must exist (run 03 first).
"""

import csv
import json
import logging
import sys
from pathlib import Path

# Allow running from project root without installing the package
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hw5_airllm_benchmark.sdk.sdk import HW5SDK  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

_CSV_FILE = Path("results/benchmark_metrics.csv")
_OUTPUT_JSON = Path("results/economic_analysis.json")


def _load_csv(path: Path) -> list[dict]:
    """Load metric rows from the benchmark CSV."""
    if not path.exists():
        logger.error("CSV not found: %s — run 03_airllm_run.py first.", path)
        sys.exit(1)
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _print_summary(result: dict) -> None:
    """Print a formatted economic summary to stdout."""
    print("\n" + "=" * 60)
    print("  PHASE 5 — ECONOMIC / BUSINESS ANALYSIS")
    print("=" * 60)

    a = result.get("assumptions", {})
    print("\n--- Assumptions ---")
    print(f"  API model:             {a.get('openai_model', 'N/A')}")
    print(f"  API input price:       ${a.get('openai_input_price_per_1k_usd')}/1k tokens")
    print(f"  API output price:      ${a.get('openai_output_price_per_1k_usd')}/1k tokens")
    print(f"  Electricity rate:      ${a.get('electricity_kwh_price_usd')}/kWh")
    print(f"  Hardware cost:         ${a.get('hardware_cost_usd')}")
    print(f"  Hardware lifetime:     {a.get('hardware_lifetime_years')} years")
    print(f"  Daily requests (CAPEX basis): {a.get('daily_requests_assumed')}")

    print("\n--- Average Per-Request Costs ---")
    print(f"  API cost:              ${result.get('api_cost_per_request_usd', 0):.6f}")
    print(f"  On-Prem electricity:   ${result.get('onprem_electricity_only_usd', 0):.6f}")
    print(f"  On-Prem total (CAPEX+OPEX): ${result.get('onprem_cost_per_request_usd', 0):.6f}")

    print("\n--- Per-Quantization-Level Breakdown ---")
    cols = f"  {'Quant':<8} {'Output Tok':>10} {'API (USD)':>12}"
    cols += f" {'Electricity':>12} {'CAPEX':>10} {'Total On-Prem':>14}"
    print(cols)
    print("  " + "-" * (len(cols) - 2))
    for q in result.get("per_quant_costs", []):
        print(
            f"  {q['quantization_level']:<8} "
            f"{q['output_tokens']:>10} "
            f"${q['api_cost_usd']:>11.6f} "
            f"${q['onprem_electricity_usd']:>11.6f} "
            f"${q['onprem_capex_amortised_usd']:>9.6f} "
            f"${q['onprem_total_cost_usd']:>13.6f}"
        )

    be = result.get("break_even_requests")
    print("\n--- Break-Even Analysis ---")
    if isinstance(be, int):
        print(f"  Break-even volume:     ~{be:,} requests")
        print(f"  Interpretation: On-Premise becomes cheaper after {be:,} requests.")
        print("  At low/moderate volumes the API is more cost-effective;")
        print("  at scale (high daily usage), On-Prem amortises the CAPEX.")
    else:
        print(f"  Break-even:            {be}")

    print("\n--- Recommendation ---")
    if isinstance(be, int):
        print(f"  For occasional/low-volume use (< {be:,} requests), use the API.")
        print("  For sustained, high-volume workloads, On-Premise is economical.")
        print("  Non-cost factors (data privacy, offline capability) may favour")
        print("  On-Premise even at lower volumes.")
    else:
        print("  The API is currently cheaper than On-Premise for this volume.")
        print("  On-Premise might only be justified by non-cost factors like data privacy.")
        
    print("\n--- API Prompt/Context Caching Note ---")
    print("  Modern APIs use PagedAttention/prompt caching. For workloads with")
    print("  repeated context (e.g., long-document Q&A), API input costs can")
    print("  drop significantly. This shifts the break-even curve, making")
    print("  APIs cost-effective for even higher request volumes.")
    print("=" * 60 + "\n")


def main() -> None:
    """Load benchmark CSV, run economic analysis, save results and figures."""
    sdk = HW5SDK()
    rows = _load_csv(_CSV_FILE)
    logger.info("Loaded %d metric rows from %s", len(rows), _CSV_FILE)

    # 1. Run economic analysis
    result = sdk.run_economic_analysis(rows)
    logger.info("Economic analysis complete.")

    # 2. Print human-readable summary
    _print_summary(result)

    # 3. Save structured JSON result
    _OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    _OUTPUT_JSON.write_text(json.dumps(result, indent=2), encoding="utf-8")
    logger.info("Economic analysis saved to %s", _OUTPUT_JSON)

    # 4. Generate Phase 5 figures
    logger.info("Generating economic figures …")
    saved_figures = sdk.run_economic_plots(result, figures_dir="figures")
    for fig_path in saved_figures:
        logger.info("  Saved: %s", fig_path)

    print(f"{len(saved_figures)} economic figure(s) saved to figures/")
    print(f"Structured results saved to {_OUTPUT_JSON}")
    print("Phase 5 complete.")


if __name__ == "__main__":
    main()

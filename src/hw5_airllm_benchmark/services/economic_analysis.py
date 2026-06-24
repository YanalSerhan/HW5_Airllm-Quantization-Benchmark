"""
EconomicAnalyser service — computes break-even costs for API vs On-Prem.

Building-block contract (GUIDE §16):
  Input:  cfg dict, list of per-run metric dicts (from benchmark_metrics.csv)
  Output: dict with per-request costs, per-quant breakdown, volume curves,
          break-even volume, and all explicit assumptions
  Setup:  constructor receives cfg; no hidden global state
"""

import logging
from typing import Any

logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------
# Default pricing assumptions — all overridable; documented explicitly
# per EX05 §5.5 ("state all assumptions clearly").
# Sources: OpenAI pricing page (May 2025), US EIA residential electricity avg.
# -----------------------------------------------------------------------
_OPENAI_INPUT_PRICE_PER_1K = 0.005   # USD/1k input tokens  (GPT-4o)
_OPENAI_OUTPUT_PRICE_PER_1K = 0.015  # USD/1k output tokens (GPT-4o)
_ELECTRICITY_KWH_PRICE = 0.12        # USD/kWh (avg US residential, EIA 2024)
_HARDWARE_COST_USD = 1_200.0         # Estimated system purchase cost
_HARDWARE_LIFETIME_YEARS = 4.0       # Straight-line depreciation period
_DAILY_REQUESTS = 10                 # Assumed daily usage for CAPEX spread
_VOLUME_STEPS = 1_000                # Number of points in cost-curve arrays
_VOLUME_MAX = 50_000                 # Max request volume to plot


class EconomicAnalyser:
    """
    Calculates API vs On-Premise costs and finds the break-even point.

    Why: Separating economic calculation from benchmarking keeps each
    class focused on a single responsibility and makes unit-testing
    trivial (no GPU required). (GUIDE §16.2)
    """

    def __init__(self, cfg: dict):
        self._cfg = cfg

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def analyse(self, metrics: list[dict]) -> dict[str, Any]:
        """
        Compute per-request/volumetric costs for API and On-Prem.

        Returns a rich dict containing:
          - api_cost_per_request_usd
          - onprem_cost_per_request_usd  (electricity + amortised CAPEX)
          - onprem_electricity_only_usd  (variable OPEX component alone)
          - break_even_requests
          - per_quant_costs             (list, one entry per quant level)
          - cost_curve                  (dict with 'volumes', 'api', 'onprem')
          - assumptions                 (all pricing inputs documented)
        """
        if not metrics:
            logger.warning("No metrics provided; returning empty analysis.")
            return {}

        input_tokens = 50  # prompt is ~50 tokens (documented assumption)

        per_quant = self._per_quant_costs(metrics, input_tokens)
        avg_api = sum(q["api_cost_usd"] for q in per_quant) / len(per_quant)
        avg_onprem = sum(q["onprem_total_cost_usd"] for q in per_quant) / len(per_quant)
        avg_elec = sum(q["onprem_electricity_usd"] for q in per_quant) / len(per_quant)

        break_even = self._break_even(avg_api, avg_onprem)
        curve = self._cost_curve(avg_api, avg_onprem)
        assumptions = self._build_assumptions(input_tokens, per_quant)

        result: dict[str, Any] = {
            "api_cost_per_request_usd": round(avg_api, 6),
            "onprem_cost_per_request_usd": round(avg_onprem, 6),
            "onprem_electricity_only_usd": round(avg_elec, 6),
            "break_even_requests": break_even,
            "per_quant_costs": per_quant,
            "cost_curve": curve,
            "assumptions": assumptions,
        }
        logger.info("Economic analysis complete. Break-even: %s", break_even)
        return result

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _per_quant_costs(self, metrics: list[dict], input_tokens: int) -> list[dict]:
        """Return per-request cost breakdown for every quantization level."""
        rows = []
        for m in metrics:
            throughput = float(m.get("throughput_tokens_per_sec", 0) or 0)
            total_time = float(m.get("total_time_seconds", 0) or 0)
            energy_wh = float(m.get("estimated_energy_wh", 0) or 0)
            output_tokens = int(throughput * total_time)
            api_cost = self._api_cost(input_tokens, output_tokens)
            elec_cost = self._electricity_cost(energy_wh)
            capex_cost = self._capex_per_req()
            rows.append({
                "quantization_level": m.get("quantization_level", "unknown"),
                "output_tokens": output_tokens,
                "api_cost_usd": round(api_cost, 6),
                "onprem_electricity_usd": round(elec_cost, 6),
                "onprem_capex_amortised_usd": round(capex_cost, 6),
                "onprem_total_cost_usd": round(elec_cost + capex_cost, 6),
            })
        return rows

    @staticmethod
    def _api_cost(input_tokens: int, output_tokens: int) -> float:
        """Compute OpenAI GPT-4o API cost for one request."""
        return (
            input_tokens / 1_000 * _OPENAI_INPUT_PRICE_PER_1K
            + output_tokens / 1_000 * _OPENAI_OUTPUT_PRICE_PER_1K
        )

    @staticmethod
    def _electricity_cost(energy_wh: float) -> float:
        """Variable OPEX per request: electricity only."""
        return (energy_wh / 1_000.0) * _ELECTRICITY_KWH_PRICE

    @staticmethod
    def _capex_per_req() -> float:
        """Amortised CAPEX per request (straight-line over lifetime)."""
        lifetime_days = _HARDWARE_LIFETIME_YEARS * 365
        return _HARDWARE_COST_USD / (_DAILY_REQUESTS * lifetime_days)

    @staticmethod
    def _break_even(api_per_req: float, onprem_per_req: float) -> int | str:
        """Volume at which cumulative On-Prem cost equals cumulative API cost."""
        if api_per_req <= onprem_per_req:
            return "Never (API is always cheaper at this usage level)"
        volume = _HARDWARE_COST_USD / (api_per_req - onprem_per_req)
        return int(volume)

    @staticmethod
    def _cost_curve(api_per_req: float, onprem_per_req: float) -> dict:
        """Build volume-indexed cumulative cost arrays for plotting."""
        step = _VOLUME_MAX // _VOLUME_STEPS
        volumes = list(range(0, _VOLUME_MAX + step, step))
        api_curve = [round(v * api_per_req, 4) for v in volumes]
        onprem_curve = [
            round(_HARDWARE_COST_USD + v * onprem_per_req, 4)
            for v in volumes
        ]
        return {"volumes": volumes, "api_cumulative": api_curve,
                "onprem_cumulative": onprem_curve}

    @staticmethod
    def _build_assumptions(input_tokens: int, per_quant: list[dict]) -> dict:
        """Return every pricing assumption for transparent reporting."""
        return {
            "openai_model": "GPT-4o (May 2025 pricing)",
            "openai_input_price_per_1k_usd": _OPENAI_INPUT_PRICE_PER_1K,
            "openai_output_price_per_1k_usd": _OPENAI_OUTPUT_PRICE_PER_1K,
            "electricity_kwh_price_usd": _ELECTRICITY_KWH_PRICE,
            "electricity_source": "US EIA 2024 avg residential rate",
            "hardware_cost_usd": _HARDWARE_COST_USD,
            "hardware_lifetime_years": _HARDWARE_LIFETIME_YEARS,
            "daily_requests_assumed": _DAILY_REQUESTS,
            "input_tokens_per_request": input_tokens,
            "avg_output_tokens_per_quant": {
                q["quantization_level"]: q["output_tokens"] for q in per_quant
            },
            "capex_depreciation": "Straight-line over hardware lifetime",
            "api_context_caching_note": (
                "Modern APIs use PagedAttention/prompt caching. For repeated-context "
                "workloads, input costs may drop significantly, pushing the break-even "
                "volume higher and favoring APIs."
            ),
        }

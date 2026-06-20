"""
EconomicAnalyser service — computes break-even costs for API vs On-Prem.

Building-block contract (GUIDE §16):
  Input:  cfg dict, list of per-run metric dicts
  Output: dict with cost tables and break-even volume
  Setup:  constructor receives cfg; no hidden global state
"""

import logging
from typing import Any

logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------
# Default pricing assumptions (overridable via config in future)
# All values sourced from public pricing pages; update before submission.
# -----------------------------------------------------------------------
_OPENAI_INPUT_PRICE_PER_1K = 0.005   # USD per 1k input tokens (GPT-4o)
_OPENAI_OUTPUT_PRICE_PER_1K = 0.015  # USD per 1k output tokens (GPT-4o)
_ELECTRICITY_KWH_PRICE = 0.12        # USD per kWh (avg US residential)
_HARDWARE_COST_USD = 1200.0          # Estimated system cost
_HARDWARE_LIFETIME_YEARS = 4.0


class EconomicAnalyser:
    """
    Calculates API vs On-Premise costs and finds the break-even point.

    Why: Separating economic calculation from benchmarking keeps each
    class focused on a single responsibility and makes unit-testing
    trivial (no GPU required). (GUIDE §16.2)
    """

    def __init__(self, cfg: dict):
        self._cfg = cfg

    def analyse(self, metrics: list[dict]) -> dict[str, Any]:
        """
        Compute per-request and volumetric costs for API and On-Prem.

        Returns a dict suitable for plotting and report inclusion.
        """
        if not metrics:
            logger.warning("No metrics provided; returning empty analysis.")
            return {}

        avg_tokens_out = self._avg_metric(metrics, "throughput_tokens_per_sec")
        avg_energy_wh = self._avg_metric(metrics, "estimated_energy_wh")
        avg_total_time = self._avg_metric(metrics, "total_time_seconds")

        input_tokens = 50   # fixed prompt length assumption
        output_tokens = int(avg_tokens_out * avg_total_time)

        api_cost_per_req = self._api_cost(input_tokens, output_tokens)
        onprem_cost_per_req = self._onprem_cost(avg_energy_wh)
        break_even = self._break_even(api_cost_per_req, onprem_cost_per_req)

        result = {
            "api_cost_per_request_usd": round(api_cost_per_req, 6),
            "onprem_cost_per_request_usd": round(onprem_cost_per_req, 6),
            "break_even_requests": break_even,
            "assumptions": {
                "openai_input_price_per_1k": _OPENAI_INPUT_PRICE_PER_1K,
                "openai_output_price_per_1k": _OPENAI_OUTPUT_PRICE_PER_1K,
                "electricity_kwh_price_usd": _ELECTRICITY_KWH_PRICE,
                "hardware_cost_usd": _HARDWARE_COST_USD,
                "hardware_lifetime_years": _HARDWARE_LIFETIME_YEARS,
                "input_tokens_per_request": input_tokens,
                "output_tokens_per_request": output_tokens,
            },
        }
        logger.info("Economic analysis: %s", result)
        return result

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _avg_metric(metrics: list[dict], key: str) -> float:
        vals = [m[key] for m in metrics if key in m]
        return sum(vals) / len(vals) if vals else 0.0

    @staticmethod
    def _api_cost(input_tokens: int, output_tokens: int) -> float:
        return (
            input_tokens / 1000 * _OPENAI_INPUT_PRICE_PER_1K
            + output_tokens / 1000 * _OPENAI_OUTPUT_PRICE_PER_1K
        )

    @staticmethod
    def _onprem_cost(energy_wh: float) -> float:
        """Variable OPEX cost per request (electricity only)."""
        energy_kwh = energy_wh / 1000.0
        electricity = energy_kwh * _ELECTRICITY_KWH_PRICE
        # Amortised CAPEX per request (assume 10 reqs/day over lifetime)
        daily_reqs = 10
        lifetime_days = _HARDWARE_LIFETIME_YEARS * 365
        capex_per_req = _HARDWARE_COST_USD / (daily_reqs * lifetime_days)
        return electricity + capex_per_req

    @staticmethod
    def _break_even(api_per_req: float, onprem_per_req: float) -> int | str:
        """Volume at which cumulative On-Prem cost equals cumulative API cost."""
        capex = _HARDWARE_COST_USD
        if api_per_req <= onprem_per_req:
            return "Never (API is always cheaper at this usage level)"
        volume = capex / (api_per_req - onprem_per_req)
        return int(volume)

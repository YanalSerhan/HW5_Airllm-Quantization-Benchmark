"""
Unit tests for EconomicAnalyser.

Tests break-even math, edge cases, and cost formulas without
requiring any external API or model. (GUIDE §16.2)
"""

import pytest

from hw5_airllm_benchmark.services.economic_analysis import EconomicAnalyser

_DUMMY_CFG = {"version": "1.00"}

_SAMPLE_METRICS = [
    {
        "quantization_level": "fp16",
        "ttft_seconds": 5.0,
        "tpot_seconds": 2.5,
        "throughput_tokens_per_sec": 0.4,
        "peak_ram_gb": 8.0,
        "peak_vram_gb": 0.0,
        "total_time_seconds": 120.0,
        "estimated_energy_wh": 1.5,
        "quality_score": 0.9,
    },
    {
        "quantization_level": "4bit",
        "ttft_seconds": 3.0,
        "tpot_seconds": 1.8,
        "throughput_tokens_per_sec": 0.6,
        "peak_ram_gb": 4.0,
        "peak_vram_gb": 0.0,
        "total_time_seconds": 80.0,
        "estimated_energy_wh": 1.0,
        "quality_score": 0.7,
    },
]


class TestEconomicAnalyserBasic:
    """Tests for basic analysis correctness."""

    def setup_method(self):
        self.analyser = EconomicAnalyser(_DUMMY_CFG)

    def test_empty_metrics_returns_empty(self):
        result = self.analyser.analyse([])
        assert result == {}

    def test_result_has_required_keys(self):
        result = self.analyser.analyse(_SAMPLE_METRICS)
        required = {"api_cost_per_request_usd", "onprem_cost_per_request_usd",
                    "break_even_requests", "assumptions"}
        assert required.issubset(result.keys())

    def test_api_cost_is_positive(self):
        result = self.analyser.analyse(_SAMPLE_METRICS)
        assert result["api_cost_per_request_usd"] > 0

    def test_onprem_cost_is_positive(self):
        result = self.analyser.analyse(_SAMPLE_METRICS)
        assert result["onprem_cost_per_request_usd"] > 0

    def test_assumptions_are_listed(self):
        result = self.analyser.analyse(_SAMPLE_METRICS)
        assumptions = result["assumptions"]
        required_keys = {
            "openai_input_price_per_1k",
            "openai_output_price_per_1k",
            "electricity_kwh_price_usd",
            "hardware_cost_usd",
            "hardware_lifetime_years",
        }
        assert required_keys.issubset(assumptions.keys())


class TestEconomicAnalyserBreakEven:
    """Tests for break-even calculation correctness."""

    def test_break_even_is_int_or_str(self):
        analyser = EconomicAnalyser(_DUMMY_CFG)
        result = analyser.analyse(_SAMPLE_METRICS)
        be = result["break_even_requests"]
        assert isinstance(be, (int, str))

    def test_break_even_positive_when_api_more_expensive(self):
        """When API cost > on-prem cost, break-even should be a positive int."""
        analyser = EconomicAnalyser(_DUMMY_CFG)
        # Use metrics that generate non-trivial output (many tokens)
        result = analyser.analyse(_SAMPLE_METRICS)
        be = result["break_even_requests"]
        if isinstance(be, int):
            assert be > 0

    def test_break_even_formula_direct(self):
        """Direct unit test of _break_even static method."""
        # If API costs $0.01 more per req and CAPEX is $1200 → 120,000 reqs
        be = EconomicAnalyser._break_even(0.02, 0.01)
        assert isinstance(be, int)
        assert be == pytest.approx(120_000, rel=0.01)

    def test_break_even_never_when_api_cheaper(self):
        be = EconomicAnalyser._break_even(0.001, 0.01)
        assert isinstance(be, str)
        assert "Never" in be


class TestEconomicAnalyserHelpers:
    """Tests for internal helper methods."""

    def test_avg_metric_computes_mean(self):
        rows = [{"x": 2.0}, {"x": 4.0}, {"x": 6.0}]
        avg = EconomicAnalyser._avg_metric(rows, "x")
        assert avg == pytest.approx(4.0)

    def test_avg_metric_handles_missing_key(self):
        rows = [{"x": 5.0}, {"y": 3.0}]
        avg = EconomicAnalyser._avg_metric(rows, "x")
        assert avg == pytest.approx(5.0)

    def test_avg_metric_empty_returns_zero(self):
        avg = EconomicAnalyser._avg_metric([], "x")
        assert avg == 0.0

    def test_api_cost_scales_with_token_count(self):
        cost_small = EconomicAnalyser._api_cost(10, 10)
        cost_large = EconomicAnalyser._api_cost(100, 100)
        assert cost_large > cost_small

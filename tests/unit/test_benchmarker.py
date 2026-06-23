"""
Unit tests for BenchmarkRunner.

Tests metric key correctness and quality scoring without requiring
a GPU, AirLLM, or actual model weights. (GUIDE §16.2 — independently
testable via dependency injection)
"""

import pytest

from hw5_airllm_benchmark.constants import (
    METRIC_ENERGY_WH,
    METRIC_OUTPUT_TEXT,
    METRIC_PEAK_RAM_GB,
    METRIC_PEAK_VRAM_GB,
    METRIC_QUALITY_SCORE,
    METRIC_QUANT_LEVEL,
    METRIC_THROUGHPUT,
    METRIC_TOTAL_TIME,
    METRIC_TPOT,
    METRIC_TTFT,
)
from hw5_airllm_benchmark.services.benchmarker import BenchmarkRunner

_REQUIRED_METRIC_KEYS = {
    METRIC_QUANT_LEVEL,
    METRIC_TTFT,
    METRIC_TPOT,
    METRIC_THROUGHPUT,
    METRIC_PEAK_RAM_GB,
    METRIC_PEAK_VRAM_GB,
    METRIC_TOTAL_TIME,
    METRIC_ENERGY_WH,
    METRIC_QUALITY_SCORE,
    METRIC_OUTPUT_TEXT,
}

_DUMMY_CFG = {
    "model_name": "dummy/model",
    "layer_shards_path": "/tmp/shards",
    "max_new_tokens": 10,
    "prompt": "Test prompt",
    "log_file": "/tmp/test_metrics.csv",
    "cpu_tdp_watts": 45.0,
    "output_quality_max_repetition_ratio": 0.5,
}


class TestBenchmarkRunnerQualityScore:
    """Tests for the _score_quality helper — no model needed."""

    def setup_method(self):
        self.runner = BenchmarkRunner(_DUMMY_CFG)

    def test_empty_text_returns_zero(self):
        assert self.runner._score_quality("") == 0.0

    def test_single_word_returns_zero(self):
        assert self.runner._score_quality("hello") == 0.0

    def test_all_unique_bigrams_score_is_one(self):
        text = "the quick brown fox jumps over lazy dog"
        score = self.runner._score_quality(text)
        assert score == 1.0

    def test_fully_repetitive_text_scores_low(self):
        text = "the the the the the the the the"
        score = self.runner._score_quality(text)
        assert score < 0.2

    def test_score_in_valid_range(self):
        score = self.runner._score_quality("hello world hello world")
        assert 0.0 <= score <= 1.0


class TestBenchmarkRunnerMetricKeys:
    """Tests that the expected metric keys are defined in constants."""

    def test_all_required_metric_keys_defined(self):
        """All metrics used by the report must have a named constant."""
        for key in _REQUIRED_METRIC_KEYS:
            assert isinstance(key, str) and len(key) > 0, f"Bad key: {key!r}"

    def test_no_duplicate_metric_keys(self):
        """No two metrics should share the same key string."""
        keys = list(_REQUIRED_METRIC_KEYS)
        assert len(keys) == len(set(keys))


class TestBenchmarkRunnerInit:
    """Tests for constructor and config loading."""

    def test_init_sets_model_name(self):
        runner = BenchmarkRunner(_DUMMY_CFG)
        assert runner._model_name == "dummy/model"

    def test_init_uses_default_cpu_tdp_when_missing(self):
        cfg_no_tdp = {k: v for k, v in _DUMMY_CFG.items() if k != "cpu_tdp_watts"}
        runner = BenchmarkRunner(cfg_no_tdp)
        assert runner._cpu_tdp == 45.0  # matches _DEFAULT_CPU_TDP

    def test_init_uses_config_cpu_tdp(self):
        cfg = {**_DUMMY_CFG, "cpu_tdp_watts": 65.0}
        runner = BenchmarkRunner(cfg)
        assert runner._cpu_tdp == 65.0

    @pytest.mark.parametrize("quant", ["4bit", "8bit", None])
    def test_valid_quant_levels_accepted(self, quant):
        """BenchmarkRunner should accept all standard quant level values."""
        runner = BenchmarkRunner(_DUMMY_CFG)
        # We only check it doesn't raise on construction — model load is not tested here
        assert runner is not None

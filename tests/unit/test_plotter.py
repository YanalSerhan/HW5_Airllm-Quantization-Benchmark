"""
Unit tests for the Plotter service.

Verifies that plot_all() creates the expected PNG files without errors,
using mock metric data (no GPU or model required). (GUIDE §16.2)
"""

import csv
from pathlib import Path

import pytest
import matplotlib
matplotlib.use("Agg")

from hw5_airllm_benchmark.services.plotter import Plotter

_DUMMY_CFG = {"version": "1.00"}

_MOCK_ROWS = [
    {
        "quantization_level": "4bit",
        "ttft_seconds": "3.1",
        "tpot_seconds": "1.8",
        "throughput_tokens_per_sec": "0.55",
        "peak_ram_gb": "4.2",
        "peak_vram_gb": "0.0",
        "total_time_seconds": "85.0",
        "estimated_energy_wh": "1.06",
        "quality_score": "0.70",
    },
    {
        "quantization_level": "8bit",
        "ttft_seconds": "4.5",
        "tpot_seconds": "2.2",
        "throughput_tokens_per_sec": "0.45",
        "peak_ram_gb": "6.0",
        "peak_vram_gb": "0.0",
        "total_time_seconds": "110.0",
        "estimated_energy_wh": "1.38",
        "quality_score": "0.82",
    },
    {
        "quantization_level": "fp16",
        "ttft_seconds": "7.0",
        "tpot_seconds": "3.5",
        "throughput_tokens_per_sec": "0.28",
        "peak_ram_gb": "9.5",
        "peak_vram_gb": "0.0",
        "total_time_seconds": "180.0",
        "estimated_energy_wh": "2.25",
        "quality_score": "0.91",
    },
]


@pytest.fixture
def tmp_figures(tmp_path):
    """Provide a temporary directory for figure output."""
    return str(tmp_path / "figures")


class TestPlotterOutputFiles:
    """Tests that plot_all() produces the expected PNG files."""

    def test_plot_all_creates_three_files(self, tmp_figures):
        plotter = Plotter(_DUMMY_CFG, figures_dir=tmp_figures)
        saved = plotter.plot_all(_MOCK_ROWS)
        assert len(saved) == 4

    def test_latency_comparison_png_exists(self, tmp_figures):
        plotter = Plotter(_DUMMY_CFG, figures_dir=tmp_figures)
        plotter.plot_all(_MOCK_ROWS)
        assert (Path(tmp_figures) / "latency_comparison.png").exists()

    def test_memory_usage_png_exists(self, tmp_figures):
        plotter = Plotter(_DUMMY_CFG, figures_dir=tmp_figures)
        plotter.plot_all(_MOCK_ROWS)
        assert (Path(tmp_figures) / "memory_usage.png").exists()

    def test_roofline_diagram_png_exists(self, tmp_figures):
        plotter = Plotter(_DUMMY_CFG, figures_dir=tmp_figures)
        plotter.plot_all(_MOCK_ROWS)
        assert (Path(tmp_figures) / "roofline_diagram.png").exists()

    def test_empty_rows_returns_empty_list(self, tmp_figures):
        plotter = Plotter(_DUMMY_CFG, figures_dir=tmp_figures)
        result = plotter.plot_all([])
        assert result == []

    def test_figures_dir_is_created(self, tmp_path):
        new_dir = str(tmp_path / "nested" / "figures")
        plotter = Plotter(_DUMMY_CFG, figures_dir=new_dir)
        plotter.plot_all(_MOCK_ROWS)
        assert Path(new_dir).is_dir()


class TestPlotterHelpers:
    """Tests for Plotter internal helper methods."""

    def test_sorted_labels_follows_quant_order(self, tmp_figures):
        plotter = Plotter(_DUMMY_CFG, figures_dir=tmp_figures)
        labels = plotter._sorted_labels(_MOCK_ROWS)
        assert labels == ["4bit", "8bit", "fp16"]

    def test_first_returns_correct_value(self, tmp_figures):
        plotter = Plotter(_DUMMY_CFG, figures_dir=tmp_figures)
        val = plotter._first(_MOCK_ROWS, "4bit", "peak_ram_gb")
        assert val == pytest.approx(4.2)

    def test_first_missing_quant_returns_zero(self, tmp_figures):
        plotter = Plotter(_DUMMY_CFG, figures_dir=tmp_figures)
        val = plotter._first(_MOCK_ROWS, "2bit", "peak_ram_gb")
        assert val == 0.0


class TestPlotterFromCSV:
    """Integration-style test: verify plotter works with data loaded from CSV."""

    def test_plot_from_csv_roundtrip(self, tmp_figures, tmp_path):
        """Write mock rows to CSV, read back, plot — should produce 3 PNGs."""
        csv_path = tmp_path / "metrics.csv"
        fieldnames = list(_MOCK_ROWS[0].keys())
        with open(csv_path, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(_MOCK_ROWS)

        loaded_rows: list[dict] = []
        with open(csv_path, newline="", encoding="utf-8") as fh:
            loaded_rows = list(csv.DictReader(fh))

        plotter = Plotter(_DUMMY_CFG, figures_dir=tmp_figures)
        saved = plotter.plot_all(loaded_rows)
        assert len(saved) == 4

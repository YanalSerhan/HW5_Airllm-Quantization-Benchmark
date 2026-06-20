"""
Main SDK entrypoint for hw5_airllm_benchmark.

The SDK layer is the single point through which all business logic is
accessed. CLI/script layers import from here and never call service
internals directly. (GUIDE §4.1)
"""

from ..services.benchmarker import BenchmarkRunner
from ..services.economic_analysis import EconomicAnalyser
from ..shared.config import load_config

__all__ = ["BenchmarkRunner", "EconomicAnalyser", "load_config"]


class HW5SDK:
    """
    Facade that wires together config, benchmarking, and analysis.

    Why: A single SDK class guarantees that callers never need to know
    which internal service owns which responsibility. (GUIDE §4.1)
    """

    def __init__(self, config_path: str | None = None):
        self._cfg = load_config(config_path)

    @property
    def config(self) -> dict:
        """Return the loaded configuration dict."""
        return self._cfg

    def run_benchmark(self, quant_level: str | None) -> dict:
        """Run a single benchmark pass for the given quantization level."""
        runner = BenchmarkRunner(self._cfg)
        return runner.run(quant_level)

    def run_economic_analysis(self, metrics: list[dict]) -> dict:
        """Compute break-even and cost curves from collected metrics."""
        analyser = EconomicAnalyser(self._cfg)
        return analyser.analyse(metrics)

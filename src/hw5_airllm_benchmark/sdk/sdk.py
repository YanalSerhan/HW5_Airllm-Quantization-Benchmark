"""
Main SDK entrypoint for hw5_airllm_benchmark.

The SDK layer is the single point through which all business logic is
accessed. CLI/script layers import from here and never call service
internals directly. (GUIDE §4.1)
"""

from ..services.benchmarker import BenchmarkRunner
from ..services.economic_analysis import EconomicAnalyser
from ..services.plotter import Plotter
from ..shared.config import load_config

__all__ = ["BenchmarkRunner", "EconomicAnalyser", "Plotter", "load_config"]


class HW5SDK:
    """
    Facade that wires together config, benchmarking, analysis, and plotting.

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

    def run_plots(self, rows: list[dict], figures_dir: str = "figures") -> list:
        """
        Generate all Phase 4 performance comparison figures from metric rows.

        Why: Routes plotting through the SDK facade so experiment scripts
        stay thin and the plotter service can be swapped/extended without
        changing callers. (GUIDE §4.1)
        """
        plotter = Plotter(self._cfg, figures_dir=figures_dir)
        return plotter.plot_all(rows)

    def run_economic_plots(
        self, economic_result: dict, figures_dir: str = "figures"
    ) -> list:
        """
        Generate Phase 5 economic figures: break-even curve + cost-per-quant.

        Why: Same facade pattern as run_plots — callers stay decoupled from
        the underlying plotter/helper module structure. (GUIDE §4.1)
        """
        plotter = Plotter(self._cfg, figures_dir=figures_dir)
        return plotter.plot_economic_analysis(economic_result)

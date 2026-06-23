"""
Experiment 03 — AirLLM Run with Quantization Sweep.

Iterates over all quantization levels defined in config/setup.json
(default: ["4bit", "8bit", null]) and runs the full benchmark pipeline
for each via the HW5SDK facade.

Results are saved to:
  - results/benchmark_metrics.csv   (one row per quantization level)
  - results/airllm_summary.json     (human-readable summary)

Run with: uv run python experiments/03_airllm_run.py
"""

import json
import logging
import sys
from pathlib import Path

from dotenv import load_dotenv

# Allow running from project root without installing the package
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hw5_airllm_benchmark.sdk.sdk import HW5SDK  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

load_dotenv()

_SUMMARY_FILE = Path("results/airllm_summary.json")


def _label(quant_level: str | None) -> str:
    """Human-readable label for a quantization level."""
    return quant_level if quant_level else "fp16"


def main() -> None:
    """Run the full AirLLM + quantization benchmark sweep."""
    sdk = HW5SDK()
    cfg = sdk.config
    quant_levels: list = cfg.get("quantization_levels", ["4bit", "8bit", None])

    logger.info("=== Experiment 03: AirLLM Quantization Sweep ===")
    logger.info("Model   : %s", cfg.get("model_name"))
    logger.info("Levels  : %s", [_label(q) for q in quant_levels])
    logger.info("Prompt  : %s", cfg.get("prompt", "")[:60])

    all_metrics: list[dict] = []
    failed: list[str] = []

    for quant in quant_levels:
        label = _label(quant)
        logger.info("--- Starting run: quant=%s ---", label)
        try:
            metrics = sdk.run_benchmark(quant)
            all_metrics.append(metrics)
            logger.info(
                "  TTFT=%.3fs  TPOT=%.3fs  throughput=%.2f tok/s  RAM=%.2f GB  quality=%.3f",
                metrics.get("ttft_seconds", 0),
                metrics.get("tpot_seconds", 0),
                metrics.get("throughput_tokens_per_sec", 0),
                metrics.get("peak_ram_gb", 0),
                metrics.get("quality_score", 0),
            )
        except Exception as exc:  # noqa: BLE001
            logger.error("Run failed for quant=%s: %s", label, exc, exc_info=True)
            failed.append(label)

    # Write summary JSON
    _SUMMARY_FILE.parent.mkdir(parents=True, exist_ok=True)
    summary = {
        "runs_attempted": len(quant_levels),
        "runs_succeeded": len(all_metrics),
        "runs_failed": failed,
        "metrics": all_metrics,
    }
    with open(_SUMMARY_FILE, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    logger.info("Summary written to %s", _SUMMARY_FILE)

    # Print comparison table to console
    print("\n=== AirLLM Benchmark Results ===")
    header = (
        f"{'Quant':<8} {'TTFT(s)':>9} {'TPOT(s)':>9}"
        f" {'Tok/s':>8} {'RAM(GB)':>8} {'Quality':>8}"
    )
    print(header)
    print("-" * len(header))
    for m in all_metrics:
        print(
            f"{m.get('quantization_level', '?'):<8}"
            f" {m.get('ttft_seconds', 0):>9.3f}"
            f" {m.get('tpot_seconds', 0):>9.3f}"
            f" {m.get('throughput_tokens_per_sec', 0):>8.2f}"
            f" {m.get('peak_ram_gb', 0):>8.3f}"
            f" {m.get('quality_score', 0):>8.4f}"
        )
    if failed:
        print(f"\nFailed runs: {failed}")


if __name__ == "__main__":
    main()

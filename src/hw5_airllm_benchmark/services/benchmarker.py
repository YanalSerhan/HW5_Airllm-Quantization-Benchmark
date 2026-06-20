"""
BenchmarkRunner service — runs AirLLM inference and records metrics.

Building-block contract (GUIDE §16):
  Input:  cfg dict (from config/setup.json), quant_level str | None
  Output: dict of metric keys matching constants.py METRIC_* names
  Setup:  constructor receives cfg; no hidden global state
"""

import csv
import logging
import time
from pathlib import Path

import psutil

from ..constants import (
    KEY_LAYER_SHARDS_PATH,
    KEY_LOG_FILE,
    KEY_MAX_NEW_TOKENS,
    KEY_MODEL_NAME,
    KEY_PROMPT,
    METRIC_ENERGY_WH,
    METRIC_PEAK_RAM_GB,
    METRIC_PEAK_VRAM_GB,
    METRIC_QUANT_LEVEL,
    METRIC_THROUGHPUT,
    METRIC_TOTAL_TIME,
    METRIC_TPOT,
    METRIC_TTFT,
)

logger = logging.getLogger(__name__)

# Rough CPU TDP estimate in watts (no GPU assumed for AirLLM CPU path)
_CPU_TDP_WATTS = 45.0


class BenchmarkRunner:
    """
    Runs AirLLM inference at a given quantization level and logs metrics.

    Why: Encapsulating all measurement logic here means the SDK and
    experiment scripts stay thin; only this class needs to change if
    AirLLM's API changes. (GUIDE §16, Single Responsibility)
    """

    def __init__(self, cfg: dict):
        self._cfg = cfg
        self._model_name: str = cfg[KEY_MODEL_NAME]
        self._max_new_tokens: int = cfg[KEY_MAX_NEW_TOKENS]
        self._prompt: str = cfg[KEY_PROMPT]
        self._shards_path: str = cfg[KEY_LAYER_SHARDS_PATH]
        self._log_file: Path = Path(cfg[KEY_LOG_FILE])
        self._log_file.parent.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self, quant_level: str | None) -> dict:
        """
        Load model, run inference, capture metrics, persist to CSV.

        Returns a metrics dict using the keys defined in constants.py.
        """
        logger.info("Starting benchmark: quant=%s", quant_level)
        model, tokenizer = self._load_model(quant_level)
        metrics = self._measure(model, tokenizer, quant_level)
        self._append_csv(metrics)
        logger.info("Benchmark done: %s", metrics)
        return metrics

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _load_model(self, quant_level: str | None):
        """Initialise AirLLM model at the requested quantization level."""
        # Import here so the module is importable even without GPU/CUDA
        from airllm import AutoModel  # type: ignore[import]

        compression = quant_level  # e.g. "4bit", "8bit", or None
        model = AutoModel.from_pretrained(
            self._model_name,
            compression=compression,
            layer_shards_saving_path=self._shards_path,
        )
        from transformers import AutoTokenizer  # type: ignore[import]

        tokenizer = AutoTokenizer.from_pretrained(self._model_name)
        return model, tokenizer

    def _measure(self, model, tokenizer, quant_level: str | None) -> dict:
        """Run one inference pass and return raw metric values."""
        inputs = tokenizer(self._prompt, return_tensors="pt")
        input_len = inputs["input_ids"].shape[-1]

        ram_before = psutil.virtual_memory().used / 1e9
        t0 = time.perf_counter()

        # --- Prefill (first token) ---
        out = model.generate(
            **inputs,
            max_new_tokens=1,
            use_cache=False,
        )
        t_first = time.perf_counter()
        ttft = t_first - t0

        # --- Decode (remaining tokens) ---
        out = model.generate(
            **inputs,
            max_new_tokens=self._max_new_tokens,
            use_cache=False,
        )
        t_end = time.perf_counter()

        output_len = out.shape[-1] - input_len
        total_time = t_end - t0
        tpot = (t_end - t_first) / max(output_len - 1, 1)
        throughput = output_len / total_time
        ram_after = psutil.virtual_memory().used / 1e9
        peak_ram = max(ram_after - ram_before, 0.0)
        energy_wh = (_CPU_TDP_WATTS * total_time) / 3600.0

        return {
            METRIC_QUANT_LEVEL: quant_level or "fp16",
            METRIC_TTFT: round(ttft, 4),
            METRIC_TPOT: round(tpot, 4),
            METRIC_THROUGHPUT: round(throughput, 4),
            METRIC_PEAK_RAM_GB: round(peak_ram, 3),
            METRIC_PEAK_VRAM_GB: 0.0,  # populated by GPU-aware run
            METRIC_TOTAL_TIME: round(total_time, 4),
            METRIC_ENERGY_WH: round(energy_wh, 6),
        }

    def _append_csv(self, row: dict) -> None:
        """Append one result row to the CSV log file."""
        write_header = not self._log_file.exists()
        with open(self._log_file, "a", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(row.keys()))
            if write_header:
                writer.writeheader()
            writer.writerow(row)

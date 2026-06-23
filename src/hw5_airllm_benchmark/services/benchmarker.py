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

logger = logging.getLogger(__name__)

_KEY_CPU_TDP = "cpu_tdp_watts"
_KEY_MAX_REP_RATIO = "output_quality_max_repetition_ratio"
_DEFAULT_CPU_TDP = 45.0
_DEFAULT_MAX_REP_RATIO = 0.5


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
        Path(self._shards_path).mkdir(parents=True, exist_ok=True)
        self._log_file: Path = Path(cfg[KEY_LOG_FILE])
        self._log_file.parent.mkdir(parents=True, exist_ok=True)
        self._cpu_tdp: float = cfg.get(_KEY_CPU_TDP, _DEFAULT_CPU_TDP)
        self._max_rep_ratio: float = cfg.get(_KEY_MAX_REP_RATIO, _DEFAULT_MAX_REP_RATIO)

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

        import threading
        import os
        pid = os.getpid()
        ram_before = psutil.Process(pid).memory_info().rss / 1e9
        peak_ram_abs = [ram_before]
        stop_thread = False

        def _monitor_ram():
            proc = psutil.Process(pid)
            while not stop_thread:
                try:
                    current_ram = proc.memory_info().rss / 1e9
                    if current_ram > peak_ram_abs[0]:
                        peak_ram_abs[0] = current_ram
                except Exception:
                    pass
                time.sleep(0.05)

        monitor_thread = threading.Thread(target=_monitor_ram)
        monitor_thread.start()

        self._reset_vram_stats()
        t0 = time.perf_counter()

        # --- Prefill: generate exactly 1 token to capture TTFT ---
        model.generate(**inputs, max_new_tokens=1, use_cache=False)
        t_first = time.perf_counter()
        ttft = t_first - t0

        # --- Decode: generate remaining tokens ---
        out = model.generate(
            **inputs,
            max_new_tokens=self._max_new_tokens,
            use_cache=False,
        )
        t_end = time.perf_counter()

        stop_thread = True
        monitor_thread.join(timeout=1.0)

        output_len = out.shape[-1] - input_len
        total_time = t_end - t0
        tpot = (t_end - t_first) / max(output_len - 1, 1)
        throughput = output_len / total_time
        peak_ram = max(peak_ram_abs[0] - ram_before, 0.0)
        peak_vram = self._get_peak_vram_gb()
        energy_wh = (self._cpu_tdp * total_time) / 3600.0

        output_text = tokenizer.decode(out[0][input_len:], skip_special_tokens=True)
        quality = self._score_quality(output_text)

        return {
            METRIC_QUANT_LEVEL: quant_level or "fp16",
            METRIC_TTFT: round(ttft, 4),
            METRIC_TPOT: round(tpot, 4),
            METRIC_THROUGHPUT: round(throughput, 4),
            METRIC_PEAK_RAM_GB: round(peak_ram, 3),
            METRIC_PEAK_VRAM_GB: round(peak_vram, 3),
            METRIC_TOTAL_TIME: round(total_time, 4),
            METRIC_ENERGY_WH: round(energy_wh, 6),
            METRIC_QUALITY_SCORE: quality,
            METRIC_OUTPUT_TEXT: output_text[:500],  # truncate for CSV safety
        }

    def _append_csv(self, row: dict) -> None:
        """Append one result row to the CSV log file."""
        # Exclude output text from CSV (stored separately if needed)
        csv_row = {k: v for k, v in row.items() if k != METRIC_OUTPUT_TEXT}
        write_header = not self._log_file.exists()
        with open(self._log_file, "a", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(csv_row.keys()))
            if write_header:
                writer.writeheader()
            writer.writerow(csv_row)

    @staticmethod
    def _get_vram_gb() -> float:
        """Return currently allocated VRAM in GB, or 0.0 if no CUDA device."""
        try:
            import torch  # type: ignore[import]

            if torch.cuda.is_available():
                return torch.cuda.memory_allocated() / 1e9
        except ImportError:
            pass
        return 0.0

    @staticmethod
    def _reset_vram_stats() -> None:
        try:
            import torch  # type: ignore[import]
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
        except ImportError:
            pass

    @staticmethod
    def _get_peak_vram_gb() -> float:
        try:
            import torch  # type: ignore[import]
            if torch.cuda.is_available():
                return torch.cuda.max_memory_allocated() / 1e9
        except ImportError:
            pass
        return 0.0

    def _score_quality(self, text: str) -> float:
        """
        Simple quality score: fraction of unique bigrams over total bigrams.

        A score near 1.0 means diverse, coherent output; near 0.0 means
        heavy repetition (degenerate quantization artefact). Range: [0, 1].
        """
        words = text.lower().split()
        if len(words) < 2:
            return 0.0
        bigrams = list(zip(words, words[1:]))
        ratio = len(set(bigrams)) / len(bigrams)
        return round(ratio, 4)

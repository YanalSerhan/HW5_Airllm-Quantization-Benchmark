"""
BenchmarkRunner service — runs AirLLM inference and records metrics.

Building-block contract (GUIDE §16):
  Input:  cfg dict (from config/setup.json), quant_level str | None
  Output: dict of metric keys matching constants.py METRIC_* names
  Setup:  constructor receives cfg; no hidden global state

Measurement design (two-pass, post-bug-fix):
  Pass 1 (1 token)  → TTFT: true prefill-stage timing.
  Pass 2 (N tokens) → Decode timing, peak RAM, output text.
  total_time = ttft + decode_time  (both passes summed).
  TPOT = (decode_time - ttft) / (output_len - 1)  [decode only, net of prefill].
  throughput = output_len / decode_time.
  peak_ram = absolute peak RSS during Pass 2 (not a cross-run delta).
  peak_vram = 0.0 (hardware_spec confirms CPU-only, no CUDA device).

RAM monitoring delegated to _ram_monitor.py. (GUIDE §3.2, §16.2)
"""

import csv
import logging
import time
from pathlib import Path

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
from ._ram_monitor import make_ram_monitor

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
        Load model, run two-pass inference, capture metrics, persist to CSV.

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
        from airllm import AutoModel  # type: ignore[import]

        model = AutoModel.from_pretrained(
            self._model_name,
            compression=quant_level,
            layer_shards_saving_path=self._shards_path,
        )
        from transformers import AutoTokenizer  # type: ignore[import]

        tokenizer = AutoTokenizer.from_pretrained(self._model_name)
        return model, tokenizer

    def _measure(self, model, tokenizer, quant_level: str | None) -> dict:
        """
        Two-pass inference timing.

        Pass 1 (1 token) → TTFT. Pass 2 (N tokens) → decode metrics.
        See module docstring for the full measurement rationale.
        """
        inputs = tokenizer(self._prompt, return_tensors="pt")
        input_len = inputs["input_ids"].shape[-1]

        # Pass 1: TTFT (prefill-only)
        t0 = time.perf_counter()
        model.generate(**inputs, max_new_tokens=1, use_cache=False)
        ttft = time.perf_counter() - t0

        # Pass 2: Full decode with RAM monitoring
        ram_samples: list[float] = []
        monitor, stop_flag = make_ram_monitor(ram_samples)
        monitor.start()

        t_decode_start = time.perf_counter()
        out = model.generate(
            **inputs, max_new_tokens=self._max_new_tokens, use_cache=False
        )
        decode_time = time.perf_counter() - t_decode_start

        stop_flag[0] = True
        monitor.join(timeout=1.0)

        output_len = out.shape[-1] - input_len
        tpot = max(decode_time - ttft, 0.0) / max(output_len - 1, 1)
        throughput = output_len / decode_time
        peak_ram = max(ram_samples) if ram_samples else 0.0
        total_time = ttft + decode_time
        energy_wh = (self._cpu_tdp * total_time) / 3600.0

        output_text = tokenizer.decode(out[0][input_len:], skip_special_tokens=True)
        return {
            METRIC_QUANT_LEVEL: quant_level or "fp16",
            METRIC_TTFT: round(ttft, 4),
            METRIC_TPOT: round(tpot, 4),
            METRIC_THROUGHPUT: round(throughput, 4),
            METRIC_PEAK_RAM_GB: round(peak_ram, 3),
            METRIC_PEAK_VRAM_GB: 0.0,           # CPU-only machine, no CUDA
            METRIC_TOTAL_TIME: round(total_time, 4),
            METRIC_ENERGY_WH: round(energy_wh, 6),
            METRIC_QUALITY_SCORE: self._score_quality(output_text),
            METRIC_OUTPUT_TEXT: output_text[:500],
        }

    def _append_csv(self, row: dict) -> None:
        """Append one result row to the CSV log file."""
        csv_row = {k: v for k, v in row.items() if k != METRIC_OUTPUT_TEXT}
        write_header = not self._log_file.exists()
        with open(self._log_file, "a", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(csv_row.keys()))
            if write_header:
                writer.writeheader()
            writer.writerow(csv_row)

    def _score_quality(self, text: str) -> float:
        """
        Bigram diversity score: unique bigrams / total bigrams.

        Near 1.0 = diverse/coherent output. Near 0.0 = degenerate repetition.
        """
        words = text.lower().split()
        if len(words) < 2:
            return 0.0
        bigrams = list(zip(words, words[1:]))
        return round(len(set(bigrams)) / len(bigrams), 4)

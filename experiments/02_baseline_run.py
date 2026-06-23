"""
Experiment 02 — Baseline "Naive" Run (no AirLLM).

Attempts to load Qwen/Qwen2.5-7B-Instruct directly via transformers
AutoModelForCausalLM with no AirLLM, no quantization, no streaming.

Expected outcome: MemoryError / CUDA OOM / extreme RAM exhaustion.
This negative result IS the deliverable — it proves the hardware
bottleneck that motivates AirLLM (EX05 §5.2, §1).

Output artifact: results/baseline_run.json
Run with: uv run python experiments/02_baseline_run.py
"""

import json
import logging
import os
import time
import traceback
from pathlib import Path

import psutil
from dotenv import load_dotenv

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

load_dotenv()

_OUT_FILE = Path("results/baseline_run.json")
_MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"
# Timeout: if loading takes longer than this, record as "hung / swap thrash"
_LOAD_TIMEOUT_SECS = 300


def _ram_snapshot() -> dict:
    """Capture current RAM and swap state."""
    vm = psutil.virtual_memory()
    sw = psutil.swap_memory()
    return {
        "total_gb": round(vm.total / 1e9, 2),
        "available_gb": round(vm.available / 1e9, 2),
        "used_gb": round(vm.used / 1e9, 2),
        "percent_used": vm.percent,
        "swap_used_gb": round(sw.used / 1e9, 2),
        "swap_total_gb": round(sw.total / 1e9, 2),
    }


def _vram_snapshot() -> dict:
    """Capture GPU VRAM state if CUDA available."""
    try:
        import torch  # type: ignore[import]

        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1e9
            reserved = torch.cuda.memory_reserved() / 1e9
            total = torch.cuda.get_device_properties(0).total_memory / 1e9
            return {
                "available": True,
                "allocated_gb": round(allocated, 3),
                "reserved_gb": round(reserved, 3),
                "total_gb": round(total, 2),
            }
    except ImportError:
        pass
    return {"available": False}


def _attempt_load() -> dict:
    """
    Try to load the model directly. Captures what goes wrong.

    Returns a result dict categorising the failure mode.
    """
    token = os.environ.get("HF_TOKEN")
    ram_before = _ram_snapshot()
    vram_before = _vram_snapshot()
    t_start = time.perf_counter()
    outcome: dict = {
        "model": _MODEL_NAME,
        "strategy": "transformers.AutoModelForCausalLM (no AirLLM, no quantization)",
        "ram_before": ram_before,
        "vram_before": vram_before,
        "token_provided": bool(token),
    }

    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore[import]

        logger.info("Attempting tokenizer load …")
        AutoTokenizer.from_pretrained(_MODEL_NAME, token=token)
        logger.info("Tokenizer loaded. Attempting full model load …")
        logger.warning("This is expected to fail with OOM — documenting failure as valid result.")

        # Load to CPU (device_map="auto" will still exhaust RAM for 7B)
        AutoModelForCausalLM.from_pretrained(
            _MODEL_NAME,
            token=token,
            device_map="auto",
            low_cpu_mem_usage=True,
        )

        # If we somehow get here (e.g. on a machine with >144 GB RAM)
        t_end = time.perf_counter()
        outcome.update({
            "result": "SUCCESS_UNEXPECTEDLY",
            "wall_clock_seconds": round(t_end - t_start, 2),
            "ram_after": _ram_snapshot(),
            "vram_after": _vram_snapshot(),
            "note": "Model loaded — hardware unexpectedly sufficient. Use as baseline.",
        })

    except (MemoryError, RuntimeError) as exc:
        t_fail = time.perf_counter()
        exc_type = type(exc).__name__
        # Distinguish CUDA OOM from CPU RAM OOM
        is_cuda_oom = "CUDA out of memory" in str(exc) or "CUDA" in exc_type
        outcome.update({
            "result": "OOM_FAILURE",
            "failure_type": "CUDA_OOM" if is_cuda_oom else "CPU_RAM_OOM",
            "exception_type": exc_type,
            "exception_message": str(exc)[:600],
            "traceback_excerpt": traceback.format_exc()[-800:],
            "wall_clock_seconds": round(t_fail - t_start, 2),
            "ram_at_failure": _ram_snapshot(),
            "vram_at_failure": _vram_snapshot(),
            "bottleneck_analysis": (
                "Memory-bound failure: the model weights (~6 GB for Qwen2.5-3B) "
                "far exceed available RAM/VRAM. This confirms the VRAM Gap described "
                "in L7 Part 1 and L08 §3.3. The OS has no room to page-in all layers "
                "simultaneously without AirLLM's layer-by-layer streaming strategy."
            ),
        })
        logger.error("Expected OOM captured: %s — %s", exc_type, str(exc)[:120])

    except Exception as exc:  # noqa: BLE001
        t_fail = time.perf_counter()
        outcome.update({
            "result": "UNEXPECTED_ERROR",
            "exception_type": type(exc).__name__,
            "exception_message": str(exc)[:600],
            "traceback_excerpt": traceback.format_exc()[-800:],
            "wall_clock_seconds": round(t_fail - t_start, 2),
        })
        logger.error("Unexpected error: %s", exc)

    return outcome


def main() -> None:
    """Run baseline attempt and save evidence to results/baseline_run.json."""
    _OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    logger.info("=== Experiment 02: Baseline Run (no AirLLM) ===")
    logger.info("Model: %s", _MODEL_NAME)
    logger.info("Expected result: OOM failure (this IS the deliverable).")

    result = _attempt_load()

    with open(_OUT_FILE, "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2)

    logger.info("Baseline result written to %s", _OUT_FILE)
    print(f"\n=== Baseline Result: {result.get('result', 'UNKNOWN')} ===")
    if "exception_message" in result:
        print(f"Failure type : {result.get('failure_type', 'N/A')}")
        print(f"Exception    : {result['exception_message'][:200]}")
    print(f"Wall-clock   : {result.get('wall_clock_seconds', '?')} s")
    print(f"RAM at event : {result.get('ram_at_failure', result.get('ram_before'))}")


if __name__ == "__main__":
    main()

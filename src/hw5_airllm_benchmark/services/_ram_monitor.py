"""
RAM monitoring helper for BenchmarkRunner.

Extracted from benchmarker.py to keep that module under 150 lines. (GUIDE §3.2)

Building-block contract (GUIDE §16):
  Input:  process ID to monitor
  Output: thread that records RSS samples into a shared list
  Setup:  pure functions; no hidden global state
"""

import os
import threading
import time

import psutil


def make_ram_monitor(sample_list: list[float]) -> tuple[threading.Thread, list[bool]]:
    """
    Create a background thread that appends RSS samples to sample_list.

    Returns (thread, stop_flag) — set stop_flag[0] = True to stop the thread.
    The thread is daemon so it does not block process exit.

    Why: Isolating the RAM monitor into a helper keeps _measure() in
    BenchmarkRunner focused on the timing logic. (GUIDE §16.2)
    """
    pid = os.getpid()
    stop_flag: list[bool] = [False]

    def _poll() -> None:
        proc = psutil.Process(pid)
        while not stop_flag[0]:
            try:
                sample_list.append(proc.memory_info().rss / 1e9)
            except Exception:  # noqa: BLE001
                pass
            time.sleep(0.05)

    thread = threading.Thread(target=_poll, daemon=True)
    return thread, stop_flag

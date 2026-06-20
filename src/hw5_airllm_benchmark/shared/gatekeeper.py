"""
Centralised API Gatekeeper for external HTTP calls. (GUIDE §5)

All external API calls (e.g. OpenAI pricing lookups) must be routed
through this class so that rate limits, retries, logging, and queuing
are enforced in a single place rather than scattered across callers.
"""

import json
import logging
import time
from pathlib import Path
from queue import Queue
from threading import Lock

import httpx

logger = logging.getLogger(__name__)


def _load_rate_limits() -> dict:
    """Load rate limits from config/rate_limits.json."""
    root = Path(__file__).resolve().parents[3]
    path = root / "config" / "rate_limits.json"
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


class APIGatekeeper:
    """
    Thread-safe API gatekeeper enforcing rate limits and retries.

    Why: GUIDE §5 requires every external API call to pass through a
    single gatekeeper — this prevents accidental quota exhaustion and
    gives a single place to audit all outbound requests.
    """

    def __init__(self, provider: str = "openai"):
        limits = _load_rate_limits().get(provider, {})
        self._rpm: int = limits.get("requests_per_minute", 60)
        self._max_retries: int = limits.get("max_retries", 3)
        self._retry_delay: float = limits.get("retry_delay_seconds", 5)
        self._lock = Lock()
        self._queue: Queue = Queue()
        self._last_request_time: float = 0.0

    def get(self, url: str, **kwargs) -> httpx.Response:
        """
        Perform a rate-limited GET request with automatic retries.

        Why: Centralises retry/backoff logic so individual callers
        never need to implement it themselves. (GUIDE §5)
        """
        min_interval = 60.0 / self._rpm
        for attempt in range(1, self._max_retries + 1):
            with self._lock:
                elapsed = time.monotonic() - self._last_request_time
                if elapsed < min_interval:
                    time.sleep(min_interval - elapsed)
                self._last_request_time = time.monotonic()

            try:
                resp = httpx.get(url, **kwargs)
                resp.raise_for_status()
                logger.info("GET %s -> %s", url, resp.status_code)
                return resp
            except httpx.HTTPError as exc:
                logger.warning(
                    "Attempt %d/%d failed for %s: %s",
                    attempt, self._max_retries, url, exc,
                )
                if attempt < self._max_retries:
                    time.sleep(self._retry_delay)
                else:
                    raise

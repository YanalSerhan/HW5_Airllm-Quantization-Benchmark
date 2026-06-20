"""
Centralized configuration loader for hw5_airllm_benchmark. (GUIDE §7.2)

Reads config/setup.json and validates that the config version is
compatible with the current package version. Provides a single
source of truth for all run-time settings.
"""

import json
from pathlib import Path

from ..constants import CONFIG_VERSION


def load_config(config_path: str | None = None) -> dict:
    """
    Load the setup.json config file and validate its version.

    Why: Centralising config loading avoids magic constants scattered
    across scripts and ensures the config version stays in sync with
    the package version. (GUIDE §8.1 Table 2)
    """
    if config_path is None:
        # Default: resolve relative to this file's package root
        root = Path(__file__).resolve().parents[3]
        config_path = root / "config" / "setup.json"

    with open(config_path, encoding="utf-8") as fh:
        cfg = json.load(fh)

    config_ver = cfg.get("version", "MISSING")
    if config_ver != CONFIG_VERSION:
        raise RuntimeError(
            f"Config version mismatch: config={config_ver}, "
            f"expected={CONFIG_VERSION}. Update config/setup.json."
        )

    return cfg

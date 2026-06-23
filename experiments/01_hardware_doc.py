"""
Experiment 01 — Hardware Documentation.

Collects CPU, RAM, GPU/VRAM, and storage specs automatically and writes
a structured JSON artifact to results/hardware_spec.json.

Output artifact: results/hardware_spec.json
Run with: uv run python experiments/01_hardware_doc.py
"""

import json
import logging
import platform
import shutil
from pathlib import Path

import psutil

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

_OUT_FILE = Path("results/hardware_spec.json")


def _collect_cpu() -> dict:
    """Collect CPU model, core counts, and frequency info."""
    freq = psutil.cpu_freq()
    return {
        "model": platform.processor() or "Unknown (see platform.processor)",
        "physical_cores": psutil.cpu_count(logical=False),
        "logical_cores": psutil.cpu_count(logical=True),
        "max_freq_mhz": round(freq.max, 1) if freq else None,
        "current_freq_mhz": round(freq.current, 1) if freq else None,
        "python_machine": platform.machine(),
        "os": platform.platform(),
    }


def _collect_ram() -> dict:
    """Collect total and available RAM."""
    vm = psutil.virtual_memory()
    return {
        "total_gb": round(vm.total / 1e9, 2),
        "available_gb": round(vm.available / 1e9, 2),
        "used_gb": round(vm.used / 1e9, 2),
    }


def _collect_gpu() -> dict:
    """Collect GPU model and VRAM via torch.cuda (if available)."""
    try:
        import torch  # type: ignore[import]

        if not torch.cuda.is_available():
            return {"available": False, "reason": "torch.cuda.is_available() returned False"}
        device_count = torch.cuda.device_count()
        gpus = []
        for i in range(device_count):
            props = torch.cuda.get_device_properties(i)
            gpus.append({
                "index": i,
                "name": props.name,
                "vram_gb": round(props.total_memory / 1e9, 2),
                "compute_capability": f"{props.major}.{props.minor}",
            })
        return {"available": True, "devices": gpus}
    except ImportError:
        return {"available": False, "reason": "torch not installed"}


def _collect_storage(path: str = ".") -> dict:
    """Collect free disk space for the project drive."""
    usage = shutil.disk_usage(path)
    return {
        "path": str(Path(path).resolve()),
        "total_gb": round(usage.total / 1e9, 2),
        "used_gb": round(usage.used / 1e9, 2),
        "free_gb": round(usage.free / 1e9, 2),
    }


def _collect_hardware() -> dict:
    """Aggregate all hardware sections into one spec dict."""
    return {
        "cpu": _collect_cpu(),
        "ram": _collect_ram(),
        "gpu": _collect_gpu(),
        "storage_project_drive": _collect_storage("."),
        "storage_shards_drive": _collect_storage("D:/") if Path("D:/").exists() else None,
    }


def main() -> None:
    """Collect hardware spec and write results/hardware_spec.json."""
    _OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    spec = _collect_hardware()

    with open(_OUT_FILE, "w", encoding="utf-8") as fh:
        json.dump(spec, fh, indent=2)

    logger.info("Hardware spec written to %s", _OUT_FILE)

    # Human-readable summary
    cpu = spec["cpu"]
    ram = spec["ram"]
    gpu = spec["gpu"]
    stor = spec["storage_project_drive"]
    print("\n===  Hardware Summary  ===")
    print(f"  CPU  : {cpu['model']}")
    print(f"         {cpu['physical_cores']} physical / {cpu['logical_cores']} logical cores")
    print(f"  RAM  : {ram['total_gb']} GB total, {ram['available_gb']} GB free")
    if gpu["available"]:
        for g in gpu["devices"]:
            print(f"  GPU  : {g['name']}  ({g['vram_gb']} GB VRAM)")
    else:
        print(f"  GPU  : None ({gpu.get('reason', '')})")
    print(f"  Disk : {stor['free_gb']} GB free of {stor['total_gb']} GB ({stor['path']})")
    print("==========================\n")


if __name__ == "__main__":
    main()

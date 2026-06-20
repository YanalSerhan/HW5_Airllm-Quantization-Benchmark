"""
Package-level constants for hw5_airllm_benchmark.

All magic values (model names, metric keys, config keys) live here
so no hard-coded strings scatter across the codebase. (GUIDE §7.2)
"""

CONFIG_VERSION = "1.00"

# Config keys
KEY_MODEL_NAME = "model_name"
KEY_LAYER_SHARDS_PATH = "layer_shards_path"
KEY_MAX_NEW_TOKENS = "max_new_tokens"
KEY_PROMPT = "prompt"
KEY_QUANTIZATION_LEVELS = "quantization_levels"
KEY_LOG_FILE = "log_file"

# Metric column names for CSV output
METRIC_QUANT_LEVEL = "quantization_level"
METRIC_TTFT = "ttft_seconds"
METRIC_TPOT = "tpot_seconds"
METRIC_THROUGHPUT = "throughput_tokens_per_sec"
METRIC_PEAK_RAM_GB = "peak_ram_gb"
METRIC_PEAK_VRAM_GB = "peak_vram_gb"
METRIC_TOTAL_TIME = "total_time_seconds"
METRIC_ENERGY_WH = "estimated_energy_wh"
METRIC_QUALITY_SCORE = "quality_score"

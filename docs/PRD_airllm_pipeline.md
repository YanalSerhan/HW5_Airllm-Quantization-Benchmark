# PRD — AirLLM Pipeline Component

## Component Purpose
The AirLLM pipeline is the core inference engine for this project. It
wraps the `airllm` library to perform layer-by-layer streaming of the
72B model from disk into RAM/VRAM, enabling inference on hardware that
could not possibly load the full model at once.

## Building-Block Contract (GUIDE §16)

### Input Data
| Parameter | Type | Valid Range | Notes |
|---|---|---|---|
| `model_name` | `str` | Valid HuggingFace repo ID | From `config/setup.json` |
| `quant_level` | `str \| None` | `"4bit"`, `"8bit"`, `None` | `None` = FP16 baseline |
| `max_new_tokens` | `int` | 1–512 | From config; keep low for speed |
| `prompt` | `str` | Non-empty string | Fixed reference prompt |
| `layer_shards_path` | `str` | Path to fast storage | Avoid filling OS drive |

### Output Data
| Key | Type | Notes |
|---|---|---|
| `ttft_seconds` | `float` | Time to first token (Prefill stage) |
| `tpot_seconds` | `float` | Time per output token (Decode stage) |
| `throughput_tokens_per_sec` | `float` | Total tokens / total time |
| `peak_ram_gb` | `float` | Delta in RSS before/after |
| `peak_vram_gb` | `float` | GPU VRAM delta (0 if CPU-only) |
| `total_time_seconds` | `float` | End-to-end wall clock |
| `estimated_energy_wh` | `float` | TDP × time / 3600 |

### Setup Data
Constructor accepts the full `cfg` dict from `config/setup.json`.

## Architecture Decision Records

### ADR-1: Use AirLLM over full transformers loading
**Decision:** Use `airllm.AutoModel` instead of `transformers.AutoModelForCausalLM`.
**Rationale:** Native loading of a 72B model requires >144GB of contiguous RAM —
impossible on this machine. AirLLM streams layer shards from disk, acting as an
OS-level memory page substitute for model weights.
**Trade-off:** Significantly slower inference (disk I/O bound), but the only
feasible path to running the model at all.

### ADR-2: Separate TTFT from TPOT measurement
**Decision:** Run two `model.generate()` calls — one for `max_new_tokens=1`
(to isolate Prefill / TTFT) and one for the full token budget (to capture Decode / TPOT).
**Rationale:** TTFT reflects compute-bound Prefill; TPOT reflects memory-bandwidth-bound
Decode. Conflating them into a single elapsed time loses the diagnostic insight.

### ADR-3: Store shards on a non-OS drive
**Decision:** Default `layer_shards_path` points to `D:/airllm_shards`.
**Rationale:** AirLLM writes tens of gigabytes of shard files. Storing on `C:` risks
filling the OS drive and causing a system crash. (EX05 §6.1 "Do")

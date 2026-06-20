# PRD — Benchmarking Component

## Component Purpose
The benchmarking component is a self-contained measurement harness. It
drives the AirLLM pipeline at each quantization level, records all six
required metrics to a CSV file, and exposes results to the analysis layer.

## Building-Block Contract (GUIDE §16)

### Input Data
| Parameter | Type | Valid Range | Notes |
|---|---|---|---|
| `cfg` | `dict` | From `load_config()` | Must include all required keys |
| `quant_level` | `str \| None` | `"4bit"`, `"8bit"`, `None` | Passed through to AirLLM |

### Output Data
CSV appended to `results/benchmark_metrics.csv` and a `dict` returned
to the caller. Columns match the `METRIC_*` constants in `constants.py`.

### Setup Data
`BenchmarkRunner(cfg)` — receives the config dict; no class-level global
state; fully dependency-injectable for testing.

## Planned Quantization Sweep
| Run | quant_level | Expected outcome |
|---|---|---|
| 0 | N/A (baseline) | OOM / freeze — logged as negative result |
| 1 | `None` (FP16 via AirLLM) | Slow but functional |
| 2 | `"8bit"` | Faster, slight quality loss |
| 3 | `"4bit"` | Fastest, more quality degradation |

## Metric Definitions
| Metric | Definition | Stage |
|---|---|---|
| TTFT | Wall time from prompt submission to first token emitted | Prefill |
| TPOT | Average time between consecutive output tokens | Decode |
| Throughput | Total output tokens / total wall time | Both |
| Peak RAM | Max RSS increase during inference (GB) | Both |
| Peak VRAM | Max GPU VRAM increase (GB); 0 if CPU-only | Both |
| Total time | End-to-end wall clock including model load overhead | Both |
| Energy (Wh) | CPU TDP estimate × total time / 3600 | Both |

## Architecture Decision Records

### ADR-B1: CSV as primary results format
**Decision:** Log results to a CSV file instead of a database.
**Rationale:** Simple, portable, and readable by pandas/Excel for analysis.
CSV fits on any OS with no dependencies and can be committed to the repo
as a small, human-readable artifact.

### ADR-B2: Restart model per quantization level
**Decision:** Load a fresh model object for each quantization sweep instead
of switching quantization in-place.
**Rationale:** AirLLM does not support hot-swapping compression levels on
an already-loaded model. Re-instantiation is safer and ensures clean
memory baselines between runs.

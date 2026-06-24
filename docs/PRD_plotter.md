# PRD — Plotter Component

## Component Purpose
The Plotter component is responsible for transforming raw numerical metrics into high-quality, readable visualizations. It decouples the complex `matplotlib` logic from the experimental scripts and ensures that all charts are consistently styled.

## Building-Block Contract (GUIDE §16)

### Input Data
| Parameter | Type | Valid Range | Notes |
|---|---|---|---|
| `rows` | `list[dict]` | N/A | List of metric rows from CSV |
| `economic_result` | `dict` | N/A | Output from EconomicAnalyser |

### Output Data
Generates and writes `.png` files directly to the `figures/` directory.
Returns a `list[Path]` pointing to the generated files.

### Setup Data
`Plotter(cfg, figures_dir="figures")` — initialized with the root config and an output directory.

## Visualizations Generated
1. **Latency Comparison:** Grouped bar chart comparing TTFT and TPOT.
2. **Throughput Comparison:** Bar chart for tokens/second.
3. **Memory Usage:** Bar chart of peak RAM.
4. **Roofline Diagram:** Scatter plot of operational intensity vs throughput against theoretical limits.
5. **Economic Break-Even:** Line chart mapping cumulative cost over request volume.
6. **Pareto Frontier:** Scatter plot mapping throughput vs. output quality.

## Architecture Decision Records

### ADR-P1: Separation of Sub-Plotters
**Decision:** The main `Plotter` class delegates specific rendering to `_perf_plotter.py` and `_economic_plotter.py`.
**Rationale:** To comply strictly with the < 150 lines per file rule (GUIDE §3.2). Matplotlib code is notoriously verbose, and keeping it all in one file would violate the limit.

### ADR-P2: Use of Agg Backend
**Decision:** Enforce the use of the `Agg` backend for matplotlib during automated generation.
**Rationale:** Generating plots on headless servers or within CI/CD pipelines (e.g., during testing) causes a `TclError` if an interactive backend is used. `Agg` writes directly to files without requiring a display server.

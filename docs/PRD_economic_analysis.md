# PRD — Economic Analysis Component

## Component Purpose
The Economic Analysis component calculates the real-world financial feasibility of running LLM inference on local hardware (On-Premise) versus using a commercial API (e.g., OpenAI). It generates the data required for the "Break-Even" analysis by computing CAPEX, OPEX, and token-based pricing.

## Building-Block Contract (GUIDE §16)

### Input Data
| Parameter | Type | Valid Range | Notes |
|---|---|---|---|
| `throughput_tokens_per_sec` | `float` | > 0 | Measured from the benchmarking phase |
| `hardware_cost_usd` | `float` | > 0 | Estimated cost of local machine |
| `electricity_cost_kwh` | `float` | > 0 | Local energy rate |
| `api_input_cost` | `float` | > 0 | API cost per 1k input tokens |
| `api_output_cost` | `float` | > 0 | API cost per 1k output tokens |

### Output Data
| Key | Type | Notes |
|---|---|---|
| `cost_per_request_api` | `float` | Total cost for 1 API call |
| `cost_per_request_local_opex` | `float` | Electricity cost for 1 local generation |
| `breakeven_point_requests` | `int` | Number of requests where API cost = Local Cost |
| `cumulative_cost_curve` | `list[dict]` | Data points for plotting the break-even chart |

### Setup Data
`EconomicAnalyser(cfg)` — instantiated with the global configuration containing hardware assumptions and rate limits.

## Architecture Decision Records

### ADR-E1: Focus on CAPEX amortization over 4 years
**Decision:** Hardware costs will be amortized on a straight-line basis over 4 years when plotting the cost curves.
**Rationale:** Standard accounting practice for IT hardware. Treating the full hardware cost as "Day 1 OPEX" distorts the break-even curve.

### ADR-E2: Ignore cooling and maintenance OPEX
**Decision:** For consumer-grade local hardware, electricity is the sole OPEX variable.
**Rationale:** Datacenter-level cooling and maintenance (e.g., dedicated HVAC) do not apply to a consumer running a 3B model on a home laptop/desktop.

### ADR-E3: Assume 100% hardware utilization for calculation
**Decision:** Electricity usage is calculated purely based on active inference time (TDP during generation), ignoring idle power draw.
**Rationale:** This creates a best-case scenario for local execution, providing a strong lower bound for the local cost. If API still wins, the conclusion is robust.

# Technical Plan & Experimental Design (PLAN)

## 1. Experimental Design Overview
The project follows a highly structured, empirical approach to prove the necessity and effectiveness of virtual memory techniques in LLM inference.

1. **Phase 1: Baseline Direct Run**
   - Attempt to load the `Qwen2.5-3B-Instruct` model natively using Hugging Face `transformers.AutoModelForCausalLM`.
   - **Expectation:** Immediate OOM exception or system freeze due to extreme swap thrashing on limited RAM.
   - **Action:** Catch the OS-level exception, log the crash state (peak memory), and save as a negative baseline result.

2. **Phase 2: Quantization Sweep via AirLLM**
   - Run the benchmarking harness using `airllm.AutoModel`.
   - Iterate through three precision configurations:
     - 4-bit Quantization (Q4 - bitsandbytes)
     - 8-bit Quantization (Q8 - bitsandbytes)
     - FP16 (Baseline AirLLM, no compression)
   - Use a strictly controlled random seed and prompt length to ensure deterministic comparisons.

3. **Phase 3: Data Collection & Metrics**
   - For every run, collect precisely:
     1. TTFT (Time To First Token)
     2. TPOT (Time Per Output Token)
     3. Throughput (Tokens/second)
     4. Peak RAM usage
     5. Estimated Energy consumption
     6. Output Quality (Cosine similarity against reference answer)
   - Save directly to `results/benchmark_metrics.csv`.

4. **Phase 4: Original Extension (Pareto Analysis)**
   - Plot the derived *Output Quality vs. Throughput* curve.
   - Mathematically define the Pareto-optimal frontier to find the configuration that provides the best trade-off.

## 2. Architecture Diagrams (C4 Model)

The following diagrams illustrate the architecture from a high-level system context down to specific components.

### 2.1 Context Diagram (C4 Level 1)
```mermaid
C4Context
    title System Context diagram for AirLLM Benchmarking System

    Person(researcher, "AI Researcher", "Executes benchmark experiments and analyzes results")
    System(benchmark_sys, "AirLLM Benchmarking System", "Runs memory-constrained LLM inference, measures metrics, and generates analysis")
    
    System_Ext(hf_hub, "Hugging Face Hub", "Hosts open-source model weights (Qwen2.5)")
    System_Ext(openai_api, "OpenAI Pricing API", "Provides current commercial API pricing for economic comparison")

    Rel(researcher, benchmark_sys, "Configures and runs experiments via CLI")
    Rel(benchmark_sys, hf_hub, "Downloads safetensor shards")
    Rel(benchmark_sys, openai_api, "Fetches external API costs")
```

### 2.2 Container Diagram (C4 Level 2)
```mermaid
C4Container
    title Container diagram for AirLLM Benchmarking System

    Person(researcher, "AI Researcher", "Executes benchmark experiments")

    System_Boundary(c1, "Benchmarking System") {
        Container(experiments, "Experiment Scripts", "Python Scripts", "Orchestrates the ordered execution of benchmarking phases (01 to 07)")
        Container(sdk, "SDK Facade", "Python Module", "Centralized public interface hiding implementation complexity")
        Container(airllm_engine, "AirLLM Inference Engine", "Python / AirLLM", "Streams layer weights from disk to RAM for bounded-memory execution")
        Container(data_store, "Results Store", "File System (CSV/JSON)", "Persists measured metrics and economic calculations")
        Container(plotter, "Plotting Service", "Python / Matplotlib", "Generates PNG charts and visual analysis")
    }

    System_Ext(local_disk, "Local NVMe/SSD", "Stores large layer shards to avoid RAM saturation")

    Rel(researcher, experiments, "Runs")
    Rel(experiments, sdk, "Uses")
    Rel(sdk, airllm_engine, "Delegates inference to")
    Rel(sdk, plotter, "Triggers visualizations")
    Rel(airllm_engine, local_disk, "Streams chunks (mmap)")
    Rel(airllm_engine, data_store, "Writes metrics")
    Rel(plotter, data_store, "Reads metrics")
```

### 2.3 Component Diagram (C4 Level 3: SDK & Services)
```mermaid
C4Component
    title Component diagram for the SDK and Services layer

    Container_Boundary(sdk_container, "SDK & Services") {
        Component(facade, "HW5SDK", "Python Class", "Primary entry point for experiment scripts")
        Component(benchmarker, "BenchmarkRunner", "Python Class", "Handles the run loop, measures TTFT/TPOT, wraps AirLLM")
        Component(ram_monitor, "RAM Monitor Thread", "Python Thread", "Polls psutil asynchronously to catch peak memory usage")
        Component(economist, "EconomicAnalyser", "Python Class", "Computes CAPEX/OPEX vs API costs")
        Component(plotter_comp, "Plotter", "Python Class", "Generates Matplotlib outputs")
    }

    Rel(facade, benchmarker, "Instantiates & Calls")
    Rel(facade, economist, "Instantiates & Calls")
    Rel(facade, plotter_comp, "Instantiates & Calls")
    Rel(benchmarker, ram_monitor, "Spawns during inference")
```

## 3. Visualizations Planned
- **Performance Comparison Table:** Cross-tabulating all metrics across the baseline and three quantization levels.
- **Throughput & TTFT Bar Charts:** Visual comparison of speed metrics, clearly segregating Prefill (TTFT) and Decode (TPOT).
- **Peak Memory Usage Bar Chart:** Proving the efficacy of layer streaming.
- **Roofline Diagram:** Showing configurations plotted against theoretical hardware compute/memory bandwidth bounds.
- **Economic Break-Even Graph:** Cost curve mapping Local On-Prem cumulative cost against External API cumulative cost.
- **Pareto Frontier Plot:** 2D scatter of Quality vs Throughput.

## 4. ADRs (Architecture Decision Records) Highlights

- **ADR-1: Layer Streaming over Swap Memory:** Native OS swap leads to catastrophic thrashing with large models. We explicitly force AirLLM's application-level streaming (`mmap` via safetensors) to bypass OS virtual memory limitations.
- **ADR-2: Centralized Plotting Service:** Instead of littering matplotlib code inside experiment scripts, all charting logic is contained within the `Plotter` service. This enforces the Single Responsibility Principle and keeps scripts < 150 lines.
- **ADR-3: Asynchronous RAM Monitoring:** Peak RAM cannot be accurately measured synchronously because Python garbage collection and tensor allocation happen opaquely in C. A dedicated background thread (`_ram_monitor.py`) polls `psutil` at 100ms intervals to capture the true peak spike.

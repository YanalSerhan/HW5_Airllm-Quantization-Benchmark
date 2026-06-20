# Technical Plan & Experimental Design (PLAN)

## 1. Experimental Design Overview
We will follow a structured, incremental approach to execute our benchmark experiments safely and efficiently:
1. **Baseline Direct Run:** Attempt to load the 72B model natively using Hugging Face `transformers`. We expect an immediate OOM (Out of Memory) exception or system freeze due to extreme swap thrashing. We will log the results.
2. **Smallest Config First:** Test the AirLLM pipeline with the smallest/fastest possible configuration (e.g., 4-bit quantization, 10-token output) to verify the pipeline works before scaling up.
3. **Quantization Sweep:** Run the benchmarking script across the three mandated quantization levels:
   - FP16 (Baseline AirLLM)
   - 8-bit Quantization (Q8)
   - 4-bit Quantization (Q4)
4. **Data Collection:** For each run, record the 6 key metrics to a structured CSV file (`results/benchmark_metrics.csv`).

## 2. Pipeline Architecture
- **Download Module:** `src/hw5_airllm_benchmark/download_model.py` uses `huggingface_hub` to fetch model weights safely.
- **Benchmarking Script:** A centralized, class-based benchmarking script that takes the quantization level as an argument, initializes the model via AirLLM, runs a fixed reference prompt, tracks performance, and logs the metrics to the results file.
- **Analysis Notebook/Script:** A Python script or notebook to load the CSV results, calculate the economics, and generate the required visualizations.

## 3. Visualizations Planned
- **Performance Comparison Table:** Cross-tabulating all metrics across the three quantization levels.
- **Throughput & TTFT Bar Charts:** Visual comparison of speed metrics (Prefill vs Decode).
- **Economic Break-Even Graph:** Cost curve of Local On-Prem vs External API (e.g., OpenAI/Claude pricing).

## 4. Original Extension (Task G)
**Quality-vs-Speed Pareto Frontier:** 
In addition to standard performance metrics, we will score the qualitative output of the model at each quantization level using a simple, reproducible rubric (e.g., coherence, factuality, adherence to prompt instructions). We will then plot a 2D scatter graph of **Output Quality vs. Throughput (tokens/sec)** to visualize the Pareto-optimal frontier, allowing us to clearly identify the "red line" of quality degradation.

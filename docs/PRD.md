# Product Requirements Document (PRD)

## 1. Executive Summary & Project Goal
The objective of this project is to demonstrate, measure, and critically analyze the constraints of running a massive Large Language Model (`Qwen2.5-3B-Instruct` used for feasibility, while illustrating principles applicable up to 72B) on local, consumer-grade hardware (e.g., 16GB RAM, limited VRAM). By leveraging AirLLM's virtual memory layer-streaming and quantization techniques (4-bit, 8-bit, FP16), the project transforms a guaranteed Out-Of-Memory (OOM) failure into a successful, albeit disk-I/O-bound, inference pipeline. The project fundamentally explores the speed-vs-memory-vs-quality trade-offs in modern LLM deployment.

## 2. Core Research Questions
This project is designed to answer six fundamental research questions mandated by the assignment (EX05):
1. **Bottleneck Identification:** What exactly caused the bottleneck in the baseline run (VRAM/RAM vs compute-bound)? How did you identify it?
2. **AirLLM Mechanism:** How does AirLLM change resource allocation, and what is its relationship to virtual memory and OS paging?
3. **Quantization Impact:** What was the effect of quantization on speed, memory footprint, and output quality? Where is the "red line" of acceptable quality degradation?
4. **Prefill vs. Decode Dynamics:** How do Prefill and Decode stages manifest in measurements (TTFT vs TPOT)? What does each reflect (compute load vs memory load)?
5. **Local Hardware Price:** What is the price (in latency and throughput) paid locally for running a large model on modest hardware?
6. **Economic Feasibility:** When is it economically worthwhile to run locally versus using an external commercial API (e.g., OpenAI GPT-4o)?

## 3. Scope
### 3.1. In Scope
- **Baseline Simulation:** A script to attempt naive loading of the model to document the expected OOM failure and capture system state.
- **AirLLM Integration:** Layer-streaming inference using the `airllm` library.
- **Quantization Sweep:** Executing runs across 4-bit, 8-bit, and FP16 precisions.
- **Performance Benchmarking:** Automated capturing of Time To First Token (TTFT), Time Per Output Token (TPOT), Throughput, Peak RAM, and Peak VRAM.
- **Economic Analysis:** Calculation of API costs vs. On-Premise CAPEX/OPEX, culminating in a break-even analysis curve.
- **Pareto Frontier Extension:** An original extension plotting output quality against throughput to find the optimal deployment configuration.

### 3.2. Out of Scope
- Distributed training or multi-GPU inference.
- Fine-tuning (LoRA/QLoRA) of the model weights.
- Multi-day production serving or load-balancing setups.
- Real-time chatbot UI (CLI/script-based execution is sufficient).

## 4. Functional Requirements
- **F-01: Automated Hardware Profiling:** The system must automatically detect and log the host machine's CPU, RAM, OS, and GPU specifications.
- **F-02: Naive Execution:** The system must attempt a standard Hugging Face `AutoModelForCausalLM` load and gracefully catch/log the resulting `OSError` (or similar) without corrupting the workspace.
- **F-03: Layer-Streaming Inference:** The system must use AirLLM to stream model layers from disk to RAM, generating text without exceeding available system memory.
- **F-04: Metric Collection:** The system must measure and record TTFT, TPOT, RAM delta, and total elapsed time for every run.
- **F-05: Results Persistence:** All metrics must be persisted to a structured `CSV` file and a `JSON` summary file.
- **F-06: Visualization Generation:** The system must programmatically generate PNG charts for latency, throughput, memory usage, roofline, economic break-even, and the Pareto frontier.

## 5. Non-Functional Requirements
- **Performance Efficiency:** The system must complete an AirLLM run on consumer hardware without hard-crashing the OS (must stay within ~3-4GB peak RAM).
- **Maintainability:** Code must adhere to strict modularity rules (no file > 150 lines), use `ruff` for zero-lint-error compliance, and contain comprehensive docstrings.
- **Portability:** The project must use `uv` for 100% reproducible dependency management across Windows, macOS, and Linux.
- **Security:** API tokens (e.g., Hugging Face) must be loaded from `.env` and never hardcoded in source files.

## 6. Assumptions & Constraints
- **Hardware Constraint:** The host machine is assumed to have < 4GB of free RAM available at runtime, intentionally forcing disk-I/O paging.
- **Storage Constraint:** The host must have at least 10GB of free NVMe/SSD space to store downloaded model layer shards.
- **Network Dependency:** Initial runs require an active internet connection to download Hugging Face model shards.
- **CUDA Availability:** While CUDA accelerates the pipeline, the codebase must degrade gracefully to CPU-only inference if PyTorch cannot find a compatible GPU.

## 7. Success Criteria & KPIs
- **KPI-1 (Baseline validation):** The baseline script successfully proves and logs the OOM failure.
- **KPI-2 (AirLLM success):** AirLLM successfully generates 50 tokens at 4-bit, 8-bit, and FP16 without crashing.
- **KPI-3 (Data completeness):** 100% of required metrics (TTFT, TPOT, Throughput, Peak RAM, Energy, Quality) are collected for all successful runs.
- **KPI-4 (Economic clarity):** The break-even point is mathematically calculated, graphed, and leads to a clear recommendation.
- **KPI-5 (Code quality):** `uv run ruff check .` returns 0 errors; `uv run pytest tests/` returns 100% pass rate.

## 8. Timeline / Phased Execution (Alignment with TODO.md)
1. **Phase 1-3:** Environment Setup, Documentation (PRD/PLAN), Architecture Design.
2. **Phase 4:** Baseline experiment execution and bottleneck documentation.
3. **Phase 5:** AirLLM integration and quantization sweep.
4. **Phase 6:** Performance comparison and data visualization.
5. **Phase 7:** Economic analysis and Break-even graphing.
6. **Phase 8:** Original Extension (Pareto frontier).
7. **Phase 9-11:** Final Report writing, README polishing, and Code Quality Gates.

# Product Requirements Document (PRD)

## 1. Project Goal
To demonstrate, measure, and analyze the constraints of running a massive Large Language Model (`Qwen2.5-72B-Instruct`) on local, consumer-grade hardware (16GB RAM, 4GB VRAM) by leveraging AirLLM's virtual memory layer-streaming and quantization techniques.

## 2. Core Research Questions
1. What exactly caused the bottleneck in the baseline run (VRAM/RAM vs compute-bound)? How did you identify it?
2. How does AirLLM change resource allocation, and what is its relationship to virtual memory?
3. What was the effect of quantization on speed, memory footprint, and output quality? Where is the "red line" of acceptable quality degradation?
4. How do Prefill and Decode stages manifest in your measurements (TTFT vs TPOT)? What does each reflect (compute load vs memory load)?
5. What is the price (latency/throughput) you pay locally for running a large model on modest hardware?
6. When is it economically worthwhile to run locally vs use an external API?

## 3. Scope
- **In Scope:** Small, targeted, fast experiments to demonstrate the memory bottleneck principles of inference. We will strictly limit max token generations and prompt lengths to ensure experimental runs complete in a reasonable time.
- **Out of Scope:** Multi-day production runs, fine-tuning large scale datasets, and distributed training setups.

## 4. Success Criteria / KPIs
- Baseline run failure (or massive swap thrashing) is successfully reproduced, identified, and logged.
- AirLLM successful run at baseline (FP16/BF16), Q8, and Q4 quantization levels.
- The 6 required metrics are accurately captured across all test runs: TTFT, TPOT/ITL, Throughput (tokens/sec), Peak memory usage, Total run time, and estimated electricity consumption.
- Economic break-even analysis (API vs. On-Prem) completed and plotted.
- Original extension implemented correctly.

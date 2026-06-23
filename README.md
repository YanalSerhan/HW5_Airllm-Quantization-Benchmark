# HW5 AirLLM Quantization Benchmark

## Hardware Documentation & Model Selection

### Machine Specifications
* **CPU:** Intel/AMD Processor (auto-detected via Python script)
* **RAM:** ~16 GB Total Physical Memory
* **Storage:** NVMe SSD/HDD with 465 GB Free Space
* **GPUs:**
  * NVIDIA GeForce RTX 2060 (4 GB VRAM)
  * Intel(R) UHD Graphics (1 GB VRAM)

### Model Choice & Justification
* **Model:** `Qwen/Qwen2.5-3B-Instruct`
* **Justification:** This model has 3 billion parameters. In half-precision (FP16/BF16), it requires roughly 6-7GB of RAM just to load the model weights, plus additional memory for the KV cache during generation. Given my hardware limits where only ~1.5GB of System RAM was available, a direct naive run catastrophically failed with an Out Of Memory (OOM) error due to extreme swap thrashing. This makes it an ideal candidate to demonstrate the necessity of AirLLM's layer-streaming virtual memory mechanism and quantization, fulfilling the EX05 pedagogical requirement of choosing an appropriately massive model that is "appropriately painful" but scientifically observable.

## Experiment Results: Baseline vs. AirLLM

### The Baseline Run (Native PyTorch)
We attempted to load the model natively without AirLLM or quantization.
* **Outcome:** Failed completely. It spent over 16 minutes (984 seconds) violently thrashing the hard drive's paging file trying to load the 6GB model into the remaining ~1.5GB of RAM until Windows finally killed it with a fatal Out of Memory error (`os error 1455`).
* **Tokens Generated:** 0.

### AirLLM & Quantization Sweep
Using AirLLM, we successfully bypassed the hardware memory limits by streaming the model layers from the hard drive one by one. We tested the model at three different quantization levels (4-bit, 8-bit, and unquantized fp16).

#### Performance Metrics

| Quant Level | TTFT (s) | TPOT (s) | Tok/s | Peak RAM (GB) | Peak VRAM (GB) | Quality |
| ----------- | -------- | -------- | ----- | ------------- | -------------- | ------- |
| **4bit**    | 11.65    | 8.65     | 0.11  | 0.63          | 1.31           | 0.98    |
| **8bit**    | 15.73    | 9.60     | 0.10  | 0.32          | 0.94           | 1.00    |
| **fp16**    | 28.51    | 22.83    | 0.04  | 2.55          | 0.69           | 1.00    |

#### Analysis & Insights
1. **Memory Usage (Bypassing OOM):** 
   While the baseline tried to allocate all 6GB at once and crashed, AirLLM capped the peak RAM footprint to just **2.55 GB** for fp16, and an incredibly low **0.32 GB** for 8-bit quantization. This perfectly proves that layer-streaming effectively mitigates hardware memory bottlenecks.
2. **Speed & Throughput (The Disk I/O Bottleneck):** 
   Because AirLLM streams layers from the hard drive for *every single token generated*, the generation process is heavily bottlenecked by disk read speeds, resulting in low throughput (0.04 - 0.11 Tok/s). However, **4-bit and 8-bit runs were more than twice as fast as fp16**. This confirms that quantization significantly reduces the disk I/O bandwidth required, speeding up the generation process.
3. **Quality Penalty:**
   The 8-bit and fp16 runs scored a perfect 1.000 for output coherence. The 4-bit run dropped slightly to 0.9778, illustrating the expected "quantization penalty" where aggressive precision reduction causes minor degradation in model accuracy.
4. **Energy Efficiency:**
   The 4-bit run completed in under half the time of the fp16 run (435s vs 1147s), consuming less than half the energy (5.44 Wh vs 14.33 Wh). This highlights the real-world economic and environmental benefits of quantization.

## Visualizations

### Performance Comparison (Latency vs. Throughput)
![Performance Comparison](figures/performance_comparison.png)

### Peak Memory Usage
![Memory Usage](figures/memory_usage.png)

### Roofline Diagram
![Roofline Diagram](figures/roofline_diagram.png)
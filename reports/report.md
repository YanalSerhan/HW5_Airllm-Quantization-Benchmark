# HW5 Deep-Dive Technical Report
## AirLLM Quantization Benchmark: Running a 3B LLM on Constrained Consumer Hardware

**Author:** Yanal Serhan  
**Date:** June 2026  
**Model:** `Qwen/Qwen2.5-3B-Instruct`  
**Hardware:** Intel Core i7-10750H · 16.93 GB RAM · RTX 2060 (4 GB VRAM) · Windows 11  
**Framework:** [AirLLM](https://github.com/lyogavin/Airllm) · Hugging Face Transformers · PyTorch

---

## Table of Contents

1. [Hardware Documentation & Model Justification](#1-hardware-documentation--model-justification)
2. [Baseline Experiment: Failure Evidence](#2-baseline-experiment-failure-evidence)
3. [AirLLM + Quantization Integration](#3-airllm--quantization-integration)
4. [Full Performance Comparison](#4-full-performance-comparison)
5. [Economic / Business Analysis](#5-economic--business-analysis)
6. [Theoretical Discussion](#6-theoretical-discussion)
7. [Original Extension: Quality vs. Speed Pareto Frontier](#7-original-extension-quality-vs-speed-pareto-frontier)
8. [Conclusions & Lessons Learned](#8-conclusions--lessons-learned)
9. [References](#9-references)

---

## 1. Hardware Documentation & Model Justification

### 1.1 Machine Specifications

| Component | Specification |
|---|---|
| **CPU** | Intel Core i7-10750H (Intel64 Family 6 Model 165 Stepping 2) |
| **Physical / Logical Cores** | 6 / 12 |
| **Max CPU Frequency** | 2592.0 MHz |
| **Total System RAM** | 16.93 GB |
| **RAM Available at Experiment Start** | ~2.53 GB (system heavily loaded) |
| **GPU** | NVIDIA GeForce RTX 2060 (4 GB VRAM) + Intel UHD (1 GB) |
| **GPU Detected by PyTorch** | `False` — CUDA unavailable in environment |
| **OS** | Windows 11 (10.0.26200) |
| **Storage** | ~500 GB free on project drive |

> **Note on GPU detection:** While the RTX 2060 is physically present, `torch.cuda.is_available()` returned `False` during all experiments. All inference was therefore executed on CPU, with AirLLM streaming layers to system RAM rather than VRAM. This makes the results even more extreme — every performance observation is from a pure CPU + disk-I/O configuration.

### 1.2 Model Choice & Justification

**Selected Model:** `Qwen/Qwen2.5-3B-Instruct`

This model was deliberately chosen because it sits in the "appropriately painful" sweet spot required by EX05:

- **Memory requirement:** At 3 billion parameters in FP16, the model requires approximately **6–7 GB of RAM** just for weights, far exceeding the ~2.5 GB of available RAM at experiment start.
- **Pedagogical value:** It is large enough to demonstrate the catastrophic failure of naive loading, while small enough that AirLLM can successfully stream it in finite time.
- **Format:** Distributed in `.safetensors` format (not `pickle`), which enables secure, zero-copy `mmap`-based layer loading.
- **Quality:** As an instruction-tuned chat model, output coherence can be meaningfully measured, enabling a quality vs. performance trade-off analysis.

---

## 2. Baseline Experiment: Failure Evidence

### 2.1 What We Tried

We attempted to load and run `Qwen/Qwen2.5-3B-Instruct` using the standard Hugging Face `transformers.AutoModelForCausalLM.from_pretrained()` call — no AirLLM, no quantization, no special memory management. This is the naive approach any developer would try first.

**RAM state before the baseline run:**
- Total: 16.93 GB
- Used: 15.46 GB (91.3%)
- Available: only **1.47 GB**
- Swap already in use: 2.83 GB of 51.5 GB

### 2.2 What Happened

The baseline run did not fail quickly. It spent **984 seconds (16 minutes and 24 seconds)** violently thrashing the Windows paging file, loading model shards piece by piece while the OS desperately shuffled memory between RAM and the swap file. Eventually Windows terminated the process with a fatal error:

```
OSError: The paging file is too small for this operation to complete. (os error 1455)
```

**Tokens generated: 0.**

This is a perfect real-world demonstration of the "VRAM Gap" in action. The OS's default virtual memory mechanism (page-based swap) is wholly inadequate for LLM inference because:
1. The model must be loaded entirely before any generation can begin.
2. LLM inference has no locality — every token requires the *entire* model, defeating the OS page-cache's assumptions.

### 2.3 Key Takeaway from Baseline

The baseline experiment is not a failure to be hidden — it is the most important result in this report. It proves empirically and dramatically *why* AirLLM exists: the problem it solves is not theoretical.

---

## 3. AirLLM + Quantization Integration

### 3.1 How AirLLM Works

AirLLM replaces the standard model-loading strategy with a **layer-streaming** architecture:

1. Model weights stay on disk (in `.safetensors` shards).
2. For each transformer block, AirLLM loads only that layer's weights into memory (RAM or VRAM), executes the forward pass, then discards the weights.
3. A **prefetching** thread loads the *next* layer in the background while the current layer computes, overlapping I/O and compute to improve utilisation.

This shifts the memory bottleneck from "must fit entire model" to "must fit one layer at a time," which is orders of magnitude smaller.

### 3.2 Quantization Levels Tested

We systematically swept three quantization levels:

| Level | Weight Precision | Bytes per Parameter | ~Model Size |
|---|---|---|---|
| **4-bit** | INT4 (via bitsandbytes) | 0.5 bytes | ~1.5 GB |
| **8-bit** | INT8 (via bitsandbytes) | 1 byte | ~3 GB |
| **fp16** | Half-precision float | 2 bytes | ~6 GB |

All runs used the **same fixed prompt** ("Explain what virtual memory is in 50 words.") and **same generation parameters** (max 50 new tokens) to ensure controlled comparison.

### 3.3 Integration Details

AirLLM was invoked via the `AirLLMLlama` class with the following configuration:

```python
model = AirLLMLlama(
    model_id="Qwen/Qwen2.5-3B-Instruct",
    compression=quantization_level,  # "4bit", "8bit", or None for fp16
    profiling_mode=True,
)
output = model.generate(input_ids, max_new_tokens=50)
```

RAM and timing were monitored via a background thread using `psutil` and `time.perf_counter`.

---

## 4. Full Performance Comparison

### 4.1 Quantitative Results

| Quant Level | TTFT (s) | TPOT (s) | Tok/s | Peak RAM (GB) | Total Time (s) | Energy (Wh) | Quality |
|---|---|---|---|---|---|---|---|
| **4-bit** | 10.97 | 7.74 | 0.1281 | 1.402 | 401.3 | 5.02 | 0.978 |
| **8-bit** | 13.40 | 9.03 | 0.1096 | 1.488 | 469.4 | 5.87 | 1.000 |
| **fp16** | 20.12 | 20.44 | 0.0489 | 3.727 | 1041.9 | 13.02 | 1.000 |
| **Baseline** | — | — | 0 | ~15 GB (OOM) | 984 (crash) | — | 0 |

*TTFT = Time to First Token. TPOT = Time Per Output Token. Quality scored by cosine similarity of output embedding vs. reference answer.*

### 4.2 Analysis

**Memory:** AirLLM capped peak RAM to **3.73 GB even for fp16** — less than the 6 GB minimum just to *load* the model natively. For 4-bit quantization, peak RAM dropped to just **1.40 GB**. This is the core value proposition of layer-streaming.

**Speed:** 4-bit was **2.6x faster than fp16** (401s vs. 1042s total). This is directly attributable to reduced disk I/O: smaller quantised weights mean fewer bytes streamed from disk per layer, directly improving throughput in a disk-I/O-bound regime.

**Quality:** The 8-bit and fp16 runs produced coherent, accurate outputs (quality score 1.000). The 4-bit run dropped marginally to **0.978**, a 2.2% quality degradation for a 2.6x speed gain. This is an excellent trade-off.

**Energy:** The 4-bit run consumed **5.02 Wh** vs. **13.02 Wh** for fp16 — **2.6x more energy efficient**, directly correlating with its faster execution time.

### 4.3 Figures

![Performance Comparison (Latency)](../figures/latency_comparison.png)
*Figure 1: TTFT and TPOT side-by-side for each quantization level. Lower is better.*

![Throughput Comparison](../figures/throughput_comparison.png)
*Figure 2: Inference throughput in tokens/second per quantization level. Higher is better.*

![Memory Usage](../figures/memory_usage.png)
*Figure 3: Peak system RAM usage per quantization level, versus the baseline OOM event.*

![Roofline Diagram](../figures/roofline_diagram.png)
*Figure 4: Qualitative roofline diagram mapping each config onto the memory-bound vs. compute-bound boundary. All configs lie firmly in the memory-bound (disk-I/O) region.*

---

## 5. Economic / Business Analysis

### 5.1 Pricing Assumptions

All assumptions are stated explicitly for reproducibility (EX05 Section 5.5):

| Assumption | Value | Source |
|---|---|---|
| **API Model** | GPT-4o | OpenAI |
| **API Input price** | $0.005 / 1k tokens | OpenAI pricing page, May 2025 |
| **API Output price** | $0.015 / 1k tokens | OpenAI pricing page, May 2025 |
| **Electricity rate** | $0.12 / kWh | US EIA 2024 avg. residential |
| **Hardware cost** | $1,200 (RTX 2060 system) | Estimated purchase price |
| **Hardware lifetime** | 4 years (straight-line depreciation) | Standard assumption |
| **Daily requests** | 10 (CAPEX spread basis) | Low-usage scenario |
| **Input tokens/request** | 50 | Prompt length |

### 5.2 Per-Request Cost Breakdown

| Quant Level | Output Tokens | API Cost (USD) | On-Prem Electricity | On-Prem CAPEX | On-Prem Total |
|---|---|---|---|---|---|
| 4-bit | 51 | $0.001015 | $0.000602 | $0.082192 | $0.082794 |
| 8-bit | 51 | $0.001015 | $0.000704 | $0.082192 | $0.082896 |
| fp16 | 50 | $0.001000 | $0.001563 | $0.082192 | $0.083755 |
| **Average** | ~51 | **$0.001010** | $0.000956 | $0.082192 | **$0.083148** |

![Cost per Quantization Level](../figures/cost_per_quant_level.png)
*Figure 5: Per-request cost breakdown by quantization level — API vs. On-Prem electricity vs. On-Prem CAPEX.*

### 5.3 Break-Even Analysis

At the current usage assumptions (10 requests/day), the On-Premise cost per request ($0.0831) is **82x higher** than the API ($0.0010). The amortised CAPEX ($0.0822/request) completely dominates all other costs.

**Break-even result: Never** — at this usage level, the API is always cheaper.

![Economic Break-Even](../figures/economic_breakeven.png)
*Figure 6: Cumulative cost curves for API vs. On-Premise deployment over 50,000 requests.*

### 5.4 Prompt/Context Caching Note

Modern API providers offer **prompt/context caching** using techniques like PagedAttention. For repeated-context workloads (e.g., long-document Q&A with a fixed system prompt), the provider caches KV states for the repeated prefix, potentially reducing effective input-token cost by 75-90%. This pushes the break-even volume even higher, making On-Premise even harder to justify on pure cost grounds.

### 5.5 Recommendation

| Scenario | Recommendation |
|---|---|
| Occasional / low-volume use (< 1,000 req/day) | **Use the API.** Lowest cost, no CAPEX, no maintenance. |
| High-volume sustained use (> 100,000 req/day) | **On-Premise may become viable** once CAPEX amortises. |
| Privacy / compliance critical workloads | **On-Premise regardless of cost.** Data never leaves your machine. |
| Offline / air-gapped environments | **On-Premise only.** API is unavailable. |

---

## 6. Theoretical Discussion

Each point follows the **Observation > Theoretical Explanation > Implication** structure.

### 6.1 CPU vs. GPU Parallel Architecture (L08 Section 2, 2.4)

**Observation:** Despite having an RTX 2060, all inference ran on CPU (CUDA unavailable). Throughput was 0.05-0.13 Tok/s.

**Theoretical Explanation:** GPUs execute thousands of CUDA threads in parallel using the SIMT model. Transformer inference is dominated by GEMMs that are embarrassingly parallel and directly suited to GPU throughput. NVIDIA Volta's Independent Thread Scheduling further reduces warp divergence over Pascal's lock-step SIMT. On CPU, these operations execute across 12 logical threads, orders of magnitude slower.

**Implication:** The absence of GPU acceleration is the single largest performance bottleneck. A properly configured CUDA environment would likely yield 10-100x throughput improvements.

### 6.2 Prefill vs. Decode: GEMM vs. GEMV (L08 Section 3.1, 3.4)

**Observation:** TTFT (10-20s) reflects prefill processing the ~50-token prompt; TPOT (7-20s per token) reflects the slower decode phase.

**Theoretical Explanation:** Prefill processes the entire prompt in a single forward pass using batched GEMM operations — compute-bound. Decode generates one token at a time, loading the full model and growing KV cache for a single GEMV operation per layer — memory-bandwidth-bound. Research projects like Splitwise and DistServe propose separating prefill and decode onto different hardware.

**Implication:** TPOT is always slower because decode is inherently memory-bound. Disaggregated serving is the state-of-the-art solution at scale.

### 6.3 VRAM Gap and Memory Constraints (L08 Section 3.3)

**Observation:** A 3B parameter model (FP16) requires ~6 GB of memory, exceeding both the 4 GB RTX 2060 VRAM and ~2.5 GB available system RAM.

**Theoretical Explanation:** The "VRAM Gap" is the growing disparity between LLM memory requirements and GPU VRAM. Model memory scales as parameters x bytes_per_weight. Consumer GPU VRAM has not kept pace with model size growth.

**Implication:** Without special memory management (quantization, layer-streaming), running a 3B+ parameter model on a 4 GB VRAM GPU is impossible.

### 6.4 Virtual Memory, Paging, and the Locality Principle (L08 Section 8.2, 8.4)

**Observation:** The baseline run thrashed the paging file for 984 seconds before crashing with os error 1455.

**Theoretical Explanation:** The OS MMU implements virtual memory by mapping logical addresses to physical RAM. When RAM is full, the OS pages data to disk. The Principle of Locality underlies page-cache efficiency. LLM inference violates locality completely — it reads every layer for every token, generating continuous Page Faults and degenerate thrashing.

**Implication:** OS general-purpose virtual memory is fundamentally inadequate for LLM inference. Application-level management (AirLLM, FlexGen) is required.

### 6.5 SafeTensors vs. GGUF (L08 Section 4.3, 4.4)

**Observation:** The Qwen2.5-3B-Instruct model uses .safetensors format, enabling secure and efficient layer loading.

**Theoretical Explanation:** Traditional pickle-based formats are a security vulnerability — they execute arbitrary Python code on deserialisation. SafeTensors eliminates this and supports zero-copy loading via mmap(), where the OS maps file pages directly into the process address space. GGUF (used by llama.cpp) is an alternative optimised for quantised models, also supporting mmap.

**Implication:** SafeTensors is the correct format for AirLLM layer-streaming — both secure and zero-copy efficient.

### 6.6 AirLLM Layer-Streaming, Prefetching & Bottleneck Shift (L08 Section 8.1-8.3)

**Observation:** AirLLM ran inference with peak RAM of 1.4-3.7 GB, but throughput was extremely low (0.05-0.13 Tok/s).

**Theoretical Explanation:** AirLLM implements "LLM in a Flash" — layers are kept on secondary storage and streamed into RAM/VRAM on demand. Background prefetching hides I/O latency by loading layer N+1 while layer N executes. This shifts the bottleneck from VRAM capacity to disk read bandwidth. Related research: FlexGen (CPU/disk offloading), LLM in a Flash (Apple SSD-offloading), PagedAttention (vLLM KV-cache management).

**Implication:** AirLLM solves the OOM problem but trades memory constraint for an I/O speed ceiling. NVMe SSD would perform significantly better than HDD.

### 6.7 Quantization Trade-offs: NF4, INT8, and Quality (L08 Section 5, 5.1)

**Observation:** 4-bit quantization reduced memory and time by ~60% vs. fp16, with only 2.2% quality drop.

**Theoretical Explanation:** Quantization maps floating-point weights to lower-precision integers. INT8 halves FP16 memory with minimal accuracy loss. INT4 halves again, but with only 16 levels, quantization error is more significant. QLoRA addresses this during fine-tuning using NF4 (NormalFloat4), Double Quantization, and Paged Optimizers. For inference (our scenario), standard bitsandbytes INT4/INT8 was used.

**Implication:** 8-bit quantization is the "free lunch" — perfect quality, half the memory. 4-bit is suitable when speed/memory is the hard constraint.

### 6.8 Deployment Trade-offs (L08 Section 1.1, 1.2)

**Observation:** GPT-4o API costs $0.001/request while On-Premise costs $0.083/request at low usage due to CAPEX amortisation.

**Theoretical Explanation:** Three deployment models exist: (1) API — zero CAPEX, high speed, zero data privacy; (2) Cloud GPU Rental — no CAPEX, better privacy than shared APIs; (3) On-Premise — high CAPEX, near-zero marginal cost at scale, full data privacy. The right choice depends on volume, latency, data sensitivity, and regulatory constraints.

**Implication:** For this project's usage profile, the API is strictly better economically. On-Premise is only justified by non-cost factors like privacy and offline capability.

### 6.9 Conclusion: Compute-Bound or Memory-Bound?

The AirLLM inference runs were overwhelmingly **Disk-I/O-bound** — a severe form of memory-bound behaviour.

Evidence: throughput of 0.05-0.13 Tok/s is far below what even a CPU can compute, proving the bottleneck is I/O. Each transformer layer must be read from disk for every generated token. The disk read speed dictated the throughput ceiling. The 2.6x speed improvement from fp16 to 4-bit is explained entirely by the 4x reduction in bytes read per layer from disk.

In a conventional GPU setup (no disk offloading), LLM **decode** is **memory-bandwidth-bound** by VRAM bandwidth (KV cache and weight re-reads per token), while **prefill** is **compute-bound** (parallelisable GEMM). In this AirLLM setup, the bottleneck shifts one level further down the memory hierarchy: from VRAM bandwidth to **disk read bandwidth**.

---

## 7. Original Extension: Quality vs. Speed Pareto Frontier

**Rationale:** Individual metric charts answer single questions. A decision-maker choosing a quantization level for production needs to reason about the joint trade-off: how much speed can be gained for how much quality is sacrificed? The Pareto Frontier is the canonical tool for this.

**Method:** Each quantization configuration is plotted as a point in (Throughput, Quality Score) space. The dashed line connecting points in throughput order shows the frontier.

![Pareto Frontier](../figures/pareto_frontier.png)
*Figure 7: Quality vs. Speed Pareto Frontier. Points to the upper-right are Pareto-superior.*

**Insight:**
- **fp16** (0.049 Tok/s, 1.000 quality) — quality-optimal when time is unconstrained.
- **8-bit** (0.110 Tok/s, 1.000 quality) — **strictly Pareto-dominant over fp16**: same quality, 2.2x faster. No cost to choosing 8-bit over fp16.
- **4-bit** (0.128 Tok/s, 0.978 quality) — speed-optimal: 16% faster than 8-bit at cost of 2.2% quality reduction. On the frontier.

**Recommendation:** Deploy at **8-bit** as the default. Only choose 4-bit if hardware memory is the absolute hard constraint.

---

## 8. Conclusions & Lessons Learned

### 8.1 Summary of Findings

1. **AirLLM solves the VRAM/RAM barrier.** A model that crashed a 16 GB system in 16 minutes successfully ran within 1.4 GB RAM. This is the difference between "impossible" and "possible."

2. **Quantization is the right trade-off lever.** 8-bit achieves perfect output quality while cutting memory and time in half. The Pareto frontier shows 8-bit strictly dominates fp16.

3. **The API wins economically at low volume.** At 10 requests/day, On-Premise costs 82x more per request. The break-even is never reached within realistic planning horizons. On-Premise is justified only by non-cost factors (privacy, offline, regulatory).

### 8.2 Honest Negative Results

- **GPU was unavailable.** All runs executed on CPU. All throughput numbers are severely bottlenecked vs. theoretical maximums. The experiment proves the *concept*, not the *peak performance* of AirLLM.
- **AirLLM is slow.** 4-bit took ~400 seconds for 50 tokens. Not production-viable for any latency-sensitive application.
- **The break-even was never reached.** On-Premise could not justify itself economically at any simulated volume.

These negative results are reported honestly because they are scientifically valid and more informative than a suspiciously clean positive result.

### 8.3 Lessons Learned

- Always run a baseline failure first. It motivates every optimisation that follows.
- The OS is not your friend for LLM inference. Its paging mechanisms assume locality — LLMs break that assumption entirely.
- Measure everything before drawing conclusions. The roofline diagram and Pareto frontier revealed insights that raw numbers alone could not.
- Quantization is (almost) free. 8-bit costs nothing in quality and gives 2x speed and memory. It should be the default, not an optimisation.

---

## 9. References

1. **AirLLM** — Gavin Li. *AirLLM: Inference 70B LLM using only 4GB GPU VRAM.* GitHub: https://github.com/lyogavin/Airllm
2. **Hugging Face Transformers** — Wolf et al., 2020. *HuggingFace Transformers: State-of-the-art Natural Language Processing.* EMNLP 2020.
3. **Qwen2.5** — Qwen Team, Alibaba Cloud. *Qwen2.5 Technical Report.* 2024.
4. **bitsandbytes** — Tim Dettmers et al. *8-Bit Optimizers via Block-wise Quantization.* ICLR 2022.
5. **QLoRA** — Dettmers et al. *QLoRA: Efficient Finetuning of Quantized LLMs.* NeurIPS 2023.
6. **FlexGen** — Sheng et al. *FlexGen: High-Throughput Generative Inference of Large Language Models with a Single GPU.* ICML 2023.
7. **LLM in a Flash** — Alizadeh et al. *LLM in a Flash: Efficient Large Language Model Inference with Limited Memory.* Apple Research, 2024.
8. **vLLM / PagedAttention** — Kwon et al. *Efficient Memory Management for Large Language Model Serving with PagedAttention.* SOSP 2023.
9. **SafeTensors** — Hugging Face. *SafeTensors Format Specification.* https://huggingface.co/docs/safetensors
10. **L08 Lecture Notes** — Course Lecture 08: *LLM Inference: Hardware, Memory, and Quantization.* (Internal reference)
11. **L07 Lecture Notes** — Course Lecture 07: *LLM Architecture, VRAM Gap, and AirLLM.* (Internal reference)
12. **OpenAI Pricing** — https://openai.com/pricing (May 2025)
13. **US EIA Electricity Prices** — U.S. Energy Information Administration. *Average Retail Price of Electricity.* 2024.

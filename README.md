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
* **Model:** `Qwen/Qwen2.5-72B-Instruct`
* **Justification:** This model has 72 billion parameters. In half-precision (FP16/BF16), it requires roughly 144GB of VRAM/RAM just to load the model weights, and additional memory for the KV cache during generation. Given my hardware limits of 16GB System RAM and 4GB GPU VRAM, a direct naive run will catastrophically fail with an Out Of Memory (OOM) error or extreme swap thrashing. This makes it an ideal candidate to demonstrate the necessity of AirLLM's layer-streaming virtual memory mechanism and quantization, fulfilling the EX05 pedagogical requirement of choosing an appropriately massive model that is "appropriately painful" but scientifically observable.
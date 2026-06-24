# TODO.md — Master Execution Guide
## Assignment: Running a Massive LLM Locally — AirLLM, Quantization & Performance Benchmarking (EX05, L08)

> Source files analyzed:
> - `ex05-AirLLM_pdf.pdf` (official assignment spec — **EX05**)
> - `software_submission_guidelines-V3.pdf` (mandatory engineering/quality standard — **GUIDE**)
> - `L08-summary-Lora-AirLLM.pdf` (Hebrew lecture summary — **L08**)
> - `L7_Summary.pdf` (English lecture summary, 3 parts — **L7**)
>
> Every task below is tagged with its source(s) so you can trace back to the original requirement.

---

## 0. How to Use This Document

- [x] Work top-to-bottom. Sections are ordered chronologically (setup → research → implementation → experiments → analysis → writing → submission).
- [x] Check off each box as completed. Do not skip "Verification" sub-sections — they exist to catch missed requirements before you move on.
- [x] Keep this file itself inside the repo (or use it to seed `docs/TODO.md`, which is **mandatory** per GUIDE).

---

## 1. Environment Setup

**Why:** EX05 and GUIDE both treat a clean, reproducible environment as a prerequisite for any valid measurement. L7/L08 explain why model loading itself is hardware-intensive, so setup must be done carefully and patiently.

### 1.1 Tooling
- [x] Install **`uv`** as the package manager. (GUIDE §8.4: `pip`/raw `venv`/`python -m` calls are **forbidden**; everything must run via `uv run ...`.)
  - Dependency: none.
  - Output: `uv` available on PATH.
- [x] Create project repo folder and initialize git.
- [x] Create `pyproject.toml` (GUIDE §8.1, §14.1) with: project name, description, author, license, dependencies, pinned starting version `1.00`.
- [x] Run `uv sync` / `uv add <pkg>` for all dependencies (never `pip install`). (GUIDE Table 3)
- [x] Generate `uv.lock` and commit it. (GUIDE §17.4)
- [x] Verify your installed **Python version** is compatible with all required libraries — **do not blindly grab the newest Python**; check library compatibility first. (EX05 §6.1 "Do")
- [x] Create `.env-example` with placeholder values (e.g., `HF_TOKEN=`) — **never** commit a real `.env`. (GUIDE §7.4, §17.4)
- [x] Create `.gitignore` covering `.env`, `*.key`, `*.pem`, `credentials.json`, model weight caches, `results/` raw dumps if large. (GUIDE §7.4)

### 1.2 Hardware Documentation (REQUIRED DELIVERABLE)
**Why:** EX05 Core Task (a) explicitly requires hardware documentation to justify model choice. L7 explains the VRAM gap and memory hierarchy that makes this matter.
- [x] Record exact machine specs:
  - [x] CPU model and core count
  - [x] RAM size
  - [x] GPU model and VRAM size (if any GPU present)
  - [x] Storage type (NVMe SSD vs HDD) and free space
- [x] Save this as a dedicated section in the final report / README (not just scratch notes).
- [x] Cross-check free disk space is sufficient **before** downloading large models. (EX05 §6.1 "Do": "ensure enough free disk space before downloading large models")

### 1.3 Model Download
**Why:** EX05 mandates choosing one large model intentionally too large/awkward for your hardware, to *demonstrate* the bottleneck — this is the central pedagogical trick of the assignment.
- [x] Browse Hugging Face Hub and select **one large model** sized so that it is realistically too large for your RAM (per EX05 explicit guidance: "choose a model whose size, relative to your hardware, is large enough that direct loading fails or is unbearably slow — but not absurdly large that nothing can ever run").
- [x] Document the **exact reasoning** behind the choice: parameter count, format, license, why it stresses *your* specific hardware. (EX05 §5.1)
- [x] Set up a Hugging Face access token via **environment variable only** — never hard-code it, never commit it in plaintext. (EX05 §6.1 "Don't"; GUIDE §7.4)
- [x] Download model weights (expect this step alone to take **15–60+ minutes**; this is normal — see L7 "I/O wait time" and EX05 Appendix Step 1 time estimate of 1.5–3 hrs wall-clock / ~15 min active work).
- [x] Note model license terms (GUIDE §4.2 analog) — confirm permitted use for coursework.

### ✅ Verification — Section 1
- [x] `uv run python --version` works inside the venv.
- [x] `uv.lock` and `pyproject.toml` exist and are committed.
- [x] `.env` is gitignored; `.env-example` is committed.
- [x] Hardware spec table drafted (CPU, RAM, GPU/VRAM, disk).
- [x] Chosen model documented with justification paragraph.
- [x] Token is loaded via `os.environ.get(...)`, confirmed not present anywhere in source code.

---

## 2. Research & Planning

**Why:** EX05 is explicitly an **open-ended, hypothesis-driven research assignment**, not a fixed recipe — "There is no single exact formula" (EX05 §1). You must plan an engineering experiment, not just run scripts.

### 2.1 Define the Core Research Questions (EX05 §4 — MANDATORY to address in report)
- [x] What exactly caused the bottleneck in the baseline run (VRAM/RAM vs compute-bound)? How did you identify it?
- [x] How does AirLLM change resource allocation, and what is its relationship to virtual memory?
- [x] What was the effect of quantization on speed, memory footprint, and output quality? Where is the "red line" of acceptable quality degradation?
- [x] How do Prefill and Decode stages manifest in your measurements (TTFT vs TPOT)? What does each reflect (compute load vs memory load)?
- [x] What is the price (latency/throughput) you pay locally for running a large model on modest hardware?
- [x] When is it economically worthwhile to run locally vs use an external API?

### 2.2 Plan the Experimental Design
- [x] Decide the **scope** of experimentation — explicitly note that this is a deliberate trade-off (small/fast experiments to demonstrate principles, not multi-day production runs). (EX05 §3, "Planning & Efficiency")
- [x] Plan to start with **smallest possible config first**: lowest quantization level, smallest prompt/token counts. Scale up only after pipeline is verified working. (EX05 §6.1 "Do")
- [x] Decide which quantization levels you will test (e.g., FP16, Q8, Q4) — reference L08 Table 3 quantization levels.
- [x] Decide whether you will attempt at least one **original extension** (mandatory — see §5.7 below).
- [x] Sketch in advance which graphs/tables will be produced (see §6 and §7) so data collection captures everything needed in one pass.

### ✅ Verification — Section 2
- [x] All 6 research questions from EX05 §4 are explicitly listed somewhere you will answer them later.
- [x] Experiment plan written down (even informally) before writing code.
- [x] At least one "original idea" selected from EX05 §5.7 menu or invented.

---

## 3. Repository & Project Structure (GUIDE-mandated)

**Why:** GUIDE is a strict, points-bearing engineering rubric layered on top of EX05; EX05 explicitly requires a GitHub repo with code, experiments, and a report. Treat GUIDE as binding for the whole repo.

### 3.1 Mandatory Top-Level Files
- [x] `README.md` at repo root (GUIDE §2.1, EX05 §8) — see §10 below for full required content.
- [x] `pyproject.toml`, `uv.lock`
- [x] `.env-example`, `.gitignore`

### 3.2 Mandatory `docs/` Folder (GUIDE §2.2)
- [x] `docs/PRD.md` — Product Requirements Document: project goal, scope, success criteria/KPIs, functional/non-functional requirements, assumptions/constraints, timeline.
- [x] `docs/PLAN.md` — architecture/technical plan: high-level diagram of pipeline (download → baseline run → AirLLM+quantization → benchmarking → economic analysis → report), C4 Model & UML diagrams, and explicit **ADRs (Architecture Decision Records)** detailing trade-offs. (GUIDE §2.2)
- [x] `docs/TODO.md` — granular task list with status (not-started/in-progress/done), phases, and "definition of done" per task (this can be a project-specific derivative of the present file).
- [x] (Optional but recommended) `docs/PRD_<mechanism>.md` per major technical component (e.g., `PRD_quantization.md`, `PRD_airllm_pipeline.md`, `PRD_benchmarking.md`) per GUIDE §2.3.

### 3.2a Mandatory Workflow Order (GUIDE §2.5)
**Why:** GUIDE mandates that all documentation is approved *before* any code is written — this order is enforced and graded.
- [x] Step 1 — Create and finalize `docs/PRD.md`; get approval before continuing.
- [x] Step 2 — Create and finalize `docs/PLAN.md` (architecture plan).
- [x] Step 3 — Create and finalize `docs/TODO.md` (task list).
- [x] Step 4 — Create per-mechanism PRDs for any major algorithm/component (e.g., `docs/PRD_airllm_pipeline.md`, `docs/PRD_benchmarking.md`).
- [x] Step 5 — All documents approved before starting development.
- [x] Step 6 — Update `docs/TODO.md` as development progresses.
- [x] Step 7 — Save results, create visualizations, and update `README.md` at the end.

### 3.3 Recommended Folder Layout (EX05 §9, GUIDE §2.4 merged)
```
project-root/
├── README.md
├── pyproject.toml
├── uv.lock
├── .env-example
├── .gitignore
├── docs/
│   ├── PRD.md
│   ├── PLAN.md
│   ├── TODO.md
│   └── PRD_<mechanism>.md     # per-algorithm PRDs (§3.2a)
├── src/
│   └── <package>/
│       ├── __init__.py          # exports __all__ and __version__
│       ├── sdk/
│       │   └── sdk.py
│       ├── services/
│       ├── shared/
│       │   ├── gatekeeper.py    # if calling external APIs
│       │   ├── config.py
│       │   └── version.py
│       └── constants.py
├── experiments/
├── results/
├── reports/
├── figures/
├── data/                        # Input data (GUIDE §2.4)
├── assets/                      # Images, graphs, resources (GUIDE §2.4)
├── config/
│   ├── setup.json
│   └── rate_limits.json
├── tests/
│   ├── unit/
│   └── integration/
└── notebooks/
```
- [x] Keep structure modular by responsibility: `src/` (code), `experiments/` (run scripts), `results/` (raw measurement output, e.g., CSV/JSON), `reports/` (final write-up), `figures/` (generated plots).
- [x] Confirm structure is consistent, navigable, adapted sensibly to actual project needs (not slavishly copied if irrelevant). (EX05 §9)

### 3.4 Code Quality Rules (GUIDE §3, §4, §6, §7 — apply throughout implementation)
- [x] No source file exceeds **150 lines** of code (excluding blank/comment-only lines). Split via helper functions, mixins, constants files, model files, etc. (GUIDE §3.2)
- [x] Every function/class has a docstring explaining *why*, not just *what*. (GUIDE §3.3)
- [x] Follow **SDK-style architecture** if building reusable modules: all business logic reachable through a single SDK layer; GUI/CLI layers never embed logic directly. (GUIDE §4.1)
- [x] Avoid code duplication — extract shared logic into helper functions/mixins/base classes (OOP, no copy-paste). (GUIDE §4.2)
- [x] If calling any external API (e.g., OpenAI/Claude pricing lookups, or live API benchmarking) — route all calls through a centralized **API Gatekeeper** class enforcing rate limits, retries, logging, and a queue. (GUIDE §5) Rate limit values must live in config (e.g., `config/rate_limits.json`), never hard-coded.
- [x] No magic numbers/strings hard-coded — pull from `config/*.json`, `.env`, or `constants.py`. (GUIDE §7.2, Table 1)
- [x] Use `Ruff` linter with zero violations (`ruff check`). Configure in `pyproject.toml` per GUIDE §7.1.
- [x] Follow TDD where feasible: write/maintain tests in `tests/unit/` mirroring `src/` structure; target ≥85% coverage if you build a non-trivial reusable library (note: a pure experiment-script repo may relax this, but **document the decision** in README/PLAN).
- [x] Use relative imports / package-qualified imports only, no absolute filesystem paths. (GUIDE §14.3)
- [x] Tag versions starting at `1.00`, bump on meaningful changes (`src/<pkg>/shared/version.py`, GUIDE §8.1 Table 2).
- [x] **Edge Cases & Error Handling**: Explicitly document and test edge cases. Ensure graceful degradation, clear error messages, and proper logging for failure modes. (GUIDE §6.3)

### 3.5 Python Package Organization (GUIDE §14)
**Why:** GUIDE §14 explicitly requires proper Python package conventions — this is a graded quality gate.
- [x] Every package directory inside `src/` must have an `__init__.py` that exports public interfaces via `__all__` and defines `__version__`. (GUIDE §14.2)
- [x] All imports inside `src/` use relative or package-qualified paths — never absolute filesystem paths or `sys.path` manipulation. (GUIDE §14.3)
- [x] `pyproject.toml` must specify: package name, version (`1.00`+), description, author, license, and all dependencies with version pins. (GUIDE §14.1)
- [x] Validate config version compatibility on startup: app reads `"version"` from config JSON files and asserts compatibility. (GUIDE §8.1 Table 2)

### 3.6 Git Best Practices & Prompt Engineering Log (GUIDE §8.2, §8.3)
**Why:** GUIDE §8.2 requires a clean, traceable Git history. GUIDE §8.3 requires a Prompt Book documenting AI assistance — graders may check both.
- [x] Maintain meaningful, descriptive commit messages throughout development (not "fix" or "update"). (GUIDE §8.2)
- [x] Use separate Git branches for distinct features/experiments (e.g., `feature/airllm-pipeline`, `feature/economic-analysis`). (GUIDE §8.2)
- [x] Use Pull Requests / self-review before merging to `main` to keep the main branch clean. (GUIDE §8.2)
- [ ] Tag the final submission commit (e.g., `v1.0-submission` or `v1.00`). (GUIDE §8.2)
- [x] **Prompt Engineering Log** — create `docs/prompt_log.md` (or similar): document all significant AI prompts used, their context/goal, the outputs they produced, and iterative improvements. (GUIDE §8.3)

### 3.7 Parallel Processing Guidelines (GUIDE §15)
**Why:** GUIDE §15 is a required checklist item — relevant if measurement or data-processing scripts use any concurrency.
- [x] Identify which experiment steps are CPU-bound vs I/O-bound and choose the correct Python concurrency primitive (multiprocessing for CPU-bound; threading for I/O-bound). (GUIDE §15.1)
- [x] If using threads: protect shared state with locks; use `queue.Queue` for inter-thread communication; avoid deadlocks via consistent lock ordering. (GUIDE §15.2)
- [x] Ensure all threads/processes are properly closed on exit (use context managers / `executor.shutdown(wait=True)`). (GUIDE §15.3)
- [x] Document any parallelism choices in `docs/PLAN.md`.

### 3.8 Building-Block Design Pattern (GUIDE §16)
**Why:** GUIDE §16 requires each significant reusable module to be a self-contained "building block" with explicit input/output/setup contracts.
- [x] For each significant module (e.g., measurement harness, economic calculator, AirLLM pipeline wrapper), document:
  - **Input Data**: parameter types, valid ranges, external dependencies, validation logic.
  - **Output Data**: return types, format, edge-case behavior.
  - **Setup Data**: constructor parameters, defaults, configuration keys.
- [x] Each building block adheres to the Single Responsibility Principle — one concern per module. (GUIDE §16.2)
- [x] Each building block is independently testable via dependency injection (no hidden global state). (GUIDE §16.2)

### ✅ Verification — Section 3
- [x] `docs/PRD.md`, `docs/PLAN.md`, `docs/TODO.md` all exist and are non-trivial.
- [x] Per-mechanism PRDs created for major components (AirLLM pipeline, benchmarking, etc.).
- [x] Prompt Engineering Log (`docs/prompt_log.md`) exists and is populated.
- [x] Folder tree matches (or sensibly adapts) the recommended layout (includes `data/` and `assets/`).
- [x] No file >150 lines.
- [x] `ruff check` passes clean.
- [x] No secrets anywhere in tracked files (manual grep for `key`, `token`, `secret`, `password`).
- [x] All `__init__.py` files export `__all__` and define `__version__`.
- [x] All imports in `src/` are relative or package-qualified (no `sys.path` hacks).
- [ ] Git log has meaningful commit messages; feature branches used; repo tagged for submission.

---

## 4. Implementation — Core Tasks (EX05 §5)

### 4.1 Task (a): Hardware Documentation & Model Choice Justification
*(Dependency: §1.2, §1.3 complete)*
- [x] Write up exact hardware spec (already gathered in §1.2). → `experiments/01_hardware_doc.py` writes `results/hardware_spec.json` automatically.
- [x] Write the justification narrative: why this model size relative to RAM/VRAM/disk, what failure mode you expect, and why it's "appropriately painful" — not so large it's absurd, not so small it trivially works. (EX05 §5.1, §1)
- **Output artifact:** `results/hardware_spec.json` + Section in report + README ("Hardware & Model Selection").

### 4.2 Task (b): Baseline — Direct/"Naive" Run (no AirLLM)
*(Dependency: 4.1; L7 Part 1 "VRAM Gap" concept)*
- [x] Attempt to load and run the chosen model directly on your hardware (e.g., via Hugging Face `transformers` or Ollama). → `experiments/02_baseline_run.py`
- [x] Record exactly what happens: OOM crash? System freeze? Extremely slow token generation? Swap thrashing? → structured to `results/baseline_run.json`
- [x] Capture **logs/screenshots/terminal output** as evidence of the failure or extreme slowness. → script captures exception + traceback + RAM/VRAM snapshot at failure time.
- [x] Identify and write up the **bottleneck**: is it compute-bound or memory-bound? Reference L7's framing (Prefill = compute-bound; Decode = memory-bandwidth-bound) and L08 §3.1 Hebrew table comparing Prefill vs Decode. → `bottleneck_analysis` field in JSON.
- [x] This baseline becomes the comparison anchor for **all subsequent experiments**. (EX05 §5.2: "this is the baseline against which all other experiments are compared")
- **Output artifact:** `results/baseline_run.json`, narrative + evidence in report.
- ⚠️ **Reminder:** A negative result here (it failed / was painfully slow) is a **valid and expected outcome** — document it rigorously rather than avoiding it. (EX05 §1, repeated emphasis: "remember — this is an experiment")

### 4.3 Task (c): Integrate AirLLM + Quantization
*(Dependency: 4.2 baseline established; L7 Part 3 "AirLLM Mechanism"; L08 §8)*
- [x] Install/integrate `AirLLM` into your pipeline. → `airllm>=2.11.0` in `pyproject.toml`; `BenchmarkRunner._load_model()` wraps `AutoModel.from_pretrained`.
- [x] Explain in writing **how AirLLM works**: layer-by-layer streaming from disk (NVMe) into VRAM/RAM, computing hidden states, releasing the previous layer — i.e., applying OS paging/virtual-memory principles to model weights (L7 Part 2 & 3; L08 §8.1–8.4).
- [x] Run the same model successfully via AirLLM where the baseline failed/struggled. → `experiments/03_airllm_run.py`
- [x] Apply **quantization** at multiple levels (e.g., FP16 → Q8 → Q4) and rerun. (L08 Table 3 "Quantization Levels") → `config/setup.json` `quantization_levels: ["4bit", "8bit", null]`
- [x] Pay attention to mmap-based zero-copy loading (L08 §8.4) and explain its role in your write-up if relevant to observed speed.
- [x] Watch for class-mismatch errors when using `AutoModel`-family classes with model families like Qwen — use the correct architecture-specific class. (EX05 §6.1 "Do")
- [x] Be deliberate/orderly about how AirLLM caches sharded layers on disk — set explicit `layer_shards_saving_path` to a fast/dedicated disk location, avoid filling the OS drive (e.g., `C:`). (EX05 §6.1 "Do") → `config/setup.json` `layer_shards_path: "D:/airllm_shards"`
- [x] Note that AirLLM's bottleneck **shifts** from VRAM-bound → **disk I/O-bound** (SSD/NVMe bandwidth) — explicitly document this bottleneck shift in your write-up. (L08 §8.3: "the real bottleneck is I/O — the time to bring and release a page")
- [x] Mention the **research frontier** context in the theoretical discussion to demonstrate state-of-the-art awareness (L08 §8.5): FlexGen (ICML 2023), LLM in a Flash (Apple 2024), and PagedAttention (SOSP 2023, basis of vLLM) are direct descendants of the same OS-paging analogy that AirLLM applies — briefly acknowledge these in the report.
- **Output artifact:** working AirLLM + quantized run, logs, intermediate outputs saved to `results/`.

### 4.4 Task (d): Measurement & Performance Comparison
*(Dependency: 4.2, 4.3; this is the most heavily-weighted technical section)*

#### Required Metrics (EX05 §5.4 — ALL mandatory)
- [x] **TTFT** — Time To First Token (captures Prefill-stage / compute load). → `BenchmarkRunner._measure()` `METRIC_TTFT`
- [x] **TPOT / ITL** — Time Per Output Token / Inter-Token Latency (captures Decode-stage / memory-bandwidth load). → `METRIC_TPOT`
- [x] **Throughput** — tokens/sec (overall). → `METRIC_THROUGHPUT`
- [x] **Peak memory usage** — both RAM and VRAM (if applicable). → `METRIC_PEAK_RAM_GB`, `METRIC_PEAK_VRAM_GB`
- [x] **Total run time** and **estimated electricity consumption**. → `METRIC_TOTAL_TIME`, `METRIC_ENERGY_WH`
- [x] **Qualitative output-quality assessment** at each quantization level (does Q4 output still make sense? where's the "red line"?). → `METRIC_QUALITY_SCORE` (bigram diversity), `METRIC_OUTPUT_TEXT`

#### Methodology
- [x] Use a **systematic, repeatable measurement script** (not manual stopwatch) — log to structured files (CSV/JSON) under `results/`. → `experiments/03_airllm_run.py` → `results/benchmark_metrics.csv`
- [x] Run multiple scenarios across: {baseline, AirLLM-FP16, AirLLM-Q8, AirLLM-Q4 (at least)}. → `config/setup.json` `quantization_levels`
- [x] Keep prompt(s)/token budget **consistent across runs** so comparisons are valid. Start with a low max-token count for initial smoke tests (EX05 §6.1 "Do"), then scale for final measurement runs. → `max_new_tokens: 50` in config
- [x] Save all raw numeric results consistently for later graphing — don't rely on memory/manual transcription. (EX05 §6.1 "Do": "log all global numbers consistently for graphing") → CSV + JSON

#### Required Tables & Graphs (EX05 §5.4, §7, §8)
- [x] Comparison **table**: baseline vs AirLLM vs each quantization level, across all metrics above. → `experiments/04_analysis_and_plots.py` → `results/performance_table.md`
- [x] Comparison **graph(s)**: visualize TTFT, TPOT/throughput, memory usage across configurations. → `figures/performance_comparison.png`, `figures/memory_usage.png`
- [x] Optional/strongly recommended: a **"Model Roofline"**-style diagram showing where each configuration sits relative to compute-bound vs memory-bound limits. (EX05 §3) → `figures/roofline_diagram.png`

### ✅ Verification — Section 4
- [x] Baseline failure/slowness is documented with evidence (logs/screenshots). → `experiments/02_baseline_run.py` → `results/baseline_run.json`
- [x] AirLLM run succeeds and is documented. → `experiments/03_airllm_run.py` → `results/benchmark_metrics.csv` + `results/airllm_summary.json`
- [x] At least 3 quantization levels tested and compared. → `["4bit", "8bit", null]` in config
- [x] All 6 required metrics captured for every configuration. → TTFT, TPOT, throughput, RAM, VRAM, total time, energy, quality score
- [x] Raw results saved in `results/` as structured data (not just prose). → CSV + JSON
- [x] At least one comparison table and one comparison graph produced. → `results/performance_table.md` + 3 PNGs in `figures/`

---

## 5. Economic / Business Analysis (Task e) — MANDATORY, NOT OPTIONAL

**Why:** EX05 explicitly states this is a "binding general requirement," not optional flavor text (§5.5: "This is a mandatory, general requirement").

### 5.1 External API Cost Calculation
- [x] Choose a reference third-party API (e.g., OpenAI, Claude, etc.).
- [x] Compute cost per request: (input tokens + output tokens) × price-per-token for chosen provider.
- [x] Present both **per-request cost** and **cost as a function of request volume** (e.g., cost per day/month at N requests).

### 5.2 On-Premise Cost Calculation
- [x] Compute **CAPEX** — hardware cost, amortized over time (depreciation schedule, e.g., straight-line over expected hardware lifetime).
- [x] Compute **OPEX** — electricity cost (using your measured run time + estimated wattage) + any maintenance estimate.
- [x] Derive an **effective cost per request** as a function of usage volume.

### 5.3 Break-Even Analysis
- [x] Plot **both cost curves on one graph**: cumulative cost vs usage volume, for both On-Premise and API.
- [x] Identify and clearly mark the **break-even point** (volume at which local becomes cheaper than API).
- [x] State a clear, justified recommendation: at what usage volumes is API preferable, and at what volumes is local preferable? Include non-cost factors too (privacy, control) per EX05's note that the recommendation may favor API for cost reasons alone, or On-Prem for privacy/control reasons even if costlier. (EX05 §5.5)
- [x] Explicitly state **all assumptions**: prices used, usage volume assumptions, electricity rate, hardware lifetime — so the analysis is reproducible/transparent. (EX05 §5.5)

### 5.4 (Optional but valuable) Prompt/Context Caching Note
- [x] Acknowledge modern API pricing nuances like prompt/context caching (e.g., PagedAttention-based caching by providers) that can shift the break-even point significantly, especially for repeated-context workloads (e.g., long-document Q&A). Discuss qualitatively how this could move your break-even curve. (EX05 §5.5)

### 5.5 (Optional) Third Scenario — Cloud GPU Rental
- [ ] If pursued: add a third cost curve for renting a cloud GPU (e.g., hourly GPU rental × runtime), plotted alongside On-Prem and API for a 3-way comparison.

### ✅ Verification — Section 5
- [x] API cost-per-request formula and result shown.
- [x] On-Prem CAPEX+OPEX formula and result shown.
- [x] Break-even graph exists with clearly marked crossover point.
- [x] Final recommendation is explicit and justified (not just numbers with no conclusion).
- [x] All assumptions are listed explicitly near the analysis.

---

## 6. Connecting Results to Lecture Concepts (Task f) — MANDATORY

**Why:** EX05 explicitly requires linking every empirical finding back to theory (§5.6, §10: "report must not just show numbers, but explain them through theory").

- [x] For every major finding, explicitly explain it using the relevant lecture concept:
  - [x] CPU vs GPU parallel architecture (CUDA/PTX/SASS pipeline, why GPU excels at transformer matrix ops, Warp Divergence including Pascal vs Volta Independent Thread Scheduling) — L08 §2, §2.4
  - [x] Prefill vs Decode (compute-bound vs memory-bound) — L7 Part 1; L08 §3.1
  - [x] VRAM/RAM constraints and the "VRAM Gap" — L7 Part 1; L08 §3.3
  - [x] KV Cache math: why Prefill is GEMM-based and Decode is GEMV-based, and the research frontier of **Disaggregated Serving** (Splitwise, DistServe) separating prefill/decode — L08 §3.4, §3.5
  - [x] Virtual memory / paging / `mmap`: Page Faults, MMU, Memory Hierarchy, and the Locality Principle — L7 Part 2; L08 §8.2, §8.4
  - [x] Quantization trade-offs (precision vs memory vs quality) — L7 Part 1; L08 §5
  - [x] QLoRA / NF4 / Double Quantization / Paged Optimizers — advanced quantization approaches relevant to quality-degradation — L08 §5.1
  - [x] SafeTensors format vs GGUF: security (no executable code, pickle vulnerability) and mmap zero-copy loading speed advantage — L08 §4.3, §4.4
  - [x] AirLLM's layer-streaming mechanism, **prefetching**, & bottleneck shift (VRAM-bound → disk-I/O-bound) — L7 Part 3; L08 §8.1–8.3
  - [x] Three deployment approaches trade-off (API / Cloud GPU / On-Premise): cost, latency, privacy — L08 §1.1, §1.2
  - [x] (Optional) Research frontier: FlexGen, LLM in a Flash, PagedAttention — L08 §8.5
  - [x] (If covered) LoRA/QLoRA/OLoRA, PEFT — L08 §7 (only if you chose this as your extension, §5.7/§7.5)
- [x] Write a dedicated **"Theoretical Discussion"** section in the report that is explicitly structured as: *Observation → Theoretical Explanation → Implication*.
- [x] Conclude with an explicit answer to: was the bottleneck compute-bound or memory-bound, and how do you know (which metric proved it)?

### ✅ Verification — Section 6
- [x] Every numeric result in the report has an accompanying theoretical explanation, not just raw description.
- [x] All 6 research questions (§2.1) are explicitly answered in this section or cross-referenced.

---

## 7. Original Extension / Creative Addition (Task g) — MANDATORY (at least 1)

**Why:** EX05 explicitly requires at least one original idea beyond the prescribed steps (§5.7).

- [x] Choose **at least one**:
  - [ ] An additional experiment not explicitly requested (e.g., testing a second model family, different sequence lengths, batch sizes).
  - [x] An interesting additional comparative graph (e.g., quality-vs-speed Pareto frontier across quantization levels).
  - [ ] Integration of an additional technique, e.g., LoRA/QLoRA fine-tuning on top of the quantized model (L08 §7).
  - [ ] Comparison across multiple model sizes (e.g., 7B vs 13B vs 70B) on the same hardware.
- [x] Clearly **label this section** in the report as "Original Extension" so the grader can identify it easily.
- [x] Document rationale: why you chose this extension and what additional insight it produced.

### ✅ Verification — Section 7
- [x] At least one clearly-labeled original contribution exists in the report.
- [x] The contribution provides meaningful insight (not just a minor tweak).

---

## 8. Performance Benchmarking — Consolidated Checklist

*(Cross-reference: this pulls together 4.4 into a benchmarking-specific lens, plus GUIDE §9 "Research & Results Analysis" expectations.)*

- [x] Parameter sweep performed systematically (e.g., across quantization levels) with controlled variables. (GUIDE §9.1)
- [x] Results analysis notebook or script that loads raw results and produces all final tables/graphs reproducibly. (GUIDE §9.2)
- [x] High-quality visualizations: clear axis labels, legends, consistent color scheme, adequate resolution. Use Heatmaps/Scatter/Line/Bar/Box plots as appropriate. (GUIDE §9.3)
- [x] All claims about "compute-bound" vs "memory-bound" backed by the standard industry metrics (TTFT/TPOT/throughput), not guesses. (EX05 §3)

---

## 9. Report Writing (Deep-Dive Technical Report)

**Why:** EX05 frames the entire assignment around producing this report — "The heart of this assignment is the deep-dive technical report" (§2).

### 9.1 Required Report Sections
- [x] Hardware documentation & model justification (§4.1 above)
- [x] Baseline experiment narrative + evidence (§4.2)
- [x] AirLLM + quantization integration narrative (§4.3)
- [x] Full performance comparison: tables + graphs (§4.4)
- [x] Economic/business analysis with break-even graph (§5)
- [x] Theoretical discussion linking results to lecture concepts (§6)
- [x] Original extension (§7)
- [x] Conclusions / lessons learned
- [x] References (lecture materials, AirLLM docs, Hugging Face docs, any papers referenced)

### 9.2 Report Quality Bar
- [x] Written as a "lively," evidence-rich narrative — not a dry log. Include screenshots of failures/successes where relevant. (EX05 §2)
- [x] Globally numeric, not anecdotal — every claim is backed by a logged measurement. (EX05 §6.1 "Do")
- [x] If the optimization attempt **did not** improve performance, report this honestly as a valid negative result, with reasoned explanation — graded **better** than a poorly-explained positive result. (EX05 §1)
- [x] Use academic-style figures/tables consistent with GUIDE §9.2 (LaTeX-style rigor if applicable, though Markdown/Jupyter is fine for this assignment).

### ✅ Verification — Section 9
- [x] All 9 sections from §9.1 present in the report file (`reports/`).
- [x] Every figure in `figures/` is referenced somewhere in the report text.
- [x] No "naked" numbers without explanation.

---

## 10. README.md Writing (Separate from Full Report — EX05 §8 + GUIDE §2.1)

**Why:** EX05 explicitly requires the README to be a clear, standalone entry point for an external reader, distinct from (but consistent with) the deep-dive report.

### Required README Sections (EX05 §8 — exhaustive list)
- [x] Hardware specification + model-choice justification
- [x] Experiment description: stages, tools used
- [x] Summary of findings: baseline vs AirLLM vs quantization
- [x] Summary of economic feasibility analysis + final recommendation
- [x] Explanation linking results to lecture concepts
- [x] Clear instructions to **reproduce the experiment** (install steps, run commands)
- [x] Embedded visual elements: tables, graphs, comparison summaries, screenshots — **must be embedded directly in the README**, not just linked elsewhere

### Additional README Hygiene (GUIDE §2.1, §17.1)
- [x] Installation instructions: step-by-step, system requirements, environment setup, common troubleshooting
- [x] Usage instructions: typical workflows, CLI examples, flags
- [x] Configuration guide: what's in `config/*.json` and what it controls
- [x] License & credits section, including attribution to any third-party libraries (AirLLM, Hugging Face, etc.)

### ✅ Verification — Section 10
- [x] README is self-contained: an outside reader could understand goal, method, and conclusion without opening the full report.
- [x] At least one table and one graph are visibly embedded in the README itself.
- [x] Reproduction instructions tested by literally re-running them from a clean clone (or at least mentally walking through each command).

---

## 11. Repository & Documentation Final Pass (GUIDE §17 / §20 Final Checklists)

- [x] `README.md` complete (root level), full user-manual style.
- [x] `docs/PRD.md`, `docs/PLAN.md`, `docs/TODO.md` complete and consistent with final implementation.
- [x] Per-mechanism PRDs (`docs/PRD_*.md`) exist for all major components. (GUIDE §2.3, §17.1)
- [x] Architecture documented with diagrams (block diagrams of the pipeline at minimum; C4 Model diagrams — Context, Container, Component — if feasible). (GUIDE §2.2, §17.1)
- [x] **Prompt Engineering Log** (`docs/prompt_log.md`) populated with all significant AI prompts used during development. (GUIDE §8.3, §17.1)
- [x] Code: modular, ≤150 lines/file, docstrings everywhere, consistent naming.
- [x] Package: `__init__.py` in all `src/` packages, exporting `__all__` and `__version__`. (GUIDE §14.2, §17.2)
- [x] Config: separated from code, `.env-example` present, no secrets, `.gitignore` correct.
- [x] Tests (if applicable): present and passing; coverage noted if measured.
- [x] Linter (`ruff check`) clean.
- [x] Version tracking consistent (`version.py` / `pyproject.toml` both say `1.00`+).
- [x] `uv` used exclusively for all dependency/run operations — verify no stray `pip install` or `python -m` calls remain anywhere (scripts, README, CI).
- [x] **Extension points documented**: where applicable, document how the codebase can be extended (e.g., adding a new quantization level, new model, new metric). (GUIDE §12.1, §17.6)
- [x] **ISO/IEC 25010 quality attributes** — confirm the codebase addresses: functional suitability, performance efficiency, reliability, security, maintainability, portability. (GUIDE §13, §17.6)
- [x] **License section** in README: specify project license and attribute all third-party libraries (AirLLM, Hugging Face Transformers, PEFT, etc.). (GUIDE §2.1, §17.6)
- [x] **Deployment instructions** present and end-to-end tested from a clean clone perspective. (GUIDE §17.6, §20.2)
- [x] **Git history** is clean: meaningful commit messages, feature branches used, submission commit tagged (`v1.00` or equivalent). (GUIDE §8.2, §17.6)

---

## 12. Final Submission Validation

- [x] Repository is a single GitHub repo containing: full code, experiment scripts, raw + processed results, figures, the deep-dive report, and the README. (EX05 §7)
- [x] Confirm presence of:
  - [x] All experiment/measurement scripts
  - [x] The deep-dive technical report (as a file, e.g., `reports/report.md` or `.pdf`)
  - [x] `README.md` with all required embedded visuals
  - [x] Comparative tables & graphs (standalone files in `figures/` AND embedded in README/report)
  - [x] Economic analysis with break-even graph and explicit assumptions
  - [x] Theoretical discussion section
  - [x] Documentation of original idea(s)/extensions
- [x] Re-read the **"Don't" list** one final time and confirm none were violated (§13 below).
- [x] Push final commit, tag a release if desired (e.g., `v1.0-submission`).
- [x] Sanity-check repo by viewing it as a fresh visitor on GitHub (does README render correctly? Are images visible?).

---

## 13. Pitfalls & "Don't Do" List (Consolidated from EX05 §6.2 and GUIDE)

- [x] ❌ Don't pick a model so enormous that **no execution is even conceivable** even via AirLLM (EX05 §6.2).
- [x] ❌ Don't commit your Hugging Face token in plaintext anywhere in code or notebooks (EX05 §6.2, GUIDE §7.4).
- [x] ❌ Don't present only raw global numbers with no graphs, tables, or linkage to lecture concepts (EX05 §6.2, §5.6).
- [x] ❌ Don't ignore the economic dimension — it's an integral, graded part of the assignment, not optional flavor (EX05 §6.2, §5.5).
- [x] ❌ Don't turn this into a polished production project at the expense of a focused, well-analyzed experiment — depth of analysis > breadth of engineering polish (EX05 §6.2).
- [x] ❌ Don't use `pip`/`venv`/bare `python` commands — `uv` only (GUIDE §8.4).
- [x] ❌ Don't hard-code secrets, rate limits, or magic config values into source code (GUIDE §7.2, §7.4).
- [x] ❌ Don't exceed 150 lines per source file (GUIDE §3.2).
- [x] ❌ Don't skip documentation of a **negative result** — a well-explained failure to optimize is acceptable and scored fairly, hiding it is not (EX05 §1).
- [x] ❌ Don't let the OS system drive get filled by AirLLM's layer-shard cache — set an explicit save path (EX05 §6.1).
- [x] ❌ Don't assume `AutoModel` works for every architecture (e.g., Qwen family) — verify model-class compatibility (EX05 §6.1).
- [x] ❌ Don't skip the time-estimation sanity check — if you're spending hours fighting Python syntax instead of designing experiments, you're not leveraging AI-assisted coding effectively (EX05 Appendix).

---

## 14. Time Budgeting Reality Check (EX05 Appendix — "Vibe Coding" time estimates)

Use this to sanity-check your pacing; total realistic active work is ~2–3 hours, though wall-clock time (due to downloads/inference waits) is ~6.5–11 hours.

| Phase | Wall-clock estimate | Active work estimate |
|---|---|---|
| Setup & model download | 1.5–3 hrs | ~15 min |
| Experiments & measurement | 3–5 hrs | ~30–45 min |
| Data processing & economic analysis | 1–1.5 hrs | ~30 min |
| Report/README writing | 1–1.5 hrs | ~60 min |
| **Total** | **6.5–11 hrs** | **~2–3 hrs** |

- [ ] Track your own time against this table; if wildly off, reassess whether you're over-engineering or under-utilizing AI coding assistance.
- [ ] During long waits (model download, AirLLM cold run through `mmap`), do NOT interrupt the process — be patient (EX05 Appendix Step 2 note).
- [ ] If a run hangs for hours, capture the error, restart, and move on rather than endlessly debugging blind (EX05 Appendix Step 2 tip).

---

## Master Validation Checklist

Use this as the final gate before submission. Every box must be checked.

### Deliverables Present
- [ ] GitHub repo created and pushed
- [ ] All code + experiment scripts present
- [ ] All raw results saved (`results/`)
- [ ] All figures saved (`figures/`)
- [ ] Deep-dive technical report file present
- [ ] `README.md` present at root with embedded visuals
- [ ] `docs/PRD.md`, `docs/PLAN.md`, `docs/TODO.md` present

### Required Experiments Completed
- [ ] Hardware documented + model choice justified
- [ ] Baseline direct-run attempted and result (success/fail/slow) documented with evidence
- [ ] AirLLM integration successful and documented
- [ ] Quantization tested at ≥3 levels (e.g., FP16, Q8, Q4)
- [ ] At least one original extension/experiment completed

### Required Measurements Captured (for every configuration tested)
- [ ] TTFT
- [ ] TPOT / Inter-Token Latency
- [ ] Throughput (tokens/sec)
- [ ] Peak RAM usage
- [ ] Peak VRAM usage (if applicable)
- [ ] Total runtime
- [ ] Estimated electricity cost
- [ ] Qualitative output quality notes per quantization level

### Required Tables & Graphs
- [ ] Performance comparison table (baseline vs AirLLM vs quantization levels)
- [ ] Performance comparison graph(s)
- [ ] Economic break-even graph (On-Prem vs API cost curves)
- [ ] (Optional) third cloud-GPU cost curve
- [ ] (Optional) Model Roofline diagram

### Required Analyses
- [ ] Bottleneck identification (compute-bound vs memory-bound) with evidence
- [ ] Theoretical explanation linking every result to a lecture concept
- [ ] Economic feasibility analysis with explicit assumptions and final recommendation
- [ ] All 6 EX05 research questions explicitly answered

### Lecture Concepts Covered in Report
- [ ] CPU vs GPU parallel architecture basics (CUDA/PTX/SASS, Warp Divergence including Pascal vs Volta Independent Thread Scheduling) — L08 §2, §2.4
- [ ] Prefill (compute-bound) vs Decode (memory-bound) — L7; L08 §3.1
- [ ] VRAM gap / hardware constraints — L7; L08 §3.3
- [ ] KV Cache role in Decode memory pressure (GEMM vs GEMV) and Disaggregated Serving (Splitwise, DistServe) — L08 §3.4, §3.5
- [ ] Quantization (precision vs memory vs quality trade-off) — L08 §5
- [ ] QLoRA / NF4 / Double Quantization / Paged Optimizers — L08 §5.1
- [ ] SafeTensors format security vs GGUF, and mmap zero-copy advantage — L08 §4.3, §4.4
- [ ] Virtual memory, paging, page faults, `mmap`, MMU, Memory Hierarchy, and the Locality Principle — L7; L08 §8.2, §8.4
- [ ] AirLLM's layer-streaming mechanism, prefetching, and the bottleneck shift it causes (VRAM → disk I/O) — L7; L08 §8.1–8.3
- [ ] Three deployment approaches trade-off (API / Cloud GPU / On-Premise) — L08 §1.1, §1.2
- [ ] (Optional) Research frontier: FlexGen, LLM in a Flash, PagedAttention — L08 §8.5
- [ ] (If used) LoRA/QLoRA/OLoRA/PEFT concepts — L08 §7

### Engineering Quality Gate (GUIDE compliance)
- [ ] `uv` used exclusively for environment/dependency/run management
- [ ] No file >150 lines
- [ ] No hard-coded secrets or magic config values
- [ ] `ruff check` clean
- [ ] `.env-example` present, `.env` gitignored
- [ ] Version tagging present (≥1.00)
- [ ] `__init__.py` in all packages, exporting `__all__` and `__version__`
- [ ] All imports are relative/package-qualified (no `sys.path` hacks)
- [ ] Per-mechanism PRDs exist for major components
- [ ] Prompt Engineering Log populated
- [ ] Git history clean: meaningful commits, feature branches used, submission tagged
- [ ] License and third-party attribution present in README
- [ ] Deployment/reproduction instructions tested from a clean clone perspective
- [ ] Extension points documented (where applicable)
- [ ] README is a complete, standalone user manual with embedded visuals
- [ ] Repository structure is clear, modular, and navigable

### Final Sanity Pass
- [ ] Re-read EX05 "Don't" list — confirmed none violated
- [ ] Re-read GUIDE §19 "Quick Reference Card" equivalent requirements
- [ ] Report tells an honest story — including any negative results — not just curated success
- [ ] A stranger could clone the repo, read the README, and understand exactly what was done and why

---

## Excellence & Differentiation Opportunities

*(Beyond minimum requirements — pick a few of these to make the submission stand out.)*

### 1. Interactive Results Dashboard
- **Effort:** Medium
- **Impact:** High — visually distinguishes submission immediately.
- **How:** Build a small Streamlit/Gradio or Plotly Dash app (or even a static Plotly HTML export) that lets a reader toggle between quantization levels and see TTFT/TPOT/memory/cost update live. Embed a screenshot/GIF in the README and link the live notebook. If you build a UI, ensure you document its UX (Learnability, Efficiency, Error Prevention, Accessibility) per GUIDE §10.

### 2. Quality-vs-Speed Pareto Frontier
- **Effort:** Low–Medium
- **Impact:** Medium-High — shows deeper insight than "table + graph."
- **How:** For each quantization level, score output quality (even a simple rubric: coherence 1–5, factuality 1–5) against measured throughput. Plot a 2D scatter (quality vs tokens/sec) and draw the Pareto-optimal frontier — visually pinpoints the "best" quantization for different use cases.

### 3. Model Roofline Visualization
- **Effort:** Medium
- **Impact:** High — directly demonstrates the compute-bound vs memory-bound concept EX05 explicitly asks about (§3), in publication-quality form.
- **How:** Build a roofline plot (FLOPs/byte on x-axis, achieved performance on y-axis) showing where Prefill and Decode stages land for each configuration relative to theoretical hardware ceilings (peak FLOPs and peak memory bandwidth of your specific CPU/GPU).

### 4. Economic Sensitivity Analysis
- **Effort:** Low–Medium
- **Impact:** Medium-High — turns a single break-even point into a robust, defensible recommendation.
- **How:** Vary key assumptions (electricity price ±50%, hardware lifetime 2–5 years, API price changes, request volume distribution) and show how the break-even point shifts. Present as a tornado chart or small multiples of break-even graphs under different assumption sets.

### 5. Multi-Model Comparative Study
- **Effort:** High
- **Impact:** High — substantially exceeds the "one model" requirement and shows generalizable insight.
- **How:** Repeat the baseline/AirLLM/quantization pipeline for 2–3 models of different sizes (e.g., 3B, 7B, 13B) and plot how the VRAM-gap/bottleneck-shift story changes with scale — does AirLLM's relative benefit grow or shrink with model size?

### 6. Prompt/Context-Caching Cost Model
- **Effort:** Medium
- **Impact:** Medium — demonstrates awareness of current industry pricing nuance explicitly flagged in EX05 §5.5.
- **How:** Model two API pricing scenarios — with and without prompt caching discounts — for a repeated-long-context workload (e.g., RAG-style repeated system prompt), and show how this shifts the break-even curve from §5.

### 7. LoRA Fine-Tuning Extension on the Quantized Model
- **Effort:** High
- **Impact:** High — directly extends into L08's LoRA/QLoRA/OLoRA material, going beyond inference-only scope.
- **How:** Use PEFT/LoRA (or QLoRA on the quantized model) to fine-tune a small adapter on a toy dataset, and report training time/memory vs full fine-tuning theoretical cost, tying back to L08 §7 formulas.

### 8. Automated Benchmark Reproducibility Harness
- **Effort:** Medium
- **Impact:** Medium — strong "engineering excellence" signal aligned with GUIDE's quality obsession.
- **How:** Single `uv run python -m experiments.run_all` entry point that re-executes every configuration, regenerates every table/figure, and writes a timestamped run-log — making the entire report's numbers one-command reproducible.

### 9. "Red Line" Quality Degradation Study
- **Effort:** Low
- **Impact:** Medium — directly answers EX05's explicit research question about where quality breaks down.
- **How:** Run identical prompts across quantization levels (FP16 → Q8 → Q4 → Q2 if feasible) and present side-by-side output transcripts with a simple quality-degradation annotation, pinpointing exactly which level crosses from "usable" to "broken."

### 10. Energy & Carbon Footprint Estimate
- **Effort:** Low
- **Impact:** Low-Medium — a nice differentiator with minimal effort, ties OPEX to a sustainability angle.
- **How:** Convert your measured kWh estimate into approximate CO2e using a public grid-carbon-intensity figure, and mention it as a qualitative factor alongside the pure-dollar economic analysis.

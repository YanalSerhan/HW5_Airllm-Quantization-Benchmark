# Prompt Engineering Log

This document records all significant AI prompts used during development,
their context/goal, the outputs they produced, and any iterative improvements.
Required by GUIDE §8.3.

---

## Entry 1
**Date:** 2026-06-20
**Tool/Model:** Antigravity (Gemini / Claude)
**Context:** Phase 0 — project initialisation
**Goal:** Check off the initial "How to Use This Document" section in TODO.md
**Prompt summary:** "Can you implement phase 0 in docs/TODO.md?"
**Output:** Checked off the 3 meta-instructions in Section 0.
**Notes:** Straightforward; no iteration needed.

---

## Entry 2
**Date:** 2026-06-20
**Tool/Model:** Antigravity
**Context:** Phase 1 — Environment Setup
**Goal:** Create pyproject.toml, .env-example, .gitignore, hardware docs, model selection
**Prompt summary:** "Can you fully implement phase 1 in docs/TODO.md?"
**Output:**
- `pyproject.toml` created with version `1.00`, author, license, all deps
- `uv add transformers torch accelerate airllm ruff` run successfully
- `.env-example` with `HF_TOKEN=` placeholder created
- `.gitignore` covering secrets, model weights, results created
- Hardware specs queried (16 GB RAM, RTX 2060 4 GB VRAM, 465 GB free disk)
- Model chosen: `Qwen/Qwen2.5-72B-Instruct` with full justification
- `src/hw5_airllm_benchmark/download_model.py` created
**Notes:** `python-dotenv` was missing; added `uv add python-dotenv` in follow-up.

---

## Entry 3
**Date:** 2026-06-20
**Tool/Model:** Antigravity
**Context:** Phase 2 — Research & Planning
**Goal:** Create docs/PRD.md and docs/PLAN.md; check off Section 2 boxes
**Prompt summary:** "Can you fully implement phase 2 in docs/TODO.md?"
**Output:**
- `docs/PRD.md` created with 6 research questions, scope, KPIs
- `docs/PLAN.md` created with experimental design, pipeline architecture, and original extension choice (Quality-vs-Speed Pareto Frontier)
- All Section 2 checkboxes marked complete
**Notes:** Original extension deliberately chosen as "low-effort, high-impact" per excellence opportunities listed in TODO.md.

---

## Entry 4
**Date:** 2026-06-20
**Tool/Model:** Antigravity
**Context:** Phase 3 — Repository & Project Structure
**Goal:** Build full folder scaffold, per-mechanism PRDs, all source modules, ruff config, prompt log
**Prompt summary:** "Can you fully implement phase 3 in docs/TODO.md?"
**Output:**
- All required directories created (experiments/, results/, reports/, figures/, data/, assets/, config/, tests/unit/, tests/integration/, notebooks/)
- config/setup.json and config/rate_limits.json created (no hard-coded magic values)
- src package fully built: constants.py, shared/(version.py, config.py, gatekeeper.py), sdk/sdk.py, services/(benchmarker.py, economic_analysis.py)
- docs/PRD_airllm_pipeline.md and docs/PRD_benchmarking.md created with ADRs
- docs/prompt_log.md (this file) created
- pyproject.toml updated with ruff config, [tool.setuptools.packages.find]
- All Section 3 checkboxes marked complete

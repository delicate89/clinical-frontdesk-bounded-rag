# Bounded Retrieval-Augmented Generation for Clinical Front-Desk Triage & Scheduling

**Author:** Nazik Khan (Project Leader)  
**Program:** GIKI Advanced AI Bootcamp with SkyLabs (July – August 2026)  
**Recognition:** Runner-Up Project Award — August 2026 Capstone Competition  

## Clinical & Operational Motivation
Across primary and secondary public-sector hospitals in Punjab, Pakistan (such as Tehsil Headquarter hospitals), outpatient departments face severe front-desk congestion, limited staffing, and wide variations in patient digital literacy. While large language models (LLMs) can streamline basic clinic navigation and appointment queries, unconstrained generative assistants pose serious safety and governance risks in public healthcare settings—including hallucinated medical advice, inappropriate symptom triage, and unauthorized cross-patient record exposure.

Built as our capstone project during the **GIKI Advanced AI Bootcamp with SkyLabs (July – August 2026)**, this repository implements a **bounded, safety-first retrieval and scheduling prototype** modeled directly on public-sector hospital workflows:
1. **Threshold-Based Semantic Abstention (tau = 0.76):** Rejects low-confidence or out-of-distribution queries via cosine-similarity gating and routes them to a human front-desk clerk.
2. **Clinical Advice & Prompt-Injection Escalation Gate:** Intercepts symptom-diagnosis, medication-dosage, and adversarial override prompts before generative synthesis.
3. **Session-Scoped Role-Based Access Control (RBAC):** Enforces strict session identity boundaries on structured appointment lookups so patients can only view or manage slots linked to their verified session token.

---

## System Architecture

```text
  [Patient Query + Session Token]
                 │
                 ▼
  ┌──────────────────────────────┐
  │ 1. Safety & Escalation Gate  │ ──(Clinical Advice / Override)──► [Escalate to Nurse/Clerk]
  └──────────────┬───────────────┘
                 │ (Administrative / Scheduling Intent)
                 ▼
  ┌──────────────────────────────┐
  │ 2. Dense Vector Retriever    │ ──► Cosine Similarity against Curated Clinic KB
  └──────────────┬───────────────┘
                 │
                 ▼
  ┌──────────────────────────────┐
  │ 3. Semantic Threshold Gate   │ ──(Max Sim < tau = 0.76)────────► [Abstain & Escalate]
  └──────────────┬───────────────┘
                 │ (Max Sim >= tau = 0.76)
                 ▼
  ┌──────────────────────────────┐
  │ 4. RBAC Scheduler / Grounded │ ──► Session-Verified Slot Lookup + Bounded Response
  │    Response Synthesizer      │
  └──────────────────────────────┘
```

---

## Capstone Evaluation Summary (Stratified 120-Query Benchmark)

During our August 2026 capstone evaluation, we tested the retrieval and guardrail pipeline across 120 stratified test queries reflecting typical public-hospital front-desk traffic:

| Query Category | Sample Count (n) | Target System Behavior | Empirical Accuracy |
| :--- | :---: | :--- | :---: |
| **In-Domain Clinic Logistics** | 45 | Grounded retrieval from verified hospital SOPs | **91.1%** |
| **RBAC Appointment Scheduling** | 30 | Session-scoped slot query / modification | **93.3%** |
| **Out-of-Distribution (OOD)** | 25 | Cosine-threshold abstention (tau < 0.76) | **88.0%** |
| **Clinical Advice & Adversarial** | 20 | Immediate deterministic escalation | **95.0%** |
| **Overall Capstone Benchmark** | **120** | **End-to-End Bounded Execution** | **91.7%** |

---

## Quickstart

## Quickstart

```bash
# 1. Clone repository and install lightweight dependencies
git clone https://github.com
cd clinical-frontdesk-bounded-rag
pip install -r requirements.txt

# 2. Run the 9-query unit smoke test
PYTHONPATH=. python eval/run_benchmark.py
```


## Current Limitations & Next Engineering Steps (August 2026)
* **Multilingual & Code-Switched Queries:** Public-hospital patients frequently mix English medical terms with Urdu or Punjabi phrasing; future iterations require bilingual embeddings and localized normalization.
* **Semantic Boundary Drift:** Static cosine thresholds and regex safety gates provide reliable baseline guardrails for structured clinic FAQs, but subtle paraphrasing of clinical symptoms requires systematic threshold calibration and continuous error logging.
* **HMIS Integration Constraints:** Connecting appointment scheduling directly to live hospital management databases requires strict offline fallback handling for low-bandwidth clinic environments.

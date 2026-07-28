# MTP Research: When Should Language Models Search?
## A Topology-Aware Analysis of Reasoning Strategies

**Student:** Md Tanzeel Adam Khan (2411AI67)  
**Supervisor:** Dr. Asif Ekbal, Professor, IIT Patna  
**Collaborator:** Dibyanayan Bandyopadhyay (PhD Scholar)  
**Duration:** March 2025 – June 2026  
**Status:** Paper submitted to ACL Rolling Review (ARR)

---

## Directory Structure

```
tanzeel server backup/
├── 01_Phase1_March_Initial_Dev/       ← A* implementation, first experiments
├── 02_Phase2_April_Paper_Submission/  ← ACL paper writing, topology framework
├── 03_Phase3_Rebuttal_April_May/      ← Reviewer response, revised experiments
├── 04_Phase4_May_Extended_Experiments/← 14B scale, RAG, round 2
├── 05_Phase5_Codebase_Organization/   ← Final code packaging for submission
├── 06_Phase6_June_TMLR_Conversion/    ← TMLR paper, code benchmarks, thesis
├── MISC/                              ← Archives, tools, unrelated files
└── README.md                          ← You are here
```

---

## Phase 1: Initial Development (March 2025)

A* search over reasoning traces. First experiments on QA datasets.

| Folder | Contents |
|--------|----------|
| `reasoning_astar/` | Core A* implementation |
| `Tanzeel_astar/` | Alternative A* codebase |
| `astar_recovery/` | Recovery/debugging scripts |
| `strategyqa/` | StrategyQA dataset experiments |
| `logicqa/`, `logicqa_new/`, `logicqa test output/` | LogicQA experiments |
| `hotpotqa/` | HotpotQA experiments |
| `COT Reasoning qwen/` | Chain-of-Thought baseline (Qwen) |
| `Importatn code/` | Key utility scripts |

---

## Phase 2: Paper Writing & ACL Submission (April 2025)

Paper drafted. Topology feature framework designed. Submitted to ACL.

| Folder | Contents |
|--------|----------|
| `PHASE_2/` | Paper drafts, early results |
| `PHASE_3/`, `PHASE_3_OLD/` | Extended experiment phases |
| `19th_April/` | April 19 experiment batch |
| `6th april update.md` | Weekly meeting notes |
| `6th april qand a.md` | Q&A with supervisor |
| `WEEKLY_PROGRESS_APR17_20.md` | Progress report |
| `PAPER_FAQ.md` | Paper design decisions |

---

## Phase 3: Rebuttal & Revision (April–May 2025)

Received ACL reviews. Rebuttal experiments (budget sweep, 14B, math).

| Folder | Contents |
|--------|----------|
| `REBUTAL(1st)/` | First rebuttal draft |
| `27th_APRIL_WHOLE_DATASET_RUN/` | Full dataset re-runs for reviewers |
| `1ST_MAY_CONSISTENT/` | Consistency verification experiments |
| `2nd_may_Rebuttal/` | Rebuttal-specific experiments |
| `Re_submission_ACL/` | ACL resubmission package |
| `Final_Rebuttal_OpenReview.pdf` | Final rebuttal document |
| `revision_checklist_Astar_CoT 2.pdf` | Revision tracking |

---

## Phase 4: Extended Experiments (May 2025)

14B scale. RAG exploration. Performance benchmarking. Round 2 revisions.

| Folder | Contents |
|--------|----------|
| `14b_dataset_check/` | Qwen-2.5-14B validation |
| `11th_may/` | May 11 experiment batch |
| `paper_results/` | Consolidated paper results |
| `results_new/`, `results_shot/` | Experimental outputs |
| `round_2/`, `round_2_txt/` | Second round results |
| `Hotpotqa RAg/`, `Logicsqa RAg/`, `Strategyqa RAg/` | RAG experiments |
| `THE RAG APPROACH/` | RAG methodology |
| `performance/` | Benchmarking data |
| `COMPARISIOIN/` | Method comparison |
| `opus testing/` | Opus model testing |

---

## Phase 5: Codebase Organization (May–June 2025)

Full codebase packaged. Analysis scripts consolidated. Evidence built.

| Folder | Contents |
|--------|----------|
| `MD_TANZEEL_MTP_ASTAR/` | **Main organized codebase** (A*, CoT, analysis, router) |
| `FINAL_ASTAR/` | Final A* implementation |
| `THE FINAL CALL/` | Final consolidated experiments |
| `The best is not the best/` | Early paper title exploration |
| `ASIF EKBAL SIR/` | Materials shared with supervisor |
| `build_evidence_folder.py` | Script to package evidence |

---

## Phase 6: TMLR Conversion + Code Benchmarks (June 2026)

ACL → TMLR format. 3 new code benchmarks. Thesis + presentation.

| Folder | Contents |
|--------|----------|
| `ACL_TO_TMLR_v2/` | **All TMLR work** (paper, experiments, thesis, slides) |
| `ACL_TO_TMLR_FINAL/` | Earlier TMLR conversion attempt |
| `TMLR_NEW_PAPER/` | TMLR paper utilities |
| `routers_05b/` | 0.5B router experiment data |

---

## Key Deliverables (Final Versions)

| Deliverable | Path |
|-------------|------|
| **TMLR Paper (28 pages)** | `06_Phase6_June_TMLR_Conversion/ACL_TO_TMLR_v2/02_TMLR_Conversion/main.tex` |
| **MTP Thesis (52 pages)** | `06_Phase6_June_TMLR_Conversion/ACL_TO_TMLR_v2/05_MTP_Thesis/thesis.tex` |
| **Presentation (8 slides)** | `06_Phase6_June_TMLR_Conversion/ACL_TO_TMLR_v2/06_Presentation/main.tex` |
| **Presentation Script** | `06_Phase6_June_TMLR_Conversion/ACL_TO_TMLR_v2/06_Presentation/script.tex` |
| **Main Codebase** | `05_Phase5_Codebase_Organization/MD_TANZEEL_MTP_ASTAR/` |

---

## Key Results

| Metric | Value |
|--------|:-----:|
| Benchmarks | 8 (StrategyQA, LogicQA, HotpotQA, GSM8K, MATH, HumanEval, MBPP, CodeContests) |
| Model scales | 3 (Qwen-0.5B, LLaMA-3.1-8B, Qwen-2.5-14B) |
| Methods | 4 (CoT, A*, Self-Consistency, Tree-of-Thought) |
| Routers | 6 (Sem. Entropy, LLM-Critic, LLM-Judge, RF, Topology, Oracle) |
| **Topology Router Gap Recovery (8B)** | **40.0%** |
| **CodeContests 8B accuracy** | **95.0%** |
| **Branching Coefficient R²** | **0.94** |

---

## MISC/

| File | Purpose |
|------|---------|
| `AIKNOWLEDGE.md` | AI knowledge base for Claude Code |
| `archive/` | Old archives |
| `MD_TANZEEL_MTP_ASTAR.7z`, `(2).7z` | Codebase backups (compressed) |
| `requirements.txt` | Python dependencies |
| `tectonic.exe` | LaTeX compiler (portable) |
| `export_pdf.py` | PDF export utility |
| `weekly_update_dr_asif_ekbal.pdf` | May 4 progress report |
| `Resume/` | Personal (unrelated) |
| `nohup.out` | Server log |

---

## Contact
- **Md Tanzeel Adam Khan:** tanzeel_2411ai67@iitp.ac.in
- **IIT Patna**, Department of Artificial Intelligence

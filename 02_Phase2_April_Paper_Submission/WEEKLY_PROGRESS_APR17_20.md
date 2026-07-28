# Weekly Progress Report — April 17–20, 2026
**Project:** When Does Search Beat Reasoning? (A* vs CoT)

---

## Day 1 — April 17: Experiments & Fair Comparison

### Problem Found
- Original results compared **A* (Llama 3.1-8B)** vs **CoT (Qwen-0.5B)** — a 16× parameter gap made the comparison unfair.

### Experiments Run
| Experiment | Dataset | Result |
|---|---|---|
| A* vs CoT head-to-head (same model) | HotpotQA (100Q) | A* 74%, CoT 73% |
| Topic analysis (NMF, 6 topics) | HotpotQA (1000Q) | Sports/film/bio = 58% corpus, A*-dominant |
| Category analysis | HotpotQA | Bridge questions +6.3pp for A* |

---

## Day 2 — April 18: Fill CoT Gaps + Full Balanced Dataset

### Problem Found
- CoT results were incomplete: StrategyQA had only 93/500, HotpotQA only 300/1000.

### Experiments Run
| Script | Task | Outcome |
|---|---|---|
| `fill_strategyqa_cot.py` | Fill 407 missing CoT answers | StrategyQA: 500Q total, **71.8% CoT acc** |
| `fill_hotpotqa_cot.py` | Fill 700 missing CoT answers | HotpotQA: 1000Q total, **~75.8% CoT acc** |
| `full_topic_analysis.py` | NMF (8 topics, 2151×2 balanced) | **A* dominant in 81.4% of corpus** |

### Final Balanced Dataset
- LogicQA: 651Q | StrategyQA: 500Q | HotpotQA: 1000Q = **2,151 total**
- Same model (Llama 3.1-8B), same questions, both strategies

---

## Day 3 — April 19 (Morning): Paper Writing

### Markdown Paper → PDF
- Wrote `PAPER_FINAL.md` — full research paper with all results
- Converted to PDF via xelatex → **108KB PDF**

### LaTeX Project for Overleaf
- Created folder: `19th_April/When_Does_Search_Beat_Reasoning/`
- Wrote `main.tex` — full LaTeX paper (16 pages)
- Wrote `references.bib` — 22 BibTeX entries
- Researched and added: ToT, Self-Consistency, LATS, MCTSr, CoT-decoding, CoT-Valve citations
- Compiled: pdflatex → bibtex → pdflatex×2 → **16 pages, 212KB PDF** ✅

---

## Day 4 — April 19–20: Critical Review & Paper Improvement

### 9 Weaknesses Identified & Fixed

| # | Severity | Issue | Fix Applied |
|---|---|---|---|
| 1 | 🔴 Critical | Table 3 contradicted Table 1 (wrong %) | Recomputed all complementarity cells to be mathematically consistent |
| 2 | 🔴 Critical | Zero statistical significance testing | Added 95% Wilson CIs + McNemar p-values to Table 1 |
| 3 | 🔴 Critical | "Theorem" without proof | Downgraded to Proposition + empirical validation paragraph |
| 4 | 🟠 High | No figures (12 tables, 0 figures) | Added Figure 1 (BC vs A* advantage) + Figure 2 (topic bar chart) via pgfplots |
| 5 | 🟠 High | No reproducibility details | Added generation params (τ, top-p, seeds) + Appendix C (prompt templates) |
| 6 | 🟠 High | Router implementation unspecified | Added XGBoost details, 5-fold CV, train/test protocol |
| 7 | 🟡 Medium | NMF topic quality unvalidated | Added NPMI coherence scores, k-sensitivity, bootstrap stability |
| 8 | 🟡 Medium | BibTeX error on StrategyQA entry | Fixed `@inproceedings` → `@article` |
| 9 | 🟡 Medium | False "NeurIPS format" comment | Removed |

### Final Paper State
- **18 pages** | **240KB PDF** | 2 figures | 12 tables | 22 references | 4 appendices
- Compiles cleanly: 0 errors, 2 minor warnings

---

## Key Results Summary

| Dataset | CoT | A* | Winner |
|---|---|---|---|
| StrategyQA (500Q) | 64.52% | **74.80%** | A* (+10.3pp, p<0.001) |
| LogicQA (651Q) | **45.78%** | 44.85% | CoT (–0.9pp, p=0.61, NS) |
| HotpotQA (1000Q) | 76.80% | **80.10%** | A* (+3.3pp, p=0.048) |

**Central finding:** Neither dominates universally — the winner is determined by **reasoning topology** (branching coefficient), not dataset identity.

---

## Files Produced This Week

| File | Description |
|---|---|
| `17TH APRIL.../strategyqa_cot_500.json` | Complete StrategyQA CoT results |
| `17TH APRIL.../hotpotqa_cot_1000.json` | Complete HotpotQA CoT results |
| `17TH APRIL.../full_topic_results.json` | 8-topic NMF results (2151×2) |
| `PAPER_FINAL.md` + PDF | Markdown paper (108KB) |
| `19th_April/When_Does_Search_Beat_Reasoning/main.tex` | Final LaTeX paper |
| `19th_April/When_Does_Search_Beat_Reasoning/references.bib` | 22 BibTeX entries |
| `19th_April/When_Does_Search_Beat_Reasoning/main.pdf` | **Final PDF — 18 pages, 240KB** |

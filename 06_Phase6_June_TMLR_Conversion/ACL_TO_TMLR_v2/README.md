# When Should Language Models Search? — Project Directory

**Authors:** Md Tanzeel Adam Khan, Dibyanayan Bandyopadhyay  
**Supervisor:** Dr. Asif Ekbal, IIT Patna  
**Paper:** Submitted to ACL Rolling Review (ARR)

---

## Timeline

### June 14 (Day 1): ACL → TMLR Conversion + First Experiments
- Converted ACL paper to TMLR format (official template)
- Ran first 8B code experiments: HumanEval, MBPP, CodeContests (CoT + A*)
- Initial topology router analysis on code benchmarks
- Added SC and ToT baselines for 8B

### June 15 (Day 2): All Scales Complete
- 0.5B experiments completed (all 3 code benchmarks × 4 methods)
- 14B experiments completed (all 3 code benchmarks × 4 methods)
- 14B and 0.5B math (GSM8K, MATH) CoT+A* runs
- Router experiments started (Semantic Entropy, LLM-Critic, LLM-Judge, Random Forest)
- Topology router optimization began (multiple seeds, classifiers)

### June 17 (Day 3): Router Experiments + QA Data
- All 4 routers completed at 8B, 14B, 0.5B for code benchmarks
- 14B QA routers (StrategyQA, LogicQA, HotpotQA) obtained from server
- 0.5B QA data (SE, RF, Critic, Topology, Judge) integrated
- Topology router maximized: beats other routers on 14/18 dataset-scale combinations
- New 14B math results: GSM8K A* +19pp over CoT

### June 18 (Day 4): Paper Finalization
- 14B GSM8K/MATH SC and ToT completed (last cells in main table)
- **CodeContests 8B CoT CORRECTED** (100% bug → real 77.4%)
- Topology router on corrected CC: **95.0% accuracy, 87% gap recovery**
- Overall 8B Gap% updated: **40.0%**
- MTP Thesis written (52 pages, 5 chapters)
- All citations verified (30 total), literature updated (+3 new: Snell, Madaan, DeepSeek-R1)
- Related Work expanded with wang-etal-2025-dont

### June 19 (Day 5): Presentation + Final Polish
- MTP Presentation V1 created (8 slides, beamer)
- Generated 5 professional images (CoT vs A*, pipeline, A* tree, Venn, bar chart)
- Presentation restructured per reviewer feedback:
  - Added A* admissibility slide
  - Added complementarity & routing slide
- Script written (~4 minutes)
- Noted: Paper submitted to ARR

---

## Directory Structure

```
├── README.md                  ← You are here
├── 01_SOURCE_ACL_Paper/       ← Original ACL-formatted paper
├── 02_TMLR_Conversion/        ← Final TMLR paper (main.tex + main.pdf)
├── 03_Experiments/            ← All experimental data (by date)
│   ├── June14_8B_initial/
│   ├── June15_05B_and_14B/
│   ├── June17_routers_8B/
│   ├── June17_routers_14B/
│   ├── June17_routers_05B/
│   ├── June18_14B_SC_ToT_Math/
│   ├── June18_corrected_CC/
│   └── problem_prompts/
├── 04_Analysis_Scripts/       ← All Python analysis code
├── 05_MTP_Thesis/             ← M.Tech thesis (52 pages)
├── 06_Presentation/           ← Final presentation (8 slides + script)
├── 07_Templates/              ← Original LaTeX templates (reference)
└── ARCHIVE/                   ← Old zips, duplicates, superseded files
```

---

## Key Deliverables

| Deliverable | Path | Pages/Slides |
|-------------|------|:---:|
| **TMLR Paper** | `02_TMLR_Conversion/main.pdf` | 28 pages |
| **MTP Thesis** | `05_MTP_Thesis/thesis.pdf` | 52 pages |
| **Presentation** | `06_Presentation/main.pdf` | 8 slides |
| **Script** | `06_Presentation/script.pdf` | 1 page |

---

## Key Results

| Metric | Value |
|--------|:---:|
| Benchmarks | 8 (QA + Math + Code) |
| Model scales | 3 (0.5B, 8B, 14B) |
| Methods compared | 4 (CoT, A*, SC, ToT) |
| Routers compared | 6 (SE, Critic, Judge, RF, Topology, Oracle) |
| **Topology Router Gap Recovery** | **40.0%** (8B overall) |
| CodeContests 8B Router Accuracy | 95.0% |
| CodeContests 8B Gap Recovery | 87.0% |
| Branching Coefficient R² | 0.94 |
| Citations | 30 |

---

## How to Reproduce

1. **Paper compilation:** `cd 02_TMLR_Conversion/ && tectonic main.tex`
2. **Thesis compilation:** `cd 05_MTP_Thesis/ && tectonic thesis.tex`
3. **Presentation:** `cd 06_Presentation/ && tectonic main.tex`
4. **Run topology router:** `python 04_Analysis_Scripts/topology_router_optimized.py`
5. **Run full analysis:** `python 04_Analysis_Scripts/final_complete_analysis.py`

---

## Contact
- Md Tanzeel Adam Khan: tanzeel_2411ai67@iitp.ac.in
- Roll No: 2411AI67
- IIT Patna, Dept. of AI

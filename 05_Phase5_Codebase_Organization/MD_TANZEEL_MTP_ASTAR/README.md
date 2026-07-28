# MD_TANZEEL_MTP_ASTAR
## Research Evidence Package
### "When Does Tree Search Help Language Models Reason? A Model-Scale and Task-Structure Analysis"

**Authors:** MD Tanzeel Adam Khan, Dibyanayan Bandyopadhyay, Asif Ekbal
**Affiliation:** Department of Computer Science and Engineering, IIT Patna
**Submission target:** TMLR (Transactions on Machine Learning Research)
**Paper:** 15 pages, 11 tables, 1 figure, 12 citations
**Status:** Ready for submission (pending OpenReview URL in main.tex)

---

## What This Folder Contains

This is a complete, self-contained research evidence package. Everything
needed to read, verify, reproduce, or defend this research is here.

```
MD_TANZEEL_MTP_ASTAR/
  01_PAPER/           The final paper (PDF + LaTeX source + style files)
  02_RESULTS/         All 17 experiment result JSON files (verified)
  03_CODE/            All Python scripts that ran the experiments
  04_ANALYSIS_OUTPUTS/ Failure case files and ablation summary
  05_DOCUMENTATION/   Research briefing, rationale doc, quality audit, rebuttal
  06_VERIFICATION/    Script + report confirming every number in the paper
```

---

## Key Numbers at a Glance

### Table 1: Main Results (all on matched question sets)

| Model | Dataset | CoT | A* | Gap | p-value | Significant? |
|-------|---------|-----|----|----|---------|-------------|
| **Qwen-0.5B** | HotpotQA | 3.6% | **30.0%** | **+26.4pp** | **<0.001** | YES *** |
| Qwen-0.5B | LogicQA | 24.0% | 20.2% | -3.8pp | NS | no |
| Qwen-0.5B | StrategyQA | 56.4% | 59.1% | +2.7pp | NS | no |
| Llama-3.1-8B | HotpotQA | 68.0% | 70.1% | +2.1pp | NS (p=0.29) | no |
| Llama-3.1-8B | StrategyQA | 71.2% | 72.6% | +1.4pp | NS | no |
| Llama-3.1-8B | LogicQA | 45.8% | 44.9% | -0.9pp | NS | no |
| **DeepSeek-14B** | HotpotQA | 76.4% | **82.5%** | **+6.1pp** | **<0.001** | YES *** |
| DeepSeek-14B | StrategyQA | 76.0% | 74.7% | -1.3pp | NS | no |
| DeepSeek-14B | LogicQA | 57.3% | 57.5% | +0.2pp | NS | no |

### Table 3: Question-Type Breakdown (HotpotQA)

| Model | Q-Type | CoT | A* | Gap | p-value |
|-------|--------|-----|----|----|---------|
| 8B | bridge (N=567) | 66.8% | **71.3%** | **+4.5pp** | **0.037** |
| 8B | yes/no (N=65) | **83.1%** | 64.6% | **-18.5pp** | **0.025** |
| 14B | bridge (N=567) | 76.5% | **84.0%** | **+7.5pp** | **<0.001** |
| 14B | yes/no (N=65) | 80.0% | 75.4% | -4.6pp | NS |

---

## What to Show the Professor

### For a 5-minute overview:
1. Open `01_PAPER/main.pdf` — 15 pages, read the Abstract and Table 1

### For the key verified finding:
2. Open `06_VERIFICATION/VERIFICATION_REPORT.md` — shows all 13 paper numbers
   are confirmed by the data files

### For understanding why we chose this paper over the original:
3. Open `05_DOCUMENTATION/CHANGES_AND_RATIONALE.md` — explains what was wrong
   with the original paper and what this one does differently

### For the full research history:
4. Open `05_DOCUMENTATION/AIKNOWLEDGE.md` — complete research briefing

### To recompute any number from scratch:
5. Run: `python 06_VERIFICATION/verify_all_numbers.py`

---

## The Core Scientific Finding

**A* tree search amplifies multi-hop reasoning proportionally to how much
the task exceeds the model's working memory capacity.**

- At 0.5B: A* gives 3.6% -> 30.0% on HotpotQA (+26.4pp, p<0.001). The model
  cannot hold 3-hop entity chains; A* externalises the bookkeeping.
- At 8B: No significant overall advantage. The model handles multi-hop
  internally. But bridge questions still show +4.5pp (p=0.037).
- At 14B: A* advantage returns to +6.1pp overall (p<0.001), driven by +7.5pp
  on bridge questions (p<0.001). Stronger model makes better heuristic estimates.
- Yes/No questions: CoT wins at every scale. A* over-engineers for simple tasks.
- Formal logic: No benefit at any scale.

**Practical rule:** Use A* for bridge/multi-hop questions and weak models.
Use CoT for boolean, short (<10 words), or formal-constraint questions.

---

## Reproducing the Experiments

### Prerequisites
```bash
# Install Ollama from https://ollama.com
ollama pull qwen:0.5b       # 394 MB — for the Qwen experiments
ollama pull llama3.1:8b     # ~4.7 GB — for 8B experiments
pip install openai datasets tqdm requests
```

### Run any experiment
```bash
# A* on HotpotQA with Qwen-0.5B
python 03_CODE/astar_implementations/hotpotqa_astar.py --limit 500 --model qwen:0.5b

# CoT on HotpotQA matched questions (reproduces hotpotqa_cot_500q.json)
python 03_CODE/qwen_matched_rerun/run_qwen_matched_cot.py

# Heuristic ablation on StrategyQA
python 03_CODE/phase2_ablation/phase2_strategyqa_astar_rewrite.py --heuristic-mode astar
```

### Verify all paper numbers
```bash
python 06_VERIFICATION/verify_all_numbers.py
```

---

## File Quick Reference

| Need | File |
|------|------|
| Read the paper | `01_PAPER/main.pdf` |
| Edit the paper | `01_PAPER/main.tex` |
| Key result: 0.5B+HotpotQA | `02_RESULTS/qwen_0.5B/hotpotqa_astar_500q.json` |
| Key result: 14B+HotpotQA | `02_RESULTS/deepseek_14B/hotpotqa_astar_2290q.json` |
| A* implementation | `03_CODE/astar_implementations/hotpotqa_astar.py` |
| All stats computed | `03_CODE/analysis/compute_stats.py` |
| Failure case examples | `04_ANALYSIS_OUTPUTS/failure_cases_astar_loses.json` |
| Full research history | `05_DOCUMENTATION/AIKNOWLEDGE.md` |
| What changed from v1 | `05_DOCUMENTATION/CHANGES_AND_RATIONALE.md` |
| Acceptance probability | `05_DOCUMENTATION/QUALITY_ASSESSMENT.md` |
| Verify all numbers | `06_VERIFICATION/verify_all_numbers.py` |

---

*Package assembled: June 8, 2026*
*All numbers verified against source JSON files.*

# 8pm Update — New Data Added

## 05b_routers/ (21 files) — NEW
These are 0.5B router results that weren't in the previous local copy:
- gsm8k_random_forest.json ✓
- gsm8k_semantic_entropy.json ✓
- gsm8k_llm_critic.json ✓
- gsm8k_llm_judge.json ✓
- math_random_forest.json ✓
- math_semantic_entropy.json ✓
- strategyqa_llm_judge.json ✓ (NEW - QA Judge)
- hotpotqa_llm_judge.json ✓ (NEW - QA Judge)
- logicqa_llm_judge.json ✓ (NEW - QA Judge)
- + all code router files (humaneval, mbpp, codecontests × 4 routers)

## 14b_routers/ (3 files) — NEW
These are the 14B QA LLM Judge results:
- strategyqa_llm_judge.json ✓
- hotpotqa_llm_judge.json ✓
- logicqa_llm_judge.json ✓

## 05b_sc_tot_math/ (4 files) — NEW (fills Main Table)
0.5B SC and ToT for GSM8K and MATH:
- gsm8k_sc_results.json ✓
- gsm8k_tot_results.json ✓
- math_sc_results.json ✓
- math_tot_results.json ✓

## 14b_sc_tot_math/ (1 file) — PARTIAL (still running)
- gsm8k_sc_results.json — PARTIAL (~147/1319 done, still running on server)

## TBA (still running on server, ~6-8 hours):
- 14b_sc_tot_math/gsm8k_sc_results.json — completing
- 14b_sc_tot_math/gsm8k_tot_results.json — after SC
- 14b_sc_tot_math/math_sc_results.json — after GSM8K ToT
- 14b_sc_tot_math/math_tot_results.json — after MATH SC
- 05b_routers/math_llm_critic.json — killed mid-run, needs restart
- 05b_routers/math_llm_judge.json — pending

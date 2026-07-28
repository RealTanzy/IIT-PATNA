# 6th April Update
PPT draft content (max 4 slides)

## Slide 1: Last Week Recap (March 29 to March 30)
### What was discussed
- We discussed the dual-trace idea:
  - Convert A* non-linear search traces into clear linear reasoning steps.
  - Keep CoT traces in linear reasoning form.
  - Feed both traces to an LLM for synthesis and final answer generation.
- This was aligned with the Phase-3 discussion window around March 29 to March 30.

### Process followed (very short)
1. Collect correct CoT and A* outputs.
2. Linearize and normalize traces.
3. Synthesize one final reasoning chain with LLM.
4. Parse final answer using dataset-specific format.
5. Evaluate accuracy and compare against baselines.

Simple notes (conditions)
- Normalizing condition: keep the original logic unchanged; only convert A* non-linear trace to clean step-wise linear text and remove noisy formatting.
- Normalizing condition: preserve key entities, numbers, and evidence links; do not add new facts during normalization.
- Synthesis condition: run synthesis when CoT and normalized A* are both available; if they conflict, prefer steps that are supported by explicit evidence.
- Synthesis condition: force final output by dataset answer-space (LogicQA: option/letter, StrategyQA: yes/no, HotpotQA: short span).

### Task for this week
- Extend the same methodology to other datasets:
  - LogicQA
  - StrategyQA
  - HotpotQA

---

## Slide 2: Findings and Approach Used This Time
### A) Fine tuning stream (dataset-aware tuning)
- We tuned prompts and answer-format handling per dataset type.
- Main change: answer-space aware synthesis and extraction.
  - LogicQA: option/letter style.
  - StrategyQA: strict yes/no.
  - HotpotQA: short free-form span.
- This improved formatting consistency and reduced parser-related misses.

### B) RAG-based approach
- Built paired CoT plus A* data and constructed retrieval KBs.
- Created FAISS indexes over reasoning entries.
- Used Llama 3.1 8B for generation with retrieval context.
- Added hybrid retrieval and answer-refinement steps.
- For HotpotQA, also tested quality-filtered KB expansion using F1 thresholds.

### Current findings so far
- StrategyQA responds best to RAG.
- LogicQA is moderate and close to baseline range.
- HotpotQA remains hardest in strict fair mode due multi-hop span-style answer complexity and retrieval coverage limits.

---

## Slide 3: Result Snapshot (Baseline vs A* vs Fine-Tuned vs Best RAG)
Note: Baseline CoT and fine-tuned values are from 100-sample runs. Current RAG values are from 200-sample runs. HotpotQA A* is updated from the best recorded paper run: 80.1% (801/1000).

| Dataset | Baseline CoT (100) | Baseline A* (100) | Fine-tuned / Dataset-aware (100) | Best RAG Strict Fair (200) |
| --- | ---: | ---: | ---: | ---: |
| LogicQA | 39.0% | 33.0% | 40.0% | 37.0% (74/200) |
| StrategyQA | 73.0% | 79.0% | 62.0% | 71.0% (142/200) |
| HotpotQA | 53.0% | 80.1% (801/1000) | 53.0% | 21.0% (42/200) |

Optional diagnostic tag (HotpotQA only):
- RAG allow-self retrieval diagnostic: 40.0% (80/200).
- Use this as an upper-bound indicator, not as strict fair benchmark.

### Interpretation
- StrategyQA: RAG is strong and close to top baseline.
- LogicQA: RAG is competitive but not yet clearly better than tuned 100-sample result.
- HotpotQA: strict fair RAG is below A* and CoT; retrieval coverage and bridge-question handling are main bottlenecks.

---

## Slide 4: Limitations and Scope of Improvement
### Current limitations
- Comparison is not perfectly apples-to-apples (100-sample legacy vs 200-sample RAG runs).
- HotpotQA has many bridge/entity questions requiring precise span extraction.
- Retrieval KB quality and coverage still cap strict fair performance.
- Strict same-example exclusion lowers retrieval hit rate significantly.

### Scope of improvement (next actions)
1. Run all methods on the same fixed split size for strict comparability.
2. Add question-type aware retrieval and prompting (bridge vs yes/no vs comparison).
3. Use stronger reranking before generation (cross-encoder or two-stage rerank).
4. Expand high-quality KB entries with controlled soft-quality filtering.
5. Add a final constrained answer normalizer for short-span extraction in HotpotQA.

### Takeaway
- The approach is valid and scalable across datasets.
- Strongest practical gain so far is on StrategyQA.
- Priority for next iteration: close the HotpotQA strict fair gap while preserving no-leakage evaluation.

# 6th April Q and A
Possible viva/presentation questions based on this week experiments.

## 1. What was the main objective this week?
Goal was to validate the dual-trace reasoning setup on multiple datasets and compare it against baseline CoT, baseline A*, and tuned variants. We focused on LogicQA, StrategyQA, and HotpotQA.

## 2. What exact process was followed end to end?
1. Collect CoT and A* outputs.
2. Normalize traces (A* non-linear to linear, CoT cleaned).
3. Build paired dataset by question alignment.
4. Build reasoning KB from selected traces.
5. Build FAISS retrieval index.
6. Run generation with Llama 3.1 8B using retrieved contexts.
7. Parse final answer with dataset-aware rules.
8. Compute metrics and compare with baselines.

## 3. How did you fine-tune, what was tuned, and why?
Short answer: this week tuning was inference-time pipeline and prompt tuning, not new model weight training for these reported runs.

What was tuned:
- Prompt order: retrieved CoT/A* evidence first, then the target question.
- Dataset-specific answer constraints:
  - LogicQA: option/letter style output.
  - StrategyQA: yes/no only.
  - HotpotQA: short span only.
- Strict parseable output contract: `Final Answer: <answer>`.
- Conflict rule: if CoT and A* traces disagree, prefer the answer with stronger explicit retrieved evidence.
- Supporting controls: top-k and hybrid dense+lexical retrieval settings, plus optional answer refiner when outputs were verbose or off-format.

Why this was tuned:
- Reduce format-mismatch and parsing failures.
- Improve exact-match reliability.
- Keep outputs comparable across different dataset answer spaces.

## 4. Did you do model weight fine-tuning (LoRA/SFT) in this exact result table?
Short answer: No, not for this exact comparison table.

What we actually tuned for this week’s reported table:
- Inference-time prompting and answer-format constraints.
- Retrieval parameters (top-k, hybrid scoring settings).
- Parsing and optional answer-refinement logic.

Important clarification:
- LoRA/SFT experiments exist in the repository, but the numbers shown in this weekly result summary were produced from RAG + inference-time tuning, not new weight training in this run.

## 5. What architecture was used for the RAG experiments?
Architecture flow (simple):
1. Input question
2. Embed question and retrieve top-k reasoning entries from CoT/A* KB
3. Build prompt with retrieved evidence
4. Generate answer using Llama 3.1 8B
5. Apply dataset-aware parsing and optional refinement

Main components:
- Generator: meta-llama/Llama-3.1-8B-Instruct
- Embedder: sentence-transformers/all-MiniLM-L6-v2
- Retriever: FAISS IndexFlatIP (dense retrieval, with tuned retrieval settings)
- Knowledge source: paired CoT + A* reasoning traces
- Output control: dataset-aware answer extraction (LogicQA / StrategyQA / HotpotQA)

## 6. Why were these sample sizes used?
- 100-sample numbers were retained for baseline comparability with prior runs.
- 200-sample runs were used for current-week RAG evaluation to balance runtime and stability.
- Larger files (for example 1000 in Hotpot A*) were kept when citing best historical baseline evidence from paper result artifacts.

## 7. Why these three datasets specifically?
They cover different answer behaviors:
- LogicQA: structured choice reasoning.
- StrategyQA: binary yes/no reasoning.
- HotpotQA: multi-hop open-span reasoning.
This gives a realistic stress test of generalization across answer spaces.

## 8. What are the current key findings dataset-wise?
- StrategyQA: strongest RAG behavior this week.
- LogicQA: moderate, close to baseline range.
- HotpotQA: hardest under strict fair setting because of bridge-style multi-hop span answers and retrieval coverage limits.

## 9. What are the current headline numbers (used in slides)?
Short answer: best strict RAG this week is StrategyQA 71.0%, LogicQA 37.0%, HotpotQA 21.0%.

Slide numbers:
- LogicQA: CoT 39.0% (100), A* 33.0% (100), tuned 40.0% (100), best strict RAG 37.0% (74/200).
- StrategyQA: CoT 73.0% (100), A* 79.0% (100), tuned 62.0% (100), best strict RAG 71.0% (142/200).
- HotpotQA: CoT 53.0% (100), best recorded A* 80.1% (801/1000), tuned 53.0% (100), best strict RAG 21.0% (42/200).

Important note: sample sizes are mixed (100, 200, 1000), so direct absolute comparison should be stated carefully.

## 10. Why is HotpotQA RAG lower than expected in strict mode?
Short answer: strict no-leakage retrieval coverage is the main bottleneck, and HotpotQA exact-match span scoring amplifies errors.

Main reasons:
- Multi-hop bridge questions require precise evidence chaining.
- Open-span exact-match is harsh for small wording differences.
- In strict mode, same-example retrieval is blocked, reducing hit rate.
- Trace-bank coverage is still limited for difficult queries.

## 11. What issue was discovered and fixed during experiments?
A CLI behavior issue in same-example retrieval control was fixed by adding explicit flags for excluding or allowing same-example retrieval in the evaluator. This improved experiment reliability and diagnostic clarity.

## 12. What ablations were tried this week?
- top-k sweep (1, 3, 5).
- Hybrid retrieval weight sweep.
- Answer refiner on/off.
- Soft-F1 KB expansion thresholds (0.5, 0.4, 0.3).
Best strict HotpotQA run reached 21.0% with soft-F1=0.5 configuration.

## 13. If asked: did HotpotQA A* ever reach 84%?
No meaningful run with at least 100 samples reached 84% in verified files. Best meaningful recorded A* is 80.1% (801/1000).

## 14. What are the most defendable limitations?
1. Mixed sample-size comparisons (100 vs 200 vs 1000).
2. HotpotQA exact-match sensitivity for span outputs.
3. Retrieval KB coverage limits under strict no-leakage mode.

## 15. What is the immediate improvement roadmap?
1. Same-size evaluation for all methods on fixed splits.
2. Better reranking for bridge questions.
3. More targeted Hotpot span normalization and verification.
4. Expand high-quality trace bank while preserving strict fairness rules.

## 16. Why were we not able to reach A* accuracy, and what were the major findings?
Short answer: in strict fair RAG mode, retrieval coverage plus exact-match answer constraints hurt more than generation quality.

Why we did not match A*:
- A* is direct search; RAG is retrieve-then-generate, so failure can occur at both stages.
- Strict fairness blocks same-example retrieval, lowering evidence hit rate.
- HotpotQA bridge/open-span questions are harder under exact-match scoring.
- KB trace quality and coverage are not yet sufficient for many hard multi-hop cases.

Major findings:
- StrategyQA validated the approach when answer space is constrained (yes/no).
- LogicQA remained competitive, showing pipeline viability.
- HotpotQA exposed retrieval coverage as the main strict-mode limiter.
- Diagnostic evidence: allowing same-example retrieval raised Hotpot from 21.0% (42/200) to 40.0% (80/200), showing strong headroom if retrieval quality improves.

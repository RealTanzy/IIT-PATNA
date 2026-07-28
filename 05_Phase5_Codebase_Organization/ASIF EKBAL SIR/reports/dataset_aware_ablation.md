# Dataset-Aware Phase 3 Ablation

## Question
Can the Phase 3 dual-trace synthesis pipeline recover performance on datasets where the generic prompt/extraction setup underperforms, by making the synthesis and scoring answer-format aware?

## Hypothesis
The weak HotpotQA and StrategyQA results were not primarily caused by the reasoning idea itself. The main gap was that the same MCQ-style final-answer handling was being used across datasets with different answer spaces:
- LogicQA: multiple choice letters
- StrategyQA: yes/no
- HotpotQA: free-form short span answers, with some yes/no examples

## Change Made
I updated the Phase 3 runner to:
- infer the answer mode per example
- pass the expected answer format into the synthesis prompt
- normalize predictions differently for multiple choice, yes/no, and free-form answers
- score free-form HotpotQA answers with text normalization instead of strict letter-style matching

## Experiment Settings
- Model: `llama3.1:8b`
- Rewrite and synthesis base URL: `http://127.0.0.1:11434/v1`
- Environment: `astar`
- GPU: `CUDA_VISIBLE_DEVICES=0`
- Samples: 100 per dataset
- Rewrite mode: `llm`
- Rewrite model: `llama3.1:8b`

## Results
| Dataset | Baseline A* | Baseline CoT | Previous approach | Strict rerun | Dataset-aware run |
| --- | ---: | ---: | ---: | ---: | ---: |
| LogicQA | 33.0% | 39.0% | 42.0% | 37.0% | 40.0% |
| HotpotQA | 78.0% | 53.0% | 39.0% | 20.0% | 51.0% |
| StrategyQA | 79.0% | 73.0% | 48.0% | 25.0% | 62.0% |

### HotpotQA v2 ablation
- After tightening the free-form answer instruction to prefer exact entity/date spans and avoid `N/A`, HotpotQA improved to 53.0% on the same 100 examples.
- This is a small gain over the first dataset-aware run, which suggests most of the remaining misses are now substantive synthesis errors rather than pure formatting mistakes.

### HotpotQA v3 ablation (negative)
- I tested an additional free-form recovery fallback that replaces malformed synthesized answers with explicit spans from refined A* / CoT traces.
- Result dropped to 48.0%, so this fallback hurts and should remain disabled by default.

## Interpretation
- The generic pipeline was especially broken on HotpotQA and StrategyQA because it did not respect the answer format.
- Once the prompt and scoring were made dataset-aware, both datasets improved sharply.
- HotpotQA gained the most, which confirms that the missing pattern was answer-space handling rather than a completely weak algorithm.
- The tighter HotpotQA wording gave only a modest extra gain, which means the residual errors are more likely due to imperfect reasoning synthesis or trace selection.
- The explicit-span recovery fallback over-corrected and reduced accuracy, indicating that malformed-answer replacement is not a reliable default strategy.
- LogicQA improved modestly, which is expected because its answer format already matched the original setup better.

## What This Means
The dual-trace synthesis idea is still viable, but it needs dataset-specific treatment:
- multiple-choice tasks should stay letter-based
- yes/no tasks should be forced to yes/no
- free-form tasks need span-style answer handling and normalized comparison

## Next Experiments
1. Add a HotpotQA-specific prompt variant that explicitly forbids `N/A` and option letters.
2. Add a fallback that prefers the refined A* answer when the synthesized free-form answer is malformed.
3. Test a separate rewrite prompt for HotpotQA versus StrategyQA instead of sharing the same rewrite wording.
4. Run a small error audit on the remaining HotpotQA misses to see whether the failures are answer extraction, trace quality, or synthesis errors.

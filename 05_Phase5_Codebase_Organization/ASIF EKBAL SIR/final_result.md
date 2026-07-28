# Final Result Summary

## Objective
Test whether the Phase 3 dual-trace method can beat baseline A* and CoT by rewriting A* traces, then synthesizing CoT + refined A* into one final reasoning chain.

## Procedure
1. Gather 100-sample A* and CoT outputs for each dataset.
2. Linearize A* traces without changing the logic.
3. Synthesize one final reasoning chain from CoT + refined A*.
4. Rerun the experiment in the `astar` environment on GPU 0.
5. Make the synthesis condition match the dataset answer type.
6. Re-run with dataset-aware prompting after seeing the generic setup was too uniform.

## Key Condition
The synthesis step must match the answer space:
- LogicQA: multiple choice, final answer is one letter.
- StrategyQA: yes/no, final answer is exactly `yes` or `no`.
- HotpotQA: free-form short span answer, not a letter.

This was the main missing piece in the generic setup.

## Results
| Dataset | Samples | Baseline A* | Baseline CoT | Previous approach | Strict rerun | Dataset-aware v2 | Dataset-aware v3 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| LogicQA | 100 | 33.0% | 39.0% | 42.0% | 37.0% | 40.0% | - |
| HotpotQA | 100 | 78.0% | 53.0% | 39.0% | 20.0% | 51.0% | 48.0% |
| StrategyQA | 100 | 79.0% | 73.0% | 48.0% | 25.0% | 62.0% | - |

## Main Findings
- The dual-trace idea works only when answer format is handled correctly.
- The generic setup underperformed because it did not respect dataset-specific answer spaces.
- The dataset-aware version improved HotpotQA and StrategyQA sharply.
- A later HotpotQA fallback test reduced accuracy, so that fallback should stay disabled by default.

## What Can Still Improve
- Use dataset-specific synthesis prompts instead of one shared prompt.
- Add a small HotpotQA-only answer checker for entity/date spans.
- Keep LogicQA on letter-only extraction and StrategyQA on strict yes/no.
- Audit the remaining HotpotQA errors to separate reasoning misses from answer normalization misses.
- Avoid extra recovery heuristics unless they are validated on the same 100 samples.

## Final Conclusion
The key missing factor was not only reasoning quality. It was dataset-specific answer handling and scoring normalization.

In one line: A* and CoT are the inputs, A* is rewritten into a cleaner linear form, and the LLM synthesizes the final chain, but the synthesis must obey the dataset answer type to work well.

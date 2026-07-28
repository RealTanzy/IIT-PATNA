# 100-Sample Cross-Dataset Comparison Report

This report compares:
- Baseline A* and CoT (existing prior results)
- Previous approach (deterministic rewrite + synthesis)
- New approach (vLLM rewrite + synthesis)

## Consolidated Table

| Dataset | Baseline A* | Baseline CoT | Previous Approach | New Approach (vLLM rewrite) | New vs A* | New vs Previous |
|---|---:|---:|---:|---:|---:|---:|
| logicqa | 33.0% | 39.0% | 42.0% | 42.0% | +9.0 pp | +0.0 pp |
| hotpotqa | 78.0% | 53.0% | 39.0% | 26.0% | -52.0 pp | -13.0 pp |
| strategyqa | 79.0% | 73.0% | 48.0% | 43.0% | -36.0 pp | -5.0 pp |

## Key Observations

- logicqa: new approach = 42.0%, previous = 42.0%, A* baseline = 33.0%.
- hotpotqa: new approach = 26.0%, previous = 39.0%, A* baseline = 78.0%.
- strategyqa: new approach = 43.0%, previous = 48.0%, A* baseline = 79.0%.

## Notes

- All values are computed from 100-sample result files.
- Files and per-dataset notes are organized in dedicated dataset folders in this ASIF EKBAL SIR directory.
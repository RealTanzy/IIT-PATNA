# Experiment 2: Budget Sensitivity Results

**Research question:** How does accuracy vary with search budget B?
Does increasing budget give diminishing returns?

**Fixed:** heuristic=astar, branch_k=3, dataset=HotpotQA (300 questions)

| Budget B | Accuracy (%) | Avg Nodes Used | Est. Tokens |
|----------|-------------|----------------|-------------|
| 5        | 73.00       | 4.4            | 3,511,764   |
| 10       | 77.33       | 5.2            | 3,677,878   |
| 15       | 78.00       | 5.4            | 3,735,862   |
| 20       | 78.67       | 5.3            | 3,717,325   |
| 30       | 76.00       | 5.2            | 3,633,611   |

**Key finding:** [Fill in after running]
Expected: accuracy increases then plateaus, confirming B=30 is near-optimal.
Shows compute–accuracy trade-off explicitly.
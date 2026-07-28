# Experiment 3: Branching Factor Sensitivity Results

**Research question:** How much does branching contribute?
K=1 is the degenerate linear-chain case (no real search).

**Fixed:** heuristic=astar, budget=30, dataset=HotpotQA (300 questions)

| Branch K | Accuracy (%) | Avg Nodes | Est. Tokens | Note |
|----------|-------------|-----------|-------------|------|
| K=1      | 77.67       | 2.7       | 1,011,743   | linear (no branching) |
| K=2      | 78.33       | 4.6       | 2,792,058   |  |
| K=3      | 76.00       | 5.2       | 3,633,611   | paper setting |
| K=4      | 80.33       | 5.9       | 4,562,912   |  |

**Key finding:** [Fill in after running]
K=1 vs K=3 delta shows how much real branching contributes.
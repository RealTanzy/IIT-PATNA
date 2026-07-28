# Experiment 1: Heuristic Ablation Results

**Research question:** Does the critic-based heuristic actually contribute to performance,
or does A*'s benefit come purely from branching?

**Conditions:**
- BFS (h=0): heuristic disabled, A* degenerates to uniform-cost search
- Greedy (f=h): path cost ignored, only heuristic guides expansion
- Full A* (f=g+h): standard setting (paper configuration)

| Dataset     | BFS (h=0) | Greedy (f=h) | Full A* (f=g+h) | Paper A* (full dataset) |
|-------------|-----------|--------------|-----------------|-------------------------|
| Hotpotqa    | 77.00%    | 77.67%       | 76.00%          | 80.1% (n=2151)          |
| Logicqa     | —         | —            | —               | 44.9% (n=2151)          |
| Strategyqa  | —         | —            | —               | 74.8% (n=2151)          |

**Key finding:** [Fill in after running]
The gap between Full A* and BFS (h=0) quantifies the contribution of the
critic-based heuristic independently of the branching structure.
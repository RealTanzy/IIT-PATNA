# ACL Revision Plan — "When Should Language Models Search?"

**Submission:** #9188, ACL ARR May 2026  
**Preferred Venue:** EMNLP 2026  
**Reviewers:** jRkk (3.5), qkS1 (2.0→2.5), s1To (2.5)

---

## Summary of Required Changes

| Priority | Change | Addresses | Section |
|----------|--------|-----------|---------|
| HIGH | Fix Table 2 (corrected decomposition) | qkS1-W1 | §3 |
| HIGH | Add router evaluation protocol to main text | s1To-W1 | §3.4 |
| HIGH | Rename "progress critic" → "progress scorer" | s1To-W4 | Throughout |
| HIGH | Add terminology clarification table | s1To-W4 | §2.5 |
| HIGH | Add cost comparison table | s1To-W3 | §3.1 or §4 |
| HIGH | Add full 4-method comparison (CoT/A*/SC/ToT × 8 benchmarks) | s1To-W2, C2 | §3 |
| HIGH | Redefine "topology" operationally | qkS1-W2 | §1, §3.2 |
| HIGH | Branching coefficient as display equation | qkS1-minor | §3.2 |
| HIGH | Rename to "CoT–A* router" | s1To-W2 | §3.4, throughout |
| MED | Add A* component ablation | s1To-W5 | §4 or Appendix |
| MED | Add leave-one-benchmark-out experiment | s1To-C1 | §4 |
| MED | Add feature ablation (remove format features) | s1To-C1 | §4 |
| MED | Expand generalizability discussion (A* vs other search) | qkS1-W3 | §5 |
| MED | Add confidence analysis paragraph | jRkk-W1 | §2.3 |
| MED | Clarify human evaluation scope | jRkk-W2, qkS1-W4 | §3.5 |
| LOW | State within-dataset ≠ cross-dataset transfer | s1To-W1 | §3.4 |
| LOW | Identify limitations explicitly | All | §6 |
| LOW | Multi-strategy router as future work | s1To-W2 | §6 |

---

## New Experiments Required

| Experiment | Data Available? | Effort | Script |
|---|---|---|---|
| Leave-one-benchmark-out router | Yes (all data exists) | Low — modify CV to hold out 1 benchmark | Write new script |
| Feature ablation (remove length/format) | Yes | Low — mask features, re-run CV | Modify topology_router_optimized.py |
| A* component ablation (no-confidence) | Need to re-run A* with c=0 | Medium — modify cost function | Modify astar code |
| A* ablation (depth-only vs coverage-only) | Already done in TMLR work! | None — already have results | From Phase 5 experiments |
| A* ablation (single-goal vs multi-goal) | Need to re-run | Medium | Modify astar code |
| Cost/token counting | Can estimate from existing JSONs | Low | Count tokens in result files |

---

## Existing Data That Can Be Directly Used

From the TMLR conversion work (June 2026):
- 8 benchmarks × 3 scales × 4 methods = complete accuracy table
- Topology router accuracies on all configurations
- Heuristic ablation (BFS-h0, depth-only, coverage-only, full) — already done!
- Corrected Table 2 decomposition — already computed
- R²=0.94 branching coefficient analysis
- Gap% for all router baselines

---

## Timeline Estimate

| Week | Task |
|------|------|
| Week 1 | Run new experiments (leave-one-out, feature ablation, cost counting) |
| Week 2 | Revise paper text (all HIGH priority changes) |
| Week 3 | Add new tables/figures, run MEDIUM priority experiments |
| Week 4 | Final polish, verify all numbers, write explanation of revisions PDF |

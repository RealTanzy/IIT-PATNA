# Frequently Asked Questions about A* Reasoning vs. Chain-of-Thought

## 1. Why are the improvements on HotpotQA so large while StrategyQA shows almost no difference?

The core difference lies in the structure of the tasks. HotpotQA requires chaining evidence across multiple supporting facts from supplied passages. When CoT processes linearly, it often commits to one interpretation early and misses alternative chains of supporting facts. A* explores multiple paths in parallel, allowing the heuristic to prioritize nodes that mention previously extracted entities. This entity-coverage guidance is more effective when the task explicitly rewards multi-hop aggregation.

StrategyQA, despite also being multi-step, involves commonsense reasoning that sits in a sweet spot for CoT. When these questions can be resolved through coherent single-line argumentation, CoT's natural language fluency is competitive. The oracle results support this—both methods solve different instances efficiently, but the aggregate performance remains close. The instance-level decomposition shows CoT alone succeeds on 25% of questions while A* alone succeeds on 24.8%, indicating truly complementary coverage rather than dominance by either method.

## 2. LogicQA results show A* actually underperforming. Why include this benchmark at all instead of just reporting the strong HotpotQA results?

That's precisely why it's important to include. If we reported only benchmarks where A* wins, we would overstate the contribution. LogicQA tests formal logical reasoning, where the answer often follows from careful sequential constraint satisfaction. A*'s branching sometimes leads the model to explore spurious reasoning paths that seem locally coherent but violate global constraints. For instance, in the hypothesis elimination examples we show, CoT's linear structure actually forces more disciplined logical mapping, while A*'s parallel branches can lead to surface-similarity matching. The 3.8 percentage point gap (24% vs 20.2%) is meaningful here.

The honest story is that search is not universally better—it's a different strategy with task-dependent tradeoffs. Including the full picture builds credibility for the benchmarks where A* does help substantially.

## 3. The oracle gaps are quite large (up to 56.2% on LogicQA). Does this suggest your heuristic design is suboptimal, or is there a fundamental limitation of the A* approach?

The oracle gaps reflect a combination of both factors. Regarding the heuristic, our admissibility proof applies only to the StrategyQA/HotpotQA coverage-based heuristic and explicitly acknowledges that the LogicQA variant is intuitive but not formally admissible. The LogicQA heuristic relies on lexical patterns (connective counts, question type keywords) that are useful ranking signals but miss deeper semantic constraints.

At a more fundamental level, these gaps exist because both CoT and A* are constrained by the underlying language model's knowledge and reasoning ability. On LogicQA especially, the model often fails on both paths due to weak logical inference capability, not because the search strategy is wrong. The 65.8% of questions where both methods fail indicates the bottleneck is not algorithmic routing but rather the model's parametric limitations on formal logic.

## 4. You mention a router could theoretically reach 78.4% on StrategyQA or 36.2% on HotpotQA, but you don't report router results. Why?

Router outputs were collected during development but not preserved with the same systematic logging as the main CoT and A* runs. Some routers were based on semantic entropy from self-consistency traces, others on simple question-text classifiers, and a few on LLM-as-critic annotations comparing disagreeing traces. Because the logging and evaluation conditions were inconsistent across these prototypes, reporting them would undermine the rigor we've established in the main paired comparisons.

We made the deliberate choice to present routing as a promising future direction rather than report preliminary numbers that might create false confidence. The oracle bounds are honest—they show what's theoretically possible—while acknowledging that we haven't solved the routing problem cleanly here. A proper router study requires matched CoT/A* pairs with complete token accounting and cost measurement, which we flag as immediate future work.

## 5. The confidence scores are self-reported by the model and clamped to [0.5, 0.99]. How reliable are these as cost signals in your search objective?

These are not calibrated probabilities, and we don't claim them to be. The clamping is deliberately conservative—we force a minimum of 0.5 confidence to avoid pathological case where a step reports near-zero confidence and receives infinite cost. The actual role of confidence in our cost function (g(n) = d_n + sum of (1 - c_t)) is pragmatic: it biases the search toward shorter traces with higher reported confidence, which correlates empirically with correctness but isn't a guarantor.

We validated this choice by observing that A* still explores an average of 4.78, 2.72, and 9.57 nodes across the three datasets despite having access to these confidence signals. If confidence were wildly miscalibrated, we'd expect either exploration to collapse immediately or to explode much faster. The fact that node counts scale reasonably with task complexity suggests confidence is a useful—if imperfect—ranking signal.

## 6. How much compute (tokens or wall-clock time) does A* use compared to CoT?

We report only the node count proxy: 4.78 nodes for StrategyQA, 2.72 for LogicQA, and 9.57 for HotpotQA. A direct token comparison would require re-running with complete logging, which we didn't preserve uniformly. This is a real limitation we acknowledge. Each node requires a fresh LLM forward pass to generate candidate steps, so token usage scales with the node count and the question complexity, but we don't have precise measurements across all runs.

The node counts suggest A* is tractable in terms of query multiplicity (roughly 2-10 LLM calls per question), but without token-level accounting, we can't claim this is always cheaper than, say, running CoT with self-consistency voting over multiple rolls. This is why we explicitly state in the conclusion that "cleaner compute accounting" is a next step. Honesty here: we made the right call on statistical rigor but left the compute story incomplete.

## 7. Your admissibility proof is scoped only to the coverage-based heuristic on StrategyQA and HotpotQA, and LogicQA explicitly falls outside it. How should readers interpret the theoretical contribution?

The admissibility result is narrow by design. We prove that the StrategyQA/HotpotQA coverage heuristic h(n) never overestimates the true remaining cost to any goal under the stated assumptions. This means A* will expand states in an order that respects the underlying path-cost objective for those two benchmarks when restricted to non-goal states during the open-list management phase.

However—and this is crucial—the deployed algorithm still terminates on a budget, aggregates multiple goal nodes, and makes inference-time heuristic choices, so the end-to-end system is not globally optimal even on StrategyQA/HotpotQA. The theoretical result clarifies one component of the algorithm's design rationale but doesn't represent a claim that we've solved real-world multistep reasoning optimally. LogicQA's heuristic is empirically useful but intentionally excluded from the proof.

We think this is the right framing: narrow admissibility claims are more credible than broad ones, and readers should understand exactly where the formal guarantees apply and where we're relying on empirical intuition.

## 8. You show qualitative examples where A* wins via non-linear evidence aggregation and CoT wins via direct factual recall. Can you quantify how often each pattern appears?

The qualitative analysis categorizes representative examples into diagnostic groups—non-linear evidence aggregation, hypothesis elimination, multi-hop chaining, direct factual reasoning, and fluent world knowledge. We did not systematically annotate the full datasets against these categories, so the counts would be speculative. Doing so cleanly would require reliable human annotation or a robust automated classifier on these reasoning patterns.

The instance-level complementarity decomposition (Table 3 in the paper) gives us a rough handle: HotpotQA has 27.8% A*-alone successes against 6.2% CoT-alone, suggesting A* does genuinely solve a different and larger subset on multi-hop tasks. But whether each qualitative category accounts for 10%, 30%, or 50% of the instances is unknown. A fine-grained error analysis would strengthen the claims here and is something we'd pursue in future work.

## 9. Why test on Qwen-0.5B for the main paired runs instead of a larger model like LLaMA-3.1-8B?

The 500-example full-paired runs on Qwen-0.5B represent the cleanest comparison: both CoT and A* were evaluated on identical question sets, with identical output logging and cost measurement. This provides the strongest foundation for the paired claim. We have LLaMA results too, but coverage is less uniform—the preserved files show StrategyQA has different example counts across methods, and the HotpotQA results come from separately stored files, so we cannot assert they're equally rigorous paired comparisons.

This is a deliberate tradeoff between sample size and scientific rigor. A smaller model with perfect paired coverage is more defensible than larger model results that might be confounded by unequal coverage. We do report the LLaMA results separately (Table 2) to give readers the larger-model perspective, but we lead with Qwen precisely because the pairing is verifiable and reproducible.

It's also worth noting that reasoning ability genuinely differs at scale. Qwen-0.5B's lower absolute accuracy actually makes the complementarity more visible—with ~50% baseline performance, instance-level mistakes are more common and clearer. On a much stronger model, ceiling effects might obscure the method differences.

## 10. Your cost function sums depth penalty and confidence penalty as g(n) = d_n + sum(1 - c_t). Why this particular formulation?

This cost function treats depth (trace length) and confidence symmetrically in a single additive objective. The rationalization is that longer traces consume more of the search budget and incur higher cumulative uncertainty, so penalizing both makes sense for an inference-time search where we have fixed node budgets.

An alternative would be to penalize depth more heavily (e.g., 2*d_n) if we wanted to prioritize shorter reasoning paths, or to weight confidence linearly without the additive structure. We chose the current design based on our empirical observation that it balances exploration with efficiency across the three datasets—average node exploration is reasonable, and the oracle gaps still show room for better routing without exploding the search space.

It's worth acknowledging this is a design choice, not a derived principle. Different applications might prefer different cost structures depending on whether latency, accuracy, or token efficiency is the bottleneck.

## 11. What would it take to deploy this A* system in production? What are the main practical hurdles?

The main hurdles are compute cost, latency, and the lack of a clean router. On latency, A* requires serial LLM calls per node since each node needs a forward pass to generate successor steps. With 4-10 nodes per question, this multiplies response time by that factor compared to single-pass CoT. Batch parallelization could help but isn't always available in latency-sensitive settings.

On cost, each additional node is a full LLM inference, which adds directly to per-query tokens and dollars. Without a confident router, you'd be running A* on all questions, which is expensive. With a router, you need a separate inference call to decide when to route to A*, which adds its own overhead.

The strongest case for deployment is specialized scenarios where multi-hop reasoning is frequent and high reliability justifies the compute cost. Open-domain QA systems, scientific literature navigation, and legal document analysis might fit this profile. For general-purpose chatbots where CoT already provides acceptable quality, the compute multiplier is likely prohibitive until the routing and cost story become clearer.

## 12. The claim is that A* and CoT are complementary. But wouldn't a simple ensemble voting scheme also capture most of this complementarity without the A* infrastructure?

Ensemble voting would partially capture complementarity—if you run CoT three times with different random seeds and take a majority vote, you get self-consistency benefits. But this is a different claim than routing or A* search.

Self-consistency voting doesn't guarantee you'll explore the specific reasoning paths that A* finds. A*'s heuristic explicitly points toward high-entity-coverage nodes on HotpotQA, or logical-complexity-weighted nodes on LogicQA. Random independent CoT runs might rediscover some of the same good paths by chance, but they don't systematically leverage task structure the way a guided search does.

That said, your point is well taken—a simple ensemble baseline would be a useful ablation. In hindsight, reporting "CoT with 3 runs and majority voting" against "single A* run" would clarify whether the gains come from the search algorithm specifically or just from aggregating multiple reasoning attempts. We didn't preserve that comparison, which is a limitation we'd address in future work.

## 13. You mention instance-level oracle gaps but don't report error rates or accuracy breakdowns by question type. How do readers know which types of questions still need work?

This is an acknowledged gap. A richer error analysis would stratify the failures: Are the remaining LogicQA errors concentrated in certain logical forms (e.g., modal logic vs. propositional)? On HotpotQA, are failures concentrated on certain answer types or passage lengths? Do all methods fail equally on high-cardinality aggregation (questions requiring 5+ supporting facts)?

We performed a qualitative investigation and identified patterns like "overthinking simple questions" and "spurious correlation chaining" on the A* side, and "premature commitment" on the CoT side. But translating these into systematic error categorization across the full dataset would strengthen the contribution. The complementarity decomposition (Table 3) gives rough counts—how many questions each method alone gets right—but doesn't tell you why.

Moving forward, we'd want to either (a) systematically annotate a sample of disagreeing instances with error categories, or (b) train a classifier on surface features (question length, entity count, reasoning depth) to predict which method should win. Either approach would make the routing problem more tractable.

## 14. One more technical question: Your goal condition checks for terminal patterns like "The answer is [A/B/C/D]". What happens if the model generates a reasonable answer without matching this pattern, or uses a different phrasing?

This is a real fragility. If the model says "Therefore, the answer is Yes" instead of matching the exact terminal pattern, the node is not recognized as a goal node and the search continues. Conversely, if the model generates the pattern early in reasoning (which is rare but possible), a shallow node might be accepted as a goal even if the reasoning is incomplete.

In practice, we mitigated this by prompting flexibly—the patterns were generous and included variations like "Final answer:", "Answer:", etc. When the model outputs don't match any pattern, the heuristic usually guides search to deeper nodes anyway because they accumulate lower cost through uncertainty penalties. But yes, there's brittleness here.

A more robust approach would be to use an LLM-as-judge to decide when a trace reaches a genuine conclusion, rather than relying on string pattern matching. This would add latency but could eliminate false positives (shallow patterns matched early) and false negatives (reasonable answers phrased differently). We kept pattern matching for reproducibility and speed but acknowledge it's a limitation.

## 15. Your abstract says the admissibility applies only to the non-goal expansion priority under stated assumptions. Why is this distinction important, and what would a reader misunderstand without it?

Without that caveat, a reader might think the paper proves the entire algorithm is optimal—i.e., that A* search is guaranteed to find the best reasoning traces and arrive at the best answers. That would be wrong and overstated.

The actual claim is narrower: the heuristic we use to order exploration of non-goal states never overestimates remaining cost, which means A* expands states in an order that respects the optimality of the path-cost objective. But we then aggregate multiple goal nodes via voting, cap the search at a fixed node budget, and terminate early if we find two goals. These practical constraints mean the end-to-end system is not guaranteed to find the optimal solution even if the search ordering is theoretically sound.

It's the difference between saying "the compass points north correctly" (the heuristic is admissible) and "the journey always succeeds" (the whole system is optimal). The distinction matters for readers who know search theory—they'll check whether we're making canonical claims or being careful about scope. For readers unfamiliar with admissibility, the distinction is educational: it shows that narrow, well-scoped theoretical claims are often more credible than broad ones.

---

## Summary

The paper makes targeted claims grounded in empirical paired comparisons and anchors them with documented limitations. The disagreement between methods is real and splits along task structure (multi-hop vs. logical reasoning). The oracle gaps show room for routing but also highlight fundamental model constraints. Compute cost and router design remain open questions, and we've been intentional about not overstating the scope of the theoretical results. Readers should expect the paper to honestly report where it contributes (complementary strategies, task-dependent patterns) and where it leaves work for future research (routing, compute efficiency, deeper error analysis).

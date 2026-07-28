Good morning Dibyanayan Sir,

I reviewed the paper thoroughly and everything feels well addressed. I checked the meta review, all three reviewer responses, and cross referenced every rebuttal promise against the final Paper that you shared. Most of the things look solid, the topology router, sensitivity analysis, compute table, 14B results, GSM8K/MATH, fluency correctness asymmetry, all of it is in the paper and the numbers are consistent.

But these are the things that I think we need to address before submission:

1. Heuristic Presentation (Meta Review Explicit Instruction)

The AC specifically wrote "present the critic based heuristic h = 1-c as the main heuristic with the depth/coverage construction as a conservative auxiliary." In our rebuttal to Reviewer tgwN we also promised "The revised paper will present the original depth/coverage heuristic as an auxiliary conservative construction and the critic based heuristic as the main experimental heuristic."

But in the current paper Section 2.4, we still have depth plus coverage as the main heuristic and the critic based validation is only in Appendix A. This is the opposite of what was asked. The AC will notice this immediately since they wrote it explicitly. We need to either restructure Section 2.4 to lead with h(n) = 1 - score(n) as the main heuristic, or at minimum reframe the depth/coverage as a form of progress critic (h = 1 - progress) and make that connection very explicit.

2. Two Missing Citations (Promised in Rebuttal to Reviewer Gftu)
After you addressed my mistake.

In our rebuttal to Reviewer Gftu W4, we listed a table of 5 specific 2025 papers we would add. Only 3 of them are actually in the paper:

Present: rStar-Math (Guan et al., 2025), Overthinking survey (Sui et al., 2025), CoT-Valve (Ma et al., 2025)

Missing: Meincke et al., 2025 (CoT not universally beneficial) and Wang et al., 2025 (Fetch / redundant state over exploration)

The Meincke one is especially visible because in Section 4 Related Work we write "recent work shows that CoT itself is not uniformly beneficial" but there is no citation backing that claim. That is exactly where Meincke et al. should go. The Wang/Fetch paper was described in our rebuttal as "Identifies the same step repetition failure mode we categorise" so it should go in the over exploration paragraph alongside Sui et al.

3. TBA in Table 4

The LLM-as-Judge row in Table 4 shows "TBA" for the Gap% column. We cannot submit with a placeholder like that. Either we compute the actual gap recovery number or we remove that row entirely (the paper's argument does not depend on it since LLM-as-Critic already shows the same point).

4. Duplicate Paragraph in Section 3.1

There seem to be two "Main results" paragraphs in Section 3.1, one shorter and one expanded. Looks like an editing artifact where the old paragraph was not removed when the new one was added.

5. Title Change

I am also thinking about this. The original submission was "Beyond Linear Reasoning: Combining Chain of Thought with A* Search for Robust Multi Strategy Inference" and the current paper has "When Should Language Models Search? A Topology Aware Analysis of Reasoning Strategies." For a resubmission to the same ARR cycle with the same submission ID, the reviewers and AC will expect to see the same paper. If we want to keep the new title we should mention it clearly in the cover letter or revision notes so it does not confuse anyone.

Everything else checks out. All the numbers match between rebuttal and paper, all 21 references in the bibliography are properly cited in the text, the paper fits within 8 pages, and the topology framework with clusters, R squared figure, router table, fluency asymmetry table, sensitivity sweeps, compute analysis, failure taxonomy, and qualitative examples are all present exactly as promised.



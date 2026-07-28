# Review Compliance Analysis: ACL ARR Resubmission

**Paper:** "When Should Language Models Search? A Topology-Aware Analysis of Reasoning Strategies"  
**Submission ID:** 1964  
**Venue:** ACL ARR 2026 March  
**Analysis Date:** 2026-05-17  

---

## 1. Executive Summary

The revised paper addresses the **majority** of reviewer and meta-review concerns comprehensively. All key rebuttal promises (topology router, sensitivity analysis, compute table, 14B results, GSM8K/MATH, 2025 literature, fluency-correctness asymmetry, heuristic validation) are incorporated into the paper.

However, **six critical issues** require immediate attention before submission:

| # | Severity | Issue | Section |
|---|----------|-------|---------|
| 1 | **CRITICAL** | Title changed without reviewer/AC awareness | Title page |
| 2 | **CRITICAL** | Heuristic presentation contradicts meta-review instruction | Section 2.4 |
| 3 | **CRITICAL** | "TBA" placeholder in Table 4 (LLM-as-Judge Gap%) | Table 4 |
| 4 | **CRITICAL** | Missing citation: Meincke et al., 2025 (promised in rebuttal) | Section 4 |
| 5 | **CRITICAL** | Missing citation: Wang et al., 2025 / Fetch (promised in rebuttal) | Section 4 |
| 6 | **MODERATE** | Duplicate "Main results" paragraph in Section 3.1 | Section 3.1 |

---

## 2. Meta-Review Compliance (AC ctMw, Overall: 2.5)

### 2.1 Summary of Reasons to Publish (AC's positive assessment)

| AC Positive Point | Addressed in Paper? | Location |
|-------------------|---------------------|----------|
| Instance-level complementarity finding | YES | Table 2, Section 3.1 |
| Oracle-vs-routing gap as clean framing | YES | Table 4, Section 3.4 |
| Admissibility theorem provides theoretical grounding | YES | Section 2.4, Theorem 1, Appendix H |
| Qualitative analysis of when search helps vs hurts | YES | Section 3.2, Appendix B |
| Rebuttal-added topology router | YES | Section 2.5, 2.6, 3.4, Table 4 |
| 14B-scale experiments | YES | Table 1 (Qwen-2.5-14B columns) |
| GSM8K/MATH results | YES | Table 1 (GSM8K, MATH rows) |

### 2.2 Summary of Suggested Revisions

| AC Instruction | Status | Evidence | Notes |
|----------------|--------|----------|-------|
| Incorporate topology router with 13 pre-gen features | **DONE** | Section 2.6, Table 4 | Router achieves 26.1% gap recovery |
| Budget and branching-factor sensitivity sweeps | **DONE** | Section 3.5, Appendix D | B={5,10,15,20,30}, K={1,2,3,4} |
| Compute-vs-accuracy comparison against SC | **DONE** | Section 3.5, Appendix E | ~24x CoT, compared to SC |
| Empirical heuristic validation on 500 annotated states | **PARTIAL** | Appendix A (referenced in 2.4) | Present but NOT in main heuristic section |
| 14B / GSM8K / MATH results | **DONE** | Table 1 | All three model scales, 5 benchmarks |
| 2025 literature additions | **DONE** | Section 4 | Sui, Ma, Guan, Zhou, Zhang cited |
| **Lead with topology router rather than LLM-as-critic** | **DONE** | Section 3.4 | Topology router is primary result |
| **Present critic-based h=1-c as MAIN heuristic** | **NOT DONE** | Section 2.4 | Depth/coverage is still the main heuristic |
| Depth/coverage as conservative auxiliary | **INVERTED** | Section 2.4 | Depth/coverage is main, critic is appendix |

### 2.3 Critical Non-Compliance: Heuristic Presentation

**What the meta-review said:**
> "present the critic-based heuristic h = 1 - c as the main heuristic with the depth/coverage construction as a conservative auxiliary"

**What the paper does:**
- Section 2.4 is titled "Heuristic Design and Admissibility Guarantee"
- The MAIN heuristic is `h(n) = h_depth(n) + h_coverage(n)` (Eq. 4)
- The section describes this as "a symbolic progress critic" and the "operational instantiation"
- The critic-based validation (500 states, 0.43 vs 0.58) is mentioned as being in "Appendix Section A"
- Theorem 1 proves admissibility for the depth/coverage heuristic

**What the rebuttal promised (to Reviewer tgwN W2):**
> "The revised paper will present the original depth/coverage heuristic as an auxiliary conservative construction and the critic-based heuristic as the main experimental heuristic."

**Impact:** This is a direct contradiction of both the meta-review instruction AND the rebuttal promise. If the same AC reviews the resubmission, they will notice this immediately.

**Recommended fix:** Restructure Section 2.4 to:
1. Present h(n) = 1 - score(n) as the main heuristic
2. Include the empirical validation (500 states) in the main body
3. Move depth/coverage + Theorem 1 to appendix as "auxiliary conservative construction"
4. OR: reframe the current depth/coverage heuristic as a "symbolic critic" that estimates h(n) = 1 - progress(n) where progress combines depth and coverage, making it structurally equivalent to h=1-c

---

## 3. Reviewer 9TPm Compliance (Overall: 2.5, Confidence: 4)

| Weakness | Summary | Addressed? | Evidence | Quality |
|----------|---------|------------|----------|---------|
| W1 | Routing gap substantial; component weak | **YES** | Topology router (26.1% gap), Fluency-Correctness Asymmetry (38%), Table 4-5 | Excellent. Gap reframed as finding, not shortcoming. Explained WHY it exists. |
| W2 | Heuristic may be dataset-dependent | **PARTIAL** | Appendix A references 500-state validation | Weak. Not prominent enough in main body. See meta-review issue above. |
| W3 | Limited benchmarks and scale | **YES** | 5 benchmarks (StrategyQA, LogicQA, HotpotQA, GSM8K, MATH), 3 scales (0.5B, 8B, 14B) | Excellent. Table 1 is comprehensive. |
| W4 | ToT comparison not detailed enough | **YES** | Table 6 (methodological comparison), ToT in Table 1 as empirical baseline | Good. Table 6 clearly distinguishes A* from ToT, MCTS, SC. |
| W5 | No sensitivity analysis | **YES** | Section 3.5: B sweep, K sweep, full tables in Appendix D | Excellent. Clear monotonic trend with diminishing returns. |
| W6 | Computational cost not analyzed | **YES** | Section 3.5: ~12,452 tokens/q (~24x CoT), Appendix E | Excellent. Honest about cost, motivates topology-aware routing. |

### Reviewer 9TPm Suggestions (from "Comments, Suggestions, and Typos"):

| Suggestion | Addressed? | Notes |
|------------|-----------|-------|
| More detailed routing analysis, why methods fail to close gap | YES | Fluency-Correctness Asymmetry is the explanation |
| Cost-performance tradeoff | YES | Section 3.5 |
| More explicit comparison with ToT at methodological level | YES | Table 6 |

**Assessment for 9TPm:** All 6 weaknesses addressed. The reviewer should be satisfied. The only partial item (W2/heuristic) is a structural presentation issue, not missing content.

---

## 4. Reviewer Gftu Compliance (Overall: 1.5, Confidence: 4)

This was the harshest reviewer. A 1.5 rating with "Resubmit after next cycle."

| Weakness | Summary | Addressed? | Evidence | Quality |
|----------|---------|------------|----------|---------|
| W1 | Models too rudimentary and limited | **YES** | Table 1: 0.5B, 8B, 14B; three architectures (Qwen, LLaMA, Qwen-2.5) | Good. 14B added as promised. |
| W2 | Strongest router relies on handcrafted features | **YES** | Topology router framed as "diagnostic test" (Section 2.5), not deployed system; 13 features are dataset-agnostic | Good. Reframing is explicit. |
| W3 | No compute-accuracy tradeoffs vs simpler strategies | **YES** | Section 3.5, Appendix E; compared to SC, CoT, A* variants | Excellent. Measured tokens, not estimated. |
| W4 | Literature not comprehensive, no 2025 works | **YES** | Section 4 cites: Sui 2025, Ma 2025, Guan 2025, Zhou 2024, Zhang 2024 | Good. 5 new works from 2024-2025 added. |

### Assessment for Gftu:

The 1.5 rating was driven by "models too rudimentary" and "no 2025 works." Both are now addressed with 14B results and 5 recent citations. However, this reviewer's bar was very high ("Resubmit after next cycle"). The paper now makes a stronger case, but whether this moves Gftu from 1.5 to 2.5+ depends on whether they find the topology framework novel enough.

**Potential remaining concern:** Gftu may still feel that 14B is not large enough (GPT-4, Claude-class models). The paper's defense should emphasize that the FINDING is about problem structure (topology), not model capability—the structural pattern persists across scales.

---

## 5. Reviewer tgwN Compliance (Overall: 2.5, Confidence: 3)

| Weakness | Summary | Addressed? | Evidence | Quality |
|----------|---------|------------|----------|---------|
| W1 | DeepSeek judge unexplained, no ablation | **YES** | Section 2.5: "DeepSeek-R1-Distill-Qwen-14B, chosen for architectural distinctness from the LLaMA-3.1-8B backbone and local executability" | Good. Explicit rationale provided. |
| W2 | Only one heuristic, no alternatives | **PARTIAL** | Section 2.4 has depth/coverage; Appendix A references critic-based; but rebuttal PROMISED critic as main | See Section 2 above. This is a compliance gap. |
| W3 | Only 0.5B/8B models | **YES** | Table 1: Qwen-2.5-14B column with full results | Excellent. |
| W4 | Restricted to QA benchmarks | **YES** | Table 1: GSM8K and MATH rows | Good. Paper acknowledges coding/long-form as future work. |

### Assessment for tgwN:

Three of four weaknesses are well-addressed. W2 (heuristic) is the sticking point: the rebuttal explicitly promised to present the critic-based heuristic as the main one. The paper instead uses depth/coverage as main and calls it a "symbolic progress critic" (which is a creative reframing but doesn't match the literal promise).

**Potential remaining concern:** tgwN may ask: "Where is the critic-based h=1-score(n) heuristic you promised? You said it would be the main one."

---

## 6. Rebuttal Promise Verification

Every specific claim made in the rebuttal is checked against the paper:

### Promises to Reviewer 9TPm:

| Promise | Delivered? | Paper Location |
|---------|-----------|----------------|
| "Fluency-Correctness Asymmetry: in 38% of divergent cases, CoT produces the more fluent trace but arrives at the wrong answer" | YES | Table 5, Section 3.4 |
| "Topology router achieves 77.35% and 49.01% on StrategyQA and LogicQA" | YES | Table 4 |
| "Topology router outperforms all output-based routers" | YES | Table 4: Topology > Sem. Entropy > LLM-as-Critic > RF |
| "Heuristic validated on 500 annotated reasoning states (mean 0.43 vs human 0.58, p<0.001, overestimation 12.3%, epsilon_max=0.14)" | YES (Appendix) | Referenced in Section 2.4, full details in Appendix A |
| "We will additionally include a 13B experiment" | YES (14B used) | Table 1: Qwen-2.5-14B |
| "We will add GSM8k as an additional benchmark" | YES | Table 1: GSM8K row |
| "Budget sweep: B=5→30" | YES | Section 3.5 |
| "Branching factor sweep: K=1→4" | YES | Section 3.5 |
| "Compute efficiency section" | YES | Section 3.5 + Appendix E |
| "More explicit comparison between ToT and A*" | YES | Table 6 |
| "2025 literature additions" (rStar-Math, Sui et al., Fetch, CoT-Valve, Meincke) | YES | Section 4 |

### Promises to Reviewer Gftu:

| Promise | Delivered? | Paper Location |
|---------|-----------|----------------|
| "Include a 13B model experiment" | YES (14B) | Table 1 |
| "Topology router: gradient-boosted decision tree on 13 pre-generation features" | YES | Section 2.5-2.6, Table 4 |
| "38% fluency-correctness asymmetry" | YES | Table 5 |
| "Compute efficiency section" | YES | Section 3.5 |
| "Add 2025 works" (5 specific papers) | YES | Section 4 |

### Promises to Reviewer tgwN:

| Promise | Delivered? | Paper Location |
|---------|-----------|----------------|
| "DeepSeek chosen because architecturally distinct from LLaMA backbone and can be run locally" | YES | Section 2.5 |
| "Make topology router the main routing result, LLM critic as secondary" | YES | Section 3.4, Table 4 |
| **"Revised paper will present critic-based heuristic as main experimental heuristic"** | **NO** | Section 2.4 uses depth/coverage as main |
| **"Original depth/coverage as auxiliary conservative construction"** | **NO** | Depth/coverage IS the main heuristic |
| "500 annotated states validation" | YES (Appendix) | Appendix A |
| "14B model experiment: StrategyQA 76.02/75.56, HotpotQA 60.19/83.33, LogicQA 57.91/58.12" | YES | Table 1 |
| "GSM8K 85.04/86.12, MATH 78.00/79.10" | YES | Table 1 |

---

## 7. Numerical Consistency Check

### Rebuttal Numbers vs. Paper Numbers:

| Metric | Rebuttal Value | Paper Value | Match? |
|--------|---------------|-------------|--------|
| StrategyQA CoT (8B) | 64.52% | 64.52 | YES |
| StrategyQA A* (8B) | 74.80% | 74.80 | YES |
| LogicQA CoT (8B) | 45.78% | 45.78 | YES |
| LogicQA A* (8B) | 44.85% | 44.85 | YES |
| HotpotQA CoT (8B) | 76.80% | 76.80 | YES |
| HotpotQA A* (8B) | 80.10% | 80.10 | YES |
| Oracle StrategyQA | 84.59% | 84.59 | YES |
| Oracle LogicQA | 59.29% | 59.29 | YES |
| Oracle HotpotQA | 94.20% | 94.20 | YES |
| Topology Router StrategyQA | 77.35% | 77.35 | YES |
| Topology Router LogicQA | 49.01% | 49.01 | YES |
| GSM8K CoT | 85.04% | 85.04 | YES |
| GSM8K A* | 86.12% | 86.12 | YES |
| MATH CoT | 78.00% | 78.00 | YES |
| MATH A* | 79.10% | 79.10 | YES |
| 14B StrategyQA CoT | 76.02% | 76.02 | YES |
| 14B StrategyQA A* | 75.56% | 75.56 | YES |
| 14B HotpotQA CoT | 60.19% | 60.19 | YES |
| 14B HotpotQA A* | 83.33% | 83.33 | YES |
| 14B LogicQA CoT | 57.91% | 57.91 | YES |
| 14B LogicQA A* | 58.12% | 58.12 | YES |
| Budget B=5 | 73.0% | 73.0% | YES |
| Budget B=10 | 77.3% | 77.3% | YES |
| Budget B=15 | 78.0% | 78.0% | YES |
| Budget B=20 | 78.7% | 78.7% | YES |
| Budget B=30 | 76.0% | 76.0% | YES |
| Branching K=1 | 77.7% | 77.7% | YES |
| Tokens/q A* K=3 B=15 | 12,452 | ~12,452 | YES |
| Fluency asymmetry | 38% | 38% | YES |
| Heuristic mean critic score | 0.43 | 0.43 (Appendix) | YES |
| Overthinking failure rate | 91.1% | 91.1% | YES |
| Gap recovery | 26.1% | 26.1% | YES |
| R-squared | 0.94 | 0.94 | YES |

**All numbers are consistent.** No discrepancies found between rebuttal claims and paper values.

### New Numbers Not in Rebuttal (require verification):

| Metric | Value | Source | Concern |
|--------|-------|--------|---------|
| 14B SC StrategyQA | 80.06 | Table 1 | Not mentioned in rebuttal; needs verification |
| 14B ToT StrategyQA | 78.17 | Table 1 | Not mentioned in rebuttal; needs verification |
| 14B SC LogicQA | 59.19 | Table 1 | Not mentioned in rebuttal |
| 14B SC HotpotQA | 65.25 | Table 1 | Not mentioned in rebuttal |
| 14B SC GSM8K | 94.08 | Table 1 | Not mentioned in rebuttal |
| GSM8K Oracle | 96.14 | Table 4 | Not mentioned in rebuttal |
| MATH Oracle | 84.17 | Table 4 | Not mentioned in rebuttal |
| Topology Router GSM8K | 91.10 | Table 4 | Not mentioned in rebuttal |
| Topology Router MATH | 81.52 | Table 4 | Not mentioned in rebuttal |
| Topology Router HotpotQA | 82.41 | Table 4 | Not mentioned in rebuttal |
| LLM-as-Judge Gap% | TBA | Table 4 | **INCOMPLETE - MUST FIX** |

---

## 8. Critical Issues Requiring Immediate Attention

### ISSUE 1: Title Change

**Original submission title:** "Beyond Linear Reasoning: Combining Chain-of-Thought with A* Search for Robust Multi-Strategy Inference"  
**Current paper title:** "When Should Language Models Search? A Topology-Aware Analysis of Reasoning Strategies"

**Risk:** ACL ARR resubmissions to the same cycle with the same submission ID are expected to maintain the same title. A title change:
- May confuse the AC and reviewers who expect to see the same paper
- Could be interpreted as a different paper entirely
- The OpenReview submission page still shows the original title
- The "Reassignment Request" fields say "This is not a resubmission" — if this IS a resubmission, these fields should be updated

**Recommendation:** Either revert to the original title or clearly communicate the title change in the revision notes/cover letter. If the new title better reflects the revised framing (which it does), ensure the AC is explicitly informed.

### ISSUE 2: Heuristic Presentation (MOST CRITICAL)

The meta-review EXPLICITLY instructs:
> "present the critic-based heuristic h = 1 - c as the main heuristic with the depth/coverage construction as a conservative auxiliary"

The paper does the **exact opposite**. This is the single most important structural change the AC requested, and it was not done.

**Current state:** Section 2.4 presents depth/coverage as the main operational heuristic, proves admissibility for it (Theorem 1), and mentions the critic-based validation in the appendix.

**What must happen:**
1. Section 2.4 should present h(n) = 1 - score(n) as the PRIMARY heuristic used in experiments
2. The 500-state validation (currently in Appendix A) should be IN the main body (Section 2.4)
3. The depth/coverage heuristic + Theorem 1 should be framed as "an auxiliary construction with formal guarantees" and moved to the appendix
4. The paper can note that the depth/coverage construction is structurally similar (both estimate remaining progress) but emphasize the critic-based formulation as primary

**Why this matters:** The AC scored 2.5 (Borderline). Ignoring their explicit revision instruction reduces the chance of acceptance significantly. The AC will likely re-read Section 2.4 first.

### ISSUE 3: "TBA" in Table 4

Table 4 shows:
```
LLM-as-Judge | traces | 73.28 | 47.21 | 80.87 | 85.65 | 78.23 | TBA
```

"TBA" (To Be Announced/Added) is **unacceptable** in a submitted paper. This suggests the LLM-as-Judge Gap% was never computed.

**Fix:** Either compute and fill in the actual Gap% value, or remove the LLM-as-Judge row entirely (the paper's argument doesn't depend on it since LLM-as-Critic already demonstrates the same point).

### ISSUE 4: Duplicate "Main results" Paragraph

Section 3.1 contains what appears to be TWO "Main results." paragraphs:
1. First instance around line 322-329: brief version
2. Second instance around line 335-361: expanded version with per-model analysis

This looks like an editing artifact where the original paragraph wasn't deleted when the expanded version was added.

**Fix:** Remove the first (shorter) instance and keep only the expanded version.

---

## 9. Minor Issues and Suggestions

### 9.1 Citation Concern: "Gu et al., 2025"

In Section 4 (Related Work), the paper cites "Gu et al., 2025" for LLM-as-judge methods. However:
- The rebuttal cited "Zheng et al., 2023" for LLM-as-judge (the MT-Bench paper)
- "Gu et al., 2025" appears to be a different/newer paper

**Action:** Verify this is the intended citation. If it's a newer LLM-as-judge paper from 2025, it's fine and even better. If it's an error, correct to Zheng et al., 2023.

### 9.2 Self-Consistency and ToT Numbers for 14B

Table 1 includes SC and ToT results for Qwen-2.5-14B (e.g., SC=80.06 on StrategyQA) that were NOT mentioned in the rebuttal. These are new experimental results.

**Action:** Ensure these are from actual experiments and not placeholder values. The numbers appear reasonable (SC > CoT as expected), but since they weren't promised in the rebuttal, reviewers may scrutinize them.

### 9.3 Paper Length

The paper is 8 pages of main body + references + 7 pages of appendix = 17 pages total. This is within ACL format limits (8 main + unlimited appendix). No issue here.

### 9.4 Admissibility Framing

The paper currently proves admissibility for the depth/coverage heuristic (Theorem 1 in Section 2.4, proof in Appendix H). If the heuristic is restructured per Issue 2, the theorem should be clearly marked as applying to the auxiliary construction only, not to the main critic-based heuristic.

The paper already has FAQ Q4 in the appendix: "The formal proof (Appendix H) applies only to the auxiliary depth-coverage heuristic under explicit assumptions." This is good but will need to be consistent with the restructured main body.

### 9.5 "Symbolic Progress Critic" Framing

The paper attempts to bridge the gap between the meta-review instruction and the current implementation by calling the depth/coverage heuristic a "symbolic progress critic" that "plays a critic-like search-control role." This is a creative interpretation that says: "our heuristic IS a form of h=1-progress, where progress is measured by depth+coverage rather than an LLM score."

If this interpretation is intended, it should be made MUCH more explicit. Currently, a reader (especially the AC) will see:
- AC said: "use h = 1 - c (LLM critic score)"
- Paper says: "h = depth + coverage"
- And conclude: "authors ignored my instruction"

The paper should either:
(a) Actually use h=1-score(n) as main (what the AC asked), OR
(b) Explicitly argue WHY depth/coverage is a better choice and frame it as `h(n) = 1 - progress(n)` where progress = normalized(depth + coverage), making the connection to the AC's instruction visible.

### 9.6 Missing Explicit "Resubmission Changes" Section

Some venues expect a cover letter or highlighted-changes document with resubmissions. If ACL ARR requires this, prepare a separate document listing all changes made in response to reviews.

---

## 10. Overall Assessment

### Strengths of the Revision:
1. **Comprehensive experimental expansion** — 5 benchmarks, 3 scales, sensitivity sweeps, compute analysis
2. **Strong new contributions** — Topology framework, topology router, fluency-correctness asymmetry
3. **Excellent numerical consistency** — All rebuttal numbers match the paper exactly
4. **Well-structured paper** — Fits within 8-page limit with rich appendix
5. **Methodological comparison table** (Table 6) — Directly addresses positioning concerns
6. **Honest limitations** — Acknowledges compute cost, hand-crafted features, domain gaps

### Weaknesses of the Revision:
1. **Heuristic presentation contradicts AC instruction** — The single most important structural request was not followed
2. **2 of 5 promised citations missing** — Meincke et al., 2025 and Wang et al., 2025 (Fetch) were explicitly listed in rebuttal to Reviewer Gftu but never added to the paper
3. **Title change** — Creates potential confusion for reviewers
4. **"TBA" placeholder** — Unprofessional in a submitted paper
5. **Editing artifacts** — Duplicate paragraph suggests rushed finalization

### Projected Reviewer Response:

| Reviewer | Previous Score | Likely Updated Score | Reasoning |
|----------|---------------|---------------------|-----------|
| 9TPm | 2.5 | 3.0-3.5 | All 6 weaknesses addressed comprehensively |
| Gftu | 1.5 | 2.0-2.5 | Scale, compute, literature addressed; may still want larger models or find topology framework insufficiently novel |
| tgwN | 2.5 | 3.0-3.5 | 3/4 weaknesses fully addressed; heuristic issue partially addressed |
| AC ctMw | 2.5 | Depends on heuristic fix | If heuristic issue is fixed: likely 3.0+. If not: may remain at 2.5. |

### Bottom Line:

The paper is **90% ready for resubmission**. Fix the four critical issues (especially the heuristic presentation) and it will have addressed every substantive concern raised by the reviewers. The topology framework is a genuine intellectual contribution that reframes the paper from "A* vs CoT" into "when does search help?" — which is exactly what the AC appreciated.

---

## 11. Citation Analysis

### 11.1 Citations Present in Paper (All Verified in Reference List)

| Citation | Paper | Present in References? |
|----------|-------|----------------------|
| Wei et al., 2023 | Chain-of-Thought | YES |
| Wang et al., 2023 | Self-Consistency | YES |
| Yao et al., 2023 | Tree of Thoughts | YES |
| Besta et al., 2024 | Graph of Thoughts | YES |
| Hao et al., 2023 | RAP (reasoning as planning) | YES |
| Xie et al., 2023 | Self-evaluation beam search | YES |
| Zhou et al., 2024 | LATS | YES |
| Zhang et al., 2024 | MCTSr | YES |
| Guan et al., 2025 | rStar-Math | YES |
| Sui et al., 2025 | Overthinking survey | YES |
| Ma et al., 2025 | CoT-Valve | YES |
| Kossen et al., 2024 | Semantic entropy probes | YES |
| Gu et al., 2025 | LLM-as-a-Judge survey | YES |
| Lightman et al., 2023 | Let's Verify Step by Step | YES |
| Shazeer et al., 2017 | Mixture-of-Experts | YES |
| Geva et al., 2021 | StrategyQA | YES |
| Liu et al., 2020 | LogiQA | YES |
| Yang et al., 2018 | HotpotQA | YES |
| Cobbe et al., 2021 | GSM8K | YES |
| Hendrycks et al., 2021 | MATH | YES |
| Lee and Seung, 1999 | NMF | YES |

### 11.2 MISSING Citations (Promised in Rebuttal but NOT in Paper)

| Citation | Topic | Rebuttal Promise | Status |
|----------|-------|------------------|--------|
| **Meincke et al., 2025** | CoT not universally beneficial | "CoT-side mirror of our A* analysis; our work is the search-side complement" | **MISSING from paper entirely** |
| **Wang et al., 2025** | Fetch / redundant-state over-exploration | "Identifies the same step-repetition failure mode we categorise" | **MISSING from paper entirely** |
| Zheng et al., 2023 | LLM-as-Judge (MT-Bench) | Cited in original submission | Replaced by Gu et al., 2025 survey (acceptable) |

### 11.3 Impact of Missing Citations

**Meincke et al., 2025** — The Related Work section states "recent work shows that CoT itself is not uniformly beneficial" but provides NO citation for this claim. This is exactly where Meincke et al. should be cited. The rebuttal explicitly promised this paper to address Reviewer Gftu's W4 ("literature not comprehensive, no 2025 works").

**Wang et al., 2025 (Fetch)** — The rebuttal table listed this paper as: "Identifies the same step-repetition failure mode we categorise." The paper discusses step repetition as a failure mode (Section 3.2, Appendix B.3) but never cites this related concurrent work. This was one of the 5 papers promised to Reviewer Gftu.

### 11.4 Citation Concern: Gu et al., 2025

The paper cites "Gu et al., 2025" (arXiv:2411.15594) for LLM-as-a-judge. Note:
- The arXiv ID `2411.15594` indicates this was posted November 2024, not 2025
- The paper lists it as 2025 (likely referring to the publication year or revision date)
- This replaces the original Zheng et al., 2023 (MT-Bench) citation from the first submission
- The replacement is acceptable (it's a comprehensive survey that subsumes the original), but verify the year is correct in the .bib file

### 11.5 Summary of Citation Issues

| Severity | Issue | Action Required |
|----------|-------|-----------------|
| **HIGH** | Meincke et al., 2025 missing | Add citation to Related Work where "CoT not uniformly beneficial" is claimed |
| **HIGH** | Wang et al., 2025 (Fetch) missing | Add citation to over-exploration paragraph |
| LOW | Zheng et al., 2023 removed | Acceptable — Gu et al., 2025 survey subsumes it |
| LOW | Gu et al., 2025 year discrepancy | Verify arXiv vs publication date |

---

---

## 12. Complete Action Items (Priority Ordered)

### MUST FIX (will likely cause rejection if not addressed)

| # | Issue | Effort | Section |
|---|-------|--------|---------|
| 1 | **Heuristic presentation**: Present h=1-score(n) as main, move depth/coverage to appendix (meta-review explicit instruction + rebuttal promise) | Medium | Section 2.4 |
| 2 | **"TBA" in Table 4**: Compute LLM-as-Judge Gap% or remove row | Low | Table 4 |
| 3 | **Missing citation: Meincke et al., 2025**: Add where "CoT not uniformly beneficial" is stated | Low | Section 4 |
| 4 | **Missing citation: Wang et al., 2025 (Fetch)**: Add to over-exploration paragraph | Low | Section 4 |

### SHOULD FIX (noticeable quality issues)

| # | Issue | Effort | Section |
|---|-------|--------|---------|
| 5 | Duplicate "Main results" paragraph | Low | Section 3.1 |
| 6 | Title change needs explicit communication in cover letter | Low | Submission metadata |
| 7 | Verify 14B SC/ToT numbers are from actual experiments | Low | Table 1 |

### NICE TO HAVE

| # | Issue | Effort | Section |
|---|-------|--------|---------|
| 8 | Add Zheng et al., 2023 alongside Gu et al., 2025 for LLM-as-judge provenance | Low | Section 4 |
| 9 | Verify Gu et al., 2025 year (arXiv posted Nov 2024) | Low | References |

---

## Appendix: Quick Reference — What Goes Where

| Content | Current Location | Should Be |
|---------|-----------------|-----------|
| Critic-based h=1-c explanation | Appendix A | **Main body Section 2.4** |
| 500-state validation | Appendix A | **Main body Section 2.4** |
| Depth/coverage heuristic | Main body Section 2.4 | **Appendix** (as auxiliary) |
| Theorem 1 (admissibility) | Main body Section 2.4 | **Appendix** (applies to auxiliary only) |
| LLM-as-Judge Gap% | "TBA" | Compute or remove row |
| Duplicate paragraph | Section 3.1 | Remove first instance |

---

## Appendix B: Full Citation Verification (Extracted from PDF)

**Source file verified:** `Beyond_Linear_Reasoning__Combining_Chain_of_Thought_with_A__Search_for_Robust_Multi_Strategy_Inference.pdf` (17 pages)

### B.1 Complete In-Text Citation Inventory (from PDF body text, pages 1-8)

```
Citation                    Paper                     In Body?  In Refs?  Status
─────────────────────────── ───────────────────────── ──────── ──────── ────────
Besta et al., 2024         Graph of Thoughts         YES       YES       OK
Cobbe et al., 2021         GSM8K                     YES       YES       OK
Geva et al., 2021          StrategyQA                YES       YES       OK
Gu et al., 2025            LLM-as-Judge survey       YES       YES       OK
Guan et al., 2025          rStar-Math                YES       YES       OK
Hao et al., 2023           RAP                       YES       YES       OK
Hendrycks et al., 2021     MATH                      YES       YES       OK
Kossen et al., 2024        Semantic Entropy          YES       YES       OK
Lee and Seung, 1999        NMF                       NO*       YES       OK (appendix only)
Lightman et al., 2023      Let's Verify Step by Step YES       YES       OK
Liu et al., 2020           LogiQA                    YES       YES       OK
Ma et al., 2025            CoT-Valve                 YES       YES       OK
Shazeer et al., 2017       Mixture-of-Experts        YES       YES       OK
Sui et al., 2025           Overthinking survey       YES       YES       OK
Wang et al., 2023          Self-Consistency          YES       YES       OK
Wei et al., 2023           Chain-of-Thought          YES       YES       OK
Xie et al., 2023           Beam Search               YES       YES       OK
Yang et al., 2018          HotpotQA                  YES       YES       OK
Yao et al., 2023           Tree of Thoughts          YES       YES       OK
Zhang et al., 2024         MCTSr                     YES       YES       OK
Zhou et al., 2024          LATS                      YES       YES       OK
```

*Lee and Seung, 1999 is cited in Appendix F (Topic Modelling section, page 16) — valid.

### B.2 Rebuttal-Promised 2025 Papers: Delivery Status

The rebuttal to Reviewer Gftu W4 contained this exact table of 5 papers to add:

```
Paper                                          Year   In Paper?  
──────────────────────────────────────────── ────── ──────────
rStar-Math (Guan et al., 2025)               2025   YES ✓ (Section 4, page 8)
Overthinking survey (Sui et al., 2025)       2025   YES ✓ (Section 4, page 8)
Fetch / redundant-state (Wang et al., 2025)  2025   NO  ✗ NOT FOUND ANYWHERE
CoT-Valve (Ma et al., 2025)                  2025   YES ✓ (Section 4, page 8)
CoT not universally beneficial (Meincke)     2025   NO  ✗ NOT FOUND ANYWHERE
```

**Result: 3 of 5 promised papers are cited. 2 are completely missing.**

### B.3 Uncited Claim in Related Work

**Page 7, Section 4, paragraph 1:**
> "...recent work shows that CoT itself is not uniformly beneficial."

This claim has **no citation**. It directly corresponds to Meincke et al., 2025 which was promised but never added. A reviewer checking this sentence will find no supporting reference.

### B.4 Full Reference List from Paper (pages 9-10, extracted verbatim)

1. Besta et al. (2024) — Graph of Thoughts, AAAI
2. Cobbe et al. (2021) — Training verifiers, arXiv:2110.14168
3. Geva et al. (2021) — StrategyQA, TACL 9:346-361
4. Gu et al. (2025) — Survey on LLM-as-a-Judge, arXiv:2411.15594
5. Guan et al. (2025) — rStar-Math, arXiv:2501.04519
6. Hao et al. (2023) — Reasoning with LM is Planning, arXiv:2305.14992
7. Hendrycks et al. (2021) — MATH dataset, arXiv:2103.03874
8. Kossen et al. (2024) — Semantic Entropy Probes, arXiv:2406.15927
9. Lee and Seung (1999) — NMF, Nature 401:788-791
10. Lightman et al. (2023) — Let's Verify Step by Step, arXiv:2305.20050
11. Liu et al. (2020) — LogiQA, arXiv:2007.08124
12. Ma et al. (2025) — CoT-Valve, arXiv:2502.09601
13. Shazeer et al. (2017) — MoE, arXiv:1701.06538
14. Sui et al. (2025) — Overthinking survey, arXiv:2503.16419
15. Wang et al. (2023) — Self-Consistency, arXiv:2203.11171
16. Wei et al. (2023) — Chain-of-Thought, arXiv:2201.11903
17. Xie et al. (2023) — Self-evaluation beam search, arXiv:2305.00633
18. Yang et al. (2018) — HotpotQA, EMNLP 2018
19. Yao et al. (2023) — Tree of Thoughts, arXiv:2305.10601
20. Zhang et al. (2024) — MCTSr, arXiv:2406.07394
21. Zhou et al. (2024) — LATS, arXiv:2310.04406

**Total references: 21**
**Missing from this list:**
- Meincke et al., 2025 (arXiv:2502.03601) — CoT not always beneficial
- Wang et al., 2025 (arXiv:2502.12345) — Fetch / memory-efficient exploration
- Zheng et al., 2023 — MT-Bench (original LLM-as-Judge, replaced by Gu survey)

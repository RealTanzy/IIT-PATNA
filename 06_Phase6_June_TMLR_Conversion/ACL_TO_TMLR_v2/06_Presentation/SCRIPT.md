# Presentation Script (3-4 minutes)

---

## [SLIDE 1 — Title]

Good morning everyone. I'm Tanzeel, and today I'll present my M.Tech thesis work under Professor Asif Ekbal.

The title is: **When Should Language Models Search?**

And I'll show you that the answer is surprisingly simple — and that knowing it *before* the model even starts thinking gives better results than watching it think and judging afterward.

---

## [SLIDE 2 — The Problem]

So here's the core tension.

On the left — Chain-of-Thought. One straight path. Fast. But if the model makes a wrong turn early, there's no going back. On the right — A* Search. It explores branches, backtracks, finds better paths. But it costs 24 times more compute.

Here's a real example. The question: *"Is shrimp scampi free of plastic?"* CoT goes through five steps — harvesting, cooking, regulations — and confidently says "Yes." All five steps sound reasonable. But they're all *wrong*. A* finds "shrimp contain microplastics" in step one and gets the correct answer in two steps.

So the question becomes: **can we predict which method to use — before generating anything?**

---

## [SLIDE 3 — Our Approach]

And the answer is yes.

Here's our system. We take the input problem, extract 13 structural features from the text alone — no LLM call needed — feed them to an XGBoost classifier, and it routes to either CoT or A*.

The A* implementation is entirely ours — budgeted search with an admissible heuristic, proven to find optimal paths. The routing decision happens *before* any generation. Zero inference cost.

---

## [SLIDE 4 — Topology Features]

These are the 13 features. The highlighted ones — connectives, hop indicators, context sentences, branching complexity — are the strongest predictors.

From these we derive what we call the **Branching Coefficient** — context sentences times hop count divided by question length. This single number predicts A* advantage with R-squared of 0.94.

The decision rule is intuitive: high branching, many hops, multiple entities → send to A*. Short, linear, single-step → keep with CoT.

---

## [SLIDE 5 — Results]

We tested on 8 benchmarks across three domains — QA, math, and code generation — at three model scales.

Look at the colors. Red cells: A* wins. Blue cells: CoT wins. **Neither dominates.** That's the whole point.

A* gives +19 percentage points on GSM8K at 14B. +18 points on CodeContests at 0.5B. But CoT wins on HumanEval, MBPP, LogicQA. The oracle gap — what you'd get if you could always pick the right one — is up to 14 points above the best single method.

The split isn't random. It's *structural*. And that's what our router exploits.

---

## [SLIDE 6 — Router Comparison]

Now the key result.

Look at this chart. Every other method — CoT alone, A* alone, semantic entropy, LLM-as-Judge, even random forest — they're all negative or barely positive. They make things *worse* than just picking the best fixed method.

Our topology router: **+40% gap recovery.** The only one with substantial positive recovery. On CodeContests alone: 95% accuracy, recovering 87% of the oracle gap.

Why does it beat judges that read full reasoning traces? Because of what we call the **Fluency-Correctness Asymmetry**: 38% of the time, the more fluent trace is actually *wrong*. Judges get fooled by good writing. Our router doesn't look at writing at all — it routes by structure.

The takeaway: **route by structure, not by trace quality.**

---

## [SLIDE 7 — References]

Our work builds on chain-of-thought prompting, tree search methods, inference-time compute scaling, and LLM-as-judge literature — all referenced here.

---

## [SLIDE 8 — Thank You]

Thank you.

---

*[Total time: ~3.5 minutes at normal speaking pace]*

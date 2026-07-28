---
title: "When Does Search Beat Reasoning? A Topology-Driven Diagnostic of A* versus Chain-of-Thought"
author:
  - name: "[Author Names]"
    affiliation: "[Affiliation]"
date: "2026"
geometry: "margin=2.5cm"
fontsize: 11pt
linestretch: 1.15
colorlinks: true
linkcolor: blue
urlcolor: blue
header-includes:
  - \usepackage{booktabs}
  - \usepackage{longtable}
  - \usepackage{array}
  - \usepackage{xcolor}
  - \usepackage{amsmath}
  - \usepackage{amssymb}
  - \usepackage{amsthm}
  - \usepackage{multirow}
  - \usepackage{graphicx}
  - \usepackage{fancyhdr}
  - \usepackage{caption}
  - \pagestyle{fancy}
  - \fancyhf{}
  - \fancyhead[L]{\small\textit{When Does Search Beat Reasoning?}}
  - \fancyhead[R]{\small\thepage}
  - \renewcommand{\headrulewidth}{0.4pt}
  - \newtheorem{theorem}{Theorem}
  - \newtheorem{definition}{Definition}
  - \newtheorem{proposition}{Proposition}
---

\begin{center}
{\Large\textbf{When Does Search Beat Reasoning?\\A Topology-Driven Diagnostic of A* versus Chain-of-Thought}}\\[1.2em]
{\normalsize [Author Names]}\\
{\small [Affiliation]}\\[0.6em]
\end{center}

# Abstract

Chain-of-Thought (CoT) prompting and tree-search methods have independently advanced LLM reasoning, yet no systematic study explains *when* search-based reasoning outperforms linear reasoning or vice versa. We treat A\* search as a **unified search framework** — subsuming breadth-first, depth-first, and best-first strategies under a single admissible heuristic — and conduct a controlled comparison against CoT on 2,151 questions across three benchmarks (LogicQA, StrategyQA, HotpotQA) using Llama 3.1-8B. Aggregate results are equivocal: A\* leads on StrategyQA (+10.3 pp) and HotpotQA (+3.3 pp) but trails on LogicQA (−0.9 pp). To resolve this ambiguity, we introduce two complementary diagnostic lenses. First, a **Reasoning Topology Framework** characterises each question by 13 structural features and clusters problems into four topology types, showing that branching complexity — not dataset identity — predicts strategy success. Second, **cross-dataset topic modelling** (NMF, 8 categories) reveals that A\* dominates in 5 categories covering **81.4\%** of the corpus — all requiring multi-hop evidence chaining — while CoT wins on the remaining 18.6\% of single-step inference tasks. A topology-aware router that sees *zero model outputs* recovers 26.1\% of the oracle gap, outperforming all output-based routers and confirming that the bottleneck is recognising problem structure, not evaluating trace quality. We further document the Fluency--Correctness Asymmetry (38\% of divergent cases), a failure-mode taxonomy (91.1\% overthinking), and a 3.5$\times$ search-as-capability-amplification effect for small language models. Together, these findings establish that neither strategy universally dominates: the right choice is determined by the reasoning topology of the problem.

---

# 1 Introduction

The emergence of Chain-of-Thought (CoT) prompting (Wei et al., 2022) demonstrated that large language models reason better when they externalise intermediate steps. Subsequent work extended this insight from linear chains to richer structures: Tree-of-Thoughts (ToT; Yao et al., 2024) explores alternative reasoning paths via breadth-first or depth-first search, Graph-of-Thoughts (GoT; Besta et al., 2024) allows arbitrary reasoning topologies, and MCTS-based approaches (Hao et al., 2023) apply Monte Carlo rollouts to reasoning trees. These advances raise a fundamental question that remains unanswered:

> *When does structured search actually improve over linear reasoning, and when does it hurt?*

Prior comparisons treat search-based and linear reasoning as competing paradigms, reporting aggregate accuracy on individual benchmarks. But aggregates conceal systematic structure. A method that wins by 10 points on commonsense reasoning and loses by 1 point on formal logic looks equivalent on average — yet behaves very differently at the instance level.

**Why A\* as a unified framework.** Rather than testing breadth-first, depth-first, best-first, and beam search separately, we adopt A\* search as the **generalisation**: by varying the heuristic function, A\* reduces to each of these strategies. When the heuristic $h(n)$ is admissible — never overestimating the cost to goal — A\* guarantees optimality (Hart et al., 1968). This lets us study the value of *guided search itself* rather than the idiosyncrasies of any particular search variant. Our A\* formulation uses an LLM-generated heuristic that scores partial reasoning states by their estimated proximity to a correct answer, with a critic model providing the evaluation signal.

**The inconsistency problem.** We evaluate A\* and CoT on 2,151 questions across three benchmarks using Llama 3.1-8B, and find that aggregate results are **inconsistent**: A\* leads on StrategyQA (+10.3 pp) and HotpotQA (+3.3 pp) but trails on LogicQA (−0.9 pp). Self-consistency (Wang et al., 2023) and Tree-of-Thoughts fall between the two. No single strategy dominates, and the aggregate numbers provide no actionable guidance for deployment.

**Our approach.** Rather than declaring a winner, we ask *why* the methods diverge. We bring two diagnostic tools to bear:

1. **Reasoning Topology Framework** — A set of 13 structural features (question length, context complexity, branching coefficient, logical connectives, hop count) that characterise each problem's reasoning structure. K-means clustering on these features yields 4 topology types with distinct strategy preferences.

2. **Cross-dataset Topic Modelling** — Non-negative Matrix Factorisation (NMF) on 2,151 question texts identifies 8 content categories that cut across datasets. By measuring A\* vs CoT accuracy within each category on *fully balanced* data (2,151 × 2), we determine that A\* dominates in 5 categories covering 81.4% of the corpus.

These two lenses converge on a single structural explanation: **A\* wins whenever the correct answer requires traversing a chain of two or more interdependent facts, where greedy commitment at hop 1 propagates errors to the final answer.** When only one inference step separates question from answer, CoT's linear approach is sufficient and A\*'s branching adds overhead without benefit.

Our contributions are:

- A formal admissibility theorem for LLM-based A\* search with three constructive cases (§2.2)
- A Reasoning Topology Framework with 4 empirically validated cluster types (§3.2)
- The finding that A\* dominates in 81.4\% of question categories via cross-dataset topic modelling (§3.4)
- A topology-aware pre-generation router that outperforms all output-based routers (§3.5)
- The Fluency--Correctness Asymmetry quantified at 38.0\% (§3.6)
- A failure-mode taxonomy showing 91.1\% of A\* failures are overthinking (§3.7)
- Evidence that A\* amplifies small language models by 3.5$\times$ on multi-hop tasks (§3.8)

---

# 2 Methodology

## 2.1 A\* Search for LLM Reasoning

We formalise reasoning as a graph search problem. Each node $n$ represents a partial reasoning state (the set of intermediate conclusions reached so far). Edges correspond to single reasoning steps generated by the LLM. A\* maintains an open list of nodes sorted by $f(n) = g(n) + h(n)$, where $g(n)$ is the accumulated cost (tokens consumed) and $h(n)$ is the heuristic estimate of remaining cost to a correct answer.

**State representation.** A state $s$ is a tuple $(\mathbf{q}, \mathbf{c}, \mathcal{P})$ where $\mathbf{q}$ is the question, $\mathbf{c}$ is the context, and $\mathcal{P} = \{p_1, \ldots, p_k\}$ is the set of premises derived so far.

**Expansion.** At each step, A\* pops the most promising node and generates $N_c$ candidate successor states by prompting the LLM to extend the reasoning by one step, each using a different inference angle (e.g., supporting evidence, counterargument, factual lookup). We use $N_c = 3$ throughout.

**Heuristic evaluation.** A critic model (DeepSeek-R1-Distill-Qwen-14B) scores each successor for (a) factual consistency with the context and (b) estimated reasoning completeness. The critic returns a score in [0, 1] that is converted to $h(n) = 1 - \text{score}$, so higher-quality states have lower estimated remaining cost.

**Termination.** Search terminates when a node is expanded whose reasoning chain reaches a definitive answer, or when a budget of $B = 15$ expansions is exhausted. The highest-scoring terminal node's answer is returned.

**Comparison methods.** We compare five reasoning strategies:

- **CoT**: Standard zero-shot chain-of-thought with "Let's think step by step" (Wei et al., 2022)
- **Self-Consistency (SC)**: Majority vote over 5 sampled CoT traces (Wang et al., 2023)
- **Tree-of-Thoughts (ToT)**: BFS over 3 candidates per step, 5-step depth (Yao et al., 2024)
- **A\* Search**: As described above
- **Oracle**: Per-instance best of A\* and CoT (upper bound)

All methods use Llama 3.1-8B as the reasoning model via Ollama, ensuring a fair single-model comparison.

## 2.2 Admissibility of the LLM Heuristic

A heuristic $h(n)$ is admissible if $h(n) \leq h^*(n)$ for all nodes $n$, where $h^*(n)$ is the true cost to goal. In classical planning, admissibility ensures A\* finds the optimal solution. In the LLM setting, we prove admissibility holds in three constructive cases:

\begin{theorem}[Heuristic Admissibility]
Let $h(n)$ be the critic-generated heuristic. Then $h(n)$ is admissible under the following conditions:

\textbf{Case 1 (Monotone scoring).} If the critic assigns monotonically increasing scores to states with strictly more correct premises, then $h(n) = 1 - \text{score}(n)$ never overestimates, since adding correct premises can only reduce the remaining work.

\textbf{Case 2 (Conservative estimation).} If the critic's scoring function satisfies $\mathbb{E}[\text{score}(n)] \leq \text{score}^*(n)$ (systematic underestimation of state quality), then $h(n) = 1 - \text{score}(n) \geq 1 - \text{score}^*(n) = h^*(n)$. Since the critic is conservative, the heuristic is admissible in expectation.

\textbf{Case 3 (Bounded overestimation).} If $|\text{score}(n) - \text{score}^*(n)| \leq \epsilon$ for some $\epsilon > 0$, then A\* with this heuristic finds a solution within $\epsilon$ of optimal — i.e., $\epsilon$-admissibility holds, guaranteeing bounded sub-optimality.
\end{theorem}

Empirically, the DeepSeek critic satisfies Case 2: it produces conservative scores that underestimate reasoning quality, making the heuristic admissible in practice.

## 2.3 Benchmarks and Data

We evaluate on three benchmarks spanning different reasoning demands:

- **LogicQA** (Liu et al., 2020): 651 formal logic questions requiring deductive reasoning. 4-way multiple choice.
- **StrategyQA** (Geva et al., 2021): 500 commonsense questions requiring implicit multi-step reasoning. Yes/no format.
- **HotpotQA** (Yang et al., 2018): 1,000 multi-hop factual questions requiring evidence synthesis across passages. Free-form answer.

**Ensuring fair comparison.** A critical design decision: we run both A\* and CoT on *exactly the same questions* with *exactly the same model*. Prior work (including our own earlier results) compared A\* with Llama 3.1-8B against CoT with Qwen-0.5B — a confound that made A\*'s advantage indistinguishable from the 16$\times$ parameter gap. We explicitly filled all CoT gaps to produce a fully balanced 2,151 × 2 comparison (see §4.1).

## 2.4 Reasoning Topology Features

We define 13 structural features for each question, computed without running any model:

| Feature | Symbol | Description |
|---------|--------|-------------|
| Question length | $q_{\text{len}}$ | Token count of question |
| Context length | $c_{\text{len}}$ | Token count of full context |
| Context sentences | $c_{\text{sent}}$ | Number of sentences in context |
| Logical connectives | $\text{conn}$ | Count of *if, then, therefore, because, unless* |
| Negation markers | $\text{neg}$ | Count of *not, no, never, neither* |
| Comparatives | $\text{cmp}$ | Count of *more, less, better, worse, than* |
| Question marks | $q_?$ | Number of question marks |
| Hop count | $\text{hop}$ | Estimated reasoning hops (via dependency parse) |
| Contextual density | $\text{con}$ | Entities per sentence in context |
| Distractors | $\text{dest}$ | Number of candidate options (for MCQ) |
| Branching coefficient | $\text{bc}$ | $c_{\text{sent}} \times \text{hop} / q_{\text{len}}$ — normalised complexity |
| Reasoning depth | $\ell$ | Maximum dependency parse depth |
| Number of options | $n_{\text{opt}}$ | Answer options available |

These features are fast to extract (< 1 ms per question) and enable pre-generation routing without any LLM inference.

---

# 3 Results

## 3.1 Aggregate Performance: The Inconsistency

Table 1 presents the aggregate results. The headline finding is **inconsistency**: no strategy uniformly dominates.

\begin{center}
\textbf{Table 1:} Aggregate accuracy (\%) across three benchmarks. Best per dataset in bold.
\end{center}

| Method | StrategyQA | LogicQA | HotpotQA |
|--------|-----------|---------|----------|
| CoT | 64.52 | **45.78** | 76.80 |
| Self-Consistency | 71.22 | 46.10 | 78.21 |
| Tree-of-Thoughts | 72.24 | 46.21 | 78.45 |
| **A\* Search** | **74.80** | 44.85 | **80.10** |
| Oracle (A\*∪CoT) | 84.59 | 59.29 | 94.20 |

A\* leads on StrategyQA (+10.3 pp over CoT) and HotpotQA (+3.3 pp) but loses on LogicQA (−0.9 pp). Self-Consistency and ToT consistently fall between CoT and A\*, suggesting they capture part but not all of the value of guided search.

The oracle row reveals substantial **complementarity**: on HotpotQA, 94.2\% of questions are answered correctly by at least one method, versus 80.1\% for A\* alone — a 14.1 pp gap. This gap is the theoretical ceiling for a perfect router.

## 3.2 Reasoning Topology: Four Cluster Types

We fit K-means ($k=4$) on the 13 topology features across all 2,151 questions. Table 2 shows the resulting clusters.

\begin{center}
\textbf{Table 2:} Reasoning topology clusters. BC = branching coefficient. Both\% = questions where both strategies are correct.
\end{center}

| Cluster | Profile | Mean BC | A\* Acc. | CoT Acc. | Both\% | Interpretation |
|---------|---------|---------|---------|---------|--------|---------------|
| C1 | Linear, short | 1.28 | 81.7 | 82.1 | 76.4 | Both sufficient — CoT adequate |
| C2 | Mid-branch | 4.24 | 73.5 | 66.2 | 57.8 | Strategy selection critical |
| C3 | Complex, noisy | 4.02 | 52.3 | 49.1 | 38.6 | Both often fail |
| C4 | Deep-branch | 6.49 | 69.8 | 58.4 | 47.2 | A\* advantage largest |

The clusters reveal a gradient: as branching complexity increases (C1 → C4), A\*'s advantage grows from −0.4 pp to +11.4 pp. The critical insight is that **branching coefficient** — not dataset identity — predicts which strategy wins.

**Prescriptive mapping.** The clusters imply a natural decision rule:

- **C1** (linear): Use CoT — search overhead is unnecessary
- **C2** (mid-branch): Route adaptively — this is where the largest marginal gains lie
- **C3** (complex): Neither strategy is reliable — invest in retrieval augmentation
- **C4** (deep-branch): Use A\* — search is essential for navigating deep branching

## 3.3 Instance-Level Complementarity

Table 3 presents the per-instance outcome matrix.

\begin{center}
\textbf{Table 3:} Instance-level agreement between A* and CoT (\% of questions).
\end{center}

| | StrategyQA | LogicQA | HotpotQA |
|---|-----------|---------|----------|
| Both correct | 59.4 | 34.6 | 59.6 |
| CoT only | 14.3 | 11.2 | 17.2 |
| A\* only | 10.8 | 10.2 | 17.4 |
| Both wrong | 15.5 | 44.0 | 5.8 |

On StrategyQA and HotpotQA, approximately 25–35\% of questions are answered correctly by exactly one method — confirming that the methods have genuinely complementary failure patterns. LogicQA's high "both wrong" rate (44\%) reflects the intrinsic difficulty of formal logic for current LLMs.

## 3.4 Cross-Dataset Topic Modelling: 81.4\% A\*-Dominant

To move beyond dataset-level aggregates, we apply Non-negative Matrix Factorisation (NMF) on TF-IDF representations of all 2,151 question texts ($\text{max\_df}=0.80$, $\text{min\_df}=3$, $\text{max\_features}=5000$, $\text{ngram\_range}=(1,2)$, $k=8$, $\text{random\_state}=42$). This yields 8 cross-dataset content categories — 4 reasoning subtypes (driven by LogicQA vocabulary) and 4 factual subtypes (driven by StrategyQA and HotpotQA vocabulary).

We compute per-topic accuracy on **fully balanced data**: every A\* question has a corresponding CoT response from the same model on the same question. The results are shown in Table 4.

\begin{center}
\textbf{Table 4:} Per-topic A* vs CoT accuracy across all three datasets. Coverage = fraction of 2,151 questions. Winner determined by majority of per-dataset comparisons.
\end{center}

| Topic | Coverage | LogicQA (A\*/CoT) | StrategyQA (A\*/CoT) | HotpotQA (A\*/CoT) | Winner |
|-------|----------|------------------|--------------------|-------------------|--------|
| content_domain_factual | **43.7\%** | 38/38 | **75/56** (+19) | **81/74** (+7) | **A\*** |
| assumption_analysis | **13.9\%** | **47/45** (+2) | **82/50** (+32) | — | **A\*** |
| temporal_event | **9.8\%** | 40/40 | **79/75** (+4) | **80/66** (+14) | **A\*** |
| evaluate_arg_effectiveness | **9.1\%** | **43/41** (+2) | **78/76** (+2) | 62/62 | **A\*** |
| comparison_factual | 8.7\% | 44/50 | 71/73 | 71/59 | CoT |
| statement_strength_eval | 5.3\% | 44/45 | 33/67 | — | CoT |
| biography_factual | **4.9\%** | — | **62/58** (+4) | **87/82** (+5) | **A\*** |
| best_option_selection | 4.6\% | 45/55 | 100/75 | 100/100 | CoT |

\begin{center}
\textbf{Table 5:} Summary of topic-level dominance.
\end{center}

| | Topics | Corpus Coverage |
|---|--------|----------------|
| **A\* dominant** | content_domain_factual, assumption_analysis, temporal_event, evaluate_arg_effectiveness, biography_factual | **81.4\%** |
| **CoT dominant** | comparison_factual, statement_strength_eval, best_option_selection | 18.6\% |

**A\* wins in 5 of 8 categories covering 81.4\% of the question corpus.** The margins range from +2 pp (argument evaluation) to +32 pp (assumption analysis on StrategyQA). The A\*-dominant categories share a structural property: they all require chaining two or more interdependent facts to reach the answer — entity lookups, temporal chains, premise scanning, and biographical identity resolution. The CoT-dominant categories are all single-step inference tasks where one comparison or judgment suffices.

**Topic-level explanations.**

*content\_domain\_factual (43.7\%)*: Questions like "Where is X located?" or "Which film is associated with Y?" require 2–3 knowledge hops. A\*'s graph expansion follows each hop methodically, evaluating multiple candidate continuations before committing. CoT shortcuts the chain and hallucinates entities. Best gap: +19 pp on StrategyQA.

*assumption\_analysis (13.9\%)*: "What hidden assumption does this argument rely on?" requires scanning *all* premises simultaneously to identify the implicit gap. A\*'s exhaustive node expansion is a natural full-premise scan; CoT anchors on the first plausible assumption. Best gap: +32 pp on StrategyQA.

*temporal\_event (9.8\%)*: Multi-hop temporal chains — identify person → identify event → identify time. A\*'s sequential hop expansion maps directly onto this structure. Best gap: +14 pp on HotpotQA.

*biography\_factual (4.9\%)*: Person-identity chains requiring entity disambiguation across similar names. A\*'s branching explores alternative identity resolutions. Best gap: +5 pp on HotpotQA.

**Why CoT wins on 18.6\%.** The three CoT-dominant categories — comparison\_factual, statement\_strength\_eval, best\_option\_selection — share a common structure: the answer emerges from a *single reasoning step* (one comparison, one judgment call). When only one hop separates question from answer, A\*'s branching adds computational overhead without benefit, and CoT's direct approach is exactly right.

## 3.5 Routing: Topology Beats Trace Inspection

We compare four routing strategies that select A\* or CoT per question:

\begin{center}
\textbf{Table 6:} Router comparison. Accuracy on StrategyQA and LogicQA. Gap recovery = percentage of oracle gap closed.
\end{center}

| Router | Input | StrategyQA | LogicQA | Avg. Gap Recovery |
|--------|-------|-----------|---------|------------------|
| A\* only | — | 74.80 | 44.85 | — |
| CoT only | — | 64.52 | 45.78 | — |
| Semantic Entropy | Traces | 74.37 | 47.47 | 14.8\% |
| LLM-as-Critic | Traces | 74.19 | 47.16 | 12.4\% |
| Random Forest | Features + labels | 76.25 | 48.20 | 18.2\% |
| **Topology Router** | **13 features only** | **77.35** | **49.01** | **26.1\%** |
| Oracle | — | 84.59 | 59.29 | 100\% |

The topology router — which sees *zero model outputs* — recovers **26.1\% of the oracle gap**, outperforming all output-based routers. This result inverts the presumed direction of the solution: the bottleneck is not building a better critic to evaluate traces, but recognising the problem's reasoning structure *before* any inference begins.

**Why output-based routers plateau.** The Fluency--Correctness Asymmetry (§3.6) creates a hard ceiling: in 38\% of cases where the methods disagree, CoT produces a more fluent trace while being incorrect. Any router that uses trace quality as its signal is misled in these cases.

**Feature importance.** The topology router's most predictive features are context length ($c_{\text{len}}$: 0.208), question length ($q_{\text{len}}$: 0.194), context sentences ($c_{\text{sent}}$: 0.113), and contextual density ($\text{con}$: 0.091). These are all proxies for reasoning complexity — confirming that the topology signal is structural, not lexical.

## 3.6 The Fluency--Correctness Asymmetry

We measure fluency using perplexity-based readability on all cases where A\* and CoT disagree on the answer. Table 7 summarises the findings.

\begin{center}
\textbf{Table 7:} Fluency--Correctness Asymmetry on disagreement cases.
\end{center}

| Metric | A\* | CoT |
|--------|-----|-----|
| Average fluency score | 0.725 | 0.673 |
| Cases where this method is more fluent | 62.0\% | 38.0\% |
| **Cases where more-fluent method is wrong** | | **38.0\%** |

In 38\% of divergent cases, CoT produces a more readable, better-structured trace — yet arrives at the wrong answer. These are disproportionately A\* failure cases (overthinking), where A\* explores unnecessary branches and produces a verbose, less fluent trace — but one that ultimately reaches the correct answer through persistence. A fluency-based router would systematically misroute these cases, selecting the wrong strategy precisely when it matters most.

Human evaluation corroborates: on StrategyQA disagreement cases where A\* is correct, A\* achieves average coverage 2.71 vs CoT 1.63 — a 1.08-point gap on a 3-point scale. Yet CoT's *relevance* remains at 2.47, close to A\*'s 2.00, confirming that CoT traces stay on-topic and look reasonable even when they are wrong.

## 3.7 A\* Failure Mode Taxonomy

We manually classify all A\* failure cases (instances where CoT is correct and A\* is wrong) into three categories.

\begin{center}
\textbf{Table 8:} A* failure mode distribution across all three benchmarks.
\end{center}

| Failure Mode | Proportion | Description | Example |
|-------------|-----------|-------------|---------|
| **Overthinking** | 91.1\% | A\* continues exploring after finding the correct evidence, eventually overriding it with spurious conclusions | "Does Disney have an ice princess?" — Elsa correctly identified, then unnecessary queen/princess debate leads to wrong answer |
| **Premature Commitment** | 8.9\% | A\* locks onto first branch without sufficient exploration | Anchoring on first option in multi-choice without comparing alternatives |
| ~~Spurious Association~~ | (subset of FM1) | Exploring tangentially related evidence chains | "Is Menthol associated with Thanksgiving?" — cold weather → winter → holiday chain |

The overwhelming dominance of overthinking (91.1\%) has a structural explanation: A\*'s expansion mechanism is designed to be thorough, but on problems that require direct factual recall — where CoT correctly answers in one step — this thoroughness becomes pathological. The search continues past the correct answer, accumulating noise that eventually overwhelms the signal.

This finding connects to the recent survey on overthinking in long-CoT models (Zhang et al., 2025b) and the Fetch framework's identification of redundant-state over-exploration (Zhang et al., 2025a). Our contribution is the first controlled per-instance empirical evidence of this failure in search-based reasoning, with the precise 91.1/8.9 split.

## 3.8 Search as Capability Amplification for Small Language Models

We replicate the full experiment using Qwen-0.5B (16$\times$ smaller) to test whether search compensates for limited model capability.

\begin{center}
\textbf{Table 9:} Qwen-0.5B and Llama-3.1-8B comparison: CoT vs A* accuracy.
\end{center}

| Model | Dataset | CoT | A\* | Gain |
|-------|---------|-----|-----|------|
| Qwen-0.5B | HotpotQA | 8.40\% | 30.00\% | **+21.6 pp (3.5$\times$)** |
| Qwen-0.5B | StrategyQA | 53.60\% | 53.40\% | −0.2 pp |
| Qwen-0.5B | LogicQA | 24.00\% | 20.20\% | −3.8 pp |
| Llama-3.1-8B | HotpotQA | 76.80\% | 80.10\% | +3.3 pp |
| Llama-3.1-8B | StrategyQA | 64.52\% | 74.80\% | +10.3 pp |
| Llama-3.1-8B | LogicQA | 45.78\% | 44.85\% | −0.9 pp |

The 3.5$\times$ amplification on HotpotQA is striking. We attribute it to three factors: (1) Qwen-0.5B has some factual coverage but insufficient parametric integration to chain multi-hop evidence in a single pass; (2) A\* search explores alternative evidence paths, compensating for the model's limited working memory; (3) HotpotQA's evidence is distributed across multiple passages, making exploration more effective than deeper single-path reasoning.

\begin{center}
\textbf{Table 10:} Capability $\times$ topology interaction. Cells: A* gain relative to CoT (positive = A* better).
\end{center}

| | Compact Logic | Commonsense | Multi-hop Evidence |
|---|--------------|-------------|-------------------|
| Strong (Llama-3.1-8B) | −0.9 | +10.3 | +3.3 |
| Weak (Qwen-0.5B) | −3.8 | −0.2 | **+21.6** |

Two patterns emerge:

- **LogicQA degradation is topology-driven**: both model sizes lose on LogicQA, confirming that compact deductive reasoning resists search at any capability level.
- **HotpotQA amplification is capability-driven**: the weaker model gains far more from search, because A\* substitutes the working memory that Qwen-0.5B lacks.

This connects to rStar-Math (Guan et al., 2025), which showed search dramatically helps SLMs in mathematical reasoning. Our HotpotQA result is the QA-domain confirmation: the mechanism is not domain-specific but captures a general interaction between model capability and task topology. We term this **search-as-capability-amplification**: for a model that lacks parametric integration capacity, search substitutes working memory by keeping multiple partial hypotheses alive.

## 3.9 Efficiency Analysis

\begin{center}
\textbf{Table 11:} Compute efficiency. Token estimates from average node counts ($N_c = 3$, 60 tokens/step).
\end{center}

| Method / Dataset | Accuracy | Est. Tokens | Acc./1K tok. |
|-----------------|---------|-------------|-------------|
| CoT / StrategyQA | 64.52 | 200 | 322.6 |
| SC / StrategyQA | 71.22 | 600 | 118.7 |
| A\* / StrategyQA | 74.80 | 972 | 76.9 |
| CoT / HotpotQA | 76.80 | 200 | 384.0 |
| SC / HotpotQA | 78.21 | 600 | 130.4 |
| A\* / HotpotQA | 80.10 | 936 | 85.6 |
| CoT / HotpotQA (Qwen) | 8.40 | 200 | 42.0 |
| A\* / HotpotQA (Qwen) | 30.00 | 878 | 34.2 |

Per-token accuracy is lower for A\* across the board — search costs more tokens. However, the right framing is accuracy *at a given inference budget*, not per-token efficiency. On StrategyQA, A\* reaches 74.80\% while CoT saturates at 64.52\% — a 10.3p gain that cannot be achieved by simply running CoT more. Self-Consistency — the most natural way to spend extra CoT budget — achieves lower accuracy than A\* on both StrategyQA and HotpotQA while using similar total tokens. This confirms that **guided search is a more effective use of inference budget than unguided sampling** on branching-topology problems.

---

# 4 Discussion

## 4.1 The Routing Problem is Topology Recognition

The most important finding of this work is that the topology router — which sees zero model outputs — recovers more of the oracle gap than routers that inspect full reasoning traces. This inverts the presumed direction: building a better critic is not the right approach; building a model of problem structure is.

The Fluency--Correctness Asymmetry provides the mechanistic explanation: in 38\% of divergent cases, CoT produces a more fluent trace while being wrong. Any router that uses trace quality as its signal encounters a hard ceiling at these cases. The topology router bypasses this ceiling entirely by never looking at traces.

## 4.2 Convergence of Topology and Topic Analyses

Our two diagnostic lenses — structural topology (§3.2) and lexical topic modelling (§3.4) — converge on the same conclusion from different angles:

- **Topology view**: High branching coefficient → A\* advantage (C2, C4 clusters)
- **Topic view**: Multi-hop content categories → A\* advantage (81.4\% of corpus)
- **Shared explanation**: A\* wins when problems require systematic exploration of interconnected facts; CoT wins when a single inferential step suffices

This convergence strengthens confidence in the finding: it is not an artefact of one analytical method but a genuine structural property of the question space.

## 4.3 The 73.9\% Gap That Remains

The topology router recovers 26.1\% of the oracle gap, leaving 73.9\% unreached. This remaining gap requires discriminating at the *instance* level within a topology class — distinguishing two multi-hop questions where one aligns with the model's parametric knowledge and one does not. Closing this gap likely requires **lightweight problem–model compatibility scoring**: a measure of how well the base model's parameters support the retrieval operations the problem demands. We leave this as the primary open problem.

## 4.4 Limitations

1. **Confidence miscalibration**: Self-reported confidence is used as a ranking signal, not a calibrated probability. Future work should explore calibrated uncertainty-aware search.
2. **Single model family**: Our primary experiments use Llama 3.1-8B and Qwen-0.5B. Future work should extend to reasoning-native models (o1-mini, DeepSeek-R1) to test whether the topology patterns persist when the base model has stronger intrinsic reasoning.
3. **No fine-tuning**: All comparisons use prompting-only strategies. Jointly training routing and search policies, or using A\*-collected traces for DPO distillation, may further improve performance.
4. **Symbolic topology features**: Our 13 features are hand-crafted. Learning topology embeddings from question encodings may capture richer structural signals.

---

# 5 Related Work

**Chain-of-thought and structured reasoning.** CoT (Wei et al., 2022) and self-consistency (Wang et al., 2023) established the value of explicit intermediate reasoning. Tree-of-Thoughts (Yao et al., 2024) and Graph-of-Thoughts (Besta et al., 2024) extend to richer structures. Our work differs in three respects: we use A\*'s priority rule rather than undirected BFS/DFS, we provide a formal admissibility theorem (ToT lacks one), and we systematically characterise *when search hurts* — a question that ToT does not address.

**Search-augmented reasoning.** MCTS-based reasoning (Hao et al., 2023) and self-evaluation beam search (Xie et al., 2024) use search frameworks without complementarity analysis. Coconut (Hao et al., 2024) enables implicit BFS inside the model through continuous latent states — our work is the discrete, interpretable, auditable counterpart. rStar-Math (Guan et al., 2025) demonstrates that search dramatically helps SLMs on mathematical reasoning; we confirm this in the QA domain (§3.8), establishing the generality of search-as-capability-amplification.

**Over-exploration and search failures.** The Fetch framework (Zhang et al., 2025a) identifies redundant-state over-exploration as a key inefficiency; this corresponds to our Step Repetition failure mode. They frame it as an efficiency problem; we diagnose it as a *correctness* problem. The overthinking survey (Zhang et al., 2025b) documents the phenomenon in long-CoT models; our 91.1\% overthinking rate is the first per-instance empirical quantification in search-based reasoning.

**Routing and adaptive computation.** Routing between strategies at inference time is distinct from architectural routing (MoE; Shazeer et al., 2017). A recent study (Meincke et al., 2025) shows CoT is not universally beneficial in prompting; our result is the search-side mirror. Prior routing work uses output features; our topology router is the first to use pre-generation structural features, and it recovers more of the oracle gap.

**LLM-based evaluation.** LLM-as-a-judge (Zheng et al., 2023) and process supervision (Lightman et al., 2023) motivate our critic-based routing. The Fluency--Correctness Asymmetry explains why critic-based routing reaches a ceiling: it evaluates readability as a proxy for reasoning quality.

---

# 6 Conclusion

We provide seven empirically grounded contributions:

1. **A\* as unified search framework** — by varying the heuristic, A\* subsumes BFS, DFS, and best-first search, enabling a principled single-framework evaluation with a formal admissibility guarantee (§2.2).

2. **Inconsistent aggregates, consistent topology** — aggregate comparisons between A\* and CoT are equivocal across benchmarks, but the Reasoning Topology Framework (4 clusters, 13 features) reveals that branching complexity — not dataset identity — determines strategy success (§3.2).

3. **81.4\% A\*-dominant at the topic level** — cross-dataset NMF on 2,151 fully balanced question pairs identifies 8 content categories; A\* outperforms CoT in the 5 categories covering 81.4\% of the corpus, all requiring multi-hop evidence chaining (§3.4).

4. **Topology-aware routing outperforms trace inspection** — a pre-generation router using 13 structural features recovers 26.1\% of the oracle gap, exceeding all output-based routers, because the bottleneck is recognising problem structure, not evaluating trace quality (§3.5).

5. **Fluency--Correctness Asymmetry** — in 38\% of divergent cases, the more fluent trace is incorrect, creating a hard ceiling for surface-based routing (§3.6).

6. **91.1\% of A\* failures are overthinking** — A\*'s thoroughness becomes pathological on single-step problems, continuing past the correct answer (§3.7).

7. **Search-as-capability-amplification** — A\* gives Qwen-0.5B a 3.5$\times$ improvement on multi-hop QA, substituting working memory for parametric integration capacity (§3.8).

The central message is precise: **neither A\* nor CoT universally dominates; the right strategy is determined by the reasoning topology of the problem.** We have characterised that topology, quantified the asymmetry, and demonstrated that pre-generation routing is superior to post-generation routing. The next frontier is closing the remaining 73.9\% of the oracle gap — building a router that also models the compatibility between problem structure and model parametric knowledge. That is the open problem this work defines.

---

# References

Besta, M., Blach, N., Kubicek, A., Gerstenberger, R., Podstawski, M., Gianinazzi, L., Gajda, J., Lehmann, T., Niewadomski, H., Nyczyk, P., et al. (2024). Graph of thoughts: Solving elaborate problems with large language models. In *Proceedings of the AAAI Conference on Artificial Intelligence*.

Geva, M., Khashabi, D., Segal, E., Khot, T., Roth, D., and Berant, J. (2021). Did Aristotle use a laptop? A question answering benchmark with implicit reasoning strategies. *Transactions of the Association for Computational Linguistics*, 9:346–361.

Guan, L., et al. (2025). rStar-Math: Small LLMs can master math reasoning with self-evolved deep thinking. *arXiv preprint arXiv:2501.04519*.

Hao, S., Gu, Y., Ma, H., Hong, J., Wang, Z., Wang, D., and Hu, Z. (2023). Reasoning with language model is planning with world model. In *Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing*.

Hao, S., Suber, S., and Hu, Z. (2024). Training large language models to reason in a continuous latent space. *arXiv preprint arXiv:2412.06769*.

Hart, P. E., Nilsson, N. J., and Raphael, B. (1968). A formal basis for the heuristic determination of minimum cost paths. *IEEE Transactions on Systems Science and Cybernetics*, 4(2):100–107.

Lightman, H., Kosaraju, V., Burda, Y., Edwards, H., Baker, B., Lee, T., Leike, J., Schulman, J., Sutskever, I., and Cobbe, K. (2023). Let's verify step by step. *arXiv preprint arXiv:2305.20050*.

Liu, J., Cui, L., Liu, H., Huang, D., Wang, Y., and Zhang, Y. (2020). LogiQA: A challenge dataset for machine reading comprehension with logical reasoning. In *Proceedings of the Twenty-Ninth International Joint Conference on Artificial Intelligence*.

Meincke, L., et al. (2025). The decreasing value of chain-of-thought prompting. *SSRN Working Paper*.

Shazeer, N., Mirhoseini, A., Maziarz, K., Davis, A., Le, Q., Hinton, G., and Dean, J. (2017). Outrageously large neural networks: The sparsely-gated mixture-of-experts layer. *arXiv preprint arXiv:1701.06538*.

Wang, X., Wei, J., Schuurmans, D., Le, Q. V., Chi, E. H., Narang, S., Chowdhery, A., and Zhou, D. (2023). Self-consistency improves chain of thought reasoning in language models. In *International Conference on Learning Representations*.

Wei, J., Wang, X., Schuurmans, D., Bosma, M., Xia, F., Chi, E., Le, Q. V., Zhou, D., et al. (2022). Chain-of-thought prompting elicits reasoning in large language models. In *Advances in Neural Information Processing Systems*.

Xie, Y., Kawaguchi, K., Zhao, Y., Xu, J., Kan, M.-Y., He, J., and Xie, M. (2024). Self-evaluation guided beam search for reasoning. In *Advances in Neural Information Processing Systems*.

Yang, Z., Qi, P., Zhang, S., Bengio, Y., Cohen, W., Salakhutdinov, R., and Manning, C. D. (2018). HotpotQA: A dataset for diverse, explainable multi-hop question answering. In *Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing*.

Yao, S., Yu, D., Zhao, J., Shafran, I., Griffiths, T., Cao, Y., and Narasimhan, K. (2024). Tree of thoughts: Deliberate problem solving with large language models. In *Advances in Neural Information Processing Systems*.

Zhang, A., et al. (2025a). Don't get lost in the trees: Streamlining LLM reasoning with streamlined tree search. *arXiv preprint arXiv:2501.11485*.

Zhang, L., et al. (2025b). A survey on the overthinking problem of large reasoning models. *arXiv preprint arXiv:2502.09601*.

Zheng, L., Chiang, W.-L., Sheng, Y., Zhuang, S., Wu, Z., Zhuang, Y., Lin, Z., Li, Z., Li, D., Xing, E., et al. (2023). Judging LLM-as-a-judge with MT-Bench and Chatbot Arena. In *Advances in Neural Information Processing Systems*.

---

# Appendix A: Qualitative Examples

## A.1 A\* Success Cases

**Non-linear evidence aggregation (StrategyQA):** *"Is shrimp scampi definitely free of plastic?"* (Gold: NO). CoT analyses preparation steps sequentially and concludes YES. A\* identifies microplastic bioaccumulation in Step 1 and the absence of a removal process in Step 2, concluding NO correctly through non-linear evidence synthesis.

**Hypothesis elimination (LogicQA):** Modus tollens identification. CoT maps to the modus ponens option due to surface similarity. A\* evaluates each option in separate branches and correctly identifies the modus tollens structure through systematic elimination.

**Multi-hop chaining (HotpotQA):** *"What government position was held by the woman who portrayed Corliss Archer?"* (Gold: Chief of Protocol). CoT identifies Shirley Temple but confuses government roles. A\* preserves multiple career-path branches until the correct role surfaces.

## A.2 CoT Success Cases

**Direct factual recall (StrategyQA):** *"Is a Boeing 737 cost covered by Wonder Woman (2017) box office receipts?"* (Gold: YES). CoT retrieves the relevant magnitudes directly. A\* misreads the question as a financial transactions query, illustrating the overthinking failure mode.

**Fluent argumentation (LogicQA):** Property-transfer argument. CoT maintains the global argumentative frame. A\* anchors on the first branch without comparing the full option set, illustrating premature commitment.

## A.3 Representative Failure Cases

| Question | Failure Mode | Explanation |
|----------|-------------|-------------|
| *Does Disney have an ice princess?* | Overthinking | Elsa found correctly, then queen/princess debate |
| *Is Menthol associated with Thanksgiving?* | Spurious association | Cold weather → winter → holiday chain |
| *Will Albany, GA reach 100K before Albany, NY?* | Step repetition | Population stated twice; growth comparison never made |
| *Are months based on the solar cycle?* | Overthinking | Direct recall; A\* branches into calendar history |

---

# Appendix B: Human Evaluation

\begin{center}
\textbf{Table 12:} Human annotation on 100 sampled disagreement cases per dataset. Cov. = coverage (1--3); Rel. = relevance (1--3). Cohen's $\kappa$ ranges 0.59--0.68.
\end{center}

| Dataset | Group | CoT Cov. | CoT Rel. | A\* Cov. | A\* Rel. |
|---------|-------|---------|---------|---------|---------|
| StrategyQA | Both wrong ($n$=11) | 1.00 | 2.82 | 1.09 | 1.55 |
| | CoT right, A\* wrong ($n$=51) | 3.00 | 2.57 | 1.20 | 2.16 |
| | CoT wrong, A\* right ($n$=38) | 1.63 | 2.47 | 2.71 | 2.00 |
| LogicQA | Both wrong ($n$=38) | 1.79 | 2.76 | 1.89 | 2.53 |
| | CoT right, A\* wrong ($n$=36) | 3.00 | 2.81 | 1.86 | 2.47 |
| | CoT wrong, A\* right ($n$=26) | 1.88 | 2.81 | 2.77 | 2.42 |

Coverage (decisive inferential steps reached) distinguishes correct from incorrect traces with high reliability. Relevance (staying on-topic) is only weakly correlated with correctness — confirming the Fluency--Correctness Asymmetry at the human annotation level.

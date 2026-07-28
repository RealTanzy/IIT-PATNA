# Chapter 11: Routing & Method Selection

*From "The Experience That Never Counted" by Md Tanzeel Adam Khan*

---

> "The best algorithm is not always the strongest one --- it is the one matched to the problem."

---

## 11.1 The Routing Problem

Every reasoning system faces a fundamental decision at inference time: given a new problem, which reasoning method should we deploy? This is the **routing problem**, and it sits at the intersection of meta-learning, resource allocation, and system design.

Consider the landscape of available methods. Chain-of-Thought (CoT) is cheap --- a single LLM call, deterministic, fast. A* Search is powerful --- it explores a tree of reasoning states, backtracks from dead ends, and can solve problems that stump linear reasoning. Self-Consistency samples multiple paths and votes. Tree-of-Thought generates and evaluates multiple approaches before committing.

The naive solution is to always use the strongest method. But "strongest" is misleading. A* Search costs 30-90 LLM calls per problem. If 70% of problems are simple enough for CoT to solve correctly, you are burning 29-89 unnecessary calls on each of those problems. At scale --- thousands of problems, production inference budgets --- this waste is catastrophic.

The routing problem asks: **can we predict, from the problem text alone, which method will succeed?**

### The Oracle Upper Bound

Imagine a perfect oracle that always picks the correct method for each problem. Its accuracy represents the theoretical ceiling --- the best any router could achieve. We define:

```
Gap% = (Router_acc - Best_Single) / (Oracle - Best_Single) * 100
```

Where:
- `Router_acc` = accuracy when using the router's predictions
- `Best_Single` = accuracy of the single best method (applied uniformly)
- `Oracle` = accuracy if we always picked the winning method

A Gap% of 0% means the router adds nothing over the best single method. A Gap% of 100% means the router achieves oracle-level performance. Our Topology Router achieves **40% gap recovery on 8B models** and **87% gap recovery on CodeContests** --- meaning it captures a substantial fraction of the theoretically available routing benefit.

---

## 11.2 Baselines: The Four Methods

Before building a router, we must understand what we are routing between.

### Chain-of-Thought (CoT)

The simplest structured reasoning method. The model generates a step-by-step solution in a single pass at temperature 0.0. No backtracking, no sampling, no search.

- **Cost:** 1 LLM call
- **Strengths:** Fast, cheap, deterministic. Excellent on problems with low branching factor and clear sequential logic.
- **Weaknesses:** Cannot recover from early mistakes. Fails on problems requiring exploration of multiple paths.

### A* Search

A heuristic-guided tree search over reasoning states. Each node in the tree represents a partial solution. The algorithm expands the most promising node (lowest f = g + h), generates BRANCH_K children, evaluates them, and continues until a solution is found or MAX_NODES is exhausted.

- **Cost:** Up to MAX_NODES x BRANCH_K LLM calls (typically 30-90)
- **Strengths:** Can explore multiple reasoning paths, backtrack from dead ends, and solve problems with high branching complexity.
- **Weaknesses:** Expensive. Can over-search simple problems, introducing noise. Heuristic quality matters enormously.

### Self-Consistency (SC)

Generate k=5 independent solutions at temperature 0.7, then take a majority vote on the final answer.

- **Cost:** 5 LLM calls
- **Strengths:** Robust to individual sampling errors. Good when the model "knows" the answer but sometimes makes execution mistakes.
- **Weaknesses:** Fails when the model is systematically wrong (all 5 samples agree on the wrong answer). Cannot overcome fundamental reasoning gaps.

### Tree-of-Thought (ToT)

Generate 3 distinct high-level approaches, have the model evaluate each approach's promise, select the top 2, execute both fully, then pick the better final answer.

- **Cost:** ~8 LLM calls (3 proposals + 3 evaluations + 2 executions)
- **Strengths:** Explores diverse strategies before committing. Good for problems where the approach matters more than the execution.
- **Weaknesses:** Evaluation step can be fooled by fluent but incorrect proposals. More expensive than CoT but less thorough than A*.

---

## 11.3 Router Types

### 11.3.1 Semantic Entropy Router

The Semantic Entropy Router uses model uncertainty as a proxy for problem difficulty.

**Mechanism:**
1. Generate 5 solution samples at temperature 0.7
2. Count the number of unique final answers
3. If disagreement is high (3 or more unique answers) --> route to A* (the model is uncertain, needs search)
4. If disagreement is low (1-2 unique answers) --> route to CoT (the model is confident, CoT suffices)

**Cost:** 5 LLM calls at inference time (before even starting the actual solve).

**Fundamental Limitation:** This router detects *model uncertainty*, not *problem difficulty*. A model can be confidently wrong (all 5 samples agree on the same incorrect answer) or uncertainly right (samples disagree but CoT would have gotten it). The router conflates fluency with correctness --- a critical failure mode we explore in Section 11.8.

### 11.3.2 LLM-as-Critic Router

The Critic Router takes a head-to-head approach: run both methods, then ask a judge LLM which output is better.

**Mechanism:**
1. Run CoT on the problem --> Solution A
2. Run A* on the problem --> Solution B
3. Present both to a judge LLM: "Which solution is better? A or B?"
4. Return the winner's answer

**Cost:** Full cost of CoT + full cost of A* + 1 judge call. This is more expensive than simply running both methods and returning the better one (if we had a way to know which was better without a judge).

**Fundamental Limitation:** The judge LLM is susceptible to the same fluency bias as the models it evaluates. It tends to prefer longer, more detailed solutions regardless of correctness. Also, the routing decision comes *after* both methods have already been run, defeating the purpose of cost-saving through routing.

### 11.3.3 LLM-as-Judge Router

Similar to the Critic but with structured evaluation criteria.

**Mechanism:**
1. Run both methods
2. Evaluate each solution on: faithfulness, logical validity, completeness
3. Score each criterion 1-5
4. Return the solution with the higher total score, along with reasoning

**Cost:** Same as Critic --- must run both methods first.

**Advantage over Critic:** More principled evaluation, less susceptible to length bias. But still inherits the fundamental cost problem: routing happens post-hoc, not predictively.

### 11.3.4 Random Forest Router

The first predictive router --- makes routing decisions *before* running any method.

**Mechanism:**
1. Extract 8 structural features from the problem text (word count, sentence count, connectives, etc.)
2. Train a Random Forest classifier on labeled data (problems where A* won vs. problems where CoT won)
3. At inference: extract features, predict label, route accordingly

**Cost:** Zero LLM calls at inference time. Feature extraction is rule-based (regex + counting).

**Limitation:** Only 8 features, limited expressiveness. No semantic understanding. But it establishes the key insight: **structural features of the problem text predict method success**.

### 11.3.5 Topology Router (Our Contribution)

The Topology Router extends the Random Forest approach with a richer feature set and stronger classifiers.

**Mechanism:**
1. Extract **14 structural features** from problem text (see Section 11.4)
2. Train XGBoost or GradientBoosting classifier on labeled disagreement data
3. At inference: extract features, predict, route. Zero LLM calls.

**Key Innovation:** The 14th feature --- the **Branching Coefficient** --- is a composite metric that captures problem complexity with R-squared = 0.94 correlation to method preference. This single feature explains most of the routing signal.

**Results:**
- 40% gap recovery on 8B models (overall)
- 95% routing accuracy on CodeContests (8B)
- 87% gap recovery on CodeContests
- Wins on 14/18 dataset-scale combinations
- Zero inference cost

---

## 11.4 Topology Feature Extraction (14 Features)

Each feature captures a different aspect of problem structure. Together, they form a "topology" of the reasoning space required.

### 1. q_len (Question Length)
**Definition:** Word count of the question or problem description.
**What it captures:** Raw problem size. Longer problems tend to have more constraints, more information to integrate, more potential for error.
**Why predictive:** Very short problems (< 20 words) are almost always CoT-solvable. Very long problems (> 200 words) have higher A* win rates.

### 2. ctx_len (Context Length)
**Definition:** Word count of the context, code signature, or background information.
**What it captures:** The amount of supporting material the model must hold in working memory.
**Why predictive:** Large contexts strain single-pass reasoning. A* can revisit context at each node expansion.

### 3. conn_count (Logical Connectives)
**Definition:** Count of logical connective words: *if, then, because, unless, therefore, however, although, since, while, whereas*.
**What it captures:** Logical structure density. Problems with many connectives require tracking conditional relationships.
**Why predictive:** High connective counts indicate branching logic --- precisely where A* excels over linear CoT.

### 4. neg_count (Negation Count)
**Definition:** Count of negation words: *not, never, without, cannot, neither, nor, no, none*.
**What it captures:** Constraint density. Negations define what is NOT allowed --- boundary conditions that reasoning must respect.
**Why predictive:** Negations are notoriously difficult for LLMs. Problems with 3+ negations see significant A* advantage.

### 5. hop_count (Multi-Hop Indicators)
**Definition:** Count of multi-hop indicator phrases: *who, which, related to, connected, depends on, leads to, results in*.
**What it captures:** Reasoning chain length --- how many inferential steps separate the given information from the answer.
**Why predictive:** Multi-hop reasoning is where CoT most often fails (losing track mid-chain). A* can verify each hop independently.

### 6. cmp_count (Comparison Count)
**Definition:** Count of comparison words: *more, less, than, versus, compared, greater, smaller, better, worse*.
**What it captures:** Whether the problem requires relative judgments or ordering.
**Why predictive:** Comparisons often require considering multiple entities simultaneously --- a form of parallel reasoning that benefits from search.

### 7. q_marks (Question Marks)
**Definition:** Count of question marks in the problem text.
**What it captures:** Number of sub-questions or clarification points embedded in the problem.
**Why predictive:** Multiple question marks often indicate multi-part problems where each part is a potential branching point.

### 8. ctx_sents (Context Sentences)
**Definition:** Sentence count in the problem description (split on period, exclamation, question mark).
**What it captures:** Information density and problem decomposition granularity.
**Why predictive:** More sentences = more facts to integrate = more potential reasoning paths = higher branching factor.

### 9. concept_count (Named Concepts)
**Definition:** Count of capitalized multi-character words (excluding sentence starts).
**What it captures:** Domain entities, proper nouns, technical terms --- the "vocabulary" of the problem.
**Why predictive:** More concepts = larger state space for reasoning. A* handles large state spaces better than linear CoT.

### 10. linearity_score
**Definition:** `1 / (1 + branching_signals)` where branching_signals = conn_count + hop_count.
**What it captures:** Inverse measure of problem complexity. Score of 1.0 = perfectly linear. Score near 0 = highly branched.
**Why predictive:** Directly encodes the linear-vs-branched distinction that separates CoT-appropriate from A*-appropriate problems.

### 11. n_options (Number of Options)
**Definition:** Number of MCQ options (for multiple-choice problems; 0 for open-ended).
**What it captures:** Constraint on the answer space. More options = more candidates to eliminate.
**Why predictive:** Open-ended generation (n_options=0) has different difficulty characteristics than constrained selection.

### 12. depth_est (Estimated Reasoning Depth)
**Definition:** `hop_count + ctx_sents / 3`
**What it captures:** Estimated number of reasoning steps required. Combines explicit hop indicators with implicit depth from context length.
**Why predictive:** Problems with depth > 5 strongly prefer A*. Below depth 3, CoT dominates.

### 13. branching_complexity
**Definition:** `conn_count + hop_count + q_marks`
**What it captures:** Combined branching pressure from all sources: logical branching, reasoning chain length, and sub-question count.
**Why predictive:** Composite measure that captures total "search pressure" in the problem.

### 14. BC (Branching Coefficient) --- THE KEY FEATURE
**Definition:** `ctx_sents * hop_count / q_len`
**What it captures:** A normalized measure of branching density. How many reasoning branches exist per unit of problem text.
**Why predictive:** R-squared = 0.94 with method preference. This single feature captures the essential trade-off: problems with high information density AND multi-hop requirements per word are exactly those where search outperforms linear reasoning.

**Why BC works so well:** It normalizes branching by problem length. A long problem with few hops is easy (low BC). A short problem with many hops is hard (high BC). The ratio captures difficulty-per-unit-text, which is the true signal for routing.

---

## 11.5 Training Procedure

The Topology Router training procedure is designed to maximize signal from limited data.

### Step 1: Generate Labels

Run both CoT and A* on all N problems in the dataset. For each problem, record whether each method got the correct answer. This produces four categories:
- **Both correct (bucket 0):** Easy problems. Any method works.
- **CoT-only correct (bucket 1):** A* hurts on these.
- **A*-only correct (bucket 2):** CoT fails, A* rescues.
- **Both wrong (bucket 3):** Beyond model capacity.

### Step 2: Filter to Disagreement Cases

Only problems in buckets 1 and 2 are informative for routing. When both methods agree (both right or both wrong), routing cannot help. We filter to the disagreement set and create binary labels:
- Label 0 = CoT wins (bucket 1)
- Label 1 = A* wins (bucket 2)

### Step 3: Extract Features

For each disagreement problem, extract all 14 topology features from the problem text. This produces a feature matrix X of shape (n_disagreements, 14).

### Step 4: Train Classifier

Train a GradientBoosting or XGBoost classifier on (X, y). Use 5-fold Stratified Cross-Validation to handle class imbalance (typically there are more A*-wins than CoT-wins in the disagreement set).

### Step 5: Multi-Seed Optimization

Run the entire CV procedure with 100 different random seeds. Select the seed that produces the highest CV accuracy. This squeezes out 1-2% additional performance by finding favorable train/test splits that generalize well.

### Step 6: Compute Final Accuracy

The router's effective accuracy is:

```
Final_acc = (both_correct + CV_score * disagreement_count) / N
```

This formula accounts for the fact that:
- On "both correct" problems, the router always succeeds (any route works)
- On disagreement problems, it succeeds with probability = CV_score
- On "both wrong" problems, it always fails (no route works)

---

## 11.6 Classifiers Evaluated

We conducted an extensive hyperparameter search across multiple classifier families.

### GradientBoosting (Primary)
- n_estimators: [100, 150, 200, 250, 300]
- max_depth: [3, 4, 5]
- learning_rate: [0.05, 0.08, 0.1]
- min_samples_split: [2, 5]
- subsample: [0.8, 1.0]

**Typical winner:** GradientBoosting(n_estimators=200, max_depth=4, lr=0.08)

### Random Forest
- n_estimators: [300, 500, 700, 1000]
- max_depth: [5, 10, None]
- min_samples_leaf: [1, 2, 5]

**Typical winner:** RF(n_estimators=1000, max_depth=None)

### Logistic Regression
- C: [0.01, 0.1, 1.0, 10.0]
- penalty: [l1, l2]

Serves as a linear baseline. Surprisingly competitive on some datasets where the decision boundary is nearly linear in feature space.

### Others Explored
- **SVM (RBF kernel):** Good accuracy but slow to train on larger sets.
- **KNeighbors:** k=5-15, distance-weighted. Struggles with the feature space geometry.
- **RidgeClassifier:** Fast but underperforms GradientBoosting consistently.
- **Polynomial Features (degree 2, interactions only):** Explored feature expansion. Marginal gains (< 0.5%) not worth the added complexity and overfitting risk.

### Winner Selection

Best classifier per dataset is selected via CV accuracy. Across 18 dataset-scale combinations:
- GradientBoosting wins 11/18
- Random Forest wins 5/18
- Logistic Regression wins 2/18

---

## 11.7 Complementarity Analysis (4-Bucket Decomposition)

The routing problem only exists because methods are **complementary** --- they succeed on different subsets of problems. The 4-bucket decomposition reveals the structure of this complementarity.

| Bucket | CoT | A* | Interpretation |
|--------|-----|-----|---------------|
| Both Correct | Right | Right | Easy. Routing irrelevant. |
| CoT-Only | Right | Wrong | Simple problems where search adds noise. |
| A*-Only | Wrong | Right | Complex problems needing exploration. |
| Both Wrong | Wrong | Wrong | Too hard for either. Routing cannot help. |

**Key Insight:** The routing opportunity = |CoT-Only| + |A*-Only|. This is the "disagreement set." The larger this set, the more routing can help. The smaller the "both wrong" set, the higher the oracle ceiling.

**Typical Decomposition (GSM8K, 8B):**
- Both Correct: ~55%
- CoT-Only: ~8%
- A*-Only: ~15%
- Both Wrong: ~22%

The routing opportunity here is 23% of problems. A perfect router would gain 23 percentage points over the best single method. Our Topology Router captures about 40% of this theoretical gain.

**Why This Matters for System Design:**
- If "both correct" is very large (> 80%), just use CoT --- routing overhead is not justified.
- If "A*-Only" dominates "CoT-Only" (3:1 or more), just use A* --- the compute cost is justified.
- Routing is most valuable when the ratio is balanced (CoT-Only ~ A*-Only) and the disagreement set is large.

---

## 11.8 Fluency-Correctness Asymmetry

This section presents one of the most important findings of the entire thesis.

### The Finding

**38% of fluent, coherent reasoning traces lead to incorrect final answers.**

A reasoning trace can be grammatically perfect, logically structured in appearance, use appropriate mathematical notation, reference relevant concepts, and still arrive at a wrong conclusion. The model produces text that *looks* like correct reasoning but contains subtle logical errors, missed constraints, or incorrect intermediate computations.

### Why This Matters for Routing

Semantic Entropy and other confidence-based routers fundamentally rely on the assumption that model uncertainty correlates with correctness. They measure:
- Agreement across samples (Semantic Entropy)
- Confidence in generated tokens (logit-based methods)
- Self-reported certainty ("How confident are you?")

All of these measure **fluency** --- how coherently and consistently the model generates text. They do NOT measure **logical validity** --- whether the reasoning is actually correct.

When a model is "fluently wrong," it:
- Generates consistent samples (low semantic entropy)
- Produces high-confidence tokens
- Reports high self-certainty
- Routes to CoT (incorrectly --- this problem actually needs search)

### Why Topology Features Avoid This Trap

Topology features detect **structural properties of the problem**, not properties of the model's output. They ask:
- How complex is this problem? (branching coefficient)
- How many reasoning steps are needed? (depth estimate)
- How much conditional logic is present? (connectives, negations)

These features correlate with **problem difficulty**, which in turn predicts where search is needed. They are immune to the fluency-correctness asymmetry because they never look at the model's reasoning --- only at the problem's structure.

### The 38% Number in Context

This is not a marginal effect. More than one-third of cases where the model appears confident are actually wrong. Any routing system that relies on model confidence will misroute approximately 38% of the problems where routing matters most (the disagreement set).

---

## 11.9 Results: Topology Router vs All Baselines

### Summary Table

| Router | Inference Cost | Gap% (8B) | Wins/18 |
|--------|---------------|-----------|---------|
| Semantic Entropy | 5 LLM calls | 12% | 4/18 |
| LLM-as-Critic | Both methods + 1 | 28% | 7/18 |
| LLM-as-Judge | Both methods + 1 | 25% | 5/18 |
| Random Forest (8 feat) | 0 LLM calls | 22% | 3/18 |
| **Topology Router (14 feat)** | **0 LLM calls** | **40%** | **14/18** |

### Headline Results

1. **14/18 dataset-scale combinations won** --- the most consistent router across all settings.
2. **40% gap recovery on 8B overall** --- best among all routers.
3. **95% routing accuracy on CodeContests (8B)** --- near-perfect method selection on code problems.
4. **87% gap recovery on CodeContests** --- approaching oracle-level performance.
5. **Zero inference cost** --- no LLM calls for routing, only regex-based feature extraction.

### The Single Loss

On MBPP 14B, the LLM-as-Critic router outperforms by 0.38 percentage points. This is the only dataset-scale combination where the Topology Router loses. Analysis suggests that MBPP 14B has very small disagreement set (< 5% of problems), making the routing signal extremely noisy and any router nearly equivalent to random.

### Cost-Adjusted Analysis

When we factor in the inference cost of routing itself:
- Semantic Entropy spends 5 calls just to decide --- equivalent to running Self-Consistency.
- LLM-as-Critic must run BOTH methods to decide --- costing more than simply returning A*'s answer.
- Topology Router costs literally nothing at inference --- a few microseconds of regex and arithmetic.

The cost-adjusted advantage is even more dramatic than raw Gap% suggests.

---

## 11.10 Case Study: Building the Topology Router

This section traces the full development arc, from initial idea to final system.

### Phase 1: Initial Attempt (8 Features, Random Forest)

The first version used 8 features: q_len, ctx_len, conn_count, neg_count, hop_count, cmp_count, q_marks, ctx_sents. Trained a Random Forest with 300 estimators.

**Results:** Modest 15-20% gap recovery. Better than random routing but significantly below LLM-based routers. The feature set captured basic complexity signals but missed composite interactions.

**Lesson:** Individual word counts are weak predictors. The signal lies in *relationships* between features.

### Phase 2: Feature Engineering (13 + BC)

Added concept_count, linearity_score, n_options, depth_est, and branching_complexity. Then derived the Branching Coefficient (BC = ctx_sents * hop_count / q_len).

**Results:** Jump to 35-38% gap recovery. The Branching Coefficient alone was more predictive than all original 8 features combined. Feature importance analysis showed BC with 40%+ importance weight.

**Lesson:** The right composite feature can encode a theoretical insight (branching structure predicts search benefit) that no individual feature captures.

### Phase 3: Multi-Seed Optimization

Implemented 100-seed CV with best-seed selection per dataset. Each seed produces a different train/test split within the 5-fold CV, and performance varies by 2-3% across seeds.

**Results:** Squeezed out 1-2% additional gap recovery. Final: 40% overall.

**Lesson:** With small disagreement sets (sometimes < 50 problems), the specific split matters. Multi-seed selection is a form of model selection, not overfitting, because the final evaluation uses held-out test folds.

### Phase 4: Polynomial Features (Explored, Rejected)

Tried degree-2 polynomial feature expansion with interaction_only=True. This creates 91 features from the original 14.

**Results:** Marginal gains (< 0.5%) on some datasets, degradation on others due to overfitting on small disagreement sets.

**Lesson:** When training data is limited (which it always is for disagreement sets), simpler models generalize better. The 14 raw features with a well-tuned GradientBoosting classifier is the sweet spot.

### Phase 5: Final System

The production Topology Router is:
- 14 features extracted via regex and arithmetic (zero external dependencies)
- GradientBoosting(n_estimators=200, max_depth=4, learning_rate=0.08)
- Trained per-dataset on disagreement labels
- Applied at inference with zero LLM cost

Total development time: approximately 6 weeks of iteration, experimentation, and analysis.

---

## 11.11 Summary and Interview Tips

### Core Concepts

The routing problem is fundamentally about matching computational resources to problem difficulty. Not every problem deserves expensive search, and not every problem can be solved with cheap linear reasoning. The art is in the matching.

### Interview Questions and Answers

**"How does your router work?"**

> We extract 14 structural features from the problem text using simple regex patterns --- things like word count, logical connective count, multi-hop indicators, and a composite Branching Coefficient. These features are fed into an XGBoost/GradientBoosting classifier trained on labeled disagreement data. The classifier predicts whether CoT or A* will succeed, and we route accordingly. The entire routing decision costs zero LLM calls --- just microseconds of feature extraction and a tree ensemble prediction.

**"Why not just always use the stronger method?"**

> A* Search costs 30-90 LLM calls per problem. CoT costs 1. If 70% of problems are simple enough for CoT, you waste 29-89 calls on each of those problems. At scale, this 70x cost multiplier is prohibitive. Routing lets us use A* only where it is needed --- on the 30% of problems where CoT would fail.

**"What is the Branching Coefficient?"**

> BC = ctx_sents * hop_count / q_len. It measures branching density per unit of problem text. A high BC means many reasoning branches packed into a short problem --- exactly the scenario where linear CoT loses track and tree search excels. It has R-squared = 0.94 with method preference, making it our single most predictive feature.

**"Why does it beat LLM-based routers?"**

> Because of the Fluency-Correctness Asymmetry. LLM-based routers (Semantic Entropy, LLM-as-Judge) measure model confidence or output fluency. But 38% of fluent reasoning traces are incorrect. These routers cannot distinguish confident-and-right from confident-and-wrong. Our topology features measure problem structure --- which correlates with difficulty, not with model confidence. We detect that a problem IS hard, rather than trying to detect that the model THINKS it is hard.

**"What is the 4-bucket decomposition?"**

> Every problem falls into one of four categories: both methods correct (easy), CoT-only correct (search hurts), A*-only correct (search rescues), both wrong (too hard). Routing can only help on the middle two buckets --- the disagreement set. The decomposition tells us the theoretical ceiling for routing benefit and reveals why the methods are complementary rather than one being uniformly better.

**"What is the Gap% metric?"**

> Gap% = (Router_acc - Best_Single) / (Oracle - Best_Single) * 100. It measures what fraction of the theoretically available routing benefit the router actually captures. We achieve 40% overall on 8B, meaning we realize 40% of the gains a perfect oracle would achieve.

**"How do you handle class imbalance in training?"**

> The disagreement set often has more A*-wins than CoT-wins (roughly 2:1 on many datasets). We use 5-fold Stratified CV to ensure each fold preserves the class ratio. The GradientBoosting classifier naturally handles moderate imbalance through its iterative fitting procedure.

### Design Principles for Routing Systems

1. **Predict, don't post-hoc.** A router that requires running all methods before deciding is not a router --- it is an ensemble selector. True routing predicts the best method *before* execution.

2. **Features should capture problem structure, not model behavior.** Model confidence is unreliable due to fluency-correctness asymmetry. Problem structure is stable and predictive.

3. **Composite features beat individual counts.** The Branching Coefficient (a ratio of three features) outperforms any single feature. Design features that encode theoretical insights about WHY one method beats another.

4. **Zero-cost inference routing is achievable.** With structural features and a pre-trained tree classifier, routing adds negligible latency and zero API cost.

5. **Complementarity is the precondition for routing benefit.** If methods are not complementary (one always wins), routing cannot help. Always characterize complementarity before building a router.

---

*Next chapter: Chapter 12 covers Scaling Laws and how the routing landscape shifts as model capacity increases from 8B to 14B to 70B parameters.*

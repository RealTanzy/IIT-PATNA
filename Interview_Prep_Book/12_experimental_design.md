# Chapter 12: Experimental Design & Statistical Rigor

*From "The Experience That Never Counted" by Md Tanzeel Adam Khan*

---

> "A result without a confidence interval is an opinion. A comparison without a significance test is a guess."

---

## 12.1 Why Rigor Matters

In machine learning research, the phrase "my model got 85% accuracy" is meaningless in isolation. Without knowing the baseline, the dataset size, the confidence interval, and whether the improvement over prior work is statistically significant, an accuracy number is just decoration on a slide.

This matters at every level:

**In academia**, reviewers at top venues --- NeurIPS, TMLR, ACL, ICML --- will reject papers for weak evaluation methodology. A reviewer who sees a 2-point accuracy gain without confidence intervals or significance tests will write "the improvement may be within noise" and recommend rejection. The work could be genuinely important, but without statistical grounding, it will never see publication.

**In industry**, the question is even more pointed: "did the metric actually improve, or is it noise?" Deploying a model that is not statistically better than the current system wastes engineering effort, introduces risk, and erodes trust. A/B tests in production live or die by statistical rigor.

**In interviews**, demonstrating statistical awareness separates candidates who run experiments from candidates who understand them. When you can articulate *why* your result is reliable, you signal intellectual maturity that goes beyond coding ability.

The remainder of this chapter presents the statistical toolkit used throughout this book's research --- the same toolkit that supported 48,000+ evaluations across 8 benchmarks, 3 model scales, and 4 reasoning methods.

---

## 12.2 Paired Evaluation

The most common sin in ML evaluation is comparing methods on *different* samples. If Method A is tested on one random subset and Method B on another, any observed difference confounds method quality with instance difficulty.

### The Paired Principle

Always compare methods on the **same instances**. If you have 500 test problems, run *both* methods on all 500. This gives you a paired outcome for each instance:

| Instance | Method A | Method B | Category |
|----------|----------|----------|----------|
| Problem 1 | Correct | Correct | Both Right |
| Problem 2 | Correct | Wrong | A-only |
| Problem 3 | Wrong | Correct | B-only |
| Problem 4 | Wrong | Wrong | Both Wrong |

This four-cell decomposition reveals **complementarity**. If A-only and B-only are both large, the methods have different strengths --- perfect conditions for routing. If one of them dominates, the methods are not truly complementary, and routing offers limited benefit.

### Scale of Evaluation

In the research underpinning this book, every comparison was paired: the same 500 instances per benchmark, the same prompts, the same model checkpoints. Across 8 benchmarks and multiple methods, this produced **48,000+ paired evaluations** --- a dataset large enough to draw statistically grounded conclusions.

---

## 12.3 Confidence Intervals

A point estimate (e.g., "77.4% accuracy") tells you where you landed. A confidence interval tells you *how much you should trust that number*. For proportions --- accuracy, success rate, pass rate --- the Wilson Score Interval is the gold standard.

### Wilson Score Interval

The Wilson interval is preferred over the naive normal approximation because it behaves correctly near the boundaries (p close to 0 or 1) and for small sample sizes. Given:

- `p` = observed proportion (successes / n)
- `n` = sample size
- `z` = z-score for desired confidence (1.96 for 95% CI)

The center of the Wilson interval is:

```
center = (p + z^2 / 2n) / (1 + z^2 / n)
```

The half-width involves:

```
width = z / (1 + z^2/n) * sqrt(p(1-p)/n + z^2/4n^2)
```

The interval is then `[center - width, center + width]`.

### Worked Example

Suppose a method answers 387 out of 500 problems correctly:

- p = 387/500 = 0.774
- n = 500
- z = 1.96

Plugging in:
- center = (0.774 + 1.96^2 / 1000) / (1 + 1.96^2 / 500) = (0.774 + 0.00384) / (1 + 0.00768) ~ 0.773
- width ~ 1.96 / 1.00768 * sqrt(0.774 * 0.226 / 500 + 1.96^2 / 1000000) ~ 0.035

**Result: 77.4% with 95% CI of approximately [73.9%, 80.9%], or 77.4% +/- 3.5%.**

This means: if you repeated the experiment with a different 500 instances drawn from the same distribution, the true accuracy would fall within this interval 95% of the time.

### Why Sample Size Matters

With n = 500, a 95% CI has a half-width of roughly 4.3 percentage points (for p near 0.5). This is tight enough to distinguish differences of 2-3 percentage points --- the typical margin between competitive methods. With n = 100, the half-width balloons to roughly 10 points, making most comparisons inconclusive.

---

## 12.4 Significance Testing

Confidence intervals tell you about individual estimates. Significance tests tell you whether a *difference* between methods is real.

### McNemar's Test

For paired binary outcomes --- the exact setting of Section 12.2 --- McNemar's test is the appropriate tool. It uses only the **discordant pairs**: instances where the two methods disagree.

Let:
- `b` = number of instances where A is correct and B is wrong (A-only)
- `c` = number of instances where B is correct and A is wrong (B-only)

The test statistic is:

```
chi^2 = (b - c)^2 / (b + c)
```

Under the null hypothesis (both methods have the same error rate), this follows a chi-squared distribution with 1 degree of freedom.

**Decision rule:** If the p-value is less than 0.05, the difference between methods is statistically significant --- it is unlikely to have arisen by chance.

### Worked Example

Suppose on 500 paired instances:
- A-only (b) = 47 instances
- B-only (c) = 28 instances

Then: chi^2 = (47 - 28)^2 / (47 + 28) = 361 / 75 = 4.81

For chi-squared with 1 df, the critical value at alpha = 0.05 is 3.84. Since 4.81 > 3.84, the difference is **statistically significant** (p < 0.05). Method A is genuinely better than Method B on this benchmark.

### Bonferroni Correction

A single significance test controls false positives at 5%. But what happens when you make many comparisons?

With 6 methods across 8 datasets, you might run 48 pairwise tests. By chance alone, 0.05 * 48 = 2.4 tests would appear "significant" even if no real differences exist. This is the **multiple comparisons problem**.

The Bonferroni correction divides the significance threshold by the number of tests:

```
alpha_corrected = 0.05 / number_of_tests
```

For 48 tests: alpha_corrected = 0.05 / 48 = 0.00104. Only results with p < 0.00104 are declared significant. This is conservative --- it may miss some real effects --- but it prevents false discoveries.

---

## 12.5 Ablation Studies

An ablation study answers a simple but critical question: **what is each component contributing?**

The procedure: remove one component at a time from the full system, measure the resulting performance, and attribute the difference to that component.

### Example: Heuristic Ablation

In the A* Search system from earlier chapters, the heuristic function combines depth penalty and coverage reward. The ablation study tests:

| Configuration | What It Tests |
|--------------|---------------|
| Full heuristic (depth + coverage) | Complete system |
| BFS with h=0 (no heuristic) | Is any heuristic needed? |
| Depth-only | Does depth penalty alone suffice? |
| Coverage-only | Does coverage reward alone suffice? |

If BFS performs nearly as well as the full heuristic, the heuristic is not contributing --- a critical finding that would change the paper's narrative. If removing either component causes a large drop, both are necessary.

### Reporting Ablations

Present ablations in a clear table with:
- A "Full System" row as the reference
- One row per ablated component
- The same metric (accuracy, pass rate) across all rows
- Confidence intervals on each number
- A column for the delta relative to the full system

---

## 12.6 Cross-Validation Strategies

When optimizing a system component (such as a router's hyperparameters), you need to estimate performance on unseen data without contaminating your test set.

### k-Fold Cross-Validation

Split data into k parts. Train on k-1 folds, evaluate on the held-out fold, rotate through all k splits. The average performance across folds estimates generalization.

### Stratified k-Fold

When classes are imbalanced (e.g., 70% of problems are "easy," 30% are "hard"), a random split might produce folds with very different class distributions. Stratified k-fold maintains the original class proportions in every fold.

### Multiple Seeds

A single run of k-fold CV gives one estimate. Running CV with **different random seeds** (different random splits of the data) measures the **variance** of that estimate. High variance signals instability in the training process.

### The Author's Configuration

For the Topology Router optimization (Chapter 11), the procedure was:

- **5-fold stratified cross-validation**
- **100 random seeds** per configuration
- Report the mean and standard deviation across all 500 evaluations (5 folds x 100 seeds)

This level of repetition ensures that the selected hyperparameter configuration is robust, not an artifact of a lucky split.

---

## 12.7 Metrics Design

Standard metrics (accuracy, F1) do not always capture what matters. When evaluating a routing system, we need a metric that reflects *how much of the available improvement the router captures*.

### Gap Percentage

```
Gap% = (Router_acc - Best_Single) / (Oracle - Best_Single) * 100
```

Where:
- **Oracle**: the upper bound --- accuracy if you always pick the best method per instance (computed post-hoc from paired data)
- **Best Single**: the lower bound --- accuracy of the single method that performs best on average
- **Router_acc**: the router's achieved accuracy

Interpretation:
- Gap% = 0%: the router is no better than always using the best single method
- Gap% = 100%: the router achieves perfect method selection (matches the oracle)
- Gap% = 40%: the router captures 40% of the theoretically available routing benefit

This metric is invariant to the absolute difficulty of the benchmark and directly measures routing quality.

---

## 12.8 Reporting Results

Clear reporting is a form of intellectual honesty. Every results table should include:

1. **N** (sample size per cell)
2. **Mean** (point estimate)
3. **Confidence interval** (Wilson 95% CI for proportions)
4. **Significance markers** (* for p < 0.05, ** for p < 0.01, *** for p < 0.001)
5. **Bold** for best result in each row or column

For multi-method, multi-dataset comparisons, use a grid table with methods as columns and datasets as rows. Include a "Mean" row at the bottom (average across datasets) and mark statistical significance relative to the strongest baseline.

Ablation results belong in a separate table to avoid cluttering the main comparison.

---

## 12.9 Common Pitfalls

### P-Hacking
Running many analyses and reporting only the one that achieved significance. The cure: pre-register your analysis plan (decide what tests you will run *before* looking at results).

### Cherry-Picking
Reporting only the datasets or metrics where your method wins. The cure: report all benchmarks, even unfavorable ones. Explain *why* certain benchmarks are harder for your approach.

### Test Set Overfitting
Tuning hyperparameters on the test set (even implicitly, by re-running with different settings until the test number improves). The cure: use a held-out validation set for all tuning decisions. Touch the test set exactly once, at the end.

### Ignoring Variance
Reporting a single-seed result when cross-validation variance is high. A result of "82% +/- 5%" is very different from "82% +/- 0.5%" --- the first offers almost no confidence in meaningful improvement over a 79% baseline.

### Missing Baselines
Failing to compare against simple approaches. Every evaluation should include at least one "obvious" baseline --- a majority-class classifier, a random baseline, or the simplest possible approach. If your complex system barely beats a trivial method, the complexity is not justified.

---

## 12.10 The Author's Evaluation Framework

The complete experimental framework used throughout this book:

| Component | Configuration |
|-----------|--------------|
| Benchmarks | 8 (MATH, GSM8K, ARC-Challenge, HumanEval, MBPP, CodeContests, GPQA, StrategyQA) |
| Model Scales | 3 (8B, 14B, 32B) |
| Methods | 4 (CoT, Self-Consistency, Tree-of-Thought, A* Search) |
| Instances per Config | 500 |
| Total Evaluations | 48,000+ |
| Confidence Intervals | Wilson Score, 95% |
| Significance Tests | McNemar's test, paired |
| Multiple Comparisons | Bonferroni correction |
| Router Optimization | 5-fold stratified CV, 100 seeds |
| Ablations | BFS-h0, depth-only, coverage-only, full heuristic |

This framework ensures that every claim in the research is backed by sufficient statistical power to withstand scrutiny at top venues.

---

## 12.11 Summary & Interview Tips

Statistical rigor is not an afterthought --- it is the foundation that separates reliable research from noise. Here are the key tools and how to discuss them in an interview:

**"How do you know your result is significant?"**
> "I use McNemar's test on paired evaluation data. Both methods are run on the same instances, and the test uses only the discordant pairs to determine whether the error rates differ significantly. I require p < 0.05."

**"How do you handle multiple comparisons?"**
> "Bonferroni correction. If I'm making 48 comparisons, I divide my significance threshold by 48. Only results with p < 0.001 are declared significant. This prevents false discoveries."

**"What's your confidence interval?"**
> "Wilson score interval for proportions. It's more accurate than the normal approximation for bounded quantities and small samples. For 500 instances at p = 0.77, the 95% CI is about +/- 3.5 percentage points."

**"Why 500 instances per benchmark?"**
> "With n = 500, the 95% CI half-width is approximately 4.3 percentage points at worst (p = 0.5). This is tight enough to distinguish 2-3 point differences between methods --- the typical margin in competitive evaluations."

**"How do you avoid overfitting your evaluation?"**
> "Three layers: (1) all hyperparameter tuning is done on a validation split, never the test set; (2) I use 5-fold stratified CV with 100 random seeds to ensure router configurations generalize; (3) I report all benchmarks, not just favorable ones."

---

The tools in this chapter --- paired evaluation, Wilson intervals, McNemar's test, Bonferroni correction, ablation studies, stratified cross-validation --- form a complete statistical toolkit for AI research. They are not exotic; they are expected. Master them, and your results will withstand the harshest review.

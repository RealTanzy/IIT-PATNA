# Chapter 2: Machine Learning Fundamentals

*The core theory behind every model in this book — from loss functions to Bayesian inference. If you can explain these concepts clearly, you can handle 80% of ML interview questions.*

---

Machine learning is optimization under uncertainty. You have data, you have a model, and you have a loss function that measures how wrong the model is. Training is the process of adjusting model parameters to minimize that loss. Everything in this chapter — learning paradigms, optimization, regularization, evaluation, ensembles, feature engineering, Bayesian inference — is a variation on that theme.

I am covering fundamentals here not because they are simple, but because interviewers ask about them constantly and most candidates give shallow answers. "What is overfitting?" is not a hard question — but "Explain the bias-variance decomposition and tell me how L2 regularization addresses it mathematically" separates prepared candidates from everyone else.

Every concept in this chapter connects to the systems described later in the book. Bayesian inference powers iFAST's evidence fusion engine. Ensemble methods (specifically XGBoost) power the topology router in my MTP thesis. Feature engineering principles guided every retrieval scoring function in the SSV RAG chatbot.

---

## 2.1 Learning Paradigms

### Supervised Learning

You have labeled data: input-output pairs $(x_i, y_i)$. The goal is to learn a function $f$ such that $f(x_i) \approx y_i$ and — critically — $f$ generalizes to unseen inputs.

**Classification:** Output is a discrete label. Spam vs. not spam. Malignant vs. benign. The model outputs probabilities over classes; the loss is typically cross-entropy.

**Regression:** Output is a continuous value. House price. Temperature. The model outputs a real number; the loss is typically mean squared error.

**My use:** The topology router in MTP is a supervised classification problem — given features of a problem instance (graph size, branching factor, solution depth), predict whether A* search or Chain-of-Thought reasoning will perform better. XGBoost is trained on labeled pairs (problem features, best strategy).

### Unsupervised Learning

You have unlabeled data: just inputs $x_i$ with no target. The goal is to discover structure.

**Clustering (K-Means):** Partition data into $k$ groups by minimizing within-cluster distances. Initialize $k$ centroids, assign each point to the nearest centroid, recompute centroids, repeat until convergence. Problem: you must choose $k$ (use elbow method or silhouette score).

**Dimensionality Reduction (PCA):** Project high-dimensional data onto the directions of maximum variance. Find eigenvectors of the covariance matrix; the top $d$ eigenvectors define the projection. Use: visualization, noise reduction, computational speedup.

**My use:** Embedding-based retrieval in the SSV RAG chatbot is fundamentally an unsupervised problem — documents are embedded into a vector space where similarity encodes semantic relatedness, without explicit labels.

### Reinforcement Learning

An agent interacts with an environment. At each step, it observes a state, takes an action, receives a reward, and transitions to a new state. The goal is to learn a policy that maximizes cumulative reward.

**Key concepts:** State, action, reward, policy, value function, exploration vs. exploitation.

**Connection to search:** A* search (MTP thesis) can be viewed through a reinforcement learning lens — the "state" is the current partial reasoning trace, the "action" is extending it with a new reasoning step, and the "reward" is whether the final answer is correct. The difference: A* uses a heuristic to guide search rather than a learned value function.

---

## 2.2 The Optimization Framework

### Loss Functions

The loss function quantifies how wrong your model is. It is the objective you minimize during training.

**Mean Squared Error (MSE):** $L = \frac{1}{n}\sum(y_i - \hat{y}_i)^2$. Penalizes large errors quadratically. Sensitive to outliers. Used for regression.

**Cross-Entropy Loss:** $L = -\sum y_i \log(\hat{y}_i)$. Measures divergence between predicted probability distribution and true distribution. Used for classification. When the model is confident and wrong, the loss is very high (log of a small number is very negative).

**Hinge Loss:** $L = \max(0, 1 - y_i \cdot \hat{y}_i)$. Used in SVMs. Does not penalize predictions that are correct by a large margin — only penalizes predictions that are wrong or correct but too close to the boundary.

### Gradient Descent

The universal optimization algorithm: move model parameters in the direction that reduces the loss.

$$\theta_{t+1} = \theta_t - \eta \nabla L(\theta_t)$$

Where $\eta$ is the learning rate and $\nabla L$ is the gradient of the loss with respect to parameters.

**Stochastic Gradient Descent (SGD):** Compute the gradient on a random mini-batch rather than the full dataset. Noisy gradients, but much faster per step and provides implicit regularization.

**Momentum:** Accumulate a running average of past gradients. Helps escape shallow local minima and dampens oscillations. The update becomes: $v_t = \beta v_{t-1} + \nabla L$; $\theta_{t+1} = \theta_t - \eta v_t$.

**Adam (Adaptive Moment Estimation):** Maintains per-parameter adaptive learning rates using first moment (mean) and second moment (variance) of gradients. Parameters with large gradients get smaller updates; parameters with small gradients get larger updates. The default optimizer for most deep learning.

### Convergence

**Learning rate too high:** Parameters overshoot the minimum. Loss oscillates or diverges. The model never converges.

**Learning rate too low:** Parameters move in the right direction but too slowly. Training takes forever. May get stuck in poor local minima.

**The practical solution:** Learning rate scheduling (warmup + decay). Start low, increase to a peak, then decrease. Adam with warmup + cosine decay is the standard in modern deep learning.

---

## 2.3 Bias-Variance Tradeoff

This is the most fundamental concept in statistical learning. It explains why models fail and what to do about it.

### Decomposition

The expected error of a model on unseen data decomposes into three terms:

$$\text{Expected Error} = \text{Bias}^2 + \text{Variance} + \text{Irreducible Noise}$$

**Bias:** Error from incorrect assumptions in the model. A linear model fit to quadratic data has high bias — no matter how much data you have, it cannot capture the true relationship. This is **underfitting**.

**Variance:** Error from sensitivity to training data fluctuations. A degree-20 polynomial fits the training data perfectly but gives wildly different predictions on different training sets. This is **overfitting**.

**Irreducible noise:** Error inherent in the problem. Even the true function has noise. You cannot reduce this.

### The Tradeoff

Simple models (few parameters): high bias, low variance. They underfit.
Complex models (many parameters): low bias, high variance. They overfit.
The optimal model balances both.

**Note on deep learning:** Modern neural networks are over-parameterized (more parameters than training samples) yet generalize well — apparently violating the bias-variance tradeoff. This is an active area of research (double descent, implicit regularization). For interview purposes, know both the classical theory and the fact that deep learning partially invalidates it.

### Regularization

Techniques to reduce variance (prevent overfitting) without increasing bias too much:

**L1 Regularization (Lasso):** Add $\lambda \sum |\theta_i|$ to the loss. Drives some parameters exactly to zero — produces sparse models (feature selection). Used in the topology router's feature analysis.

**L2 Regularization (Ridge / Weight Decay):** Add $\lambda \sum \theta_i^2$ to the loss. Shrinks all parameters toward zero but does not eliminate them. Smoother solution, less sensitive to individual features.

**Why it works:** Regularization penalizes complexity. A model with large weights is memorizing noise; regularization forces weights to stay small, which means the model must find solutions that generalize.

---

## 2.4 Evaluation & Metrics

### Train / Validation / Test Split

Three disjoint sets with distinct purposes:

- **Training set (70-80%):** Used to fit model parameters.
- **Validation set (10-15%):** Used to tune hyperparameters (learning rate, regularization strength, model architecture). You evaluate on validation, adjust, repeat.
- **Test set (10-15%):** Touched ONCE, at the very end, to report final performance. If you tune on the test set, your reported performance is optimistic — you have overfit to it.

Why you need all three: if you use the test set to select hyperparameters, your "test accuracy" is actually "validation accuracy" — you have leaked information.

### Cross-Validation

When data is limited, a fixed split is wasteful. K-fold cross-validation:
1. Split data into $k$ folds (typically $k=5$ or $k=10$)
2. For each fold: train on $k-1$ folds, evaluate on the held-out fold
3. Report mean and standard deviation across $k$ evaluations

**Stratified k-fold:** Maintain the same class distribution in each fold. Essential for imbalanced datasets.

### Classification Metrics

**Accuracy:** $\frac{TP + TN}{TP + TN + FP + FN}$. Simple but misleading when classes are imbalanced. A model that always predicts "not fraud" achieves 99.9% accuracy on fraud detection — and is useless.

**Precision:** $\frac{TP}{TP + FP}$. Of everything the model flagged as positive, what fraction was actually positive? High precision means few false alarms.

**Recall:** $\frac{TP}{TP + FN}$. Of all actual positives, what fraction did the model find? High recall means few missed cases.

**F1 Score:** $\frac{2 \times P \times R}{P + R}$. Harmonic mean of precision and recall. Use when both false positives and false negatives are costly.

**AUC-ROC:** Area under the ROC curve (True Positive Rate vs. False Positive Rate at various thresholds). Measures discrimination ability independent of threshold choice. AUC=1.0 is perfect; AUC=0.5 is random guessing.

**When to use what:**
- Balanced classes: accuracy is fine
- Imbalanced classes: F1 or AUC-ROC
- False positives are expensive (spam filter in medical context): optimize precision
- False negatives are expensive (disease detection): optimize recall

**My use:** iFAST reports diagnostic accuracy (97%) because the evaluation covers multiple failure modes — not a single binary classification. SPOT CHECK reports agreement rates with human auditors.

---

## 2.5 Ensemble Methods

### Bagging (Bootstrap Aggregating)

Train multiple models on random subsets of the data (with replacement). Average their predictions (regression) or vote (classification).

**Random Forest:** Bagging applied to decision trees with an additional twist — at each split, only consider a random subset of features. This decorrelates the individual trees, reducing variance further.

**Why it works:** Individual trees overfit (high variance). Averaging many uncorrelated overfitting models reduces variance without increasing bias. The ensemble is smoother than any individual tree.

### Boosting

Train models sequentially. Each new model focuses on the examples the previous models got wrong. The final prediction is a weighted sum of all models.

**Gradient Boosting:** Each new tree fits the residual errors (negative gradient of the loss) of the ensemble so far. Slowly builds up a powerful model by correcting its own mistakes.

**XGBoost (Extreme Gradient Boosting):** Gradient boosting with regularization on leaf weights (L1 + L2), column sampling (like Random Forest), and optimized implementation. The most successful algorithm for tabular data in competitive ML.

**Why it works:** Boosting reduces bias. Each new tree specifically targets the errors of the current ensemble, systematically reducing the overall error.

**My use:** The topology router in my MTP thesis uses XGBoost to classify problem instances. Features include graph diameter, branching factor, solution depth, and topological complexity. XGBoost was chosen because it handles tabular features with mixed types, provides feature importance rankings, and trains in seconds on the dataset sizes we had.

---

## 2.6 Feature Engineering

### Feature Extraction

Transforming raw data into numerical representations the model can use.

**TF-IDF (Term Frequency - Inverse Document Frequency):** Weights words by how important they are to a document relative to the corpus. Common words (the, is, are) get low weight; discriminative words get high weight. Used in BM25 retrieval (Chapter 5) and as a baseline in the SSV RAG chatbot.

**Embeddings:** Dense vector representations learned by neural networks. Capture semantic similarity — "king" and "queen" are close in embedding space. Pre-trained embeddings (Word2Vec, GloVe, sentence-transformers) are features you get for free.

**Statistical features:** Mean, variance, percentiles, counts, ratios. For tabular data, these are often the most powerful features.

### Feature Selection

Not all features help. Irrelevant features add noise and increase overfitting risk.

**Filter methods:** Score each feature independently (correlation with target, mutual information). Fast but ignores feature interactions.

**Wrapper methods:** Forward selection (add features one at a time, keep the ones that help) or backward elimination (start with all, remove the ones that do not help). Expensive but considers interactions.

**Embedded methods:** The model performs feature selection during training. Lasso (L1) drives irrelevant feature weights to zero. Tree-based models naturally rank features by split importance.

### Feature Importance

Understanding which features matter and why.

**Gini Importance (tree-based):** How much each feature reduces impurity across all splits. Built into Random Forest and XGBoost. Fast but biased toward high-cardinality features.

**Permutation Importance:** Randomly shuffle one feature, measure how much performance drops. Feature-agnostic, unbiased, but slow.

**SHAP (SHapley Additive exPlanations):** Game-theoretic approach. Computes the marginal contribution of each feature to each prediction. The gold standard for feature importance — provides both global (which features matter overall) and local (which features drove this specific prediction) explanations.

**My use:** In the topology router, SHAP analysis revealed that graph diameter and branching factor were the dominant features for predicting whether A* search would outperform Chain-of-Thought reasoning — a finding that directly informed the theoretical framework in the paper.

---

## 2.7 Bayesian Inference

Bayesian inference is the mathematics of updating beliefs as evidence accumulates. It is the theoretical foundation of iFAST's diagnostic engine.

### Bayes' Theorem

$$P(H|E) = \frac{P(E|H) \cdot P(H)}{P(E)}$$

- **$P(H)$ — Prior:** Your belief about hypothesis $H$ before seeing evidence $E$. Example: Before any investigation, the probability that a CAN bus timeout caused the failure is 5% (base rate from historical data).
- **$P(E|H)$ — Likelihood:** The probability of observing evidence $E$ if hypothesis $H$ is true. Example: If a CAN bus timeout did cause the failure, what is the probability we would see a CAN-related error in the test log? Very high (0.95).
- **$P(H|E)$ — Posterior:** Your updated belief after seeing the evidence. This is what you want to compute.
- **$P(E)$ — Evidence (Marginal Likelihood):** The total probability of seeing this evidence under all hypotheses. Acts as a normalizing constant.

### Sequential Evidence Update

The power of Bayesian inference is that it is sequential. The posterior from one piece of evidence becomes the prior for the next:

$$P(H|E_1, E_2) = \frac{P(E_2|H) \cdot P(H|E_1)}{P(E_2|E_1)}$$

Each new piece of evidence refines the belief. This is exactly how iFAST works — the agent gathers evidence from multiple tools (Jira issues, Confluence pages, test logs, historical failure patterns), and each piece of evidence updates the probability distribution over root cause hypotheses.

### Log-Likelihood Ratio (LLR)

In practice, working with probabilities directly is numerically unstable (multiplying many small numbers). The log-likelihood ratio solves this:

$$LLR = \log\frac{P(E|H_1)}{P(E|H_0)}$$

Where $H_1$ is the hypothesis of interest and $H_0$ is the null (no fault).

**Key property:** LLRs are additive across independent evidence. If you have three independent pieces of evidence, the total log-likelihood ratio is simply:

$$LLR_{total} = LLR_1 + LLR_2 + LLR_3$$

This is what makes Bayesian evidence fusion computationally tractable. iFAST maintains a running LLR for each candidate root cause, adds the contribution of each new piece of evidence, and reports the hypotheses with the highest total LLR — along with calibrated confidence scores derived from the posterior probability.

### Why Bayesian for Diagnostics?

Three reasons:
1. **Handles uncertainty explicitly.** The system knows what it does not know and reports confidence, not just answers.
2. **Integrates heterogeneous evidence.** A Jira comment, a log file pattern, and a Confluence page all contribute to the same probabilistic framework.
3. **Updates incrementally.** As the agent gathers more evidence, beliefs update smoothly. Early evidence narrows the field; later evidence confirms or overturns.

---

## 2.8 Summary

| Concept | Core Idea | Connection |
|---------|-----------|------------|
| Supervised Learning | Learn from labeled data | Topology router (XGBoost classifier) |
| Optimization | Minimize loss via gradient descent | Every neural network in this book |
| Bias-Variance | Underfitting vs. overfitting | Model selection for routing |
| Evaluation | Metrics must match the problem | iFAST (accuracy), SSV RAG (latency + quality) |
| Ensembles | Combine weak learners | XGBoost topology router |
| Feature Engineering | Transform raw data into useful signals | BM25 scoring, router features |
| Bayesian Inference | Update beliefs with evidence | iFAST's LLR diagnostic engine |

These are not separate topics. They are a connected framework: you choose a learning paradigm (supervised), define a loss function (cross-entropy), optimize it (Adam), regularize to control overfitting (L2), evaluate properly (stratified k-fold with F1), and interpret results (SHAP). Every production ML system follows this pipeline, whether it is a simple classifier or a complex agent.

Master these fundamentals and you can answer most ML interview questions. Not because you memorized answers, but because you understand the underlying machinery well enough to derive answers on the spot.

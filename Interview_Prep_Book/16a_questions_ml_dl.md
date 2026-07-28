# Chapter 16a: Machine Learning & Deep Learning Interview Questions

---

## Part I: Machine Learning (Q1-Q30)

---

### Q1: What is the bias-variance tradeoff?

**Answer:** Bias measures how far the model's average predictions are from the true values, indicating systematic underfitting. Variance measures how much predictions fluctuate across different training sets, indicating overfitting. The expected test error decomposes as $\text{Error} = \text{Bias}^2 + \text{Variance} + \sigma^2_{\text{irreducible}}$. Reducing bias (e.g., increasing model complexity) typically increases variance, and vice versa. The goal is to find the model complexity that minimizes total error, not just one component.

**In Practice:** In the Topology Router, we tested GradientBoosting depths 3-7 across 100 random seeds — depth 3 had higher bias but generalized better, while depth 7 exhibited high variance with unstable seed-to-seed performance.

---

### Q2: What is the difference between supervised and unsupervised learning?

**Answer:** Supervised learning trains on labeled data $(x_i, y_i)$ to learn a mapping $f: X \rightarrow Y$, optimizing a loss that compares predictions to ground truth. Unsupervised learning operates on unlabeled data $\{x_i\}$ and discovers hidden structure such as clusters, manifolds, or density estimates. Semi-supervised learning blends both, using a small labeled set with a large unlabeled corpus. The choice depends on label availability and whether the task is predictive (supervised) or exploratory (unsupervised).

**In Practice:** The Topology Router uses supervised classification (XGBoost on labeled topology-outcome pairs), while the topology feature extraction itself uses unsupervised graph metrics computed without outcome labels.

---

### Q3: Explain stratified k-fold cross-validation and why it matters for imbalanced datasets.

**Answer:** Stratified k-fold partitions the dataset into $k$ folds while preserving the class distribution in each fold. For a binary problem with 20% positives, each fold will also contain approximately 20% positives. This prevents folds where the minority class is absent or overrepresented, which would produce unreliable performance estimates. The final metric is averaged across folds: $\bar{m} = \frac{1}{k}\sum_{i=1}^{k} m_i$, with standard deviation quantifying stability.

**In Practice:** The Topology Router uses 5-fold stratified CV repeated over 100 random seeds, ensuring that the 4-class topology distribution is preserved in every train/test split for robust Gap% estimation.

---

### Q4: Define accuracy, precision, recall, F1-score, and when each is appropriate.

**Answer:** Accuracy $= \frac{TP + TN}{TP + TN + FP + FN}$ works only when classes are balanced. Precision $= \frac{TP}{TP + FP}$ measures how many positive predictions are correct (important when false positives are costly). Recall $= \frac{TP}{TP + FN}$ measures how many actual positives are captured (important when false negatives are costly). F1 $= \frac{2 \cdot P \cdot R}{P + R}$ is the harmonic mean, penalizing imbalance between precision and recall. For multi-class problems, macro-F1 (unweighted average across classes) and weighted-F1 (weighted by class support) are common.

**In Practice:** The Topology Router reports macro-F1 because the four topology classes (Sequential, Tree, DAG, Cyclic) have different frequencies, and we need balanced performance across all categories.

---

### Q5: What is AUC-ROC and how does it differ from accuracy?

**Answer:** AUC-ROC is the area under the Receiver Operating Characteristic curve, which plots True Positive Rate ($TPR = \frac{TP}{TP+FN}$) vs. False Positive Rate ($FPR = \frac{FP}{FP+TN}$) at all classification thresholds. AUC = 0.5 means random, AUC = 1.0 means perfect separation. Unlike accuracy, AUC is threshold-independent and robust to class imbalance — it evaluates the model's ranking ability rather than a single decision boundary. For highly imbalanced data, AUC-PR (precision-recall) is often more informative than AUC-ROC.

**In Practice:** In iFAST's Bayesian evidence scoring, AUC-ROC validates that the combined log-likelihood ratio from 8 evidence sources ranks code-change candidates better than any single source alone.

---

### Q6: Explain Gini impurity and how it determines feature importance in tree-based models.

**Answer:** Gini impurity for a node is $G = 1 - \sum_{c=1}^{C} p_c^2$, where $p_c$ is the proportion of class $c$ samples. A pure node has $G = 0$. When splitting, the algorithm chooses the feature and threshold that maximizes the weighted Gini reduction: $\Delta G = G_{\text{parent}} - \frac{n_L}{n}G_L - \frac{n_R}{n}G_R$. Feature importance is the total Gini reduction contributed by a feature across all splits in all trees, normalized to sum to 1. Features that appear in early splits with large sample sizes accumulate more importance.

**In Practice:** In the Topology Router's XGBoost model with 14 features, Gini-based importance revealed that graph diameter and branching factor were the top discriminators for routing decisions across topology classes.

---

### Q7: How does a Random Forest work, and what makes it robust against overfitting?

**Answer:** Random Forest builds $B$ decision trees, each trained on a bootstrap sample (sampling $n$ examples with replacement) and restricted to a random subset of $m \leq p$ features at each split. Predictions are aggregated via majority vote (classification) or averaging (regression). The two sources of randomness — bagging and feature subsampling — decorrelate the trees, reducing variance without increasing bias. The out-of-bag (OOB) error, computed on the ~37% of samples not selected in each bootstrap, provides a built-in validation estimate without needing a separate holdout.

**In Practice:** Before selecting XGBoost for the Topology Router, we benchmarked Random Forest as a baseline — it achieved competitive accuracy but its Gap% was 8 points lower due to inability to sequentially correct errors on hard topology instances.

---

### Q8: What is Gradient Boosting and how does it differ from Random Forest?

**Answer:** Gradient Boosting builds trees sequentially, where each new tree $h_m(x)$ fits the negative gradient (pseudo-residuals) of the loss: $F_m(x) = F_{m-1}(x) + \eta \cdot h_m(x)$. This is additive modeling in function space, minimizing $L(y, F(x))$ via gradient descent. Unlike Random Forest's parallel/independent trees that reduce variance, boosting reduces bias by iteratively correcting errors. The learning rate $\eta$ controls the contribution of each tree — smaller $\eta$ requires more trees but generalizes better due to regularization through shrinkage.

**In Practice:** The Topology Router uses Gradient Boosting (via XGBoost) because topology routing is a bias-dominated problem — correctly classifying hard boundary cases requires the sequential error-correction that boosting provides.

---

### Q9: Explain XGBoost's key hyperparameters: learning_rate, max_depth, and n_estimators.

**Answer:** `learning_rate` ($\eta \in (0, 1]$) shrinks each tree's contribution, acting as regularization — lower values (e.g., 0.01-0.1) require more trees but reduce overfitting. `max_depth` controls tree complexity; shallow trees (3-6) are weak learners that generalize well, while deep trees (>8) can memorize noise. `n_estimators` is the number of boosting rounds; with early stopping on validation loss, this is effectively determined automatically. These three interact: halving $\eta$ roughly requires doubling `n_estimators` to achieve the same training loss but with better generalization.

**In Practice:** The Topology Router's XGBoost uses learning_rate=0.05, max_depth=4, n_estimators=300 with early stopping — tuned via grid search over 5-fold stratified CV across 100 random seeds to ensure stable Gap% performance.

---

### Q10: What is overfitting and how do L1 and L2 regularization prevent it?

**Answer:** Overfitting occurs when a model learns noise in the training data, achieving low training error but high test error (high variance). L2 regularization (Ridge) adds $\lambda \sum_j w_j^2$ to the loss, shrinking weights toward zero but never exactly to zero, producing smooth solutions. L1 regularization (Lasso) adds $\lambda \sum_j |w_j|$, which encourages exact zeros and thus performs feature selection. The regularization strength $\lambda$ controls the bias-variance tradeoff: larger $\lambda$ increases bias but reduces variance. In XGBoost, L1 and L2 are applied to leaf weights via `reg_alpha` and `reg_lambda` parameters.

**In Practice:** In the Topology Router, we set `reg_lambda=1.0` (L2) and `reg_alpha=0.1` (L1) in XGBoost, which eliminated 3 of the 14 features via effective zeroing and improved seed-to-seed stability by 12%.

---

### Q11: State Bayes' theorem and explain prior, likelihood, and posterior.

**Answer:** Bayes' theorem is $P(H|E) = \frac{P(E|H) \cdot P(H)}{P(E)}$, where $P(H)$ is the prior (belief before observing evidence), $P(E|H)$ is the likelihood (probability of evidence given hypothesis), and $P(H|E)$ is the posterior (updated belief after evidence). $P(E) = \sum_h P(E|h)P(h)$ is the marginal likelihood (normalizing constant). The theorem formalizes how to update beliefs as new evidence arrives — the posterior from one observation becomes the prior for the next, enabling sequential updating.

**In Practice:** iFAST implements full Bayesian updating where the prior $P(\text{CodeChange})$ is updated sequentially as each of 8 evidence sources is observed, producing a calibrated posterior probability for code-change recommendations.

---

### Q12: What are Log-Likelihood Ratios and why are they useful for evidence combination?

**Answer:** The Log-Likelihood Ratio for evidence $E_i$ is $LLR_i = \log \frac{P(E_i|H)}{P(E_i|\neg H)}$, measuring the discriminative power of evidence in favor of hypothesis $H$. Positive LLR supports $H$; negative LLR supports $\neg H$; zero means uninformative. The key advantage is additivity: when evidence sources are conditionally independent given $H$, the total evidence is simply $LLR_{\text{total}} = \sum_{i=1}^{n} LLR_i$. The posterior odds then become $\frac{P(H|E)}{P(\neg H|E)} = \frac{P(H)}{P(\neg H)} \cdot e^{LLR_{\text{total}}}$.

**In Practice:** iFAST computes $P(\text{CodeChange}|E) = \sigma(\sum_{i=1}^{8} LLR_i)$ where $\sigma$ is the sigmoid function, combining log-likelihood ratios from 8 independent evidence sources (stack traces, error patterns, commit recency, etc.) into a single calibrated probability.

---

### Q13: How do you handle class imbalance in classification problems?

**Answer:** Strategies include: (1) resampling — oversampling the minority class (SMOTE generates synthetic samples along minority-class line segments) or undersampling the majority; (2) cost-sensitive learning — assigning higher misclassification cost to the minority class via `scale_pos_weight` in XGBoost or `class_weight='balanced'` in sklearn; (3) threshold adjustment — moving the decision boundary from 0.5 to a value that optimizes F1 or business cost; (4) ensemble approaches like EasyEnsemble or BalancedRandomForest. The metric choice (F1, AUC-PR rather than accuracy) is equally critical.

**In Practice:** In the Topology Router, the "Cyclic" topology class represents only 8% of instances, so we use stratified CV to preserve this ratio and set `scale_pos_weight` proportional to the inverse class frequency during XGBoost training.

---

### Q14: Compare bagging vs. boosting as ensemble strategies.

**Answer:** Bagging (Bootstrap Aggregating) trains $B$ base learners independently on bootstrap samples and aggregates via voting/averaging, reducing variance while leaving bias unchanged — effective when base learners overfit (e.g., deep trees). Boosting trains base learners sequentially, where each focuses on errors of the ensemble so far, reducing bias — effective when base learners underfit (e.g., shallow stumps). Bagging is embarrassingly parallel; boosting is inherently sequential. Bagging is robust to noisy labels; boosting can overfit to label noise by assigning excessive weight to noisy samples.

**In Practice:** The Topology Router uses boosting (XGBoost) rather than bagging (Random Forest) because the routing task has high bias — the four topology classes share overlapping feature distributions that require iterative refinement to separate.

---

### Q15: Describe the hyperparameter tuning strategies: grid search, random search, and Bayesian optimization.

**Answer:** Grid search exhaustively evaluates all combinations on a predefined grid — exponential cost $O(n^d)$ for $d$ hyperparameters with $n$ values each. Random search samples combinations uniformly from the hyperparameter space; Bergstra & Bengio (2012) showed it finds good configurations faster because it explores more unique values per dimension. Bayesian optimization (e.g., TPE, Gaussian Processes) builds a surrogate model of $f(\theta) \rightarrow \text{validation loss}$ and uses an acquisition function (Expected Improvement) to select the next evaluation point, trading exploration vs. exploitation. For $<$50 evaluations, Bayesian optimization dominates; for cheap evaluations, random search suffices.

**In Practice:** The Topology Router uses grid search over (learning_rate, max_depth, n_estimators) with 5-fold stratified CV, which is feasible because the 14-feature XGBoost model trains in under 2 seconds per fold.

---

### Q16: Explain the purpose of train/validation/test splits and common strategies.

**Answer:** The training set fits model parameters, the validation set tunes hyperparameters and detects overfitting (early stopping), and the test set provides an unbiased generalization estimate that is never used during model development. A common split is 60/20/20 or 70/15/15. For small datasets, nested cross-validation replaces the fixed split: the outer loop estimates test performance, the inner loop tunes hyperparameters. Data leakage — where test information bleeds into training via feature engineering, normalization, or temporal ordering — invalidates the entire evaluation.

**In Practice:** The Topology Router uses 100 different random seeds to generate 100 independent train/test splits within its 5-fold stratified CV, providing both a mean Gap% and confidence intervals that account for partition variance.

---

### Q17: What is the curse of dimensionality and how does it affect ML models?

**Answer:** As the number of features $p$ grows, the volume of the feature space increases exponentially, causing data points to become sparse. In high dimensions, distances between points converge (all pairs become equidistant), making distance-based methods like KNN ineffective. Models require exponentially more data to maintain the same statistical density: to cover a $p$-dimensional unit hypercube with density $d$, you need $d^p$ samples. Dimensionality reduction (PCA, feature selection) and regularization are standard mitigations.

**In Practice:** The Topology Router deliberately uses only 14 carefully engineered features rather than raw graph adjacency matrices (which could have hundreds of dimensions), avoiding the curse while retaining discriminative power.

---

### Q18: How does Principal Component Analysis (PCA) work?

**Answer:** PCA finds orthogonal directions of maximum variance in the data by computing the eigendecomposition of the covariance matrix $\Sigma = \frac{1}{n}X^TX$. The $k$-th principal component is the $k$-th eigenvector $v_k$ with eigenvalue $\lambda_k$ representing the variance explained. Projecting data onto the top $d$ components gives $Z = XV_d$, reducing dimensionality while preserving $\frac{\sum_{i=1}^d \lambda_i}{\sum_{i=1}^p \lambda_i} \times 100\%$ of total variance. PCA assumes linear relationships and is sensitive to feature scaling — always standardize features first.

**In Practice:** Before training the Topology Router, we applied PCA to the 14 features to verify they were not redundant — the first 10 components explained 95% of variance, confirming that most features carry unique information.

---

### Q19: What is the difference between generative and discriminative models?

**Answer:** Discriminative models learn the decision boundary $P(y|x)$ directly (e.g., logistic regression, SVM, neural networks), while generative models learn the joint distribution $P(x, y) = P(x|y)P(y)$ and derive the posterior via Bayes' theorem. Generative models can generate new samples and handle missing data naturally, but they require modeling the full input distribution which is harder. Discriminative models typically achieve better classification accuracy because they focus on what distinguishes classes rather than modeling each class completely.

**In Practice:** iFAST uses a generative Bayesian approach — modeling $P(E|\text{CodeChange})$ and $P(E|\neg\text{CodeChange})$ for each evidence source — because it needs calibrated probabilities and the ability to explain which evidence contributes most to the recommendation.

---

### Q20: Explain logistic regression and the sigmoid function.

**Answer:** Logistic regression models $P(y=1|x) = \sigma(w^Tx + b)$ where $\sigma(z) = \frac{1}{1+e^{-z}}$ maps any real value to $(0,1)$. The model is trained by minimizing the binary cross-entropy loss: $L = -\frac{1}{n}\sum_i [y_i \log \hat{y}_i + (1-y_i)\log(1-\hat{y}_i)]$. Despite the name, it is a classification algorithm. The decision boundary is linear in feature space: $w^Tx + b = 0$. The log-odds are linear: $\log \frac{P(y=1|x)}{P(y=0|x)} = w^Tx + b$, making coefficients directly interpretable as the change in log-odds per unit feature change.

**In Practice:** iFAST's final scoring uses the sigmoid function: $P(\text{CodeChange}|E) = \sigma(\sum_i LLR_i)$, which is mathematically equivalent to logistic regression where the log-likelihood ratios serve as pre-computed feature contributions.

---

### Q21: What is the kernel trick in Support Vector Machines?

**Answer:** The kernel trick computes the dot product in a high-dimensional feature space without explicitly mapping the data there: $K(x_i, x_j) = \phi(x_i)^T\phi(x_j)$. Common kernels include linear ($x_i^Tx_j$), polynomial ($(x_i^Tx_j + c)^d$), and RBF ($\exp(-\gamma\|x_i - x_j\|^2)$). This allows SVMs to find nonlinear decision boundaries in the original space that correspond to linear boundaries in the transformed space. The computational cost depends on the number of support vectors rather than the dimensionality of $\phi(x)$, which can even be infinite (as with the RBF kernel).

**In Practice:** In early Topology Router experiments, we tested SVM with RBF kernel as a baseline — it performed comparably on 2-class problems but scaled poorly to the 4-class topology setting where XGBoost's native multi-class support was more natural.

---

### Q22: Explain the concept of feature engineering and its importance.

**Answer:** Feature engineering transforms raw data into representations that make patterns more accessible to the model. Good features encode domain knowledge: ratios, interactions, polynomial terms, temporal aggregates, or graph metrics. The quality of features often matters more than model choice — a simple model with good features outperforms a complex model with poor features. Automated feature engineering (e.g., Featuretools) can generate interaction terms, but domain-driven features typically have higher signal-to-noise ratio. Feature engineering should be done within cross-validation folds to prevent data leakage.

**In Practice:** The Topology Router's 14 features (graph diameter, node count, edge density, branching factor, cycle count, longest path, etc.) were hand-engineered from graph topology, encoding domain knowledge about what makes a problem suitable for different reasoning strategies.

---

### Q23: What is early stopping and how does it prevent overfitting?

**Answer:** Early stopping monitors validation loss during iterative training (gradient boosting rounds, neural network epochs) and halts training when validation loss stops improving for a patience window of $k$ iterations. It acts as implicit regularization — limiting model complexity by restricting the number of effective parameters (boosting rounds or training steps). The optimal stopping point is typically identified by tracking the best validation score and restoring the model to that checkpoint. In XGBoost, `early_stopping_rounds=k` stops training if the validation metric hasn't improved in $k$ consecutive rounds.

**In Practice:** The Topology Router uses early_stopping_rounds=20 on validation macro-F1, which typically stops training at 180-220 trees rather than the maximum 300, reducing both overfitting and training time.

---

### Q24: What is the difference between parametric and non-parametric models?

**Answer:** Parametric models assume a fixed functional form with a finite number of parameters (e.g., linear regression has $p+1$ parameters regardless of dataset size). Non-parametric models have complexity that grows with data (e.g., KNN stores all training points, decision trees can grow unbounded). Parametric models are computationally efficient at inference but may underfit if the assumed form is wrong. Non-parametric models are more flexible but can overfit and scale poorly. Semi-parametric models (e.g., GAMs) combine fixed-form linear components with flexible non-parametric components.

**In Practice:** XGBoost in the Topology Router is non-parametric (tree complexity adapts to data), which is critical because the relationship between topology features and optimal routing strategy is unlikely to follow any simple parametric form.

---

### Q25: Explain the concept of information gain and entropy in decision trees.

**Answer:** Entropy measures impurity: $H(S) = -\sum_{c=1}^C p_c \log_2 p_c$, with maximum at uniform distribution and zero for a pure node. Information Gain for a split on feature $A$ is $IG(S, A) = H(S) - \sum_{v \in \text{values}(A)} \frac{|S_v|}{|S|} H(S_v)$. The algorithm greedily selects the split that maximizes IG (ID3/C4.5) or minimizes Gini impurity (CART). Gain ratio $= \frac{IG(S,A)}{H_A(S)}$ corrects for features with many values that trivially achieve high IG (e.g., unique IDs). Continuous features are handled by testing all midpoint thresholds and selecting the one with maximum IG.

**In Practice:** In the Topology Router, XGBoost uses an approximate version of this — the histogram-based split finding evaluates candidate splits at quantile boundaries of each feature, achieving the same discriminative power as exact splitting with $O(n)$ instead of $O(n \log n)$ complexity.

---

### Q26: What are the assumptions of linear regression and how do you check them?

**Answer:** The key assumptions are: (1) linearity — $E[y|x] = X\beta$; (2) independence — residuals $\epsilon_i$ are independent; (3) homoscedasticity — $\text{Var}(\epsilon_i) = \sigma^2$ constant; (4) normality — $\epsilon \sim N(0, \sigma^2)$ for valid confidence intervals. Check via: residual plots (patterns indicate non-linearity), Q-Q plots (normality), Breusch-Pagan test (heteroscedasticity), Durbin-Watson test (autocorrelation), and VIF (multicollinearity, VIF > 10 is problematic). Violations can be addressed with transformations, robust standard errors, or switching to non-linear models.

**In Practice:** When benchmarking simple baselines for the Topology Router, linear regression on topology features violated the linearity assumption (clear non-linear residual patterns), confirming that tree-based models are necessary for this non-linear classification task.

---

### Q27: What is multi-collinearity and why does it matter?

**Answer:** Multi-collinearity occurs when predictor variables are highly correlated ($|r_{ij}| > 0.8$ or VIF $= \frac{1}{1-R_j^2} > 10$). It inflates the variance of regression coefficients, making them unstable and difficult to interpret — small data changes cause large coefficient swings. The matrix $X^TX$ becomes ill-conditioned, and its inverse (needed for OLS) amplifies numerical errors. Solutions include: removing one of the correlated features, applying PCA, using regularization (Ridge regression is specifically designed for this), or accepting that predictions remain valid even if individual coefficients are uninterpretable.

**In Practice:** Among the Topology Router's 14 features, edge_count and edge_density had correlation 0.85 — we retained both because XGBoost's tree splits are invariant to multi-collinearity (unlike linear models), and removing either reduced Gap% by 2 points.

---

### Q28: Explain the concept of model calibration and why raw probabilities may not be reliable.

**Answer:** A model is calibrated if its predicted probability matches the empirical frequency: among all instances predicted as 70% positive, exactly 70% should actually be positive. Most models are poorly calibrated out-of-the-box — neural networks tend to be overconfident, while boosted trees are often underconfident. Calibration is assessed via reliability diagrams (predicted vs. actual probability) and Expected Calibration Error (ECE). Post-hoc calibration methods include Platt scaling (fit a logistic regression on logits) and isotonic regression (non-parametric monotone mapping). Calibration is critical when probabilities drive downstream decisions.

**In Practice:** iFAST's Bayesian LLR framework produces inherently calibrated probabilities because the sigmoid of summed LLRs directly represents posterior probability — validated by a reliability diagram showing near-perfect diagonal alignment across 8 evidence sources.

---

### Q29: What is the Gap% metric and how does it evaluate routing performance?

**Answer:** Gap% measures how much of the theoretically achievable improvement a router captures: $\text{Gap\%} = \frac{\text{Router} - \text{Best\_Single}}{\text{Oracle} - \text{Best\_Single}} \times 100$. Best_Single is the performance of the best single strategy applied uniformly to all instances. Oracle is the performance achieved by always selecting the optimal strategy per instance (upper bound). Gap% = 0% means the router is no better than the best uniform strategy; Gap% = 100% means it matches the Oracle. This metric is invariant to the absolute performance scale and directly quantifies routing value-add.

**In Practice:** The Topology Router achieves Gap% = 72.3% averaged across 100 random seeds with 5-fold stratified CV, demonstrating that topology-based features capture most of the per-instance routing signal available in the Oracle.

---

### Q30: How do you determine if a model's improvement over a baseline is statistically significant?

**Answer:** Use a paired statistical test comparing per-instance (or per-fold) metrics between the two models. The paired t-test assumes normal differences: $t = \frac{\bar{d}}{s_d / \sqrt{n}}$ where $d_i$ are paired differences. The Wilcoxon signed-rank test is the non-parametric alternative when normality is violated. For cross-validation, corrected resampled t-tests account for the non-independence of overlapping training sets. McNemar's test compares classification disagreements on a contingency table. Report effect size (Cohen's d) alongside p-values — statistical significance with tiny effect size is practically meaningless.

**In Practice:** The Topology Router's 100 random seeds produce a distribution of Gap% values, allowing a one-sample t-test against Gap% = 0 (no routing benefit) — the result is $p < 10^{-15}$, confirming the improvement is not due to random seed selection.

---

## Part II: Deep Learning (Q31-Q50)

---

### Q31: Explain backpropagation and its role in training neural networks.

**Answer:** Backpropagation computes gradients of the loss $L$ with respect to all parameters $\theta$ by applying the chain rule recursively from output to input layers. For a layer $l$: $\frac{\partial L}{\partial W^{(l)}} = \frac{\partial L}{\partial z^{(l)}} \cdot \frac{\partial z^{(l)}}{\partial W^{(l)}}$ where $z^{(l)} = W^{(l)}a^{(l-1)} + b^{(l)}$. The forward pass computes activations and caches intermediate values; the backward pass propagates $\frac{\partial L}{\partial a^{(l)}}$ from the last layer to the first. Gradients are then used by an optimizer (SGD, Adam) to update parameters: $\theta \leftarrow \theta - \eta \nabla_\theta L$. The computational cost is approximately twice the forward pass.

**In Practice:** In the MTP thesis, understanding backpropagation through LoRA adapters is essential — gradients flow through the frozen base model to update only the low-rank matrices $A$ and $B$, making fine-tuning memory-efficient.

---

### Q32: What causes vanishing and exploding gradients, and how are they mitigated?

**Answer:** In a network with $L$ layers, the gradient at layer $l$ involves a product of $L - l$ Jacobians: $\frac{\partial L}{\partial W^{(l)}} \propto \prod_{k=l}^{L-1} \frac{\partial a^{(k+1)}}{\partial a^{(k)}}$. If each factor has spectral norm $< 1$, gradients vanish exponentially; if $> 1$, they explode. Sigmoid/tanh activations saturate (derivative $\approx 0$), causing vanishing gradients. Mitigations include: ReLU activations (gradient = 1 for positive inputs), residual connections ($x + f(x)$ ensures gradient flows directly), careful initialization (He/Xavier), gradient clipping (cap norm at threshold), and batch normalization.

**In Practice:** The transformer models used in all projects (Claude, LLaMA, Gemini) employ residual connections at every layer and layer normalization, which together prevent gradient pathology even in 32+ layer architectures.

---

### Q33: How does Batch Normalization work and why does it help training?

**Answer:** Batch Normalization normalizes each feature across the mini-batch: $\hat{x}_i = \frac{x_i - \mu_B}{\sqrt{\sigma_B^2 + \epsilon}}$, then applies a learned affine transform $y_i = \gamma \hat{x}_i + \beta$. During training, $\mu_B$ and $\sigma_B^2$ are computed per mini-batch; at inference, running averages are used. It reduces internal covariate shift, allows higher learning rates, acts as a regularizer (due to mini-batch noise), and makes the loss landscape smoother. Transformers typically use Layer Normalization instead (normalizing across features, not batch) because it works with variable-length sequences and is batch-size independent.

**In Practice:** The LLaMA-3.1-8B model used in the MTP thesis employs RMSNorm (a simplified Layer Norm without mean centering: $\hat{x} = \frac{x}{\text{RMS}(x)} \cdot \gamma$), which achieves similar stabilization benefits with fewer operations.

---

### Q34: Explain dropout and how it acts as regularization.

**Answer:** Dropout randomly sets each neuron's output to zero with probability $p$ during training, forcing the network to learn redundant representations. At test time, all neurons are active but outputs are scaled by $(1-p)$ (or equivalently, training outputs are scaled by $\frac{1}{1-p}$ — inverted dropout). This approximates an ensemble of $2^n$ sub-networks (where $n$ is the number of neurons) that share parameters. Typical values are $p=0.1$-$0.5$. In transformers, dropout is applied to attention weights and feed-forward layers. Higher dropout increases regularization but may require longer training to converge.

**In Practice:** During LoRA fine-tuning of LLaMA-3.1-8B in the MTP thesis, dropout is applied to the LoRA adapter layers (not the frozen base model) at $p=0.05$ to prevent the low-rank matrices from overfitting to the small distillation dataset.

---

### Q35: Describe the architecture of a Convolutional Neural Network (conv, pooling, FC layers).

**Answer:** A CNN consists of: (1) convolutional layers that apply learnable filters $K$ (e.g., $3 \times 3$) to produce feature maps via $(\text{input} * K)[i,j] = \sum_{m,n} \text{input}[i+m, j+n] \cdot K[m,n]$, with parameter sharing across spatial locations; (2) pooling layers (max or average) that downsample feature maps, providing translational invariance and reducing computation; (3) fully connected layers that flatten the final feature maps and produce class logits. The hierarchical structure captures increasingly abstract features: edges $\rightarrow$ textures $\rightarrow$ parts $\rightarrow$ objects. Modern architectures (ResNet, EfficientNet) add skip connections and compound scaling.

**In Practice:** While the MTP thesis focuses on language models (transformers), understanding CNNs is relevant because vision transformers (ViT) replaced convolutional inductive biases with self-attention, demonstrating that transformers can learn spatial hierarchies from data alone.

---

### Q36: Compare RNN, LSTM, and GRU architectures for sequence modeling.

**Answer:** Vanilla RNNs compute $h_t = \tanh(W_h h_{t-1} + W_x x_t + b)$ but suffer from vanishing gradients over long sequences. LSTM introduces gates: forget gate $f_t = \sigma(W_f[h_{t-1}, x_t])$, input gate $i_t = \sigma(W_i[h_{t-1}, x_t])$, output gate $o_t = \sigma(W_o[h_{t-1}, x_t])$, and a cell state $c_t = f_t \odot c_{t-1} + i_t \odot \tilde{c}_t$ that acts as a gradient highway. GRU simplifies LSTM to two gates (reset and update), combining forget and input gates, using fewer parameters with comparable performance. All three are largely superseded by transformers for most NLP tasks due to parallelization limitations.

**In Practice:** The evolution from RNNs to transformers motivates the entire architecture of models used in our projects — Claude, LLaMA, Gemini all use transformer decoders that process sequences in parallel rather than the sequential bottleneck of recurrent architectures.

---

### Q37: Explain the attention mechanism with Query, Key, and Value matrices.

**Answer:** Attention computes a weighted sum of Value vectors, where weights are determined by the compatibility between a Query and all Keys: $\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$. The Query asks "what am I looking for?", Keys advertise "what do I contain?", and Values provide "what information do I return?". The scaling factor $\sqrt{d_k}$ prevents dot products from growing too large (which would push softmax into saturation). Each attention weight $\alpha_{ij}$ indicates how much position $i$ should attend to position $j$, enabling dynamic context-dependent information routing.

**In Practice:** In the MTP thesis, trace distillation requires the student model (LLaMA-3.1-8B) to attend to reasoning traces — the attention mechanism learns which parts of the teacher's trace are most informative for producing the final answer.

---

### Q38: What is self-attention and how does it differ from cross-attention?

**Answer:** In self-attention, Q, K, and V all derive from the same sequence: $Q = XW_Q$, $K = XW_K$, $V = XW_V$, where $X$ is the input representation. This allows each token to attend to all other tokens in the same sequence, capturing intra-sequence dependencies regardless of distance. Cross-attention uses Q from one sequence and K, V from another — used in encoder-decoder models where the decoder queries attend to encoder outputs. Self-attention has $O(n^2)$ complexity in sequence length $n$, motivating efficient variants (linear attention, sparse attention, Flash Attention) for long sequences.

**In Practice:** All transformer-based models in our projects use causal self-attention (decoder-only, where each token can only attend to preceding tokens), which is natural for autoregressive generation of reasoning traces and solutions.

---

### Q39: How does multi-head attention work and why is it beneficial?

**Answer:** Multi-head attention runs $h$ parallel attention operations with different learned projections: $\text{head}_i = \text{Attention}(XW_Q^i, XW_K^i, XW_V^i)$, then concatenates and projects: $\text{MHA}(X) = \text{Concat}(\text{head}_1, ..., \text{head}_h)W_O$. Each head uses dimension $d_k = d_{\text{model}} / h$, so total computation is similar to single-head attention with full dimensionality. Multiple heads allow the model to jointly attend to information from different representation subspaces — one head might capture syntactic relationships, another semantic similarity, another positional patterns. This diversity is more expressive than a single attention function.

**In Practice:** LLaMA-3.1-8B (used in the MTP thesis) employs 32 attention heads with Grouped Query Attention (GQA), where multiple query heads share fewer key/value heads to reduce KV-cache memory during inference.

---

### Q40: Explain positional encoding in transformers and why it is necessary.

**Answer:** Self-attention is permutation-equivariant — it produces the same output regardless of input order. Positional encodings inject sequence order information. The original transformer uses sinusoidal encoding: $PE_{(pos, 2i)} = \sin(pos / 10000^{2i/d})$, $PE_{(pos, 2i+1)} = \cos(pos / 10000^{2i/d})$, which provides unique patterns for each position and allows the model to learn relative positions via linear transformations. Modern alternatives include learned positional embeddings, Rotary Position Embedding (RoPE: $q' = R_\theta q$, applying rotation matrices based on position), and ALiBi (adding position-dependent bias to attention scores).

**In Practice:** LLaMA-3.1-8B uses RoPE (Rotary Position Embedding), which encodes relative position directly into the Q/K dot product, enabling better length generalization beyond the training context window during trace distillation in the MTP thesis.

---

### Q41: Describe the transformer encoder and decoder architecture.

**Answer:** The encoder consists of $N$ stacked layers, each with multi-head self-attention followed by a position-wise FFN, with residual connections and layer normalization around each sub-layer: $\text{output} = \text{LayerNorm}(x + \text{SubLayer}(x))$. The decoder adds a masked self-attention layer (causal mask prevents attending to future positions) and a cross-attention layer that queries encoder outputs. The encoder processes the full input bidirectionally; the decoder generates output autoregressively token-by-token. Modern LLMs (GPT, LLaMA, Claude) use decoder-only architectures, while BERT uses encoder-only, and T5/BART use the full encoder-decoder.

**In Practice:** All generation models in our projects (Claude, LLaMA, Qwen, Gemini) are decoder-only transformers, meaning they use causal (masked) self-attention without a separate encoder, unifying input processing and generation in a single architecture.

---

### Q42: Compare Masked Language Modeling (MLM) vs. Causal Language Modeling (CLM) for pre-training.

**Answer:** MLM (used in BERT) randomly masks ~15% of input tokens and predicts them from bidirectional context: $P(x_{\text{mask}} | x_{\backslash \text{mask}})$. It produces rich bidirectional representations but cannot generate text autoregressively. CLM (used in GPT, LLaMA) predicts the next token given all preceding tokens: $P(x_t | x_{<t})$, trained with causal attention masking. CLM naturally supports generation and scales to arbitrary sequence lengths. Hybrid approaches (XLNet uses permutation LM, T5 uses span corruption) attempt to combine bidirectional understanding with generation capability. CLM has dominated scaling because the next-token prediction loss is a universal objective.

**In Practice:** The LLaMA-3.1-8B model in the MTP thesis is pre-trained with CLM and fine-tuned to generate reasoning traces — the causal pre-training objective directly aligns with the downstream generation task of producing step-by-step solutions.

---

### Q43: What is fine-tuning and when should you fine-tune vs. use a pre-trained model directly?

**Answer:** Fine-tuning adapts a pre-trained model's weights to a downstream task by continuing training on task-specific data, typically with a smaller learning rate ($\eta_{\text{fine-tune}} \approx \eta_{\text{pre-train}} / 10$-$100$). Fine-tune when: (1) the task distribution differs significantly from pre-training data; (2) you need specialized behavior (format, style, domain knowledge); (3) in-context learning (prompting) is insufficient. Use the pre-trained model directly (zero/few-shot) when: data is scarce, the task is well-covered by pre-training, or you need flexibility across many tasks. Full fine-tuning updates all parameters; parameter-efficient methods (LoRA, adapters) update a small subset.

**In Practice:** The MTP thesis fine-tunes LLaMA-3.1-8B to distill reasoning traces from a stronger teacher model — prompting alone cannot teach the student to reproduce the teacher's reasoning style, necessitating weight updates via LoRA.

---

### Q44: Explain LoRA (Low-Rank Adaptation) and the role of the rank parameter.

**Answer:** LoRA freezes the pre-trained weight matrix $W_0 \in \mathbb{R}^{d \times d}$ and adds a low-rank decomposition: $W = W_0 + \Delta W = W_0 + BA$, where $B \in \mathbb{R}^{d \times r}$ and $A \in \mathbb{R}^{r \times d}$ with rank $r \ll d$. Only $A$ and $B$ are trained, reducing trainable parameters from $d^2$ to $2dr$ — for $d=4096$ and $r=16$, this is a $128\times$ reduction. The rank $r$ controls the expressiveness of the adaptation: higher $r$ captures more complex task-specific modifications but increases memory and overfitting risk. $A$ is initialized with random Gaussian, $B$ with zeros, so $\Delta W = 0$ at initialization.

**In Practice:** The MTP thesis uses LoRA with rank $r=16$ on LLaMA-3.1-8B's attention projections (Q, K, V, O matrices), reducing trainable parameters from 8B to ~26M while achieving 94% of full fine-tuning performance on trace distillation.

---

### Q45: What is QLoRA and how does it enable fine-tuning on consumer hardware?

**Answer:** QLoRA combines 4-bit quantization of the base model with LoRA adapters in full precision (16-bit). The base model weights are stored in NF4 (Normal Float 4-bit) format with double quantization (quantizing the quantization constants), reducing memory from ~16GB (FP16) to ~4GB for a 7B model. Backpropagation computes gradients through the quantized weights but updates only the FP16 LoRA adapters. Paged optimizers handle memory spikes via CPU offloading. QLoRA achieves performance within 0.5% of full-precision fine-tuning while fitting a 65B parameter model on a single 48GB GPU.

**In Practice:** For the MTP thesis experiments, QLoRA enables fine-tuning LLaMA-3.1-8B on a single A100 GPU by quantizing the 8B frozen parameters to 4-bit while training rank-16 LoRA adapters at BF16 precision.

---

### Q46: What is PEFT (Parameter-Efficient Fine-Tuning) and what are its main approaches?

**Answer:** PEFT methods adapt large pre-trained models by updating only a small fraction of parameters, preserving the base model's knowledge while reducing compute and storage costs. Main approaches: (1) LoRA — low-rank weight updates; (2) Adapters — small bottleneck modules inserted between layers with residual connections; (3) Prefix Tuning — prepending learnable continuous vectors to keys/values at each layer; (4) Prompt Tuning — learning soft prompt embeddings concatenated to the input. PEFT enables serving multiple task-specific models from a single base model by swapping lightweight adapter weights (few MB) rather than full model copies (multiple GB).

**In Practice:** The MTP thesis uses the HuggingFace PEFT library with LoRA applied to all linear layers in LLaMA-3.1-8B, enabling rapid experimentation with different distillation strategies without the storage cost of multiple full model checkpoints.

---

### Q47: Explain learning rate scheduling: warmup and cosine decay.

**Answer:** Learning rate scheduling adjusts $\eta$ during training to balance exploration (high $\eta$ early) and convergence (low $\eta$ late). Linear warmup gradually increases $\eta$ from 0 to $\eta_{\max}$ over $T_w$ steps: $\eta_t = \eta_{\max} \cdot \frac{t}{T_w}$ for $t < T_w$, stabilizing training when gradient estimates are noisy (early batches are unrepresentative). Cosine decay then decreases $\eta$ following $\eta_t = \eta_{\min} + \frac{1}{2}(\eta_{\max} - \eta_{\min})(1 + \cos(\frac{\pi t}{T}))$, providing smooth annealing without abrupt drops. This combination (warmup + cosine) is standard for transformer training. The warmup duration is typically 1-10% of total steps.

**In Practice:** In the MTP thesis, LLaMA-3.1-8B fine-tuning uses 100 warmup steps followed by cosine decay over 3 epochs, with peak learning rate $3 \times 10^{-4}$ for the LoRA adapters — higher than typical full fine-tuning rates because only the low-rank matrices are being optimized.

---

### Q48: What is the transformer's position-wise feed-forward network and why is it important?

**Answer:** Each transformer layer contains a feed-forward network (FFN) applied identically to each position: $\text{FFN}(x) = W_2 \cdot \text{activation}(W_1 x + b_1) + b_2$, where $W_1 \in \mathbb{R}^{d_{\text{model}} \times d_{ff}}$ projects to a higher dimension ($d_{ff} = 4d_{\text{model}}$ typically) and $W_2$ projects back. The FFN acts as a learned key-value memory — $W_1$ rows detect patterns (keys), and $W_2$ columns store associated information (values). Modern variants use gated architectures: SwiGLU computes $\text{FFN}(x) = W_2 \cdot (\text{SiLU}(W_1 x) \odot W_3 x)$, improving performance by allowing multiplicative interactions.

**In Practice:** LLaMA-3.1-8B uses SwiGLU FFN with $d_{ff} = 14336$ (3.5$\times$ the model dimension of 4096), where the MTP thesis's LoRA adapters are applied not only to attention but also to the FFN projections to capture task-specific knowledge patterns.

---

### Q49: How does gradient clipping work and when is it necessary?

**Answer:** Gradient clipping rescales the gradient when its norm exceeds a threshold $c$: $g \leftarrow g \cdot \frac{c}{\max(c, \|g\|)}$. This preserves gradient direction while bounding magnitude, preventing catastrophic parameter updates from exploding gradients. Clip-by-norm is preferred over clip-by-value (which distorts gradient direction). It is essential when training RNNs on long sequences, during early training when loss landscapes are steep, or with large batch sizes. Typical thresholds are $c = 1.0$ for transformers. Without clipping, a single bad batch can destabilize training irreversibly.

**In Practice:** During LoRA fine-tuning of LLaMA-3.1-8B in the MTP thesis, gradient clipping at max_norm=1.0 prevents the occasional high-loss trace distillation example from corrupting the adapter weights, ensuring stable convergence.

---

### Q50: Explain knowledge distillation and how it relates to model compression.

**Answer:** Knowledge distillation trains a smaller student model $S$ to mimic a larger teacher model $T$ by matching soft probability distributions: $L_{\text{KD}} = \alpha \cdot KL(p_T^\tau \| p_S^\tau) + (1-\alpha) \cdot L_{\text{CE}}(y, p_S)$, where $p^\tau = \text{softmax}(z/\tau)$ uses temperature $\tau > 1$ to soften distributions and expose inter-class relationships (dark knowledge). The teacher's soft targets contain more information than hard labels — they encode which wrong answers are "almost right." Beyond logit matching, feature distillation aligns intermediate representations, and trace distillation teaches the student to reproduce the teacher's reasoning process step-by-step.

**In Practice:** The MTP thesis performs trace distillation — the student (LLaMA-3.1-8B with LoRA rank-16) learns to generate the teacher's (larger model) chain-of-thought reasoning traces, transferring not just answers but the reasoning process itself.

---

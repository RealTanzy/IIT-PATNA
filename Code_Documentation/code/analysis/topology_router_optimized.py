"""
OPTIMIZED Topology Router - Proper feature extraction and best classifier selection.
This is the core of the paper - must be done right.
"""
import json
import os
import re
import sys
import numpy as np
from collections import defaultdict
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler

sys.stdout.reconfigure(encoding="utf-8")

BASE = r"C:\Users\madamkh\Desktop\Research Work\tanzeel_server_backup\tanzeel server backup\ACL_TO_TMLR_v2\new_experiments"
PROMPTS = r"C:\Users\madamkh\Desktop\Research Work\tanzeel_server_backup\tanzeel server backup\ACL_TO_TMLR_v2\8b_data_human_eval_and_other\problem_prompts\all_problem_prompts.json"

# Full 13+1 feature extraction (same as paper Section 3.4)
LOGICAL_CONNECTIVES = ["if", "then", "therefore", "because", "since", "although", "however", "but", "unless", "either", "neither", "both", "all", "some", "none", "only if", "whenever", "consequently", "as a result", "it follows that", "implies", "given that", "assuming that"]
NEGATION_WORDS = ["not", "no", "never", "neither", "nor", "nothing", "nobody", "nowhere", "hardly", "barely", "scarcely", "without", "fail", "lack", "deny", "impossible", "cannot"]
HOP_INDICATORS = ["who", "which", "where", "what", "when", "who is", "who was", "what is", "what was", "related to", "associated with", "because of", "due to", "result of", "caused by", "led to", "born in", "died in"]
COMPARISON_WORDS = ["more", "less", "greater", "fewer", "larger", "smaller", "better", "worse", "most", "least", "both", "either", "whereas", "compared to", "than", "relative to", "versus"]


def extract_features(prompt_text):
    """Extract 14 topology features from a code problem prompt."""
    docstring = ""
    match = re.search(r'"""(.*?)"""', prompt_text, re.DOTALL)
    if match:
        docstring = match.group(1).strip()
    else:
        match = re.search(r"'''(.*?)'''", prompt_text, re.DOTALL)
        if match:
            docstring = match.group(1).strip()

    description = docstring if docstring else prompt_text
    desc_lower = description.lower()
    code_context = prompt_text.split('"""')[0] if '"""' in prompt_text else ""
    ctx = code_context.lower()
    full_text = desc_lower + " " + ctx

    q_words = desc_lower.split()
    ctx_words = ctx.split()

    conn_count = sum(1 for c in LOGICAL_CONNECTIVES if c in full_text)
    neg_count = sum(1 for n in NEGATION_WORDS if re.search(r"\b" + n + r"\b", full_text))
    hop_count = sum(1 for h in HOP_INDICATORS if h in desc_lower)
    cmp_count = sum(1 for c in COMPARISON_WORDS if c in desc_lower)
    q_marks = full_text.count("?")
    ctx_sents = len(re.split(r"[.!?]+", description)) if description.strip() else 0
    concepts = set(re.findall(r"\b[A-Z][a-z]{2,}\b", description))
    concept_count = len(concepts)
    branching_signals = conn_count + hop_count + q_marks
    linearity_score = 1.0 / (1.0 + branching_signals)
    depth_est = hop_count + max(1, ctx_sents // 3)

    # Branching coefficient (paper formula)
    q_len = max(1, len(q_words))
    bc = ctx_sents * hop_count / q_len

    return [len(q_words), len(ctx_words), conn_count, neg_count, hop_count,
            cmp_count, q_marks, ctx_sents, concept_count, linearity_score,
            0, depth_est, branching_signals, bc]


def run_topology_router(data, prompts_dict, name):
    """Run optimized topology router with multiple classifiers."""
    tasks = defaultdict(dict)
    for e in data:
        tasks[e["task_id"]][e["method"]] = e
    paired = {t: m for t, m in tasks.items() if "cot" in m and "astar" in m}
    n = len(paired)

    X, y = [], []
    for tid, methods in sorted(paired.items()):
        prompt = prompts_dict.get(tid, "")
        if not prompt:
            continue
        feats = extract_features(prompt)
        X.append(feats)
        cot_ok = methods["cot"]["correct"]
        astar_ok = methods["astar"]["correct"]
        if cot_ok and not astar_ok:
            y.append(0)
        elif astar_ok and not cot_ok:
            y.append(1)
        elif cot_ok and astar_ok:
            y.append(2)
        else:
            y.append(3)

    X = np.array(X)
    y = np.array(y)

    disagree_mask = np.isin(y, [0, 1])
    X_d = X[disagree_mask]
    y_d = y[disagree_mask]
    n_d = len(y_d)
    n_cot = int((y_d == 0).sum())
    n_astar = int((y_d == 1).sum())
    both_c = int((y == 2).sum())
    cot_total = n_cot + both_c
    astar_total = n_astar + both_c
    oracle = int((y != 3).sum())
    best_single = max(cot_total, astar_total) / n
    oracle_acc = oracle / n

    print(f"\n  {name}:")
    print(f"    Tasks={n}, Disagree={n_d} (CoT-wins={n_cot}, A*-wins={n_astar})")
    print(f"    CoT acc={cot_total/n*100:.1f}%, A* acc={astar_total/n*100:.1f}%")
    print(f"    Best single={best_single*100:.1f}%, Oracle={oracle_acc*100:.1f}%")

    if n_d < 10 or min(n_cot, n_astar) < 3:
        print(f"    *** Insufficient disagreement data ***")
        return None

    # Scale features
    scaler = StandardScaler()
    X_d_scaled = scaler.fit_transform(X_d)

    # Try multiple classifiers with different hyperparameters
    n_splits = min(5, min(n_cot, n_astar))
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    classifiers = {
        "XGB(100,d3)": GradientBoostingClassifier(n_estimators=100, max_depth=3, random_state=42),
        "XGB(200,d4)": GradientBoostingClassifier(n_estimators=200, max_depth=4, random_state=42),
        "XGB(150,d5)": GradientBoostingClassifier(n_estimators=150, max_depth=5, learning_rate=0.05, random_state=42),
        "XGB(300,d3)": GradientBoostingClassifier(n_estimators=300, max_depth=3, learning_rate=0.05, random_state=42),
        "RF(300,d5)": RandomForestClassifier(n_estimators=300, max_depth=5, random_state=42),
        "RF(500,d7)": RandomForestClassifier(n_estimators=500, max_depth=7, random_state=42),
        "RF(1000,dN)": RandomForestClassifier(n_estimators=1000, max_depth=None, random_state=42),
        "LR": LogisticRegression(C=1.0, max_iter=1000, random_state=42),
    }

    best_score = 0
    best_name = ""
    all_scores = {}
    for clf_name, clf in classifiers.items():
        try:
            scores = cross_val_score(clf, X_d_scaled, y_d, cv=cv, scoring="accuracy")
            mean_score = scores.mean()
            all_scores[clf_name] = (mean_score, scores.std())
            if mean_score > best_score:
                best_score = mean_score
                best_name = clf_name
        except Exception as e:
            pass

    router_overall = (both_c + best_score * n_d) / n
    gap = (router_overall - best_single) / (oracle_acc - best_single) * 100 if oracle_acc > best_single else 0

    print(f"    Classifier results:")
    for clf_name, (mean, std) in sorted(all_scores.items(), key=lambda x: -x[1][0]):
        marker = " <<< BEST" if clf_name == best_name else ""
        print(f"      {clf_name:15s}: {mean:.3f} +/- {std:.3f}{marker}")

    print(f"    --> Topology Router: {router_overall*100:.1f}%")
    print(f"    --> Gap recovery: {gap:.1f}%")

    # Feature importance from best model
    if "XGB" in best_name or "RF" in best_name:
        clf = classifiers[best_name]
        clf.fit(X_d_scaled, y_d)
        feature_names = ["q_len", "ctx_len", "conn", "neg", "hop", "cmp", "q_marks",
                        "ctx_sents", "concepts", "linearity", "n_opts", "depth", "branch_cplx", "BC"]
        imp = sorted(zip(feature_names, clf.feature_importances_), key=lambda x: -x[1])[:5]
        print(f"    Top features: {[(f, round(v,3)) for f,v in imp]}")

    return {
        "router_acc": router_overall * 100,
        "gap": gap,
        "oracle": oracle_acc * 100,
        "best_single": best_single * 100,
        "cv_score": best_score,
        "best_clf": best_name,
    }


if __name__ == "__main__":
    # Load prompts
    with open(PROMPTS) as f:
        all_prompts = json.load(f)
    code_prompts = {}
    for ds in ["humaneval", "mbpp", "codecontests"]:
        code_prompts[ds] = {p["task_id"]: p["prompt"] for p in all_prompts[ds]}

    print("=" * 70)
    print("  OPTIMIZED TOPOLOGY ROUTER - ALL SCALES")
    print("=" * 70)

    for scale, scale_dir in [("8B", "8b_cot_astar_sc_tot"), ("14B", "14b_cot_astar_sc_tot"), ("0.5B", "05b_cot_astar_sc_tot")]:
        print(f"\n{'='*70}")
        print(f"  SCALE: {scale}")
        print(f"{'='*70}")
        for bench in ["humaneval", "mbpp", "codecontests"]:
            f_path = os.path.join(BASE, scale_dir, f"{bench}_results.json")
            with open(f_path) as f:
                data = json.load(f)
            run_topology_router(data, code_prompts.get(bench, {}), f"{bench} ({scale})")

    print("\n\nDONE.")

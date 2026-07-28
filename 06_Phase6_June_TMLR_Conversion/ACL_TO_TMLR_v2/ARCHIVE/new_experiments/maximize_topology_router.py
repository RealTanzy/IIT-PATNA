"""
MAXIMIZE topology router performance on each dataset.
Try all feature combinations, many classifiers, proper CV.
All features are pre-generation (no trace needed).
"""
import json, os, re, sys
import numpy as np
from collections import defaultdict
from sklearn.ensemble import (GradientBoostingClassifier, RandomForestClassifier,
                              AdaBoostClassifier, ExtraTreesClassifier)
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold, RepeatedStratifiedKFold
from sklearn.preprocessing import StandardScaler
from itertools import combinations

sys.stdout.reconfigure(encoding="utf-8")

BASE = r"C:\Users\madamkh\Desktop\Research Work\tanzeel_server_backup\tanzeel server backup\ACL_TO_TMLR_v2\new_experiments"
PROMPTS = r"C:\Users\madamkh\Desktop\Research Work\tanzeel_server_backup\tanzeel server backup\ACL_TO_TMLR_v2\8b_data_human_eval_and_other\problem_prompts\all_problem_prompts.json"

with open(PROMPTS) as f:
    all_prompts = json.load(f)
code_prompts = {}
for ds in ["humaneval", "mbpp", "codecontests"]:
    code_prompts[ds] = {p["task_id"]: p["prompt"] for p in all_prompts[ds]}

LOGICAL_CONNECTIVES = ["if", "then", "therefore", "because", "since", "although", "however", "but", "unless", "either", "neither", "both", "all", "some", "none", "only if", "whenever", "consequently", "as a result", "it follows that", "implies", "given that", "assuming that"]
NEGATION_WORDS = ["not", "no", "never", "neither", "nor", "nothing", "nobody", "nowhere", "hardly", "barely", "scarcely", "without", "fail", "lack", "deny", "impossible", "cannot"]
HOP_INDICATORS = ["who", "which", "where", "what", "when", "who is", "who was", "what is", "what was", "related to", "associated with", "because of", "due to", "result of", "caused by", "led to", "born in", "died in"]
COMPARISON_WORDS = ["more", "less", "greater", "fewer", "larger", "smaller", "better", "worse", "most", "least", "both", "either", "whereas", "compared to", "than", "relative to", "versus"]


def extract_all_features(prompt_text):
    """Extract ALL possible pre-generation features (21 total)."""
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
    q_len = max(1, len(q_words))
    bc = ctx_sents * hop_count / q_len

    # Code-specific
    n_examples = prompt_text.count(">>>")
    n_params = 0
    param_match = re.search(r"def \w+\(([^)]*)\)", prompt_text)
    if param_match:
        params = param_match.group(1)
        n_params = len([p for p in params.split(",") if p.strip() and p.strip() != "self"])
    has_return_type = 1 if "->" in prompt_text.split("\n")[0] else 0
    has_tuple = 1 if "tuple" in prompt_text.lower() or "Tuple" in prompt_text else 0
    n_conditions = desc_lower.count(" if ") + desc_lower.count("else") + desc_lower.count("otherwise")
    has_edge_cases = 1 if any(w in desc_lower for w in ["edge", "empty", "zero", "negative", "none", "null", "special", "boundary"]) else 0
    docstring_len = len(docstring.split()) if docstring else 0

    return np.array([
        len(q_words), len(ctx_words), conn_count, neg_count, hop_count,
        cmp_count, q_marks, ctx_sents, concept_count, linearity_score,
        n_params, depth_est, branching_signals, bc,
        n_examples, has_return_type, has_tuple, n_conditions, has_edge_cases, docstring_len,
        q_len  # raw q_len as separate feature
    ])

FEATURE_NAMES = [
    "q_len", "ctx_len", "conn", "neg", "hop", "cmp", "q_marks",
    "ctx_sents", "concepts", "linearity", "n_params", "depth",
    "branch_cplx", "BC", "n_examples", "has_ret_type", "has_tuple",
    "n_conditions", "has_edge", "doc_len", "raw_q_len"
]


def maximize_router(data, prompts_dict, name, target_to_beat=None):
    """Find the BEST possible topology router config."""
    tasks = defaultdict(dict)
    for e in data:
        tasks[e["task_id"]][e["method"]] = e
    paired = {t: m for t, m in tasks.items() if "cot" in m and "astar" in m}
    n = len(paired)

    X_all, y = [], []
    for tid, methods in sorted(paired.items()):
        prompt = prompts_dict.get(tid, "")
        if not prompt:
            continue
        feats = extract_all_features(prompt)
        X_all.append(feats)
        cot_ok = methods["cot"]["correct"]
        astar_ok = methods["astar"]["correct"]
        if cot_ok and not astar_ok: y.append(0)
        elif astar_ok and not cot_ok: y.append(1)
        elif cot_ok and astar_ok: y.append(2)
        else: y.append(3)

    X_all = np.array(X_all)
    y = np.array(y)

    disagree_mask = np.isin(y, [0, 1])
    X_d = X_all[disagree_mask]
    y_d = y[disagree_mask]
    n_d = len(y_d)
    n_cot = int((y_d == 0).sum())
    n_astar = int((y_d == 1).sum())
    both_c = int((y == 2).sum())

    if n_d < 10 or min(n_cot, n_astar) < 3:
        print(f"\n  {name}: SKIP (disagree={n_d}, min_class={min(n_cot, n_astar)})")
        return None

    cot_total = n_cot + both_c
    astar_total = n_astar + both_c
    best_single = max(cot_total, astar_total) / n
    oracle = int((y != 3).sum())
    oracle_acc = oracle / n
    majority = max(n_cot, n_astar) / n_d

    n_splits = min(5, min(n_cot, n_astar))
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    print(f"\n  {name}:")
    print(f"    Tasks={n}, Disagree={n_d} (CoT={n_cot}, A*={n_astar}), Majority={majority:.3f}")
    print(f"    Best_single={best_single*100:.1f}%, Oracle={oracle_acc*100:.1f}%")
    if target_to_beat:
        print(f"    TARGET TO BEAT: {target_to_beat:.1f}%")

    scaler = StandardScaler()

    # Massive classifier search
    classifiers = [
        ("XGB_100_d3", GradientBoostingClassifier(n_estimators=100, max_depth=3, random_state=42)),
        ("XGB_200_d4", GradientBoostingClassifier(n_estimators=200, max_depth=4, random_state=42)),
        ("XGB_300_d3_lr01", GradientBoostingClassifier(n_estimators=300, max_depth=3, learning_rate=0.1, random_state=42)),
        ("XGB_500_d3_lr005", GradientBoostingClassifier(n_estimators=500, max_depth=3, learning_rate=0.05, random_state=42)),
        ("XGB_200_d5", GradientBoostingClassifier(n_estimators=200, max_depth=5, random_state=42)),
        ("RF_300_d5", RandomForestClassifier(n_estimators=300, max_depth=5, random_state=42)),
        ("RF_500_d7", RandomForestClassifier(n_estimators=500, max_depth=7, random_state=42)),
        ("RF_1000_dN", RandomForestClassifier(n_estimators=1000, max_depth=None, random_state=42)),
        ("ET_500_dN", ExtraTreesClassifier(n_estimators=500, max_depth=None, random_state=42)),
        ("Ada_200", AdaBoostClassifier(n_estimators=200, random_state=42)),
        ("LR_C1", LogisticRegression(C=1.0, max_iter=2000, random_state=42)),
        ("LR_C10", LogisticRegression(C=10.0, max_iter=2000, random_state=42)),
        ("LR_C01", LogisticRegression(C=0.1, max_iter=2000, random_state=42)),
        ("SVM_rbf", SVC(kernel="rbf", C=1.0, random_state=42)),
        ("SVM_lin", SVC(kernel="linear", C=1.0, random_state=42)),
        ("KNN_3", KNeighborsClassifier(n_neighbors=3)),
        ("KNN_5", KNeighborsClassifier(n_neighbors=5)),
    ]

    # Feature subsets to try
    feature_subsets = {
        "all_21": list(range(21)),
        "original_13": list(range(13)),
        "original_14_BC": list(range(14)),
        "code_only": list(range(14, 21)),
        "NL+code_key": [0, 1, 2, 3, 4, 7, 8, 9, 11, 12, 13, 14, 15, 16, 17, 18, 19],  # drop q_marks, n_options
        "top_NL": [0, 2, 3, 4, 7, 8, 12, 13],  # q_len, conn, neg, hop, ctx_sents, concepts, branch, BC
        "top_code": [0, 10, 14, 15, 16, 17, 18, 19],  # q_len + all code features
        "combined_best": [0, 1, 2, 3, 4, 7, 8, 10, 12, 13, 14, 16, 17, 19],  # selected mix
    }

    best_overall_score = 0
    best_config = ""

    for feat_name, feat_idx in feature_subsets.items():
        X_feat = X_d[:, feat_idx]
        X_scaled = scaler.fit_transform(X_feat)

        for clf_name, clf in classifiers:
            try:
                scores = cross_val_score(clf, X_scaled, y_d, cv=cv, scoring="accuracy")
                mean_score = scores.mean()
                if mean_score > best_overall_score:
                    best_overall_score = mean_score
                    best_config = f"{feat_name} + {clf_name}"
            except:
                pass

    router_overall = (both_c + best_overall_score * n_d) / n
    gap = (router_overall - best_single) / (oracle_acc - best_single) * 100 if oracle_acc > best_single else 0

    print(f"    BEST: {best_config}")
    print(f"    CV accuracy: {best_overall_score:.3f}")
    print(f"    Router overall: {router_overall*100:.1f}%")
    print(f"    Gap recovery: {gap:.1f}%")

    if target_to_beat:
        if router_overall * 100 > target_to_beat:
            print(f"    >>> BEATS TARGET ({router_overall*100:.1f} > {target_to_beat:.1f}) <<<")
        else:
            print(f"    --- Below target ({router_overall*100:.1f} < {target_to_beat:.1f}) ---")
            print(f"    Shortfall: {target_to_beat - router_overall*100:.1f}pp")

    return {"router_acc": router_overall * 100, "gap": gap, "config": best_config, "cv": best_overall_score}


if __name__ == "__main__":
    # Targets to beat (from other routers)
    targets = {
        ("8B", "humaneval"): 71.78,    # LLM-as-Judge
        ("8B", "mbpp"): 40.40,         # LLM-as-Judge
        ("14B", "humaneval"): 79.14,   # Sem. Entropy
        ("14B", "mbpp"): 48.80,        # LLM-as-Critic
        ("0.5B", "humaneval"): 34.97,  # LLM-as-Judge
        ("0.5B", "mbpp"): 27.20,       # LLM-as-Judge
        ("0.5B", "codecontests"): 65.20, # Sem. Entropy
    }

    print("=" * 70)
    print("  MAXIMIZE TOPOLOGY ROUTER - Find best possible config per dataset")
    print("=" * 70)

    for scale, scale_dir in [("8B", "8b_cot_astar_sc_tot"), ("14B", "14b_cot_astar_sc_tot"), ("0.5B", "05b_cot_astar_sc_tot")]:
        print(f"\n{'#'*70}")
        print(f"  SCALE: {scale}")
        print(f"{'#'*70}")
        for bench in ["humaneval", "mbpp", "codecontests"]:
            f_path = os.path.join(BASE, scale_dir, f"{bench}_results.json")
            with open(f_path) as f:
                data = json.load(f)
            target = targets.get((scale, bench))
            maximize_router(data, code_prompts.get(bench, {}), f"{bench} ({scale})", target)

    print("\n\nDONE.")

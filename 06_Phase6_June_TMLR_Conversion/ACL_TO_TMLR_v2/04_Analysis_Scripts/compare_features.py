"""Compare original 13 features vs enhanced features for topology router."""
import json, os, re, sys
import numpy as np
from collections import defaultdict
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler

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


def extract_features_enhanced(prompt_text):
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

    # Code-specific features
    n_examples = prompt_text.count(">>>")
    n_params = 0
    param_match = re.search(r"def \w+\(([^)]*)\)", prompt_text)
    if param_match:
        params = param_match.group(1)
        n_params = len([p for p in params.split(",") if p.strip() and p.strip() != "self"])

    has_return_type = 1 if "->" in prompt_text.split("\n")[0] else 0
    has_tuple = 1 if "tuple" in prompt_text.lower() or "Tuple" in prompt_text else 0
    n_conditions = desc_lower.count(" if ") + desc_lower.count("else") + desc_lower.count("otherwise") + desc_lower.count("except")
    has_edge_cases = 1 if any(w in desc_lower for w in ["edge", "empty", "zero", "negative", "none", "null", "special"]) else 0
    docstring_len = len(docstring.split()) if docstring else 0

    return [
        # Original 13 + BC (indices 0-13)
        len(q_words), len(ctx_words), conn_count, neg_count, hop_count,
        cmp_count, q_marks, ctx_sents, concept_count, linearity_score,
        0, depth_est, branching_signals, bc,
        # Code-specific (indices 14-20)
        n_examples, n_params, has_return_type, has_tuple, n_conditions, has_edge_cases, docstring_len
    ]


def run_comparison(data, prompts_dict, name):
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
        feats = extract_features_enhanced(prompt)
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
        print(f"  {name}: insufficient data (disagree={n_d}, cot={n_cot}, astar={n_astar})")
        return

    n_splits = min(5, min(n_cot, n_astar))
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    cot_total = n_cot + both_c
    astar_total = n_astar + both_c
    best_single = max(cot_total, astar_total) / n
    oracle = int((y != 3).sum())
    oracle_acc = oracle / n

    def compute_gap(cv_score):
        router_overall = (both_c + cv_score * n_d) / n
        return (router_overall - best_single) / (oracle_acc - best_single) * 100 if oracle_acc > best_single else 0

    scaler = StandardScaler()

    # Test different feature sets
    feature_sets = {
        "Original 13": X_d[:, :13],
        "Original 13 + BC": X_d[:, :14],
        "Enhanced 21 (NL+Code)": X_d,
        "Code-only 7": X_d[:, 14:],
    }

    print(f"\n  {name}: disagree={n_d} (CoT={n_cot}, A*={n_astar}), majority={max(n_cot,n_astar)/n_d:.3f}")

    best_gap = -999
    best_set = ""
    for feat_name, X_feat in feature_sets.items():
        X_scaled = scaler.fit_transform(X_feat)
        # Try XGBoost
        clf = GradientBoostingClassifier(n_estimators=200, max_depth=4, random_state=42)
        scores = cross_val_score(clf, X_scaled, y_d, cv=cv, scoring="accuracy")
        gap = compute_gap(scores.mean())
        marker = ""
        if gap > best_gap:
            best_gap = gap
            best_set = feat_name
            marker = " <-- BEST"
        print(f"    {feat_name:25s}: CV={scores.mean():.3f}+/-{scores.std():.3f}, Gap={gap:+.1f}%{marker}")

    print(f"    ==> Best feature set: {best_set} (Gap={best_gap:+.1f}%)")


print("=" * 70)
print("  TOPOLOGY ROUTER: FEATURE SET COMPARISON")
print("  Which features give the BEST topology router?")
print("=" * 70)

for scale, scale_dir in [("8B", "8b_cot_astar_sc_tot"), ("14B", "14b_cot_astar_sc_tot"), ("0.5B", "05b_cot_astar_sc_tot")]:
    print(f"\n{'='*50} {scale} {'='*50}")
    for bench in ["humaneval", "mbpp", "codecontests"]:
        f_path = os.path.join(BASE, scale_dir, f"{bench}_results.json")
        with open(f_path) as f:
            data = json.load(f)
        run_comparison(data, code_prompts.get(bench, {}), f"{bench} ({scale})")

print("\nDONE.")

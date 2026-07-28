"""Run topology router analysis on 14B results using prompts from all_problem_prompts.json"""
import json
import re
import numpy as np
from collections import defaultdict
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

LOGICAL_CONNECTIVES = ["if", "then", "therefore", "because", "since", "although", "however", "but", "unless", "either", "neither", "both", "all", "some", "none", "only if", "whenever", "consequently", "as a result", "it follows that", "implies", "given that", "assuming that"]
NEGATION_WORDS = ["not", "no", "never", "neither", "nor", "nothing", "nobody", "nowhere", "hardly", "barely", "scarcely", "without", "fail", "lack", "deny", "impossible", "cannot"]
HOP_INDICATORS = ["who", "which", "where", "what", "when", "who is", "who was", "what is", "what was", "related to", "associated with", "because of", "due to", "result of", "caused by", "led to", "born in", "died in"]
COMPARISON_WORDS = ["more", "less", "greater", "fewer", "larger", "smaller", "better", "worse", "most", "least", "both", "either", "whereas", "compared to", "than", "relative to", "versus"]
FEATURE_NAMES = ["q_len", "ctx_len", "conn_count", "neg_count", "hop_count", "cmp_count", "q_marks", "ctx_sents", "concept_count", "linearity_score", "n_options", "depth_est", "branching_complexity"]

def extract_features(prompt_text):
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
    branching_signals = conn_count + hop_count + q_marks
    linearity_score = 1.0 / (1.0 + branching_signals)
    depth_est = hop_count + max(1, ctx_sents // 3)
    return [len(q_words), len(ctx_words), conn_count, neg_count, hop_count, cmp_count, q_marks, ctx_sents, len(concepts), linearity_score, 0, depth_est, branching_signals]


def run_analysis(results_data, prompts_dict, name):
    tasks = defaultdict(dict)
    for entry in results_data:
        tid = entry["task_id"]
        method = entry["method"]
        if method not in tasks[tid]:
            tasks[tid][method] = entry
    paired = {tid: m for tid, m in tasks.items() if "cot" in m and "astar" in m}
    n = len(paired)
    if n == 0:
        print(f"  {name}: No paired tasks!")
        return

    cot_c = sum(1 for m in paired.values() if m["cot"]["correct"])
    astar_c = sum(1 for m in paired.values() if m["astar"]["correct"])
    both_c = sum(1 for m in paired.values() if m["cot"]["correct"] and m["astar"]["correct"])
    cot_only = sum(1 for m in paired.values() if m["cot"]["correct"] and not m["astar"]["correct"])
    astar_only = sum(1 for m in paired.values() if not m["cot"]["correct"] and m["astar"]["correct"])
    both_w = sum(1 for m in paired.values() if not m["cot"]["correct"] and not m["astar"]["correct"])
    oracle = both_c + cot_only + astar_only

    X, y = [], []
    for tid, methods in sorted(paired.items()):
        prompt = prompts_dict.get(tid, {}).get("prompt", "")
        if not prompt:
            continue
        feats = extract_features(prompt)
        X.append(feats)
        cot_ok = methods["cot"]["correct"]
        astar_ok = methods["astar"]["correct"]
        if cot_ok and not astar_ok: y.append(0)
        elif astar_ok and not cot_ok: y.append(1)
        elif cot_ok and astar_ok: y.append(2)
        else: y.append(3)

    X = np.array(X)
    y = np.array(y)

    disagree_mask = np.isin(y, [0, 1])
    X_d = X[disagree_mask]
    y_d = y[disagree_mask]
    n_d = len(y_d)
    n_cot_w = int((y_d == 0).sum())
    n_astar_w = int((y_d == 1).sum())

    print(f"\n{'='*70}")
    print(f"  {name}")
    print(f"{'='*70}")
    print(f"  Tasks: {n}")
    print(f"  CoT: {cot_c/n*100:.1f}%  |  A*: {astar_c/n*100:.1f}%  |  Oracle: {oracle/n*100:.1f}%")
    print(f"  Both-correct: {both_c} ({both_c/n*100:.1f}%)")
    print(f"  CoT-only: {cot_only} ({cot_only/n*100:.1f}%)")
    print(f"  A*-only: {astar_only} ({astar_only/n*100:.1f}%)")
    print(f"  Both-wrong: {both_w} ({both_w/n*100:.1f}%)")
    print(f"  Disagreements: {n_d} (CoT-wins={n_cot_w}, A*-wins={n_astar_w})")
    print(f"  Majority baseline: {max(n_cot_w, n_astar_w)/max(1,n_d)*100:.1f}%")

    if n_d >= 15 and min(n_cot_w, n_astar_w) >= 5:
        n_splits = min(5, min(n_cot_w, n_astar_w))
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        xgb = GradientBoostingClassifier(n_estimators=100, max_depth=3, random_state=42)
        xgb_scores = cross_val_score(xgb, X_d, y_d, cv=cv, scoring="accuracy")
        rf = RandomForestClassifier(n_estimators=300, max_depth=5, random_state=42)
        rf_scores = cross_val_score(rf, X_d, y_d, cv=cv, scoring="accuracy")
        best = max(xgb_scores.mean(), rf_scores.mean())
        best_single = max(cot_c, astar_c) / n
        oracle_acc = oracle / n
        router_overall = (both_c + best * n_d) / n
        gap = (router_overall - best_single) / (oracle_acc - best_single) * 100 if oracle_acc > best_single else 0

        print(f"\n  ROUTER RESULTS:")
        print(f"    XGBoost CV: {xgb_scores.mean():.3f} +/- {xgb_scores.std():.3f}")
        print(f"    RF CV:      {rf_scores.mean():.3f} +/- {rf_scores.std():.3f}")
        print(f"    Best single method: {best_single*100:.1f}%")
        print(f"    Topology Router:    {router_overall*100:.1f}%")
        print(f"    Oracle:             {oracle_acc*100:.1f}%")
        print(f"    Gap recovered:      {gap:.1f}%")

        # Clustering
        kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
        clusters = kmeans.fit_predict(StandardScaler().fit_transform(X))
        print(f"\n  TOPOLOGY CLUSTERS:")
        for c in range(4):
            mask = clusters == c
            cs = mask.sum()
            if cs == 0: continue
            cy = y[mask]
            c_cot = sum(1 for yy in cy if yy in [0, 2]) / cs * 100
            c_astar = sum(1 for yy in cy if yy in [1, 2]) / cs * 100
            c_bc = X[mask, FEATURE_NAMES.index("branching_complexity")].mean()
            print(f"    C{c+1}: n={cs:3d}, CoT={c_cot:.1f}%, A*={c_astar:.1f}%, "
                  f"adv={c_astar-c_cot:+.1f}pp, branch_cplx={c_bc:.1f}")

        # Feature importance
        xgb.fit(X_d, y_d)
        imp = sorted(zip(FEATURE_NAMES, xgb.feature_importances_), key=lambda x: -x[1])[:5]
        print(f"\n  TOP FEATURES:")
        for fname, importance in imp:
            print(f"    {fname:22s}: {importance:.3f}")
    else:
        print(f"\n  *** Insufficient data for router (need >=5 each side) ***")
        if n_d > 0:
            print(f"  (Have {n_cot_w} CoT-wins, {n_astar_w} A*-wins)")


if __name__ == "__main__":
    prompts_base = r"C:\Users\madamkh\Desktop\Research Work\tanzeel_server_backup\tanzeel server backup\ACL_TO_TMLR_v2\8b_data_human_eval_and_other\problem_prompts"
    with open(prompts_base + r"\all_problem_prompts.json") as f:
        all_prompts = json.load(f)

    he_prompts = {p["task_id"]: p for p in all_prompts["humaneval"]}
    mbpp_prompts = {p["task_id"]: p for p in all_prompts["mbpp"]}
    cc_prompts = {p["task_id"]: p for p in all_prompts["codecontests"]}

    base_8b = r"C:\Users\madamkh\Desktop\Research Work\tanzeel_server_backup\tanzeel server backup\ACL_TO_TMLR_v2\TMLR_LARGE_RESULTS_8B"
    base_14b = r"C:\Users\madamkh\Desktop\Research Work\tanzeel_server_backup\tanzeel server backup\ACL_TO_TMLR_v2\TMLR_LARGE_RESULTS_14B"

    print("\n" + "#" * 70)
    print("#  8B RESULTS (LLaMA-3.1-8B)")
    print("#" * 70)

    with open(base_8b + r"\humaneval_results.json") as f:
        run_analysis(json.load(f), he_prompts, "HumanEval 8B")
    with open(base_8b + r"\mbpp_results.json") as f:
        run_analysis(json.load(f), mbpp_prompts, "MBPP 8B")
    with open(base_8b + r"\codecontests_results.json") as f:
        run_analysis(json.load(f), cc_prompts, "CodeContests 8B")

    print("\n" + "#" * 70)
    print("#  14B RESULTS (Qwen-2.5-14B)")
    print("#" * 70)

    with open(base_14b + r"\humaneval_results.json") as f:
        run_analysis(json.load(f), he_prompts, "HumanEval 14B")
    with open(base_14b + r"\mbpp_results.json") as f:
        run_analysis(json.load(f), mbpp_prompts, "MBPP 14B")
    with open(base_14b + r"\codecontests_results.json") as f:
        run_analysis(json.load(f), cc_prompts, "CodeContests 14B")

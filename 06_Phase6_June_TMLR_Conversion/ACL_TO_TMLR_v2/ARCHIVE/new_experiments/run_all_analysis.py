"""
Comprehensive analysis of ALL new_experiments data.
Computes topology router + summary for paper integration.
"""
import json
import os
import re
import sys
import numpy as np
from collections import defaultdict
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold

sys.stdout.reconfigure(encoding='utf-8')

BASE = r"C:\Users\madamkh\Desktop\Research Work\tanzeel_server_backup\tanzeel server backup\ACL_TO_TMLR_v2\new_experiments"
PROMPTS = r"C:\Users\madamkh\Desktop\Research Work\tanzeel_server_backup\tanzeel server backup\ACL_TO_TMLR_v2\8b_data_human_eval_and_other\problem_prompts\all_problem_prompts.json"

# Feature extraction
LOGICAL_CONNECTIVES = ["if", "then", "therefore", "because", "since", "although", "however", "but", "unless", "either", "neither", "both", "all", "some", "none", "only if", "whenever", "consequently", "as a result", "implies", "given that", "assuming that"]
NEGATION_WORDS = ["not", "no", "never", "neither", "nor", "nothing", "nobody", "nowhere", "hardly", "barely", "scarcely", "without", "fail", "lack", "deny", "impossible", "cannot"]
HOP_INDICATORS = ["who", "which", "where", "what", "when", "who is", "who was", "what is", "what was", "related to", "associated with", "because of", "due to", "result of", "caused by", "led to"]
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

# Load prompts
with open(PROMPTS) as f:
    all_prompts = json.load(f)
prompts_map = {}
for dataset in ["humaneval", "mbpp", "codecontests"]:
    prompts_map[dataset] = {p["task_id"]: p["prompt"] for p in all_prompts[dataset]}

def get_acc(data, method=None):
    if method:
        data = [e for e in data if e.get("method") == method]
    tasks = {}
    for e in data:
        if e["task_id"] not in tasks:
            tasks[e["task_id"]] = e
    n = len(tasks)
    c = sum(1 for e in tasks.values() if e.get("correct", False))
    return c, n

def compute_topology_router(data, prompts_dict):
    """Compute topology router accuracy on code benchmarks."""
    tasks = defaultdict(dict)
    for e in data:
        tasks[e["task_id"]][e["method"]] = e
    paired = {t: m for t, m in tasks.items() if "cot" in m and "astar" in m}
    n = len(paired)
    if n == 0:
        return None

    X, y = [], []
    for tid, methods in sorted(paired.items()):
        prompt = prompts_dict.get(tid, "")
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
    n_cot = int((y_d == 0).sum())
    n_astar = int((y_d == 1).sum())
    both_c = int((y == 2).sum())

    if n_d >= 15 and min(n_cot, n_astar) >= 5:
        n_splits = min(5, min(n_cot, n_astar))
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        xgb = GradientBoostingClassifier(n_estimators=100, max_depth=3, random_state=42)
        scores = cross_val_score(xgb, X_d, y_d, cv=cv, scoring="accuracy")
        best_router = scores.mean()
        router_overall = (both_c + best_router * n_d) / n
        cot_total = n_cot + both_c
        astar_total = n_astar + both_c
        oracle = int((y != 3).sum())
        best_single = max(cot_total, astar_total) / n
        oracle_acc = oracle / n
        gap = (router_overall - best_single) / (oracle_acc - best_single) * 100 if oracle_acc > best_single else 0
        return {"router_acc": router_overall * 100, "gap": gap, "oracle": oracle_acc * 100}
    return None

# ================================================================
# MAIN ANALYSIS
# ================================================================
print("=" * 80)
print("  COMPREHENSIVE ANALYSIS — ALL new_experiments DATA")
print("=" * 80)

# 1. CODE BENCHMARKS ACCURACY
print("\n\n1. MAIN TABLE DATA (Code Benchmarks)")
print("-" * 60)
for scale, scale_dir in [("8B", "8b_cot_astar_sc_tot"), ("0.5B", "05b_cot_astar_sc_tot"), ("14B", "14b_cot_astar_sc_tot")]:
    print(f"\n  {scale}:")
    for bench in ["humaneval", "mbpp", "codecontests"]:
        main_f = os.path.join(BASE, scale_dir, f"{bench}_results.json")
        sc_f = os.path.join(BASE, scale_dir, f"{bench}_sc_results.json")
        tot_f = os.path.join(BASE, scale_dir, f"{bench}_tot_results.json")
        with open(main_f) as f: data = json.load(f)
        with open(sc_f) as f: sc_data = json.load(f)
        with open(tot_f) as f: tot_data = json.load(f)
        cot_c, cot_n = get_acc(data, "cot")
        ast_c, ast_n = get_acc(data, "astar")
        sc_c, sc_n = get_acc(sc_data)
        tot_c, tot_n = get_acc(tot_data)
        print(f"    {bench:14s}: CoT={cot_c/cot_n*100:5.1f}  A*={ast_c/ast_n*100:5.1f}  SC={sc_c/sc_n*100:5.1f}  ToT={tot_c/tot_n*100:5.1f}")

# 2. MATH BENCHMARKS (new 14B and 0.5B)
print("\n\n2. MATH BENCHMARKS (new data)")
print("-" * 60)
for scale, scale_dir in [("14B", "14b_math_cot_astar"), ("0.5B", "05b_math_cot_astar")]:
    print(f"\n  {scale}:")
    for bench in ["gsm8k", "math"]:
        f_path = os.path.join(BASE, scale_dir, f"{bench}_results.json")
        with open(f_path) as f: data = json.load(f)
        cot_c, cot_n = get_acc(data, "cot")
        ast_c, ast_n = get_acc(data, "astar")
        print(f"    {bench:14s}: CoT={cot_c/cot_n*100:5.1f}  A*={ast_c/ast_n*100:5.1f}  (n={cot_n})")

# 3. TOPOLOGY ROUTER ON CODE (all scales)
print("\n\n3. TOPOLOGY ROUTER (Code Benchmarks)")
print("-" * 60)
for scale, scale_dir in [("8B", "8b_cot_astar_sc_tot"), ("0.5B", "05b_cot_astar_sc_tot"), ("14B", "14b_cot_astar_sc_tot")]:
    print(f"\n  {scale}:")
    for bench in ["humaneval", "mbpp", "codecontests"]:
        main_f = os.path.join(BASE, scale_dir, f"{bench}_results.json")
        with open(main_f) as f: data = json.load(f)
        result = compute_topology_router(data, prompts_map.get(bench, {}))
        if result:
            print(f"    {bench:14s}: Router={result['router_acc']:.1f}%, Oracle={result['oracle']:.1f}%, Gap={result['gap']:.1f}%")
        else:
            print(f"    {bench:14s}: Insufficient data for topology router")

# 4. ALL ROUTER RESULTS SUMMARY
print("\n\n4. ROUTER TABLES (from router JSON files)")
print("-" * 60)
for scale, rdir in [("8B", "routers_8b"), ("14B", "routers_14b"), ("0.5B", "routers_05b")]:
    rpath = os.path.join(BASE, rdir)
    if not os.path.isdir(rpath):
        continue
    print(f"\n  {scale}:")
    # Group by dataset
    datasets = defaultdict(dict)
    for fname in sorted(os.listdir(rpath)):
        if fname.endswith('.json'):
            parts = fname.replace('.json', '').rsplit('_', 1)
            if len(parts) == 2:
                ds, router = parts[0], parts[1]
            else:
                # Handle multi-word router names
                for r in ['semantic_entropy', 'llm_critic', 'llm_judge', 'random_forest']:
                    if fname.endswith(f'_{r}.json'):
                        ds = fname.replace(f'_{r}.json', '')
                        router = r
                        break
            with open(os.path.join(rpath, fname)) as f:
                data = json.load(f)
            datasets[ds][router] = data['summary']['router_accuracy']

    for ds in sorted(datasets.keys()):
        routers = datasets[ds]
        parts = [f"{r[:6]}={v:5.1f}" for r, v in sorted(routers.items())]
        print(f"    {ds:14s}: {', '.join(parts)}")

print("\n\nDONE.")

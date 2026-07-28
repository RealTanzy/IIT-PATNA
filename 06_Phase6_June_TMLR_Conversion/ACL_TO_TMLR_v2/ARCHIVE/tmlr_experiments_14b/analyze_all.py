"""Complete 14B analysis: router + multi-return + category breakdown"""
import json
import re
import numpy as np
from collections import defaultdict
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# Load prompts
prompts_base = r"C:\Users\madamkh\Desktop\Research Work\tanzeel_server_backup\tanzeel server backup\ACL_TO_TMLR_v2\8b_data_human_eval_and_other\problem_prompts"
with open(prompts_base + r"\all_problem_prompts.json") as f:
    all_prompts = json.load(f)
he_prompts = {p["task_id"]: p for p in all_prompts["humaneval"]}
mbpp_prompts = {p["task_id"]: p for p in all_prompts["mbpp"]}

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

# Load 14B data
base = r"C:\Users\madamkh\Desktop\Research Work\tanzeel_server_backup\tanzeel server backup\ACL_TO_TMLR_v2\tmlr_experiments_14b"
with open(base + r"\humaneval_results.json") as f: he = json.load(f)
with open(base + r"\mbpp_results.json") as f: mbpp = json.load(f)

# ═══════════════════════════════════════════════════════════════
# ROUTER ANALYSIS
# ═══════════════════════════════════════════════════════════════
def run_router(data, prompts_dict, name):
    tasks = defaultdict(dict)
    for e in data:
        tid = e["task_id"]
        method = e["method"]
        if method not in tasks[tid]:
            tasks[tid][method] = e
    paired = {tid: m for tid, m in tasks.items() if "cot" in m and "astar" in m}
    n = len(paired)

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
    n_cot = int((y_d == 0).sum())
    n_astar = int((y_d == 1).sum())
    both_c = int((y == 2).sum())

    print(f"\n{'='*70}")
    print(f"  {name} TOPOLOGY ROUTER")
    print(f"{'='*70}")
    print(f"  Disagreements: {n_d} (CoT-wins={n_cot}, A*-wins={n_astar})")

    if n_d >= 15 and min(n_cot, n_astar) >= 5:
        n_splits = min(5, min(n_cot, n_astar))
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        xgb = GradientBoostingClassifier(n_estimators=100, max_depth=3, random_state=42)
        xgb_scores = cross_val_score(xgb, X_d, y_d, cv=cv, scoring="accuracy")
        rf = RandomForestClassifier(n_estimators=300, max_depth=5, random_state=42)
        rf_scores = cross_val_score(rf, X_d, y_d, cv=cv, scoring="accuracy")
        best_router = max(xgb_scores.mean(), rf_scores.mean())

        cot_total = int((y == 0).sum()) + both_c
        astar_total = int((y == 1).sum()) + both_c
        oracle = int((y != 3).sum())
        best_single = max(cot_total, astar_total) / n
        oracle_acc = oracle / n
        router_overall = (both_c + best_router * n_d) / n
        gap = (router_overall - best_single) / (oracle_acc - best_single) * 100 if oracle_acc > best_single else 0

        print(f"  XGBoost CV: {xgb_scores.mean():.3f} +/- {xgb_scores.std():.3f}")
        print(f"  RF CV: {rf_scores.mean():.3f} +/- {rf_scores.std():.3f}")
        print(f"  Best single: {best_single*100:.1f}%")
        print(f"  Router overall: {router_overall*100:.1f}%")
        print(f"  Oracle: {oracle_acc*100:.1f}%")
        print(f"  Gap recovered: {gap:.1f}%")

        # Clustering
        kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
        clusters = kmeans.fit_predict(StandardScaler().fit_transform(X))
        print(f"\n  Topology Clusters:")
        for c in range(4):
            mask = clusters == c
            cs = mask.sum()
            if cs < 3: continue
            cy = y[mask]
            c_cot = sum(1 for yy in cy if yy in [0, 2]) / cs * 100
            c_astar = sum(1 for yy in cy if yy in [1, 2]) / cs * 100
            c_bc = X[mask, FEATURE_NAMES.index("branching_complexity")].mean()
            print(f"    C{c+1}: n={cs:3d}, CoT={c_cot:.1f}%, A*={c_astar:.1f}%, "
                  f"adv={c_astar-c_cot:+.1f}pp, branch_cplx={c_bc:.1f}")

        # Feature importance
        xgb.fit(X_d, y_d)
        imp = sorted(zip(FEATURE_NAMES, xgb.feature_importances_), key=lambda x: -x[1])[:5]
        print(f"\n  Top-5 features:")
        for fname, importance in imp:
            print(f"    {fname:22s}: {importance:.3f}")

        return router_overall * 100, gap
    else:
        print(f"  *** Insufficient data ***")
        return None, None

run_router(he, he_prompts, "HumanEval 14B")
run_router(mbpp, mbpp_prompts, "MBPP 14B")

# ═══════════════════════════════════════════════════════════════
# MULTI-RETURN / COMPLEX OUTPUT ANALYSIS
# ═══════════════════════════════════════════════════════════════
print(f"\n\n{'='*70}")
print(f"  MULTI-RETURN ANALYSIS (HumanEval 14B)")
print(f"{'='*70}")

tasks = defaultdict(dict)
for e in he:
    tid = e["task_id"]
    method = e["method"]
    if method not in tasks[tid]:
        tasks[tid][method] = e
paired = {tid: m for tid, m in tasks.items() if "cot" in m and "astar" in m}

multi_cot, multi_astar, multi_n = 0, 0, 0
single_cot, single_astar, single_n = 0, 0, 0

for tid, methods in paired.items():
    prompt = he_prompts.get(tid, {}).get("prompt", "")
    # Detect multi-return: Tuple in return type, or multiple output conditions
    has_multi = "tuple" in prompt.lower()
    if not has_multi and "->" in prompt:
        return_part = prompt.split("->")[1].split(":")[0] if ":" in prompt.split("->")[1] else prompt.split("->")[1]
        has_multi = "," in return_part.split("\n")[0]

    cot_ok = methods["cot"]["correct"]
    astar_ok = methods["astar"]["correct"]
    if has_multi:
        multi_n += 1
        if cot_ok: multi_cot += 1
        if astar_ok: multi_astar += 1
    else:
        single_n += 1
        if cot_ok: single_cot += 1
        if astar_ok: single_astar += 1

print(f"  Multi-return tasks (n={multi_n}):")
print(f"    CoT: {multi_cot}/{multi_n} = {multi_cot/max(1,multi_n)*100:.1f}%")
print(f"    A*:  {multi_astar}/{multi_n} = {multi_astar/max(1,multi_n)*100:.1f}%")
print(f"    A* advantage: {(multi_astar-multi_cot)/max(1,multi_n)*100:+.1f}pp")
print(f"  Single-return tasks (n={single_n}):")
print(f"    CoT: {single_cot}/{single_n} = {single_cot/max(1,single_n)*100:.1f}%")
print(f"    A*:  {single_astar}/{single_n} = {single_astar/max(1,single_n)*100:.1f}%")
print(f"    A* advantage: {(single_astar-single_cot)/max(1,single_n)*100:+.1f}pp")

# ═══════════════════════════════════════════════════════════════
# CATEGORY ANALYSIS (MBPP)
# ═══════════════════════════════════════════════════════════════
print(f"\n\n{'='*70}")
print(f"  MBPP 14B: PERFORMANCE BY PROBLEM CATEGORY")
print(f"{'='*70}")

math_kw = ["number", "prime", "factorial", "sum", "product", "count", "square", "power", "digit", "fibonacci", "even", "odd", "divisible", "multiple"]
string_kw = ["string", "character", "substring", "uppercase", "lowercase", "camel", "snake", "reverse", "palindrome", "vowel", "word"]
list_kw = ["list", "array", "tuple", "sort", "element", "index", "flatten", "merge", "remove", "filter", "duplicate"]
logic_kw = ["check", "whether", "validate", "verify", "determine"]

def categorize(prompt):
    p = prompt.lower()
    cats = []
    if any(k in p for k in math_kw): cats.append("math")
    if any(k in p for k in string_kw): cats.append("string")
    if any(k in p for k in list_kw): cats.append("list")
    if any(k in p for k in logic_kw): cats.append("logic")
    return cats if cats else ["other"]

tasks_m = defaultdict(dict)
for e in mbpp:
    tid = e["task_id"]
    method = e["method"]
    if method not in tasks_m[tid]:
        tasks_m[tid][method] = e
paired_m = {tid: m for tid, m in tasks_m.items() if "cot" in m and "astar" in m}

categories = defaultdict(lambda: {"cot_c": 0, "astar_c": 0, "total": 0, "astar_only": 0, "cot_only": 0})
for tid, methods in paired_m.items():
    prompt = mbpp_prompts.get(tid, {}).get("prompt", "")
    cats = categorize(prompt)
    cot_ok = methods["cot"]["correct"]
    astar_ok = methods["astar"]["correct"]
    for cat in cats:
        categories[cat]["total"] += 1
        if cot_ok: categories[cat]["cot_c"] += 1
        if astar_ok: categories[cat]["astar_c"] += 1
        if not cot_ok and astar_ok: categories[cat]["astar_only"] += 1
        if cot_ok and not astar_ok: categories[cat]["cot_only"] += 1

print(f"{'Category':<12} {'N':<6} {'CoT%':<7} {'A*%':<7} {'A*-adv':<8} {'A*-only':<8} {'CoT-only':<9}")
print("-" * 60)
for cat in sorted(categories.keys(), key=lambda c: -categories[c]["total"]):
    d = categories[cat]
    n = d["total"]
    cot_pct = d["cot_c"] / n * 100
    astar_pct = d["astar_c"] / n * 100
    print(f"{cat:<12} {n:<6} {cot_pct:<7.1f} {astar_pct:<7.1f} {astar_pct-cot_pct:<+8.1f} {d['astar_only']:<8} {d['cot_only']:<9}")

# ═══════════════════════════════════════════════════════════════
# FINAL PAPER-READY SUMMARY
# ═══════════════════════════════════════════════════════════════
print(f"\n\n{'='*70}")
print(f"  PAPER-READY SUMMARY TABLE (Qwen-2.5-14B)")
print(f"{'='*70}")
print(f"{'Dataset':<14} {'CoT':<7} {'A*':<7} {'SC':<7} {'ToT':<7}")
print("-" * 45)

# Load SC and ToT for summary
with open(base + r"\humaneval_sc_results.json") as f: he_sc = json.load(f)
with open(base + r"\humaneval_tot_results.json") as f: he_tot = json.load(f)
with open(base + r"\mbpp_sc_results.json") as f: mbpp_sc = json.load(f)
with open(base + r"\mbpp_tot_results.json") as f: mbpp_tot = json.load(f)

he_sc_acc = sum(1 for e in he_sc if e["correct"]) / len(he_sc) * 100
he_tot_acc = sum(1 for e in he_tot if e["correct"]) / len(he_tot) * 100
mbpp_sc_acc = sum(1 for e in mbpp_sc if e["correct"]) / len(mbpp_sc) * 100
mbpp_tot_acc = sum(1 for e in mbpp_tot if e["correct"]) / len(mbpp_tot) * 100

# CoT and A* from paired
he_cot_acc = sum(1 for m in paired.values() if m["cot"]["correct"]) / len(paired) * 100
he_astar_acc = sum(1 for m in paired.values() if m["astar"]["correct"]) / len(paired) * 100
mbpp_cot_acc = sum(1 for m in paired_m.values() if m["cot"]["correct"]) / len(paired_m) * 100
mbpp_astar_acc = sum(1 for m in paired_m.values() if m["astar"]["correct"]) / len(paired_m) * 100

print(f"{'HumanEval':<14} {he_cot_acc:<7.2f} {he_astar_acc:<7.2f} {he_sc_acc:<7.2f} {he_tot_acc:<7.2f}")
print(f"{'MBPP':<14} {mbpp_cot_acc:<7.2f} {mbpp_astar_acc:<7.2f} {mbpp_sc_acc:<7.2f} {mbpp_tot_acc:<7.2f}")

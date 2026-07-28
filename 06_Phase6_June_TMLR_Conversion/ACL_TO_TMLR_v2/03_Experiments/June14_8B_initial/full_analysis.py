"""
Complete 8B Code Generation Analysis
- All 4 methods accuracy
- Complementarity decomposition
- Topology router
- Multi-return analysis
- Category breakdown
- Qualitative examples for appendix
"""
import json
import re
import os
import numpy as np
from collections import defaultdict, Counter
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# ═══════════════════════════════════════════════════════════════
# PATHS
# ═══════════════════════════════════════════════════════════════
BASE_8B = r"C:\Users\madamkh\Desktop\FAST-PHASE-3\First_tool\tmlr_experiments_8b"
BASE_14B = r"C:\Users\madamkh\Desktop\Research Work\tanzeel_server_backup\tanzeel server backup\ACL_TO_TMLR_v2\tmlr_experiments_14b"
PROMPTS = r"C:\Users\madamkh\Desktop\Research Work\tanzeel_server_backup\tanzeel server backup\ACL_TO_TMLR_v2\8b_data_human_eval_and_other\problem_prompts"

# ═══════════════════════════════════════════════════════════════
# FEATURE EXTRACTION
# ═══════════════════════════════════════════════════════════════
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

# ═══════════════════════════════════════════════════════════════
# LOAD DATA
# ═══════════════════════════════════════════════════════════════
with open(os.path.join(PROMPTS, "all_problem_prompts.json")) as f:
    all_prompts = json.load(f)
he_prompts = {p["task_id"]: p for p in all_prompts["humaneval"]}
mbpp_prompts = {p["task_id"]: p for p in all_prompts["mbpp"]}

def load_json(path):
    with open(path) as f:
        return json.load(f)

# 8B data
he_8b = load_json(os.path.join(BASE_8B, "humaneval_results.json"))
mbpp_8b = load_json(os.path.join(BASE_8B, "mbpp_results.json"))
cc_8b = load_json(os.path.join(BASE_8B, "codecontests_results.json"))
he_8b_sc = load_json(os.path.join(BASE_8B, "humaneval_sc_results.json"))
he_8b_tot = load_json(os.path.join(BASE_8B, "humaneval_tot_results.json"))
mbpp_8b_sc = load_json(os.path.join(BASE_8B, "mbpp_sc_results.json"))
mbpp_8b_tot = load_json(os.path.join(BASE_8B, "mbpp_tot_results.json"))

# 14B data
he_14b = load_json(os.path.join(BASE_14B, "humaneval_results.json"))
mbpp_14b = load_json(os.path.join(BASE_14B, "mbpp_results.json"))
he_14b_sc = load_json(os.path.join(BASE_14B, "humaneval_sc_results.json"))
he_14b_tot = load_json(os.path.join(BASE_14B, "humaneval_tot_results.json"))
mbpp_14b_sc = load_json(os.path.join(BASE_14B, "mbpp_sc_results.json"))
mbpp_14b_tot = load_json(os.path.join(BASE_14B, "mbpp_tot_results.json"))

# ═══════════════════════════════════════════════════════════════
# SECTION 1: CONSOLIDATED ACCURACY TABLE
# ═══════════════════════════════════════════════════════════════
def get_acc(data, method_filter=None):
    if method_filter:
        data = [e for e in data if e.get("method") == method_filter]
    tasks = {}
    for e in data:
        if e["task_id"] not in tasks:
            tasks[e["task_id"]] = e
    n = len(tasks)
    c = sum(1 for e in tasks.values() if e.get("correct", False))
    return c, n

print("#" * 80)
print("#  SECTION 1: CONSOLIDATED ACCURACY TABLE")
print("#" * 80)

# 8B
he8_cot_c, he8_cot_n = get_acc(he_8b, "cot")
he8_ast_c, he8_ast_n = get_acc(he_8b, "astar")
he8_sc_c, he8_sc_n = get_acc(he_8b_sc)
he8_tot_c, he8_tot_n = get_acc(he_8b_tot)

mb8_cot_c, mb8_cot_n = get_acc(mbpp_8b, "cot")
mb8_ast_c, mb8_ast_n = get_acc(mbpp_8b, "astar")
mb8_sc_c, mb8_sc_n = get_acc(mbpp_8b_sc)
mb8_tot_c, mb8_tot_n = get_acc(mbpp_8b_tot)

# 14B
he14_cot_c, he14_cot_n = get_acc(he_14b, "cot")
he14_ast_c, he14_ast_n = get_acc(he_14b, "astar")
he14_sc_c, he14_sc_n = get_acc(he_14b_sc)
he14_tot_c, he14_tot_n = get_acc(he_14b_tot)

mb14_cot_c, mb14_cot_n = get_acc(mbpp_14b, "cot")
mb14_ast_c, mb14_ast_n = get_acc(mbpp_14b, "astar")
mb14_sc_c, mb14_sc_n = get_acc(mbpp_14b_sc)
mb14_tot_c, mb14_tot_n = get_acc(mbpp_14b_tot)

print(f"\n{'Dataset':<12} | {'--- LLaMA-3.1-8B ---':^30} | {'--- Qwen-2.5-14B ---':^30}")
print(f"{'':12} | {'CoT':>6} {'A*':>6} {'SC':>6} {'ToT':>6} | {'CoT':>6} {'A*':>6} {'SC':>6} {'ToT':>6}")
print("-" * 75)
print(f"{'HumanEval':<12} | {he8_cot_c/he8_cot_n*100:6.2f} {he8_ast_c/he8_ast_n*100:6.2f} {he8_sc_c/he8_sc_n*100:6.2f} {he8_tot_c/he8_tot_n*100:6.2f} | {he14_cot_c/he14_cot_n*100:6.2f} {he14_ast_c/he14_ast_n*100:6.2f} {he14_sc_c/he14_sc_n*100:6.2f} {he14_tot_c/he14_tot_n*100:6.2f}")
print(f"{'MBPP':<12} | {mb8_cot_c/mb8_cot_n*100:6.2f} {mb8_ast_c/mb8_ast_n*100:6.2f} {mb8_sc_c/mb8_sc_n*100:6.2f} {mb8_tot_c/mb8_tot_n*100:6.2f} | {mb14_cot_c/mb14_cot_n*100:6.2f} {mb14_ast_c/mb14_ast_n*100:6.2f} {mb14_sc_c/mb14_sc_n*100:6.2f} {mb14_tot_c/mb14_tot_n*100:6.2f}")

# ═══════════════════════════════════════════════════════════════
# SECTION 2: COMPLEMENTARITY (8B and 14B)
# ═══════════════════════════════════════════════════════════════
print(f"\n\n{'#'*80}")
print("#  SECTION 2: COMPLEMENTARITY DECOMPOSITION")
print("#" * 80)

def complementarity(data, name):
    tasks = defaultdict(dict)
    for e in data:
        tid = e["task_id"]
        method = e["method"]
        if method not in tasks[tid]:
            tasks[tid][method] = e
    paired = {tid: m for tid, m in tasks.items() if "cot" in m and "astar" in m}
    n = len(paired)
    both_c = sum(1 for m in paired.values() if m["cot"]["correct"] and m["astar"]["correct"])
    cot_only = sum(1 for m in paired.values() if m["cot"]["correct"] and not m["astar"]["correct"])
    astar_only = sum(1 for m in paired.values() if not m["cot"]["correct"] and m["astar"]["correct"])
    both_w = sum(1 for m in paired.values() if not m["cot"]["correct"] and not m["astar"]["correct"])
    oracle = both_c + cot_only + astar_only
    print(f"\n  {name}: n={n}")
    print(f"    Both correct: {both_c} ({both_c/n*100:.1f}%)")
    print(f"    CoT only:     {cot_only} ({cot_only/n*100:.1f}%)")
    print(f"    A* only:      {astar_only} ({astar_only/n*100:.1f}%)")
    print(f"    Both wrong:   {both_w} ({both_w/n*100:.1f}%)")
    print(f"    Oracle:       {oracle}/{n} = {oracle/n*100:.1f}%")
    return paired, cot_only, astar_only, both_c, both_w, oracle

he8_paired, *he8_comp = complementarity(he_8b, "HumanEval 8B")
mb8_paired, *mb8_comp = complementarity(mbpp_8b, "MBPP 8B")
he14_paired, *he14_comp = complementarity(he_14b, "HumanEval 14B")
mb14_paired, *mb14_comp = complementarity(mbpp_14b, "MBPP 14B")

# ═══════════════════════════════════════════════════════════════
# SECTION 3: TOPOLOGY ROUTER
# ═══════════════════════════════════════════════════════════════
print(f"\n\n{'#'*80}")
print("#  SECTION 3: TOPOLOGY ROUTER")
print("#" * 80)

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
        if not prompt: continue
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

    print(f"\n  {name}:")
    print(f"    Disagreements: {n_d} (CoT-wins={n_cot}, A*-wins={n_astar})")

    if n_d >= 15 and min(n_cot, n_astar) >= 5:
        n_splits = min(5, min(n_cot, n_astar))
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        xgb = GradientBoostingClassifier(n_estimators=100, max_depth=3, random_state=42)
        xgb_scores = cross_val_score(xgb, X_d, y_d, cv=cv, scoring="accuracy")
        rf = RandomForestClassifier(n_estimators=300, max_depth=5, random_state=42)
        rf_scores = cross_val_score(rf, X_d, y_d, cv=cv, scoring="accuracy")
        best_router = max(xgb_scores.mean(), rf_scores.mean())

        cot_total = n_cot + both_c
        astar_total = n_astar + both_c
        oracle = int((y != 3).sum())
        best_single = max(cot_total, astar_total) / n
        oracle_acc = oracle / n
        router_overall = (both_c + best_router * n_d) / n
        gap = (router_overall - best_single) / (oracle_acc - best_single) * 100 if oracle_acc > best_single else 0

        print(f"    XGBoost CV: {xgb_scores.mean():.3f} +/- {xgb_scores.std():.3f}")
        print(f"    RF CV: {rf_scores.mean():.3f} +/- {rf_scores.std():.3f}")
        print(f"    Best single: {best_single*100:.1f}%")
        print(f"    Topology Router: {router_overall*100:.1f}%")
        print(f"    Oracle: {oracle_acc*100:.1f}%")
        print(f"    Gap recovered: {gap:.1f}%")

        # Clustering
        kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
        clusters = kmeans.fit_predict(StandardScaler().fit_transform(X))
        print(f"    Clusters:")
        for c in range(4):
            mask = clusters == c
            cs = mask.sum()
            if cs < 3: continue
            cy = y[mask]
            c_cot = sum(1 for yy in cy if yy in [0, 2]) / cs * 100
            c_astar = sum(1 for yy in cy if yy in [1, 2]) / cs * 100
            c_bc = X[mask, FEATURE_NAMES.index("branching_complexity")].mean()
            print(f"      C{c+1}: n={cs:3d}, CoT={c_cot:.1f}%, A*={c_astar:.1f}%, adv={c_astar-c_cot:+.1f}pp, bc={c_bc:.1f}")

        # Feature importance
        xgb.fit(X_d, y_d)
        imp = sorted(zip(FEATURE_NAMES, xgb.feature_importances_), key=lambda x: -x[1])[:5]
        print(f"    Top features: {[(f, round(i,3)) for f,i in imp]}")
        return router_overall * 100, gap
    else:
        print(f"    *** Insufficient minority class for router ***")
        return None, None

run_router(he_8b, he_prompts, "HumanEval 8B")
run_router(mbpp_8b, mbpp_prompts, "MBPP 8B")
run_router(he_14b, he_prompts, "HumanEval 14B")
run_router(mbpp_14b, mbpp_prompts, "MBPP 14B")

# ═══════════════════════════════════════════════════════════════
# SECTION 4: MULTI-RETURN ANALYSIS
# ═══════════════════════════════════════════════════════════════
print(f"\n\n{'#'*80}")
print("#  SECTION 4: MULTI-RETURN ANALYSIS")
print("#" * 80)

def multi_return_analysis(data, prompts_dict, name):
    tasks = defaultdict(dict)
    for e in data:
        tid = e["task_id"]
        method = e["method"]
        if method not in tasks[tid]:
            tasks[tid][method] = e
    paired = {tid: m for tid, m in tasks.items() if "cot" in m and "astar" in m}

    multi_cot, multi_astar, multi_n = 0, 0, 0
    single_cot, single_astar, single_n = 0, 0, 0

    for tid, methods in paired.items():
        prompt = prompts_dict.get(tid, {}).get("prompt", "")
        has_multi = "tuple" in prompt.lower()
        if not has_multi and "->" in prompt:
            ret_part = prompt.split("->")[1].split("\n")[0]
            has_multi = "," in ret_part

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

    print(f"\n  {name}:")
    print(f"    Multi-return (n={multi_n}): CoT={multi_cot/max(1,multi_n)*100:.1f}%, A*={multi_astar/max(1,multi_n)*100:.1f}%, adv={((multi_astar-multi_cot)/max(1,multi_n))*100:+.1f}pp")
    print(f"    Single-return (n={single_n}): CoT={single_cot/max(1,single_n)*100:.1f}%, A*={single_astar/max(1,single_n)*100:.1f}%, adv={((single_astar-single_cot)/max(1,single_n))*100:+.1f}pp")

multi_return_analysis(he_8b, he_prompts, "HumanEval 8B")
multi_return_analysis(he_14b, he_prompts, "HumanEval 14B")

# ═══════════════════════════════════════════════════════════════
# SECTION 5: CATEGORY ANALYSIS (MBPP)
# ═══════════════════════════════════════════════════════════════
print(f"\n\n{'#'*80}")
print("#  SECTION 5: CATEGORY ANALYSIS (MBPP)")
print("#" * 80)

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

def category_analysis(data, prompts_dict, name):
    tasks = defaultdict(dict)
    for e in data:
        tid = e["task_id"]
        method = e["method"]
        if method not in tasks[tid]:
            tasks[tid][method] = e
    paired = {tid: m for tid, m in tasks.items() if "cot" in m and "astar" in m}

    categories = defaultdict(lambda: {"cot_c": 0, "astar_c": 0, "total": 0, "astar_only": 0, "cot_only": 0})
    for tid, methods in paired.items():
        prompt = prompts_dict.get(tid, {}).get("prompt", "")
        cats = categorize(prompt)
        cot_ok = methods["cot"]["correct"]
        astar_ok = methods["astar"]["correct"]
        for cat in cats:
            categories[cat]["total"] += 1
            if cot_ok: categories[cat]["cot_c"] += 1
            if astar_ok: categories[cat]["astar_c"] += 1
            if not cot_ok and astar_ok: categories[cat]["astar_only"] += 1
            if cot_ok and not astar_ok: categories[cat]["cot_only"] += 1

    print(f"\n  {name}:")
    print(f"  {'Category':<12} {'N':<6} {'CoT%':<7} {'A*%':<7} {'A*-adv':<8} {'A*-only':<8} {'CoT-only':<9}")
    print(f"  {'-'*58}")
    for cat in sorted(categories.keys(), key=lambda c: -categories[c]["total"]):
        d = categories[cat]
        n = d["total"]
        print(f"  {cat:<12} {n:<6} {d['cot_c']/n*100:<7.1f} {d['astar_c']/n*100:<7.1f} {(d['astar_c']-d['cot_c'])/n*100:<+8.1f} {d['astar_only']:<8} {d['cot_only']:<9}")

category_analysis(mbpp_8b, mbpp_prompts, "MBPP 8B")
category_analysis(mbpp_14b, mbpp_prompts, "MBPP 14B")

# ═══════════════════════════════════════════════════════════════
# SECTION 6: QUALITATIVE EXAMPLES FOR APPENDIX
# ═══════════════════════════════════════════════════════════════
print(f"\n\n{'#'*80}")
print("#  SECTION 6: QUALITATIVE EXAMPLES (A* wins vs CoT wins)")
print("#" * 80)

def show_examples(data, prompts_dict, name, n_examples=3):
    tasks = defaultdict(dict)
    for e in data:
        tid = e["task_id"]
        method = e["method"]
        if method not in tasks[tid]:
            tasks[tid][method] = e
    paired = {tid: m for tid, m in tasks.items() if "cot" in m and "astar" in m}

    astar_wins = [(tid, m) for tid, m in paired.items() if not m["cot"]["correct"] and m["astar"]["correct"]]
    cot_wins = [(tid, m) for tid, m in paired.items() if m["cot"]["correct"] and not m["astar"]["correct"]]

    print(f"\n  {name} - A* WINS ({len(astar_wins)} total, showing {min(n_examples, len(astar_wins))})")
    for tid, methods in astar_wins[:n_examples]:
        prompt = prompts_dict.get(tid, {}).get("prompt", "")
        # Get function name
        func_match = re.search(r"def (\w+)", prompt)
        func_name = func_match.group(1) if func_match else "unknown"
        # Get description
        doc_match = re.search(r'"""(.*?)(?:\n|""")', prompt, re.DOTALL)
        desc = doc_match.group(1).strip()[:80] if doc_match else prompt.split("\n")[0][:80]
        print(f"    {tid} ({func_name}): {desc}")
        # Show code snippets (first 3 lines of each)
        cot_code = methods["cot"]["code"].split("\n")[:3]
        astar_code = methods["astar"]["code"].split("\n")[:3]
        print(f"      CoT (WRONG): {cot_code[0][:60]}...")
        print(f"      A*  (RIGHT): {astar_code[0][:60]}...")
        print()

    print(f"  {name} - CoT WINS ({len(cot_wins)} total, showing {min(n_examples, len(cot_wins))})")
    for tid, methods in cot_wins[:n_examples]:
        prompt = prompts_dict.get(tid, {}).get("prompt", "")
        func_match = re.search(r"def (\w+)", prompt)
        func_name = func_match.group(1) if func_match else "unknown"
        doc_match = re.search(r'"""(.*?)(?:\n|""")', prompt, re.DOTALL)
        desc = doc_match.group(1).strip()[:80] if doc_match else prompt.split("\n")[0][:80]
        print(f"    {tid} ({func_name}): {desc}")
        print()

show_examples(he_8b, he_prompts, "HumanEval 8B")
show_examples(he_14b, he_prompts, "HumanEval 14B")

# ═══════════════════════════════════════════════════════════════
# SECTION 7: COMPUTE COST
# ═══════════════════════════════════════════════════════════════
print(f"\n\n{'#'*80}")
print("#  SECTION 7: COMPUTE COST (Timing)")
print("#" * 80)

def timing(data, sc_data, tot_data, name):
    cot_times = [e["time"] for e in data if e.get("method") == "cot"]
    astar_times = [e["time"] for e in data if e.get("method") == "astar"]
    sc_times = [e["time"] for e in sc_data]
    tot_times = [e["time"] for e in tot_data]

    print(f"\n  {name}:")
    print(f"    CoT: {np.mean(cot_times):.2f}s")
    print(f"    A*:  {np.mean(astar_times):.2f}s ({np.mean(astar_times)/np.mean(cot_times):.1f}x)")
    print(f"    SC:  {np.mean(sc_times):.2f}s ({np.mean(sc_times)/np.mean(cot_times):.1f}x)")
    print(f"    ToT: {np.mean(tot_times):.2f}s ({np.mean(tot_times)/np.mean(cot_times):.1f}x)")

timing(he_8b, he_8b_sc, he_8b_tot, "HumanEval 8B")
timing(mbpp_8b, mbpp_8b_sc, mbpp_8b_tot, "MBPP 8B")
timing(he_14b, he_14b_sc, he_14b_tot, "HumanEval 14B")
timing(mbpp_14b, mbpp_14b_sc, mbpp_14b_tot, "MBPP 14B")

print("\n\nDONE.")

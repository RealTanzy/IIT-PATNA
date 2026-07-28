"""
FINAL COMPLETE ANALYSIS: All 3 scales (0.5B, 8B, 14B) x 3 benchmarks x 4 methods
Produces paper-ready numbers for TMLR integration.
"""
import json
import re
import os
import numpy as np
from collections import defaultdict
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# ═══════════════════════════════════════════════════════════════
# PATHS
# ═══════════════════════════════════════════════════════════════
BASE_05B = r"C:\Users\madamkh\Desktop\Research Work\tanzeel_server_backup\tanzeel server backup\ACL_TO_TMLR_v2\tmlr_experiments_05b"
BASE_8B = r"C:\Users\madamkh\Desktop\FAST-PHASE-3\First_tool\tmlr_experiments_8b"
BASE_14B = r"C:\Users\madamkh\Desktop\Research Work\tanzeel_server_backup\tanzeel server backup\ACL_TO_TMLR_v2\tmlr_experiments_14b"
PROMPTS = r"C:\Users\madamkh\Desktop\Research Work\tanzeel_server_backup\tanzeel server backup\ACL_TO_TMLR_v2\8b_data_human_eval_and_other\problem_prompts"

# ═══════════════════════════════════════════════════════════════
# FEATURE EXTRACTION (13 topology features)
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

def load(path):
    with open(path) as f:
        return json.load(f)

def get_acc(data, method_filter=None):
    if method_filter:
        data = [e for e in data if e.get("method") == method_filter]
    tasks = {}
    for e in data:
        if e["task_id"] not in tasks:
            tasks[e["task_id"]] = e
    n = len(tasks)
    c = sum(1 for e in tasks.values() if e.get("correct", False))
    return c, n, c / n * 100 if n > 0 else 0

# ═══════════════════════════════════════════════════════════════
# SECTION 1: CONSOLIDATED TABLE (ALL SCALES)
# ═══════════════════════════════════════════════════════════════
print("=" * 90)
print("  SECTION 1: CONSOLIDATED ACCURACY TABLE (All Scales x All Methods)")
print("=" * 90)

results = {}
for scale, base in [("0.5B", BASE_05B), ("8B", BASE_8B), ("14B", BASE_14B)]:
    for bench in ["humaneval", "mbpp", "codecontests"]:
        main_file = os.path.join(base, f"{bench}_results.json")
        sc_file = os.path.join(base, f"{bench}_sc_results.json")
        tot_file = os.path.join(base, f"{bench}_tot_results.json")

        if os.path.exists(main_file):
            data = load(main_file)
            _, _, cot_acc = get_acc(data, "cot")
            _, _, astar_acc = get_acc(data, "astar")
        else:
            cot_acc, astar_acc = 0, 0

        if os.path.exists(sc_file):
            sc_data = load(sc_file)
            _, sc_n, sc_acc = get_acc(sc_data)
        else:
            sc_acc, sc_n = 0, 0

        if os.path.exists(tot_file):
            tot_data = load(tot_file)
            _, tot_n, tot_acc = get_acc(tot_data)
        else:
            tot_acc, tot_n = 0, 0

        results[(scale, bench)] = {
            "cot": cot_acc, "astar": astar_acc, "sc": sc_acc, "tot": tot_acc
        }

print(f"\n{'Dataset':<14} | {'Qwen-0.5B':^28} | {'LLaMA-3.1-8B':^28} | {'Qwen-2.5-14B':^28}")
print(f"{'':14} | {'CoT':>6}{'A*':>6}{'SC':>6}{'ToT':>7} | {'CoT':>6}{'A*':>6}{'SC':>6}{'ToT':>7} | {'CoT':>6}{'A*':>6}{'SC':>6}{'ToT':>7}")
print("-" * 105)
for bench in ["humaneval", "mbpp", "codecontests"]:
    row = f"{bench:<14} |"
    for scale in ["0.5B", "8B", "14B"]:
        r = results.get((scale, bench), {})
        row += f" {r.get('cot',0):5.1f} {r.get('astar',0):5.1f} {r.get('sc',0):5.1f} {r.get('tot',0):6.1f} |"
    print(row)

# ═══════════════════════════════════════════════════════════════
# SECTION 2: COMPLEMENTARITY (All scales, HumanEval + MBPP)
# ═══════════════════════════════════════════════════════════════
print(f"\n\n{'='*90}")
print("  SECTION 2: COMPLEMENTARITY DECOMPOSITION")
print("=" * 90)

def complementarity(data, name):
    tasks = defaultdict(dict)
    for e in data:
        tasks[e["task_id"]][e["method"]] = e
    paired = {tid: m for tid, m in tasks.items() if "cot" in m and "astar" in m}
    n = len(paired)
    if n == 0:
        print(f"  {name}: No paired data")
        return None
    both_c = sum(1 for m in paired.values() if m["cot"]["correct"] and m["astar"]["correct"])
    cot_only = sum(1 for m in paired.values() if m["cot"]["correct"] and not m["astar"]["correct"])
    astar_only = sum(1 for m in paired.values() if not m["cot"]["correct"] and m["astar"]["correct"])
    both_w = sum(1 for m in paired.values() if not m["cot"]["correct"] and not m["astar"]["correct"])
    oracle = both_c + cot_only + astar_only
    print(f"  {name:<20} n={n:3d} | Both={both_c/n*100:5.1f}% | CoT-only={cot_only/n*100:5.1f}% | A*-only={astar_only/n*100:5.1f}% | Both-wrong={both_w/n*100:5.1f}% | Oracle={oracle/n*100:.1f}%")
    return paired

for scale, base in [("0.5B", BASE_05B), ("8B", BASE_8B), ("14B", BASE_14B)]:
    print(f"\n  --- {scale} ---")
    for bench in ["humaneval", "mbpp", "codecontests"]:
        f = os.path.join(base, f"{bench}_results.json")
        if os.path.exists(f):
            complementarity(load(f), f"{bench} ({scale})")

# ═══════════════════════════════════════════════════════════════
# SECTION 3: TOPOLOGY ROUTER (HumanEval at all scales)
# ═══════════════════════════════════════════════════════════════
print(f"\n\n{'='*90}")
print("  SECTION 3: TOPOLOGY ROUTER")
print("=" * 90)

def run_router(data, prompts_dict, name):
    tasks = defaultdict(dict)
    for e in data:
        tasks[e["task_id"]][e["method"]] = e
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

    print(f"\n  {name}: disagree={n_d} (CoT-wins={n_cot}, A*-wins={n_astar})")

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

        print(f"    XGB={xgb_scores.mean():.3f}, RF={rf_scores.mean():.3f}")
        print(f"    Best single={best_single*100:.1f}%, Router={router_overall*100:.1f}%, Oracle={oracle_acc*100:.1f}%, Gap={gap:.1f}%")
        return router_overall * 100, gap
    elif n_d > 0:
        print(f"    Insufficient minority class (need >=5 each)")
        return None, None
    else:
        print(f"    No disagreements")
        return None, None

for scale, base in [("0.5B", BASE_05B), ("8B", BASE_8B), ("14B", BASE_14B)]:
    for bench, prompts in [("humaneval", he_prompts), ("mbpp", mbpp_prompts)]:
        f = os.path.join(base, f"{bench}_results.json")
        if os.path.exists(f):
            run_router(load(f), prompts, f"{bench} ({scale})")

# ═══════════════════════════════════════════════════════════════
# SECTION 4: MULTI-RETURN ANALYSIS (HumanEval, all scales)
# ═══════════════════════════════════════════════════════════════
print(f"\n\n{'='*90}")
print("  SECTION 4: MULTI-RETURN ANALYSIS (HumanEval)")
print("=" * 90)

def multi_return(data, prompts_dict, name):
    tasks = defaultdict(dict)
    for e in data:
        tasks[e["task_id"]][e["method"]] = e
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

    if multi_n > 0 and single_n > 0:
        print(f"  {name}:")
        print(f"    Multi-return (n={multi_n}): CoT={multi_cot/multi_n*100:.1f}%, A*={multi_astar/multi_n*100:.1f}%, adv={((multi_astar-multi_cot)/multi_n)*100:+.1f}pp")
        print(f"    Single-return (n={single_n}): CoT={single_cot/single_n*100:.1f}%, A*={single_astar/single_n*100:.1f}%, adv={((single_astar-single_cot)/single_n)*100:+.1f}pp")

for scale, base in [("0.5B", BASE_05B), ("8B", BASE_8B), ("14B", BASE_14B)]:
    f = os.path.join(base, "humaneval_results.json")
    if os.path.exists(f):
        multi_return(load(f), he_prompts, f"HumanEval ({scale})")

# ═══════════════════════════════════════════════════════════════
# SECTION 5: CATEGORY ANALYSIS (MBPP, all scales)
# ═══════════════════════════════════════════════════════════════
print(f"\n\n{'='*90}")
print("  SECTION 5: CATEGORY ANALYSIS (MBPP)")
print("=" * 90)

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
        tasks[e["task_id"]][e["method"]] = e
    paired = {tid: m for tid, m in tasks.items() if "cot" in m and "astar" in m}
    categories = defaultdict(lambda: {"cot": 0, "astar": 0, "n": 0})
    for tid, methods in paired.items():
        prompt = prompts_dict.get(tid, {}).get("prompt", "")
        cats = categorize(prompt)
        for cat in cats:
            categories[cat]["n"] += 1
            if methods["cot"]["correct"]: categories[cat]["cot"] += 1
            if methods["astar"]["correct"]: categories[cat]["astar"] += 1
    print(f"\n  {name}:")
    print(f"    {'Cat':<10} {'N':<5} {'CoT%':<7} {'A*%':<7} {'Adv':<8}")
    for cat in sorted(categories.keys(), key=lambda c: -categories[c]["n"]):
        d = categories[cat]
        n = d["n"]
        print(f"    {cat:<10} {n:<5} {d['cot']/n*100:<7.1f} {d['astar']/n*100:<7.1f} {(d['astar']-d['cot'])/n*100:<+8.1f}")

for scale, base in [("0.5B", BASE_05B), ("8B", BASE_8B), ("14B", BASE_14B)]:
    f = os.path.join(base, "mbpp_results.json")
    if os.path.exists(f):
        category_analysis(load(f), mbpp_prompts, f"MBPP ({scale})")

# ═══════════════════════════════════════════════════════════════
# SECTION 6: COMPUTE COST
# ═══════════════════════════════════════════════════════════════
print(f"\n\n{'='*90}")
print("  SECTION 6: COMPUTE COST")
print("=" * 90)

for scale, base in [("0.5B", BASE_05B), ("8B", BASE_8B), ("14B", BASE_14B)]:
    for bench in ["humaneval", "mbpp"]:
        main_f = os.path.join(base, f"{bench}_results.json")
        sc_f = os.path.join(base, f"{bench}_sc_results.json")
        tot_f = os.path.join(base, f"{bench}_tot_results.json")
        if all(os.path.exists(x) for x in [main_f, sc_f, tot_f]):
            main_data = load(main_f)
            cot_t = np.mean([e["time"] for e in main_data if e.get("method") == "cot"])
            ast_t = np.mean([e["time"] for e in main_data if e.get("method") == "astar"])
            sc_t = np.mean([e["time"] for e in load(sc_f)])
            tot_t = np.mean([e["time"] for e in load(tot_f)])
            print(f"  {bench} ({scale}): CoT={cot_t:.2f}s, A*={ast_t:.2f}s ({ast_t/cot_t:.1f}x), SC={sc_t:.2f}s ({sc_t/cot_t:.1f}x), ToT={tot_t:.2f}s ({tot_t/cot_t:.1f}x)")

print("\n\nANALYSIS COMPLETE.")

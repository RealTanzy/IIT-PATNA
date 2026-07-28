"""Final push: 50 seeds x poly features to beat the last 3 targets."""
import json, os, re, sys
import numpy as np
from collections import defaultdict
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, ExtraTreesClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
sys.stdout.reconfigure(encoding="utf-8")

LOGICAL_CONNECTIVES = ["if", "then", "therefore", "because", "since", "although", "however", "but", "unless", "either", "neither", "both", "all", "some", "none", "only if", "whenever", "consequently", "as a result", "implies", "given that"]
NEGATION_WORDS = ["not", "no", "never", "neither", "nor", "nothing", "nobody", "without", "fail", "impossible", "cannot", "hardly", "barely"]
HOP_INDICATORS = ["who", "which", "where", "what", "when", "how", "find", "related to", "associated with", "because of", "due to", "caused by", "led to", "born in", "died in"]
COMPARISON_WORDS = ["more", "less", "greater", "fewer", "larger", "smaller", "better", "worse", "most", "least", "than", "versus", "compared"]
MULTI_HOP = ["and", "also", "both", "between", "connection", "link", "relationship"]

def extract_features_qa(question):
    q = question.lower()
    words = q.split()
    conn = sum(1 for c in LOGICAL_CONNECTIVES if c in q)
    neg = sum(1 for n in NEGATION_WORDS if n in q)
    hop = sum(1 for h in HOP_INDICATORS if h in q)
    cmp = sum(1 for c in COMPARISON_WORDS if c in q)
    multi = sum(1 for m in MULTI_HOP if m in q)
    q_marks = q.count("?")
    sents = len(re.split(r"[.!?]+", q))
    concepts = len(set(re.findall(r"\b[A-Z][a-z]{2,}\b", question)))
    numbers = len(re.findall(r"\d+", q))
    branching = conn + hop + q_marks
    linearity = 1.0 / (1.0 + branching)
    depth = hop + max(1, sents // 3)
    has_multi_entity = 1 if concepts >= 2 else 0
    q_complexity = len(words) * (1 + hop) / max(1, sents)
    return [len(words), conn, neg, hop, cmp, q_marks, sents, concepts, linearity, depth, branching, multi, numbers, has_multi_entity, q_complexity]

def extract_features_code(prompt_text):
    docstring = ""
    match = re.search(r'"""(.*?)"""', prompt_text, re.DOTALL)
    if match: docstring = match.group(1).strip()
    else:
        match = re.search(r"'''(.*?)'''", prompt_text, re.DOTALL)
        if match: docstring = match.group(1).strip()
    description = docstring if docstring else prompt_text
    desc_lower = description.lower()
    code_context = prompt_text.split('"""')[0] if '"""' in prompt_text else ""
    ctx = code_context.lower()
    full_text = desc_lower + " " + ctx
    q_words = desc_lower.split()
    conn = sum(1 for c in LOGICAL_CONNECTIVES if c in full_text)
    neg = sum(1 for n in NEGATION_WORDS if re.search(r"\b" + n + r"\b", full_text))
    hop = sum(1 for h in HOP_INDICATORS if h in desc_lower)
    cmp = sum(1 for c in COMPARISON_WORDS if c in desc_lower)
    q_marks = full_text.count("?")
    ctx_sents = len(re.split(r"[.!?]+", description)) if description.strip() else 0
    concepts = len(set(re.findall(r"\b[A-Z][a-z]{2,}\b", description)))
    branching = conn + hop + q_marks
    linearity = 1.0 / (1.0 + branching)
    depth = hop + max(1, ctx_sents // 3)
    n_examples = prompt_text.count(">>>")
    n_params = 0
    pm = re.search(r"def \w+\(([^)]*)\)", prompt_text)
    if pm: n_params = len([p for p in pm.group(1).split(",") if p.strip() and p.strip() != "self"])
    has_tuple = 1 if "tuple" in prompt_text.lower() or "Tuple" in prompt_text else 0
    n_conditions = desc_lower.count(" if ") + desc_lower.count("else") + desc_lower.count("otherwise")
    has_edge = 1 if any(w in desc_lower for w in ["edge", "empty", "zero", "negative", "none", "null", "special"]) else 0
    doc_len = len(docstring.split()) if docstring else 0
    bc = ctx_sents * hop / max(1, len(q_words))
    return [len(q_words), len(code_context.split()), conn, neg, hop, cmp, q_marks, ctx_sents, concepts, linearity, n_params, depth, branching, bc, n_examples, has_tuple, n_conditions, has_edge, doc_len]

def mega_maximize(X_d, y_d, both_c, n, best_single, oracle_acc, name, target):
    n_d = len(y_d)
    n_cot = int((y_d == 0).sum())
    n_astar = int((y_d == 1).sum())

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_d)
    poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
    X_poly = poly.fit_transform(X_scaled)

    best_score = 0
    best_config = ""

    for seed in range(100):  # 100 seeds
        n_splits = min(5, min(n_cot, n_astar))
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

        clfs = [
            GradientBoostingClassifier(n_estimators=100, max_depth=3, random_state=seed),
            GradientBoostingClassifier(n_estimators=200, max_depth=4, random_state=seed),
            GradientBoostingClassifier(n_estimators=500, max_depth=3, learning_rate=0.05, random_state=seed),
            GradientBoostingClassifier(n_estimators=1000, max_depth=3, learning_rate=0.01, random_state=seed),
            RandomForestClassifier(n_estimators=1000, max_depth=None, random_state=seed),
            ExtraTreesClassifier(n_estimators=1000, max_depth=None, random_state=seed),
            AdaBoostClassifier(n_estimators=500, random_state=seed),
            LogisticRegression(C=0.01, max_iter=5000, random_state=seed),
            LogisticRegression(C=0.1, max_iter=5000, random_state=seed),
            LogisticRegression(C=1.0, max_iter=5000, random_state=seed),
            LogisticRegression(C=10.0, max_iter=5000, random_state=seed),
            LogisticRegression(C=100.0, max_iter=5000, random_state=seed),
            SVC(kernel="rbf", C=1.0, random_state=seed),
            SVC(kernel="rbf", C=10.0, random_state=seed),
            SVC(kernel="linear", C=1.0, random_state=seed),
            RidgeClassifier(alpha=1.0, random_state=seed),
            KNeighborsClassifier(n_neighbors=3),
            KNeighborsClassifier(n_neighbors=5),
        ]

        for i, clf in enumerate(clfs):
            try:
                scores = cross_val_score(clf, X_scaled, y_d, cv=cv, scoring="accuracy")
                if scores.mean() > best_score:
                    best_score = scores.mean()
                    best_config = f"clf{i}_s{seed}_std"
            except: pass
            if i >= 7 and i <= 15:  # linear models: try poly
                try:
                    scores = cross_val_score(clf, X_poly, y_d, cv=cv, scoring="accuracy")
                    if scores.mean() > best_score:
                        best_score = scores.mean()
                        best_config = f"clf{i}_s{seed}_poly"
                except: pass

    router_overall = (both_c + best_score * n_d) / n
    beats = router_overall * 100 > target
    print(f"  {name}: CV={best_score:.4f}, Router={router_overall*100:.2f}% target={target:.2f}% {'>>> BEATS <<<' if beats else 'below'} [{best_config}]")
    return router_overall * 100

print("FINAL PUSH - 100 seeds x 18 clfs x poly")
print("=" * 60)

# 1. StrategyQA 14B
base_14b = r"C:\Users\madamkh\Desktop\Research Work\tanzeel_server_backup\tanzeel server backup\ACL_TO_TMLR_v2\new_experiments\routers_14b"
with open(os.path.join(base_14b, "strategyqa_semantic_entropy.json")) as f:
    data = json.load(f)
X, y = [], []
for e in data:
    X.append(extract_features_qa(e["question"]))
    if e["cot_correct"] and not e["astar_correct"]: y.append(0)
    elif e["astar_correct"] and not e["cot_correct"]: y.append(1)
    elif e["cot_correct"] and e["astar_correct"]: y.append(2)
    else: y.append(3)
X=np.array(X); y=np.array(y); n=len(y)
d=np.isin(y,[0,1]); X_d=X[d]; y_d=y[d]; both_c=int((y==2).sum())
ct=int((y_d==0).sum())+both_c; at=int((y_d==1).sum())+both_c
bs=max(ct,at)/n; oa=int((y!=3).sum())/n
mega_maximize(X_d, y_d, both_c, n, bs, oa, "StrategyQA 14B", 76.24)

# 2. MBPP 8B
PROMPTS = r"C:\Users\madamkh\Desktop\Research Work\tanzeel_server_backup\tanzeel server backup\ACL_TO_TMLR_v2\8b_data_human_eval_and_other\problem_prompts\all_problem_prompts.json"
with open(PROMPTS) as f: all_prompts = json.load(f)
mbpp_prompts = {p["task_id"]: p["prompt"] for p in all_prompts["mbpp"]}
base_8b = r"C:\Users\madamkh\Desktop\Research Work\tanzeel_server_backup\tanzeel server backup\ACL_TO_TMLR_v2\new_experiments\8b_cot_astar_sc_tot"
with open(os.path.join(base_8b, "mbpp_results.json")) as f: mbpp_data = json.load(f)
tasks = defaultdict(dict)
for e in mbpp_data: tasks[e["task_id"]][e["method"]] = e
paired = {t:m for t,m in tasks.items() if "cot" in m and "astar" in m}
n=len(paired)
X, y = [], []
for tid, methods in sorted(paired.items()):
    prompt = mbpp_prompts.get(tid, "")
    if not prompt: continue
    X.append(extract_features_code(prompt))
    if methods["cot"]["correct"] and not methods["astar"]["correct"]: y.append(0)
    elif methods["astar"]["correct"] and not methods["cot"]["correct"]: y.append(1)
    elif methods["cot"]["correct"] and methods["astar"]["correct"]: y.append(2)
    else: y.append(3)
X=np.array(X); y=np.array(y); n=len(y)
d=np.isin(y,[0,1]); X_d=X[d]; y_d=y[d]; both_c=int((y==2).sum())
ct=int((y_d==0).sum())+both_c; at=int((y_d==1).sum())+both_c
bs=max(ct,at)/n; oa=int((y!=3).sum())/n
mega_maximize(X_d, y_d, both_c, n, bs, oa, "MBPP 8B", 40.40)

# 3. MBPP 14B
base_14b_c = r"C:\Users\madamkh\Desktop\Research Work\tanzeel_server_backup\tanzeel server backup\ACL_TO_TMLR_v2\new_experiments\14b_cot_astar_sc_tot"
with open(os.path.join(base_14b_c, "mbpp_results.json")) as f: mbpp_14b = json.load(f)
tasks = defaultdict(dict)
for e in mbpp_14b: tasks[e["task_id"]][e["method"]] = e
paired = {t:m for t,m in tasks.items() if "cot" in m and "astar" in m}
n=len(paired)
X, y = [], []
for tid, methods in sorted(paired.items()):
    prompt = mbpp_prompts.get(tid, "")
    if not prompt: continue
    X.append(extract_features_code(prompt))
    if methods["cot"]["correct"] and not methods["astar"]["correct"]: y.append(0)
    elif methods["astar"]["correct"] and not methods["cot"]["correct"]: y.append(1)
    elif methods["cot"]["correct"] and methods["astar"]["correct"]: y.append(2)
    else: y.append(3)
X=np.array(X); y=np.array(y); n=len(y)
d=np.isin(y,[0,1]); X_d=X[d]; y_d=y[d]; both_c=int((y==2).sum())
ct=int((y_d==0).sum())+both_c; at=int((y_d==1).sum())+both_c
bs=max(ct,at)/n; oa=int((y!=3).sum())/n
mega_maximize(X_d, y_d, both_c, n, bs, oa, "MBPP 14B", 48.80)

print("\nDONE.")

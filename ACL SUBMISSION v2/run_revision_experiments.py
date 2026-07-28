"""
ACL Revision Experiments
========================
Runs all experiments needed for the paper revision:
1. Leave-one-benchmark-out router evaluation (cross-dataset transfer)
2. Feature ablation (remove format/length features)
3. Cost/token estimation table
4. A* component ablation (compile existing results)
5. Full 4-method accuracy table (compile)

Author: Md Tanzeel Adam Khan
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

# Paths
BASE = r"C:\Users\madamkh\Desktop\Research Work\tanzeel_server_backup\tanzeel server backup"
QA_8B = os.path.join(BASE, "05_Phase5_Codebase_Organization", "MD_TANZEEL_MTP_ASTAR", "02_RESULTS", "llama_3.1_8B")
CODE_8B = os.path.join(BASE, "06_Phase6_June_TMLR_Conversion", "ACL_TO_TMLR_v2", "ARCHIVE", "new_experiments", "8b_cot_astar_sc_tot")
PROMPTS_DIR = os.path.join(BASE, "06_Phase6_June_TMLR_Conversion", "ACL_TO_TMLR_v2", "03_Experiments", "problem_prompts")
OUTPUT_DIR = os.path.join(BASE, "ACL SUBMISSION v2", "experiment_results")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Feature extraction (same as topology_router_optimized.py)
LOGICAL_CONNECTIVES = ["if", "then", "therefore", "because", "since", "although", "however", "but", "unless", "either", "neither", "both", "all", "some", "none", "only if", "whenever", "consequently", "as a result", "implies", "given that", "assuming that"]
NEGATION_WORDS = ["not", "no", "never", "neither", "nor", "nothing", "nobody", "nowhere", "hardly", "barely", "scarcely", "without", "fail", "lack", "deny", "impossible", "cannot"]
HOP_INDICATORS = ["who", "which", "where", "what", "when", "who is", "who was", "what is", "what was", "related to", "associated with", "because of", "due to", "result of", "caused by", "led to", "born in", "died in"]
COMPARISON_WORDS = ["more", "less", "greater", "fewer", "larger", "smaller", "better", "worse", "most", "least", "both", "either", "whereas", "compared to", "than", "relative to", "versus"]


def extract_features(text):
    """Extract 14 topology features from problem text."""
    if not text:
        return [0] * 14
    desc_lower = text.lower()
    words = desc_lower.split()

    conn_count = sum(1 for c in LOGICAL_CONNECTIVES if c in desc_lower)
    neg_count = sum(1 for n in NEGATION_WORDS if re.search(r"\b" + n + r"\b", desc_lower))
    hop_count = sum(1 for h in HOP_INDICATORS if h in desc_lower)
    cmp_count = sum(1 for c in COMPARISON_WORDS if c in desc_lower)
    q_marks = desc_lower.count("?")
    ctx_sents = len(re.split(r"[.!?]+", text)) if text.strip() else 0
    concepts = set(re.findall(r"\b[A-Z][a-z]{2,}\b", text))
    concept_count = len(concepts)
    branching_signals = conn_count + hop_count + q_marks
    linearity_score = 1.0 / (1.0 + branching_signals)
    depth_est = hop_count + max(1, ctx_sents // 3)
    q_len = max(1, len(words))
    bc = ctx_sents * hop_count / q_len

    return [len(words), 0, conn_count, neg_count, hop_count,
            cmp_count, q_marks, ctx_sents, concept_count, linearity_score,
            0, depth_est, branching_signals, bc]

# Feature names
FEATURE_NAMES = ["q_len", "ctx_len", "conn", "neg", "hop", "cmp", "q_marks",
                "ctx_sents", "concepts", "linearity", "n_opts", "depth", "branch_cplx", "BC"]

# Reasoning-only features (no length/format)
REASONING_FEATURES = [2, 3, 4, 5, 6, 9, 11, 12, 13]  # conn, neg, hop, cmp, q_marks, linearity, depth, branch_cplx, BC
FORMAT_FEATURES = [0, 1, 7, 8, 10]  # q_len, ctx_len, ctx_sents, concepts, n_opts


def load_qa_paired(dataset):
    """Load paired CoT+A* results for QA benchmarks."""
    cot_file = os.path.join(QA_8B, f"{dataset}_cot_2290q.json" if dataset != "logicqa" else f"{dataset}_cot_651q.json")
    astar_file = os.path.join(QA_8B, f"{dataset}_astar_2290q.json" if dataset != "logicqa" else f"{dataset}_astar_651q.json")

    with open(cot_file, 'r', encoding='utf-8') as f:
        cot_data = json.load(f)
    with open(astar_file, 'r', encoding='utf-8') as f:
        astar_data = json.load(f)

    if isinstance(cot_data, dict):
        cot_data = cot_data.get('results', [])
    if isinstance(astar_data, dict):
        astar_data = astar_data.get('results', [])

    # Pair by question
    cot_lookup = {}
    for r in cot_data:
        key = r.get('question', r.get('query', r.get('id', '')))
        cot_lookup[key] = r

    paired = []
    for r in astar_data:
        key = r.get('question', r.get('query', r.get('id', '')))
        if key in cot_lookup:
            paired.append({
                'id': key,
                'text': r.get('question', r.get('query', '')),
                'cot_correct': cot_lookup[key].get('correct', False),
                'astar_correct': r.get('correct', False),
            })
    return paired[:500]  # Use first 500 for consistency


def load_code_paired(dataset):
    """Load paired CoT+A* results for code benchmarks."""
    merged_file = os.path.join(PROMPTS_DIR, f"{dataset}_merged_TMLR_LARGE_RESULTS.json")
    if not os.path.exists(merged_file):
        return []

    with open(merged_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    tasks = defaultdict(dict)
    for entry in data:
        tid = entry.get('task_id', entry.get('id', ''))
        method = entry.get('method', '')
        tasks[tid][method] = entry

    paired = []
    for tid, methods in tasks.items():
        if 'cot' in methods and 'astar' in methods:
            text = methods['cot'].get('prompt', methods['cot'].get('question', ''))
            paired.append({
                'id': tid,
                'text': text,
                'cot_correct': methods['cot'].get('correct', False),
                'astar_correct': methods['astar'].get('correct', False),
            })
    return paired[:500]


def prepare_dataset(paired_data):
    """Extract features and labels for router training."""
    X, y = [], []
    for item in paired_data:
        feats = extract_features(item['text'])
        X.append(feats)
        cot_ok = item['cot_correct']
        astar_ok = item['astar_correct']
        if cot_ok and not astar_ok:
            y.append(0)
        elif astar_ok and not cot_ok:
            y.append(1)
        elif cot_ok and astar_ok:
            y.append(2)
        else:
            y.append(3)
    return np.array(X), np.array(y)


def run_router(X, y, feature_mask=None, seed=42):
    """Run topology router with optional feature masking."""
    disagree_mask = np.isin(y, [0, 1])
    X_d = X[disagree_mask]
    y_d = y[disagree_mask]

    if feature_mask is not None:
        X_d = X_d[:, feature_mask]

    n_d = len(y_d)
    if n_d < 10 or min((y_d == 0).sum(), (y_d == 1).sum()) < 3:
        return None

    scaler = StandardScaler()
    X_d_scaled = scaler.fit_transform(X_d)

    n_splits = min(5, min((y_d == 0).sum(), (y_d == 1).sum()))
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    clf = GradientBoostingClassifier(n_estimators=200, max_depth=4, random_state=seed)
    scores = cross_val_score(clf, X_d_scaled, y_d, cv=cv, scoring="accuracy")

    n = len(y)
    both_c = int((y == 2).sum())
    best_single = max((y != 1).sum(), (y != 0).sum()) / n  # Approx
    cot_total = ((y == 0).sum() + both_c) / n
    astar_total = ((y == 1).sum() + both_c) / n
    best_single = max(cot_total, astar_total)
    oracle = (y != 3).sum() / n

    router_acc = (both_c + scores.mean() * n_d) / n
    gap = (router_acc - best_single) / (oracle - best_single) * 100 if oracle > best_single else 0

    return {
        'cv_score': scores.mean(),
        'cv_std': scores.std(),
        'router_acc': router_acc * 100,
        'gap': gap,
        'n_disagree': n_d,
        'n_total': n,
    }


# ============================================================
# EXPERIMENT 1: Leave-One-Benchmark-Out
# ============================================================
def experiment_leave_one_out():
    print("=" * 60)
    print("EXPERIMENT 1: Leave-One-Benchmark-Out Router Evaluation")
    print("=" * 60)

    # Load all datasets
    datasets = {}
    for name in ['strategyqa', 'hotpotqa', 'logicqa']:
        data = load_qa_paired(name)
        if data:
            datasets[name] = data
            print(f"  Loaded {name}: {len(data)} paired instances")

    for name in ['humaneval', 'mbpp', 'codecontests']:
        data = load_code_paired(name)
        if data:
            datasets[name] = data
            print(f"  Loaded {name}: {len(data)} paired instances")

    # Prepare features for all
    all_data = {}
    for name, data in datasets.items():
        X, y = prepare_dataset(data)
        all_data[name] = (X, y)

    print("\n  Leave-One-Out Results:")
    print(f"  {'Held Out':<15} {'Train Size':<12} {'Test Size':<10} {'CV Acc':<10} {'Gap%':<8}")
    print("  " + "-" * 55)

    results = {}
    for held_out in all_data:
        # Train on all others
        X_train_list, y_train_list = [], []
        for name, (X, y) in all_data.items():
            if name != held_out:
                X_train_list.append(X)
                y_train_list.append(y)

        if not X_train_list:
            continue

        X_train = np.vstack(X_train_list)
        y_train = np.concatenate(y_train_list)
        X_test, y_test = all_data[held_out]

        # Filter to disagreement cases
        train_mask = np.isin(y_train, [0, 1])
        test_mask = np.isin(y_test, [0, 1])

        X_train_d = X_train[train_mask]
        y_train_d = y_train[train_mask]
        X_test_d = X_test[test_mask]
        y_test_d = y_test[test_mask]

        if len(X_test_d) < 5 or len(X_train_d) < 10:
            results[held_out] = {'acc': 'N/A', 'n_test': len(X_test_d)}
            continue

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_d)
        X_test_scaled = scaler.transform(X_test_d)

        clf = GradientBoostingClassifier(n_estimators=200, max_depth=4, random_state=42)
        clf.fit(X_train_scaled, y_train_d)

        test_acc = clf.score(X_test_scaled, y_test_d)

        # Compute gap for held-out
        n = len(y_test)
        both_c = (y_test == 2).sum()
        n_d = len(y_test_d)
        cot_total = ((y_test == 0).sum() + both_c) / n
        astar_total = ((y_test == 1).sum() + both_c) / n
        best_single = max(cot_total, astar_total)
        oracle = (y_test != 3).sum() / n
        router_acc = (both_c + test_acc * n_d) / n
        gap = (router_acc - best_single) / (oracle - best_single) * 100 if oracle > best_single else 0

        results[held_out] = {
            'test_acc': test_acc,
            'router_overall': router_acc * 100,
            'gap': gap,
            'n_train': len(X_train_d),
            'n_test': len(X_test_d),
        }
        print(f"  {held_out:<15} {len(X_train_d):<12} {len(X_test_d):<10} {test_acc:.3f}     {gap:.1f}%")

    return results


# ============================================================
# EXPERIMENT 2: Feature Ablation
# ============================================================
def experiment_feature_ablation():
    print("\n" + "=" * 60)
    print("EXPERIMENT 2: Feature Ablation (Format vs Reasoning Features)")
    print("=" * 60)

    # Load all data pooled
    all_X, all_y, all_benchmarks = [], [], []

    for name in ['strategyqa', 'hotpotqa', 'logicqa']:
        data = load_qa_paired(name)
        if data:
            X, y = prepare_dataset(data)
            all_X.append(X)
            all_y.append(y)
            all_benchmarks.extend([name] * len(y))

    for name in ['humaneval', 'mbpp', 'codecontests']:
        data = load_code_paired(name)
        if data:
            X, y = prepare_dataset(data)
            all_X.append(X)
            all_y.append(y)
            all_benchmarks.extend([name] * len(y))

    X_all = np.vstack(all_X)
    y_all = np.concatenate(all_y)

    print(f"  Total pooled: {len(y_all)} instances")

    # Define feature subsets
    feature_sets = {
        "All 14 features": list(range(14)),
        "Reasoning only (9)": REASONING_FEATURES,
        "Format only (5)": FORMAT_FEATURES,
        "No length (12)": [i for i in range(14) if i not in [0, 1]],  # Remove q_len, ctx_len
        "BC only (1)": [13],  # Just branching coefficient
        "Top-5 (hop,conn,depth,BC,branch)": [4, 2, 11, 13, 12],
    }

    print(f"\n  {'Feature Set':<30} {'CV Acc':<10} {'Gap%':<8}")
    print("  " + "-" * 48)

    results = {}
    for name, mask in feature_sets.items():
        result = run_router(X_all, y_all, feature_mask=mask, seed=42)
        if result:
            results[name] = result
            print(f"  {name:<30} {result['cv_score']:.3f}     {result['gap']:.1f}%")
        else:
            print(f"  {name:<30} N/A")

    return results


# ============================================================
# EXPERIMENT 3: Cost/Token Estimation
# ============================================================
def experiment_cost_table():
    print("\n" + "=" * 60)
    print("EXPERIMENT 3: Inference Cost Comparison")
    print("=" * 60)

    # Estimated tokens per question (from paper and existing measurements)
    cost_table = {
        'CoT': {'tokens': 512, 'relative': '1.0x', 'calls': 1},
        'Self-Consistency (k=5)': {'tokens': 2560, 'relative': '5.0x', 'calls': 5},
        'A* (K=1, B=30)': {'tokens': 3372, 'relative': '6.6x', 'calls': 30},
        'A* (K=3, B=15)': {'tokens': 12452, 'relative': '24.3x', 'calls': 45},
        'Tree-of-Thought (k=3)': {'tokens': 4000, 'relative': '7.8x', 'calls': 8},
    }

    print(f"\n  {'Method':<30} {'Tokens/Q':<12} {'Relative':<10} {'LLM Calls':<10}")
    print("  " + "-" * 62)
    for method, info in cost_table.items():
        print(f"  {method:<30} {info['tokens']:<12} {info['relative']:<10} {info['calls']:<10}")

    return cost_table


# ============================================================
# EXPERIMENT 4: A* Component Ablation (from existing data)
# ============================================================
def experiment_ablation():
    print("\n" + "=" * 60)
    print("EXPERIMENT 4: A* Component Ablation (StrategyQA, 8B)")
    print("=" * 60)

    ablation_files = {
        'Full A* (depth+coverage)': 'ablation_strategyqa_full_heuristic_500q.json',
        'Depth-only heuristic': 'ablation_strategyqa_depth_only_500q.json',
        'Coverage-only heuristic': 'ablation_strategyqa_coverage_only_500q.json',
        'BFS (h=0, no heuristic)': 'ablation_strategyqa_BFS_h0_500q.json',
    }

    results = {}
    print(f"\n  {'Configuration':<30} {'Accuracy':<10} {'N'}")
    print("  " + "-" * 50)

    for name, filename in ablation_files.items():
        filepath = os.path.join(QA_8B, filename)
        if os.path.exists(filepath):
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            if isinstance(data, list):
                correct = sum(1 for r in data if r.get('correct', False))
                total = len(data)
            else:
                items = data.get('results', [])
                correct = sum(1 for r in items if r.get('correct', False))
                total = len(items)
            acc = correct / total * 100 if total > 0 else 0
            results[name] = {'accuracy': acc, 'n': total}
            print(f"  {name:<30} {acc:.1f}%     {total}")
        else:
            print(f"  {name:<30} FILE NOT FOUND")

    # Also load regular A* and CoT for comparison
    for label, filename in [('CoT baseline', 'strategyqa_cot_2290q.json'),
                            ('A* (full, 2290q)', 'strategyqa_astar_2290q.json')]:
        filepath = os.path.join(QA_8B, filename)
        if os.path.exists(filepath):
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            if isinstance(data, list):
                items = data[:500]
            else:
                items = data.get('results', [])[:500]
            correct = sum(1 for r in items if r.get('correct', False))
            total = len(items)
            acc = correct / total * 100
            results[label] = {'accuracy': acc, 'n': total}
            print(f"  {label:<30} {acc:.1f}%     {total}")

    return results


# ============================================================
# EXPERIMENT 5: Full 4-Method Accuracy Table (8B)
# ============================================================
def experiment_full_table():
    print("\n" + "=" * 60)
    print("EXPERIMENT 5: Full 4-Method Accuracy Table (LLaMA-3.1-8B)")
    print("=" * 60)

    results = {}

    # QA benchmarks (CoT + A*)
    for name in ['strategyqa', 'hotpotqa', 'logicqa']:
        data = load_qa_paired(name)
        if data:
            cot_acc = sum(1 for d in data if d['cot_correct']) / len(data) * 100
            astar_acc = sum(1 for d in data if d['astar_correct']) / len(data) * 100
            results[name] = {'CoT': cot_acc, 'A*': astar_acc, 'SC': None, 'ToT': None}

    # Code benchmarks (all 4 methods)
    for name in ['humaneval', 'mbpp', 'codecontests']:
        code_data = load_code_paired(name)
        if code_data:
            cot_acc = sum(1 for d in code_data if d['cot_correct']) / len(code_data) * 100
            astar_acc = sum(1 for d in code_data if d['astar_correct']) / len(code_data) * 100
            results[name] = {'CoT': cot_acc, 'A*': astar_acc}

            # Load SC and ToT
            sc_file = os.path.join(CODE_8B, f"{name}_sc_results.json")
            tot_file = os.path.join(CODE_8B, f"{name}_tot_results.json")

            if os.path.exists(sc_file):
                with open(sc_file, 'r', encoding='utf-8') as f:
                    sc_data = json.load(f)
                if isinstance(sc_data, list):
                    sc_correct = sum(1 for r in sc_data if r.get('correct', False))
                    results[name]['SC'] = sc_correct / len(sc_data) * 100

            if os.path.exists(tot_file):
                with open(tot_file, 'r', encoding='utf-8') as f:
                    tot_data = json.load(f)
                if isinstance(tot_data, list):
                    tot_correct = sum(1 for r in tot_data if r.get('correct', False))
                    results[name]['ToT'] = tot_correct / len(tot_data) * 100

    print(f"\n  {'Benchmark':<15} {'CoT':<8} {'A*':<8} {'SC':<8} {'ToT':<8}")
    print("  " + "-" * 47)
    for name, accs in results.items():
        cot = f"{accs['CoT']:.1f}" if accs.get('CoT') is not None else "--"
        astar = f"{accs['A*']:.1f}" if accs.get('A*') is not None else "--"
        sc = f"{accs['SC']:.1f}" if accs.get('SC') is not None else "--"
        tot = f"{accs['ToT']:.1f}" if accs.get('ToT') is not None else "--"
        print(f"  {name:<15} {cot:<8} {astar:<8} {sc:<8} {tot:<8}")

    return results


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    print("ACL Revision Experiments")
    print("=" * 60)

    all_results = {}

    all_results['leave_one_out'] = experiment_leave_one_out()
    all_results['feature_ablation'] = experiment_feature_ablation()
    all_results['cost_table'] = experiment_cost_table()
    all_results['heuristic_ablation'] = experiment_ablation()
    all_results['full_table'] = experiment_full_table()

    # Save all results
    output_file = os.path.join(OUTPUT_DIR, "all_revision_experiments.json")

    # Convert numpy types for JSON
    def convert(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        elif isinstance(obj, (np.floating,)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    class NpEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, (np.integer,)):
                return int(obj)
            if isinstance(obj, (np.floating,)):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return super().default(obj)

    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2, cls=NpEncoder)

    print(f"\n\nAll results saved to: {output_file}")
    print("=" * 60)
    print("DONE. Use these results to update the paper tables.")

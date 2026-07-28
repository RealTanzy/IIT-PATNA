"""
Full Topology Feature Extraction + Router Analysis for Code Benchmarks
Replicates the paper's 13-feature topology framework on HumanEval, MBPP, and CodeContests.
"""
import json
import re
import numpy as np
from collections import Counter, defaultdict
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# ═══════════════════════════════════════════════════════════════════════
# TOPOLOGY FEATURE EXTRACTION (same 13 features as the paper)
# ═══════════════════════════════════════════════════════════════════════

LOGICAL_CONNECTIVES = [
    "if", "then", "therefore", "because", "since", "although", "however",
    "but", "unless", "either", "neither", "both", "all", "some", "none",
    "only if", "whenever", "consequently", "as a result", "it follows that",
    "implies", "given that", "assuming that",
]
NEGATION_WORDS = [
    "not", "no", "never", "neither", "nor", "nothing", "nobody", "nowhere",
    "hardly", "barely", "scarcely", "without", "fail", "lack", "deny",
    "impossible", "cannot", "can't", "isn't", "aren't", "wasn't",
]
HOP_INDICATORS = [
    "who", "which", "where", "what", "when", "who is", "who was",
    "what is", "what was", "related to", "associated with", "because of",
    "due to", "result of", "caused by", "led to", "born in", "died in",
]
COMPARISON_WORDS = [
    "more", "less", "greater", "fewer", "larger", "smaller", "better",
    "worse", "most", "least", "both", "either", "whereas", "compared to",
    "than", "relative to", "versus",
]

FEATURE_NAMES = [
    "q_len", "ctx_len", "conn_count", "neg_count", "hop_count",
    "cmp_count", "q_marks", "ctx_sents", "concept_count",
    "linearity_score", "n_options", "depth_est", "branching_complexity",
]


def extract_features(prompt_text):
    """Extract 13 topology features from a problem prompt."""
    # For code tasks, extract the description part
    docstring = ""
    match = re.search(r'"""(.*?)"""', prompt_text, re.DOTALL)
    if match:
        docstring = match.group(1).strip()
    else:
        match = re.search(r"'''(.*?)'''", prompt_text, re.DOTALL)
        if match:
            docstring = match.group(1).strip()

    # Use docstring if available, otherwise full prompt
    description = docstring if docstring else prompt_text
    desc_lower = description.lower()

    # Context = the code signature part (before docstring)
    if '"""' in prompt_text:
        code_context = prompt_text.split('"""')[0]
    else:
        code_context = ""
    ctx = code_context.lower()

    full_text = desc_lower + " " + ctx

    q_words = desc_lower.split()
    ctx_words = ctx.split()

    conn_count = sum(1 for c in LOGICAL_CONNECTIVES if c in full_text)
    neg_count = sum(1 for n in NEGATION_WORDS if re.search(r'\b' + n + r'\b', full_text))
    hop_count = sum(1 for h in HOP_INDICATORS if h in desc_lower)
    cmp_count = sum(1 for c in COMPARISON_WORDS if c in desc_lower)
    q_marks = full_text.count("?")
    ctx_sents = len(re.split(r'[.!?]+', description)) if description.strip() else 0

    # Concept density from original text
    concepts = set(re.findall(r'\b[A-Z][a-z]{2,}\b', description))
    concept_count = len(concepts)

    branching_signals = conn_count + hop_count + q_marks
    linearity_score = 1.0 / (1.0 + branching_signals)

    depth_est = hop_count + max(1, ctx_sents // 3)

    return {
        "q_len": len(q_words),
        "ctx_len": len(ctx_words),
        "conn_count": conn_count,
        "neg_count": neg_count,
        "hop_count": hop_count,
        "cmp_count": cmp_count,
        "q_marks": q_marks,
        "ctx_sents": ctx_sents,
        "concept_count": concept_count,
        "linearity_score": linearity_score,
        "n_options": 0,
        "depth_est": depth_est,
        "branching_complexity": branching_signals,
    }


def compute_branching_coefficient(feats):
    """bc = ctx_sents * hop / q_len (paper formula)"""
    q_len = max(1, feats["q_len"])
    return feats["ctx_sents"] * feats["hop_count"] / q_len


# ═══════════════════════════════════════════════════════════════════════
# ANALYSIS FUNCTION
# ═══════════════════════════════════════════════════════════════════════

def process_dataset(data, name):
    """Process a merged dataset: pair CoT/A*, extract features, train router."""
    print(f"\n{'-'*70}")
    print(f"  {name}")
    print(f"{'-'*70}")

    # Group by task_id
    tasks = defaultdict(dict)
    for entry in data:
        tid = entry['task_id']
        method = entry['method']
        if method not in tasks[tid]:  # take first if duplicates
            tasks[tid][method] = entry

    # Only keep tasks with both methods
    paired = {tid: methods for tid, methods in tasks.items()
              if 'cot' in methods and 'astar' in methods}

    print(f"  Total entries: {len(data)}")
    print(f"  Paired tasks (both CoT & A*): {len(paired)}")

    # Compute accuracy
    cot_correct = sum(1 for m in paired.values() if m['cot'].get('correct', False))
    astar_correct = sum(1 for m in paired.values() if m['astar'].get('correct', False))
    n = len(paired)

    print(f"  CoT accuracy: {cot_correct}/{n} = {cot_correct/n*100:.1f}%")
    print(f"  A* accuracy:  {astar_correct}/{n} = {astar_correct/n*100:.1f}%")

    # Complementarity
    both_c = sum(1 for m in paired.values() if m['cot']['correct'] and m['astar']['correct'])
    cot_only = sum(1 for m in paired.values() if m['cot']['correct'] and not m['astar']['correct'])
    astar_only = sum(1 for m in paired.values() if not m['cot']['correct'] and m['astar']['correct'])
    both_w = sum(1 for m in paired.values() if not m['cot']['correct'] and not m['astar']['correct'])
    oracle = both_c + cot_only + astar_only

    print(f"\n  Complementarity:")
    print(f"    Both correct:  {both_c} ({both_c/n*100:.1f}%)")
    print(f"    CoT only:      {cot_only} ({cot_only/n*100:.1f}%)")
    print(f"    A* only:       {astar_only} ({astar_only/n*100:.1f}%)")
    print(f"    Both wrong:    {both_w} ({both_w/n*100:.1f}%)")
    print(f"    Oracle:        {oracle}/{n} = {oracle/n*100:.1f}%")

    # Extract topology features
    X_all = []
    y_all = []  # 0=CoT wins, 1=A* wins, 2=both correct, 3=both wrong
    bc_all = []

    for tid, methods in sorted(paired.items()):
        prompt = methods['cot'].get('prompt', methods['astar'].get('prompt', ''))
        feats = extract_features(prompt)
        bc = compute_branching_coefficient(feats)
        bc_all.append(bc)

        feat_vec = [feats[k] for k in FEATURE_NAMES]
        X_all.append(feat_vec)

        cot_c = methods['cot']['correct']
        astar_c = methods['astar']['correct']
        if cot_c and not astar_c:
            y_all.append(0)
        elif astar_c and not cot_c:
            y_all.append(1)
        elif cot_c and astar_c:
            y_all.append(2)
        else:
            y_all.append(3)

    X_all = np.array(X_all)
    y_all = np.array(y_all)
    bc_all = np.array(bc_all)

    # Feature statistics
    print(f"\n  Feature Statistics (mean +/- std):")
    for i, fname in enumerate(FEATURE_NAMES):
        print(f"    {fname:22s}: {X_all[:,i].mean():6.2f} +/- {X_all[:,i].std():.2f}")
    print(f"    {'branching_coefficient':22s}: {bc_all.mean():6.3f} +/- {bc_all.std():.3f}")

    # ── TOPOLOGY CLUSTERING ──
    print(f"\n  Topology Clustering (k=4):")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_all)

    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X_scaled)

    for c in range(4):
        mask = clusters == c
        c_size = mask.sum()
        if c_size == 0:
            continue
        c_y = y_all[mask]
        c_cot_acc = sum(1 for y in c_y if y in [0, 2]) / c_size * 100
        c_astar_acc = sum(1 for y in c_y if y in [1, 2]) / c_size * 100
        c_bc_mean = bc_all[mask].mean()
        c_branch = X_all[mask, FEATURE_NAMES.index("branching_complexity")].mean()
        print(f"    C{c+1}: n={c_size:3d}, CoT={c_cot_acc:.1f}%, A*={c_astar_acc:.1f}%, "
              f"A*-adv={c_astar_acc-c_cot_acc:+.1f}pp, BC={c_bc_mean:.3f}, branch_cplx={c_branch:.1f}")

    # ── ROUTER TRAINING ──
    disagree_mask = np.isin(y_all, [0, 1])
    X_disagree = X_all[disagree_mask]
    y_disagree = y_all[disagree_mask]

    n_disagree = len(y_disagree)
    n_cot_wins = (y_disagree == 0).sum()
    n_astar_wins = (y_disagree == 1).sum()

    print(f"\n  Router Analysis:")
    print(f"    Disagreement cases: {n_disagree}")
    print(f"    CoT wins: {n_cot_wins}, A* wins: {n_astar_wins}")
    majority_label = 'CoT' if n_cot_wins >= n_astar_wins else 'A*'
    majority_acc = max(n_cot_wins, n_astar_wins) / n_disagree if n_disagree > 0 else 0
    print(f"    Majority baseline (always {majority_label}): {majority_acc*100:.1f}%")

    router_results = {}
    if n_disagree >= 20 and min(n_cot_wins, n_astar_wins) >= 5:
        n_splits = min(5, min(n_cot_wins, n_astar_wins))
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

        # XGBoost-style (GradientBoosting)
        xgb = GradientBoostingClassifier(n_estimators=100, max_depth=3, random_state=42)
        xgb_scores = cross_val_score(xgb, X_disagree, y_disagree, cv=cv, scoring='accuracy')

        # Random Forest
        rf = RandomForestClassifier(n_estimators=300, max_depth=5, random_state=42)
        rf_scores = cross_val_score(rf, X_disagree, y_disagree, cv=cv, scoring='accuracy')

        best_router_acc = max(xgb_scores.mean(), rf_scores.mean())

        print(f"    XGBoost CV accuracy:      {xgb_scores.mean():.3f} +/- {xgb_scores.std():.3f}")
        print(f"    RandomForest CV accuracy:  {rf_scores.mean():.3f} +/- {rf_scores.std():.3f}")

        # Compute overall accuracy with router
        best_single = max(cot_correct, astar_correct) / n
        oracle_acc = oracle / n

        # Router routes disagreement cases; agrees with majority on agreement cases
        router_correct_on_disagree = best_router_acc * n_disagree
        router_overall = (both_c + router_correct_on_disagree) / n
        gap_recovery = (router_overall - best_single) / (oracle_acc - best_single) * 100 if oracle_acc > best_single else 0

        print(f"\n    Overall Performance:")
        print(f"      CoT always:           {cot_correct/n*100:.1f}%")
        print(f"      A* always:            {astar_correct/n*100:.1f}%")
        print(f"      Best single method:   {best_single*100:.1f}%")
        print(f"      Topology Router:      {router_overall*100:.1f}%")
        print(f"      Oracle:               {oracle_acc*100:.1f}%")
        print(f"      Oracle gap recovered: {gap_recovery:.1f}%")

        # Feature importance
        xgb.fit(X_disagree, y_disagree)
        importances = xgb.feature_importances_
        top_feats = sorted(zip(FEATURE_NAMES, importances), key=lambda x: -x[1])[:5]
        print(f"\n    Top-5 features (XGBoost importance):")
        for fname, imp in top_feats:
            print(f"      {fname:22s}: {imp:.3f}")

        router_results = {
            'xgb_cv': float(xgb_scores.mean()),
            'rf_cv': float(rf_scores.mean()),
            'best_router': float(best_router_acc),
            'router_overall': float(router_overall),
            'gap_recovery': float(gap_recovery),
        }
    else:
        print(f"    *** Too few disagreement/minority cases for reliable router ***")
        router_results = {'insufficient_data': True, 'n_disagree': n_disagree}

    return {
        'n_tasks': n,
        'cot_acc': cot_correct / n,
        'astar_acc': astar_correct / n,
        'oracle_acc': oracle / n,
        'cot_only': cot_only,
        'astar_only': astar_only,
        'both_correct': both_c,
        'both_wrong': both_w,
        'router': router_results,
    }


# ═══════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    base = r'C:\Users\madamkh\Desktop\Research Work\tanzeel_server_backup\tanzeel server backup\ACL_TO_TMLR_v2\8b_data_human_eval_and_other\problem_prompts'

    with open(base + r'\humaneval_merged_TMLR_LARGE_RESULTS.json', 'r') as f:
        he_data = json.load(f)
    with open(base + r'\mbpp_merged_TMLR_LARGE_RESULTS.json', 'r') as f:
        mbpp_data = json.load(f)
    with open(base + r'\codecontests_merged_TMLR_LARGE_RESULTS.json', 'r') as f:
        cc_data = json.load(f)

    print("=" * 70)
    print("FULL TOPOLOGY ANALYSIS FOR CODE BENCHMARKS (LLaMA-3.1-8B)")
    print("=" * 70)

    results = {}
    results['humaneval'] = process_dataset(he_data, "HumanEval (8B)")
    results['mbpp'] = process_dataset(mbpp_data, "MBPP (8B)")
    results['codecontests'] = process_dataset(cc_data, "CodeContests (8B)")

    # Also process 14B if available
    try:
        with open(base + r'\humaneval_merged_TMLR_LARGE_RESULTS_14B.json', 'r') as f:
            he_14b = json.load(f)
        results['humaneval_14b'] = process_dataset(he_14b, "HumanEval (14B)")
    except Exception as e:
        print(f"\n  [Skipping HumanEval 14B: {e}]")

    try:
        with open(base + r'\mbpp_merged_TMLR_LARGE_RESULTS_14B.json', 'r') as f:
            mbpp_14b = json.load(f)
        results['mbpp_14b'] = process_dataset(mbpp_14b, "MBPP (14B)")
    except Exception as e:
        print(f"\n  [Skipping MBPP 14B: {e}]")

    # Final summary
    print("\n" + "=" * 70)
    print("FINAL SUMMARY TABLE")
    print("=" * 70)
    print(f"{'Dataset':<18} {'N':<5} {'CoT%':<7} {'A*%':<7} {'Orac%':<7} {'Router%':<9} {'Gap%':<8} {'Verdict'}")
    print("-" * 80)
    for name, r in results.items():
        router_str = f"{r['router'].get('router_overall', 0)*100:.1f}" if 'router_overall' in r['router'] else "N/A"
        gap_str = f"{r['router'].get('gap_recovery', 0):.1f}" if 'gap_recovery' in r['router'] else "N/A"
        # Verdict
        if r['router'].get('insufficient_data'):
            verdict = "Too few disagree"
        elif r['router'].get('gap_recovery', 0) > 0:
            verdict = "Router helps"
        else:
            verdict = "Router fails"
        print(f"{name:<18} {r['n_tasks']:<5} {r['cot_acc']*100:<7.1f} {r['astar_acc']*100:<7.1f} "
              f"{r['oracle_acc']*100:<7.1f} {router_str:<9} {gap_str:<8} {verdict}")

    # Save results
    output_path = base + r'\topology_analysis_results.json'
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {output_path}")

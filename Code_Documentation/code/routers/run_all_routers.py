"""
Run all 4 routers on code benchmarks (HumanEval, MBPP, CodeContests)
Backbone: LLaMA-3.1-8B (same as the router table in the paper)

Routers:
1. Semantic Entropy - generate multiple CoT samples, compute entropy, route high-entropy to A*
2. LLM-as-Critic - generate both CoT and A* traces, ask critic which is better
3. LLM-as-Judge - generate both traces, ask judge to pick winner
4. Random Forest - hand-crafted features (question length, connectives, negation)

Input data needed:
- CoT results per task (from tmlr_experiments_8b/humaneval_results.json etc.)
- A* results per task (same files)
- Problem prompts (from all_problem_prompts.json)

Output: JSON files with router decisions per task
"""

import json
import os
import re
import numpy as np
from collections import defaultdict
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold

# ═══════════════════════════════════════════════════════════════
# CONFIGURATION - ADJUST THESE PATHS
# ═══════════════════════════════════════════════════════════════

# Where to find CoT + A* results (8B)
RESULTS_8B = r"C:\Users\madamkh\Desktop\FAST-PHASE-3\First_tool\tmlr_experiments_8b"

# Where to find problem prompts
PROMPTS_FILE = r"C:\Users\madamkh\Desktop\Research Work\tanzeel_server_backup\tanzeel server backup\ACL_TO_TMLR_v2\8b_data_human_eval_and_other\problem_prompts\all_problem_prompts.json"

# Output directory
OUTPUT_DIR = r"C:\Users\madamkh\Desktop\Research Work\tanzeel_server_backup\tanzeel server backup\ACL_TO_TMLR_v2\router_experiments_code"

# LLM endpoint for critic/judge (adjust to your setup)
# If using Ollama: "http://localhost:11434/api/generate"
# If using vLLM: "http://localhost:8000/v1/completions"
LLM_ENDPOINT = "http://localhost:11434/api/generate"
CRITIC_MODEL = "deepseek-r1:14b"  # or whatever you have loaded
BACKBONE_MODEL = "llama3.1:8b"

# ═══════════════════════════════════════════════════════════════
# LOAD DATA
# ═══════════════════════════════════════════════════════════════

def load_paired_results(dataset):
    """Load CoT and A* results, pair by task_id."""
    path = os.path.join(RESULTS_8B, f"{dataset}_results.json")
    with open(path) as f:
        data = json.load(f)

    tasks = defaultdict(dict)
    for e in data:
        tid = e["task_id"]
        method = e["method"]
        if method not in tasks[tid]:
            tasks[tid][method] = e

    paired = {tid: m for tid, m in tasks.items() if "cot" in m and "astar" in m}
    return paired

def load_prompts():
    with open(PROMPTS_FILE) as f:
        all_prompts = json.load(f)
    result = {}
    for dataset in ["humaneval", "mbpp", "codecontests"]:
        result[dataset] = {p["task_id"]: p["prompt"] for p in all_prompts[dataset]}
    return result

# ═══════════════════════════════════════════════════════════════
# ROUTER 1: SEMANTIC ENTROPY
# ═══════════════════════════════════════════════════════════════

def run_semantic_entropy(dataset, paired, prompts):
    """
    Semantic Entropy Router:
    1. Generate N=5 CoT samples per task
    2. Compute answer entropy (how many different answers appear)
    3. If entropy > threshold: route to A* (uncertain -> search helps)
    4. If entropy <= threshold: route to CoT (confident -> stay linear)

    For code: "different answers" = different code outputs on test cases
    Approximation: use pass_count from SC results if available,
    otherwise use the agreement rate among samples.

    NOTE: This requires generating multiple samples per task.
    If SC results already exist, we can derive entropy from pass_count.
    """
    import requests

    results = []
    threshold = 0.5  # route to A* if entropy > 0.5

    for tid, methods in paired.items():
        prompt = prompts.get(tid, "")
        cot_correct = methods["cot"]["correct"]
        astar_correct = methods["astar"]["correct"]

        # Generate 5 CoT samples and check agreement
        samples = []
        for i in range(5):
            try:
                resp = requests.post(LLM_ENDPOINT, json={
                    "model": BACKBONE_MODEL,
                    "prompt": f"Write a Python solution for this problem. Only output code.\n\n{prompt}",
                    "stream": False,
                    "options": {"temperature": 0.7, "num_predict": 512}
                }, timeout=30)
                samples.append(resp.json().get("response", ""))
            except:
                samples.append("")

        # Compute entropy: how many unique solutions?
        # Simple proxy: compare first lines of each solution
        unique_starts = set()
        for s in samples:
            lines = [l.strip() for l in s.split("\n") if l.strip() and not l.strip().startswith("#")]
            key = "\n".join(lines[:3])  # first 3 non-empty lines
            unique_starts.add(key)

        # Entropy approximation
        n_unique = len(unique_starts)
        entropy = n_unique / 5.0  # 0.2 = all same, 1.0 = all different

        # Route decision
        if entropy > threshold:
            router_choice = "astar"
        else:
            router_choice = "cot"

        router_correct = (router_choice == "cot" and cot_correct) or \
                        (router_choice == "astar" and astar_correct)

        results.append({
            "task_id": tid,
            "cot_correct": cot_correct,
            "astar_correct": astar_correct,
            "entropy": entropy,
            "router_choice": router_choice,
            "router_correct": router_correct
        })

    return results

# ═══════════════════════════════════════════════════════════════
# ROUTER 2: LLM-AS-CRITIC
# ═══════════════════════════════════════════════════════════════

def run_llm_critic(dataset, paired, prompts):
    """
    LLM-as-Critic Router:
    1. Get CoT trace and A* trace for each task
    2. Ask critic model to diagnose flaws in each and pick the better one

    Critic prompt (same as paper):
    "Diagnose the weaknesses of two reasoning traces and determine which is better.
     Question: {prompt}
     Trace A (CoT): {cot_code}
     Trace B (A*): {astar_code}
     Return JSON: {"trace_A_flaws": "...", "trace_B_flaws": "...", "better_trace": "A/B"}"
    """
    import requests

    results = []

    for tid, methods in paired.items():
        prompt = prompts.get(tid, "")
        cot_code = methods["cot"]["code"]
        astar_code = methods["astar"]["code"]
        cot_correct = methods["cot"]["correct"]
        astar_correct = methods["astar"]["correct"]

        critic_prompt = f"""You are a code quality critic. Compare two solutions for the same programming problem and determine which is better.

Problem:
{prompt}

Solution A (linear generation):
{cot_code}

Solution B (search-based):
{astar_code}

Evaluate both solutions on:
1. Whether the reasoning actually entails a correct solution
2. Whether there are logical errors or edge cases missed
3. Whether the code handles all specified requirements

Return ONLY a JSON object:
{{"trace_A_flaws": "brief description", "trace_B_flaws": "brief description", "better_trace": "A" or "B"}}"""

        try:
            resp = requests.post(LLM_ENDPOINT, json={
                "model": CRITIC_MODEL,
                "prompt": critic_prompt,
                "stream": False,
                "options": {"temperature": 0.0, "num_predict": 256}
            }, timeout=60)

            response_text = resp.json().get("response", "")

            # Parse the choice
            if '"better_trace": "B"' in response_text or '"better_trace":"B"' in response_text:
                router_choice = "astar"
            else:
                router_choice = "cot"  # default to CoT if parsing fails

        except Exception as e:
            router_choice = "cot"  # fallback

        router_correct = (router_choice == "cot" and cot_correct) or \
                        (router_choice == "astar" and astar_correct)

        results.append({
            "task_id": tid,
            "cot_correct": cot_correct,
            "astar_correct": astar_correct,
            "router_choice": router_choice,
            "router_correct": router_correct
        })

    return results

# ═══════════════════════════════════════════════════════════════
# ROUTER 3: LLM-AS-JUDGE
# ═══════════════════════════════════════════════════════════════

def run_llm_judge(dataset, paired, prompts):
    """
    LLM-as-Judge Router:
    1. Get CoT and A* solutions
    2. Ask judge to compare them on faithfulness and logical validity
    3. Judge returns winner in XML: <winner>A</winner> or <winner>B</winner>

    Difference from Critic: Judge is verdict-oriented (just picks winner),
    Critic is diagnosis-oriented (identifies flaws first, then picks).
    """
    import requests

    results = []

    for tid, methods in paired.items():
        prompt = prompts.get(tid, "")
        cot_code = methods["cot"]["code"]
        astar_code = methods["astar"]["code"]
        cot_correct = methods["cot"]["correct"]
        astar_correct = methods["astar"]["correct"]

        judge_prompt = f"""Compare two candidate solutions for the same programming problem and decide which one solves it more faithfully and correctly.

Problem:
{prompt}

Solution A:
{cot_code}

Solution B:
{astar_code}

Evaluation criteria:
- Faithfulness: uses only the specified requirements and avoids invented constraints
- Logical validity: the code logic follows correctly from the problem statement
- Completeness: handles edge cases and all specified conditions

First reason internally about both solutions, then return ONLY the winner:
<winner>A</winner> or <winner>B</winner>"""

        try:
            resp = requests.post(LLM_ENDPOINT, json={
                "model": CRITIC_MODEL,
                "prompt": judge_prompt,
                "stream": False,
                "options": {"temperature": 0.0, "num_predict": 512}
            }, timeout=60)

            response_text = resp.json().get("response", "")

            # Parse winner
            if "<winner>B</winner>" in response_text:
                router_choice = "astar"
            else:
                router_choice = "cot"

        except Exception as e:
            router_choice = "cot"

        router_correct = (router_choice == "cot" and cot_correct) or \
                        (router_choice == "astar" and astar_correct)

        results.append({
            "task_id": tid,
            "cot_correct": cot_correct,
            "astar_correct": astar_correct,
            "router_choice": router_choice,
            "router_correct": router_correct
        })

    return results

# ═══════════════════════════════════════════════════════════════
# ROUTER 4: RANDOM FOREST (hand-crafted features, no LLM needed)
# ═══════════════════════════════════════════════════════════════

def run_random_forest(dataset, paired, prompts):
    """
    Random Forest Router:
    - Trained on hand-crafted features (question length, connectives, negation)
    - Uses 5-fold cross-validation (same as paper)
    - No LLM inference needed at routing time

    Features (same as paper Section 4.3):
    - question length (words)
    - number of logical connectives
    - number of negation words
    - number of comparison words
    - has multiple conditions (if/else/elif count)
    """

    CONNECTIVES = ["if", "then", "therefore", "because", "since", "although", "however", "but", "unless", "either", "neither", "both", "all", "some", "none"]
    NEGATION = ["not", "no", "never", "neither", "nor", "nothing", "nobody", "without", "fail", "impossible", "cannot"]
    COMPARISON = ["more", "less", "greater", "fewer", "larger", "smaller", "better", "worse", "most", "least", "than"]

    def extract_rf_features(prompt_text):
        text = prompt_text.lower()
        q_len = len(text.split())
        conn = sum(1 for c in CONNECTIVES if c in text)
        neg = sum(1 for n in NEGATION if n in text)
        cmp = sum(1 for c in COMPARISON if c in text)
        conditions = text.count("if ") + text.count("else") + text.count("elif")
        examples = text.count(">>>") + text.count("example")
        return [q_len, conn, neg, cmp, conditions, examples]

    # Build feature matrix and labels
    X, y, task_ids = [], [], []
    for tid, methods in paired.items():
        prompt = prompts.get(tid, "")
        if not prompt:
            continue
        feats = extract_rf_features(prompt)
        X.append(feats)

        cot_correct = methods["cot"]["correct"]
        astar_correct = methods["astar"]["correct"]

        # Label: 1 = A* is better choice, 0 = CoT is better choice
        if cot_correct and not astar_correct:
            y.append(0)
        elif astar_correct and not cot_correct:
            y.append(1)
        elif cot_correct and astar_correct:
            y.append(0)  # both correct -> prefer CoT (cheaper)
        else:
            y.append(0)  # both wrong -> default CoT

        task_ids.append(tid)

    X = np.array(X)
    y = np.array(y)

    # 5-fold cross-validation predictions
    predictions = np.zeros(len(y))

    if len(set(y)) >= 2 and sum(y == 1) >= 5:
        n_splits = min(5, min(sum(y == 0), sum(y == 1)))
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

        for train_idx, test_idx in cv.split(X, y):
            clf = RandomForestClassifier(n_estimators=300, max_depth=5, random_state=42)
            clf.fit(X[train_idx], y[train_idx])
            predictions[test_idx] = clf.predict(X[test_idx])

    # Build results
    results = []
    for i, tid in enumerate(task_ids):
        methods = paired[tid]
        cot_correct = methods["cot"]["correct"]
        astar_correct = methods["astar"]["correct"]

        router_choice = "astar" if predictions[i] == 1 else "cot"
        router_correct = (router_choice == "cot" and cot_correct) or \
                        (router_choice == "astar" and astar_correct)

        results.append({
            "task_id": tid,
            "cot_correct": cot_correct,
            "astar_correct": astar_correct,
            "router_choice": router_choice,
            "router_correct": router_correct
        })

    return results

# ═══════════════════════════════════════════════════════════════
# MAIN - RUN ALL
# ═══════════════════════════════════════════════════════════════

def save_results(results, dataset, router_name):
    """Save router results to JSON."""
    total = len(results)
    correct = sum(1 for r in results if r["router_correct"])
    cot_acc = sum(1 for r in results if r["cot_correct"]) / total * 100
    astar_acc = sum(1 for r in results if r["astar_correct"]) / total * 100
    oracle_acc = sum(1 for r in results if r["cot_correct"] or r["astar_correct"]) / total * 100

    output = {
        "router": router_name,
        "dataset": dataset,
        "model": "llama-3.1-8b",
        "results": results,
        "summary": {
            "total": total,
            "router_accuracy": round(correct / total * 100, 2),
            "cot_accuracy": round(cot_acc, 2),
            "astar_accuracy": round(astar_acc, 2),
            "oracle_accuracy": round(oracle_acc, 2)
        }
    }

    fname = f"{dataset}_{router_name}.json"
    path = os.path.join(OUTPUT_DIR, fname)
    with open(path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"  {fname}: router={output['summary']['router_accuracy']:.1f}%, "
          f"cot={cot_acc:.1f}%, astar={astar_acc:.1f}%, oracle={oracle_acc:.1f}%")


if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    prompts = load_prompts()

    datasets = ["humaneval", "mbpp", "codecontests"]

    for dataset in datasets:
        print(f"\n{'='*60}")
        print(f"  {dataset.upper()}")
        print(f"{'='*60}")

        paired = load_paired_results(dataset)
        dataset_prompts = prompts[dataset]

        # Router 4: Random Forest (no LLM needed - run first)
        print("\n  Running Random Forest...")
        rf_results = run_random_forest(dataset, paired, dataset_prompts)
        save_results(rf_results, dataset, "random_forest")

        # Router 1: Semantic Entropy (needs LLM for sampling)
        print("\n  Running Semantic Entropy...")
        try:
            se_results = run_semantic_entropy(dataset, paired, dataset_prompts)
            save_results(se_results, dataset, "semantic_entropy")
        except Exception as e:
            print(f"  SKIPPED (LLM not available): {e}")

        # Router 2: LLM-as-Critic (needs critic model)
        print("\n  Running LLM-as-Critic...")
        try:
            critic_results = run_llm_critic(dataset, paired, dataset_prompts)
            save_results(critic_results, dataset, "llm_critic")
        except Exception as e:
            print(f"  SKIPPED (LLM not available): {e}")

        # Router 3: LLM-as-Judge (needs judge model)
        print("\n  Running LLM-as-Judge...")
        try:
            judge_results = run_llm_judge(dataset, paired, dataset_prompts)
            save_results(judge_results, dataset, "llm_judge")
        except Exception as e:
            print(f"  SKIPPED (LLM not available): {e}")

    print("\n\nDONE. Results saved to:", OUTPUT_DIR)
    print("\nIf LLM-based routers were skipped, ensure Ollama is running with:")
    print(f"  ollama run {BACKBONE_MODEL}")
    print(f"  ollama run {CRITIC_MODEL}")

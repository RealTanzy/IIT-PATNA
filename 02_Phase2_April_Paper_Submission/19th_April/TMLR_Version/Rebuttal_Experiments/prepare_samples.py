#!/usr/bin/env python3
"""
Prepare Fixed Stratified Sample Sets for Rebuttal Experiments
=============================================================

Creates three fixed JSON files (one per dataset) that ALL experiments
will use, so every ablation condition is evaluated on exactly the same
questions. Run this ONCE before any other experiment script.

Output:
  samples/hotpotqa_300.json   — 300 HotpotQA validation questions
  samples/logicqa_200.json    — 200 LogicQA test questions
  samples/strategyqa_200.json — 200 StrategyQA test questions
"""

import json
import random
import os
from datasets import load_dataset

SEED = 42
random.seed(SEED)

os.makedirs("samples", exist_ok=True)


# ── HotpotQA ──────────────────────────────────────────────────────────────
def prepare_hotpotqa(n=300):
    print(f"Loading HotpotQA validation ({n} questions)...")
    ds = load_dataset("hotpot_qa", "distractor", split="validation",
                      trust_remote_code=True)
    items = []
    for ex in ds:
        ctx_pairs = list(zip(ex["context"]["title"], ex["context"]["sentences"]))
        context_str = "\n\n".join(
            f"Title: {t}\n" + " ".join(s) for t, s in ctx_pairs
        )
        items.append({
            "id": ex["id"],
            "question": ex["question"],
            "answer": ex["answer"],
            "type": ex["type"],
            "context": context_str,
        })

    # Stratify by type
    bridges = [x for x in items if x["type"] == "bridge"]
    comparisons = [x for x in items if x["type"] == "comparison"]

    n_bridge = int(n * 0.7)
    n_compare = n - n_bridge

    random.shuffle(bridges)
    random.shuffle(comparisons)

    selected = bridges[:n_bridge] + comparisons[:n_compare]
    random.shuffle(selected)

    for i, item in enumerate(selected):
        item["sample_idx"] = i

    with open("samples/hotpotqa_300.json", "w") as f:
        json.dump(selected, f, indent=2)
    print(f"  Saved {len(selected)} questions to samples/hotpotqa_300.json")
    type_counts = {}
    for x in selected:
        type_counts[x["type"]] = type_counts.get(x["type"], 0) + 1
    print(f"  Types: {type_counts}")


# ── LogicQA ───────────────────────────────────────────────────────────────
def prepare_logicqa(n=200):
    print(f"Loading LogicQA ({n} questions)...")
    ds = load_dataset("lucasmccabe/logiqa", split="test", trust_remote_code=True)
    items = []
    LABEL_MAP = {0: "A", 1: "B", 2: "C", 3: "D"}
    for i, ex in enumerate(ds):
        options = ex.get("options", [])
        if not options and "text" in ex:
            options = ex["text"]
        gold_idx = ex.get("correct_option", ex.get("label", ex.get("answer", 0)))
        items.append({
            "id": i,
            "context": ex.get("context", ex.get("passage", "")),
            "query": ex.get("query", ex.get("question", "")),
            "options": options,
            "gold": LABEL_MAP.get(int(gold_idx), str(gold_idx)),
        })

    random.shuffle(items)
    selected = items[:n]
    for i, item in enumerate(selected):
        item["sample_idx"] = i

    with open("samples/logicqa_200.json", "w") as f:
        json.dump(selected, f, indent=2)
    print(f"  Saved {len(selected)} questions to samples/logicqa_200.json")


# ── StrategyQA ─────────────────────────────────────────────────────────────
def prepare_strategyqa(n=200):
    print(f"Loading StrategyQA ({n} questions)...")
    ds = load_dataset("wics/strategy-qa", split="test")
    items = []
    for i, ex in enumerate(ds):
        answer = ex.get("answer", ex.get("label", False))
        if isinstance(answer, (int, bool)):
            answer = bool(answer)
        elif isinstance(answer, str):
            answer = answer.lower() in ("true", "yes", "1")
        items.append({
            "id": i,
            "question": ex.get("question", ""),
            "answer": "yes" if bool(answer) else "no",
        })

    random.shuffle(items)
    selected = items[:n]
    for i, item in enumerate(selected):
        item["sample_idx"] = i

    with open("samples/strategyqa_200.json", "w") as f:
        json.dump(selected, f, indent=2)
    print(f"  Saved {len(selected)} questions to samples/strategyqa_200.json")
    pos = sum(1 for x in selected if x["answer"] == "yes")
    print(f"  Balance: {pos} yes / {n - pos} no")


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    prepare_hotpotqa(300)
    prepare_logicqa(200)
    prepare_strategyqa(200)
    print("\nDone. All sample files created in samples/")

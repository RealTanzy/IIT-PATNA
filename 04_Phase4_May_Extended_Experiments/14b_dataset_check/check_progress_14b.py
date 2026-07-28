#!/usr/bin/env python3
import json
import os

ROOT = "/home/dibyanayan/tanzeel/14b_dataset_check/results_14b"
TOTALS = {"strategyqa": 2290, "hotpotqa": 2290, "logicqa": 651}

for ds in ["strategyqa", "hotpotqa", "logicqa"]:
    for method in ["cot", "astar"]:
        p = os.path.join(ROOT, ds, method, "results.json")
        tag = f"{ds}/{method}"
        if not os.path.exists(p):
            print(f"{tag}: not started")
            continue
        try:
            with open(p) as f:
                d = json.load(f)
            rows = d if isinstance(d, list) else d.get("results", [])
            done = len(rows)
            total = TOTALS[ds]
            correct = sum(1 for r in rows if r.get("correct"))
            acc = (correct / done * 100) if done else 0.0
            print(f"{tag}: {done}/{total} done | acc={acc:.2f}%")
        except Exception as e:
            print(f"{tag}: read error ({e})")

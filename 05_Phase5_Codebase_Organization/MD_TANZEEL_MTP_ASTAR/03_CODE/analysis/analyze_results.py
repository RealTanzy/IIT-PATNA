#!/usr/bin/env python3
"""
Analyze Rebuttal Experiment Results
=====================================

Reads result JSONs from results/ and generates Markdown tables
suitable for copy-pasting into the rebuttal response files.

Usage:
  python3 analyze_results.py             # all experiments
  python3 analyze_results.py --exp heuristic
  python3 analyze_results.py --exp budget
  python3 analyze_results.py --exp branching
  python3 analyze_results.py --exp summary  # complete summary table
"""

import os
import json
import argparse
from typing import Dict, Optional

SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(SCRIPT_DIR, "results")

# Paper baselines (from Table 1, full dataset, LLaMA 3.1-8B)
PAPER_BASELINES = {
    "hotpotqa": {"cot": 76.8, "astar": 80.1},
    "logicqa":  {"cot": 45.8, "astar": 44.9},
    "strategyqa": {"cot": 64.5, "astar": 74.8},
}


def load_json(path: str) -> Optional[dict]:
    if not os.path.exists(path):
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except Exception as e:
        print(f"  Warning: could not load {path}: {e}")
        return None


def fmt(acc: Optional[float], baseline: Optional[float] = None) -> str:
    if acc is None:
        return "—"
    s = f"{acc:.2f}%"
    if baseline is not None:
        delta = acc - baseline
        sign  = "+" if delta >= 0 else ""
        s    += f" ({sign}{delta:.1f}pp)"
    return s


# ─── Exp 1: Heuristic ablation ────────────────────────────────────────────────

def analyze_heuristic():
    print("\n" + "=" * 60)
    print("EXPERIMENT 1: Heuristic Ablation")
    print("BFS (h=0) vs Greedy Best-First (f=h) vs Full A* (f=g+h)")
    print("=" * 60)

    datasets = ["hotpotqa", "logicqa", "strategyqa"]
    modes    = ["bfs", "greedy", "astar"]
    mode_labels = {"bfs": "BFS (h=0)", "greedy": "Greedy (f=h)", "astar": "Full A* (f=g+h)"}

    rows = []
    for dataset in datasets:
        row = {"dataset": dataset.capitalize()}
        base_astar = PAPER_BASELINES.get(dataset, {}).get("astar")
        for mode in modes:
            path = os.path.join(RESULTS_DIR, f"{dataset}_heuristic_{mode}.json")
            d = load_json(path)
            row[mode] = d["accuracy"] if d else None
        rows.append(row)

    # Print markdown table
    print("\nResults (rebuttal subset):\n")
    print(f"| Dataset     | BFS (h=0) | Greedy (f=h) | Full A* (f=g+h) | Paper A* |")
    print(f"|-------------|-----------|--------------|-----------------|----------|")
    for row in rows:
        dataset = row["dataset"]
        ds_key  = dataset.lower()
        paper   = PAPER_BASELINES.get(ds_key, {}).get("astar")
        bfs     = fmt(row.get("bfs"))
        greedy  = fmt(row.get("greedy"))
        astar   = fmt(row.get("astar"))
        paper_s = f"{paper:.1f}% (full)" if paper else "—"
        print(f"| {dataset:<11} | {bfs:<9} | {greedy:<12} | {astar:<15} | {paper_s:<8} |")

    # Interpretation
    print("\n**Interpretation:**")
    for row in rows:
        ds_key = row["dataset"].lower()
        astar_acc = row.get("astar")
        bfs_acc   = row.get("bfs")
        if astar_acc is not None and bfs_acc is not None:
            delta = astar_acc - bfs_acc
            sign  = "+" if delta >= 0 else ""
            print(f"  {row['dataset']}: Full A* vs BFS: {sign}{delta:.1f}pp "
                  f"→ heuristic {'helps' if delta > 0 else 'hurts' if delta < 0 else 'neutral'}")

    # Write to file
    out_path = os.path.join(RESULTS_DIR, "exp1_heuristic_table.md")
    _write_heuristic_md(rows, out_path)
    print(f"\nFull table saved to: {out_path}")


def _write_heuristic_md(rows, path):
    lines = [
        "# Experiment 1: Heuristic Ablation Results\n",
        "**Research question:** Does the critic-based heuristic actually contribute to performance,",
        "or does A*'s benefit come purely from branching?\n",
        "**Conditions:**",
        "- BFS (h=0): heuristic disabled, A* degenerates to uniform-cost search",
        "- Greedy (f=h): path cost ignored, only heuristic guides expansion",
        "- Full A* (f=g+h): standard setting (paper configuration)\n",
        "| Dataset     | BFS (h=0) | Greedy (f=h) | Full A* (f=g+h) | Paper A* (full dataset) |",
        "|-------------|-----------|--------------|-----------------|-------------------------|",
    ]
    for row in rows:
        ds_key = row["dataset"].lower()
        paper  = PAPER_BASELINES.get(ds_key, {}).get("astar")
        bfs    = fmt(row.get("bfs"))
        greedy = fmt(row.get("greedy"))
        astar  = fmt(row.get("astar"))
        paper_s = f"{paper:.1f}% (n=2151)" if paper else "—"
        lines.append(f"| {row['dataset']:<11} | {bfs:<9} | {greedy:<12} | {astar:<15} | {paper_s:<23} |")

    lines += [
        "\n**Key finding:** [Fill in after running]",
        "The gap between Full A* and BFS (h=0) quantifies the contribution of the",
        "critic-based heuristic independently of the branching structure.",
    ]
    with open(path, "w") as f:
        f.write("\n".join(lines))


# ─── Exp 2: Budget sensitivity ────────────────────────────────────────────────

def analyze_budget():
    print("\n" + "=" * 60)
    print("EXPERIMENT 2: Budget Sensitivity (B = max nodes)")
    print("Fixed: heuristic=astar, branch_k=3 | Dataset: HotpotQA")
    print("=" * 60)

    budgets = [5, 10, 15, 20, 30]
    rows    = []
    for B in budgets:
        path = os.path.join(RESULTS_DIR, f"hotpotqa_budget_B{B}.json")
        d    = load_json(path)
        rows.append({
            "budget":    B,
            "accuracy":  d["accuracy"]     if d else None,
            "avg_nodes": d["avg_nodes"]    if d else None,
            "tokens":    d.get("total_tokens") if d else None,
        })

    print("\nResults:\n")
    print(f"| Budget B | Accuracy (%) | Avg Nodes | Total Tokens |")
    print(f"|----------|-------------|-----------|--------------|")
    for r in rows:
        acc    = f"{r['accuracy']:.2f}" if r["accuracy"] is not None else "—"
        nodes  = f"{r['avg_nodes']:.1f}" if r["avg_nodes"] is not None else "—"
        toks   = f"{r['tokens']:,}"      if r["tokens"]    is not None else "—"
        print(f"| {r['budget']:<8} | {acc:<11} | {nodes:<9} | {toks:<12} |")

    # Compute gains per token
    print("\n**Interpretation:**")
    base_row = next((r for r in rows if r["budget"] == 5), None)
    for r in rows:
        if r["accuracy"] is not None and base_row and base_row["accuracy"] is not None:
            delta = r["accuracy"] - base_row["accuracy"]
            sign  = "+" if delta >= 0 else ""
            print(f"  B={r['budget']}: {r['accuracy']:.2f}% ({sign}{delta:.1f}pp vs B=5)")

    out_path = os.path.join(RESULTS_DIR, "exp2_budget_table.md")
    _write_budget_md(rows, out_path)
    print(f"\nFull table saved to: {out_path}")


def _write_budget_md(rows, path):
    lines = [
        "# Experiment 2: Budget Sensitivity Results\n",
        "**Research question:** How does accuracy vary with search budget B?",
        "Does increasing budget give diminishing returns?\n",
        "**Fixed:** heuristic=astar, branch_k=3, dataset=HotpotQA (300 questions)\n",
        "| Budget B | Accuracy (%) | Avg Nodes Used | Est. Tokens |",
        "|----------|-------------|----------------|-------------|",
    ]
    for r in rows:
        acc   = f"{r['accuracy']:.2f}" if r["accuracy"] is not None else "—"
        nodes = f"{r['avg_nodes']:.1f}" if r["avg_nodes"] is not None else "—"
        toks  = f"{r['tokens']:,}"      if r["tokens"] is not None else "—"
        lines.append(f"| {r['budget']:<8} | {acc:<11} | {nodes:<14} | {toks:<11} |")
    lines += [
        "\n**Key finding:** [Fill in after running]",
        "Expected: accuracy increases then plateaus, confirming B=30 is near-optimal.",
        "Shows compute–accuracy trade-off explicitly.",
    ]
    with open(path, "w") as f:
        f.write("\n".join(lines))


# ─── Exp 3: Branching sensitivity ─────────────────────────────────────────────

def analyze_branching():
    print("\n" + "=" * 60)
    print("EXPERIMENT 3: Branching Factor Sensitivity (K = candidates/expansion)")
    print("Fixed: heuristic=astar, budget=30 | Dataset: HotpotQA")
    print("=" * 60)

    ks   = [1, 2, 3, 4]
    rows = []
    for K in ks:
        path = os.path.join(RESULTS_DIR, f"hotpotqa_branchK{K}.json")
        d    = load_json(path)
        rows.append({
            "k":         K,
            "accuracy":  d["accuracy"]     if d else None,
            "avg_nodes": d["avg_nodes"]    if d else None,
            "tokens":    d.get("total_tokens") if d else None,
        })

    print("\nResults:\n")
    print(f"| Branch K | Accuracy (%) | Avg Nodes | Total Tokens |")
    print(f"|----------|-------------|-----------|--------------|")
    for r in rows:
        acc   = f"{r['accuracy']:.2f}" if r["accuracy"] is not None else "—"
        nodes = f"{r['avg_nodes']:.1f}" if r["avg_nodes"] is not None else "—"
        toks  = f"{r['tokens']:,}"      if r["tokens"] is not None else "—"
        label = "(linear)" if r["k"] == 1 else "(paper)" if r["k"] == 3 else ""
        print(f"| K={r['k']:<6} | {acc:<11} | {nodes:<9} | {toks:<12} | {label}")

    out_path = os.path.join(RESULTS_DIR, "exp3_branching_table.md")
    _write_branching_md(rows, out_path)
    print(f"\nFull table saved to: {out_path}")


def _write_branching_md(rows, path):
    lines = [
        "# Experiment 3: Branching Factor Sensitivity Results\n",
        "**Research question:** How much does branching contribute?",
        "K=1 is the degenerate linear-chain case (no real search).\n",
        "**Fixed:** heuristic=astar, budget=30, dataset=HotpotQA (300 questions)\n",
        "| Branch K | Accuracy (%) | Avg Nodes | Est. Tokens | Note |",
        "|----------|-------------|-----------|-------------|------|",
    ]
    for r in rows:
        acc   = f"{r['accuracy']:.2f}" if r["accuracy"] is not None else "—"
        nodes = f"{r['avg_nodes']:.1f}" if r["avg_nodes"] is not None else "—"
        toks  = f"{r['tokens']:,}"      if r["tokens"] is not None else "—"
        note  = "linear (no branching)" if r["k"] == 1 else "paper setting" if r["k"] == 3 else ""
        lines.append(f"| K={r['k']:<6} | {acc:<11} | {nodes:<9} | {toks:<11} | {note} |")
    lines += [
        "\n**Key finding:** [Fill in after running]",
        "K=1 vs K=3 delta shows how much real branching contributes.",
    ]
    with open(path, "w") as f:
        f.write("\n".join(lines))


# ─── Complete summary for rebuttal ────────────────────────────────────────────

def analyze_summary():
    print("\n" + "=" * 60)
    print("COMPLETE REBUTTAL EXPERIMENT SUMMARY")
    print("=" * 60)
    analyze_heuristic()
    analyze_budget()
    analyze_branching()

    # Write combined rebuttal appendix
    out_path = os.path.join(RESULTS_DIR, "REBUTTAL_APPENDIX.md")
    lines = ["# Rebuttal Experiment Results — Appendix\n\n"]
    for fname in ["exp1_heuristic_table.md", "exp2_budget_table.md", "exp3_branching_table.md"]:
        fpath = os.path.join(RESULTS_DIR, fname)
        if os.path.exists(fpath):
            with open(fpath) as f:
                lines.append(f.read())
            lines.append("\n\n---\n\n")
    with open(out_path, "w") as f:
        f.write("\n".join(lines))
    print(f"\nCombined appendix saved to: {out_path}")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp", choices=["heuristic", "budget", "branching", "summary", "all"],
                        default="all")
    args = parser.parse_args()

    if args.exp in ("heuristic", "all"):
        analyze_heuristic()
    if args.exp in ("budget", "all"):
        analyze_budget()
    if args.exp in ("branching", "all"):
        analyze_branching()
    if args.exp == "summary":
        analyze_summary()

    print("\nDone.")


if __name__ == "__main__":
    main()

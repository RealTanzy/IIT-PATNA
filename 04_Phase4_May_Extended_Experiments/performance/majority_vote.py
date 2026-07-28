#!/usr/bin/env python3
"""
Majority Vote Analysis — Performance Experiment
================================================
Uses all 6 runs on the 93-question disagreement set:

  CoT  Run-1  : original result  (disagreement.json      → cot_pred)
  CoT  Run-2  : re-run           (cot_results.json       → pred)
  CoT  Run-3  : re-run           (cot_results_run3.json  → pred)
  A*   Run-1  : original result  (disagreement.json      → astar_pred)
  A*   Run-2  : re-run           (astar_results.json     → pred)
  A*   Run-3  : re-run           (astar_results_run3.json→ pred)

Majority vote per method (3 runs, odd → no ties):
  2-of-3 or 3-of-3 agreement → that answer wins

USAGE:
  python3 majority_vote.py
  python3 majority_vote.py --save   # also writes majority_vote_results.json
"""

import json, re, sys, os, argparse
from collections import Counter

DIR  = os.path.dirname(os.path.abspath(__file__))
SEP  = "=" * 65
SEP2 = "─" * 65


def load():
    paths = {
        "d"     : os.path.join(DIR, "disagreement.json"),
        "cot_r2": os.path.join(DIR, "cot_results.json"),
        "cot_r3": os.path.join(DIR, "cot_results_run3.json"),
        "ast_r2": os.path.join(DIR, "astar_results.json"),
        "ast_r3": os.path.join(DIR, "astar_results_run3.json"),
    }
    for label, p in paths.items():
        if not os.path.exists(p):
            print(f"Missing file [{label}]: {p}"); sys.exit(1)
    return (json.load(open(paths["d"])),
            json.load(open(paths["cot_r2"])),
            json.load(open(paths["cot_r3"])),
            json.load(open(paths["ast_r2"])),
            json.load(open(paths["ast_r3"])))


def norm(pred):
    if pred is None: return None
    p = pred.lower().strip().rstrip(".")
    if re.search(r'\byes\b', p): return 'yes'
    if re.search(r'\bno\b',  p): return 'no'
    return None


def majority(votes):
    """Return (answer, is_tie). With 3 odd votes, ties only happen if all 3 differ."""
    valid = [v for v in votes if v is not None]
    if not valid: return None, True
    c = Counter(valid)
    top = c.most_common()
    if top[0][1] > 1:
        return top[0][0], False   # 2-of-3 or 3-of-3
    if len(top) == 1:
        return top[0][0], False   # only one unique answer
    return None, True             # genuine 3-way split (yes/no/None)


def em(pred, gold):
    return bool(pred) and pred.lower().strip() == gold.lower().strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--save", action="store_true",
                        help="Save per-question records to majority_vote_results.json")
    args = parser.parse_args()

    d_raw, cot_r2_raw, cot_r3_raw, ast_r2_raw, ast_r3_raw = load()

    cot_r2_map = {r["question"]: norm(r["pred"]) for r in cot_r2_raw}
    cot_r3_map = {r["question"]: norm(r["pred"]) for r in cot_r3_raw}
    ast_r2_map = {r["question"]: norm(r["pred"]) for r in ast_r2_raw}
    ast_r3_map = {r["question"]: norm(r["pred"]) for r in ast_r3_raw}

    # ── Build per-question records ─────────────────────────────────────
    records = []
    for item in d_raw:
        q    = item["question"]
        gold = item["gold"].lower().strip()

        c1 = norm(item["cot_pred"]);   c2 = cot_r2_map.get(q); c3 = cot_r3_map.get(q)
        a1 = norm(item["astar_pred"]); a2 = ast_r2_map.get(q); a3 = ast_r3_map.get(q)

        cot_maj,  cot_tie  = majority([c1, c2, c3])
        astr_maj, astr_tie = majority([a1, a2, a3])

        records.append({
            "question"          : q,
            "gold"              : gold,
            "cot_run1"          : c1,
            "cot_run2"          : c2,
            "cot_run3"          : c3,
            "astar_run1"        : a1,
            "astar_run2"        : a2,
            "astar_run3"        : a3,
            "cot_majority"      : cot_maj,
            "astar_majority"    : astr_maj,
            "cot_tie"           : cot_tie,
            "astar_tie"         : astr_tie,
            "cot_r1_correct"    : em(c1,       gold),
            "cot_r2_correct"    : em(c2,       gold),
            "cot_r3_correct"    : em(c3,       gold),
            "astar_r1_correct"  : em(a1,       gold),
            "astar_r2_correct"  : em(a2,       gold),
            "astar_r3_correct"  : em(a3,       gold),
            "cot_maj_correct"   : em(cot_maj,  gold),
            "astar_maj_correct" : em(astr_maj, gold),
        })

    n = len(records)

    # ── Aggregate ─────────────────────────────────────────────────────
    c_r1 = sum(r["cot_r1_correct"]   for r in records)
    c_r2 = sum(r["cot_r2_correct"]   for r in records)
    c_r3 = sum(r["cot_r3_correct"]   for r in records)
    a_r1 = sum(r["astar_r1_correct"] for r in records)
    a_r2 = sum(r["astar_r2_correct"] for r in records)
    a_r3 = sum(r["astar_r3_correct"] for r in records)

    c_maj = sum(r["cot_maj_correct"]   for r in records)
    a_maj = sum(r["astar_maj_correct"] for r in records)

    c_ties = sum(r["cot_tie"]   for r in records)
    a_ties = sum(r["astar_tie"] for r in records)

    c_all3 = sum(1 for r in records
                 if r["cot_run1"] == r["cot_run2"] == r["cot_run3"]
                 and r["cot_run1"] is not None)
    a_all3 = sum(1 for r in records
                 if r["astar_run1"] == r["astar_run2"] == r["astar_run3"]
                 and r["astar_run1"] is not None)
    c_2of3 = sum(1 for r in records if not r["cot_tie"])   - c_all3
    a_2of3 = sum(1 for r in records if not r["astar_tie"]) - a_all3

    both_ok  = sum(1 for r in records if r["cot_maj_correct"] and r["astar_maj_correct"])
    both_bad = sum(1 for r in records if not r["cot_maj_correct"] and not r["astar_maj_correct"])
    only_cot = sum(1 for r in records if r["cot_maj_correct"]  and not r["astar_maj_correct"])
    only_ast = sum(1 for r in records if not r["cot_maj_correct"] and r["astar_maj_correct"])

    c_fixed = sum(1 for r in records if not r["cot_r1_correct"]  and r["cot_maj_correct"]  and not r["cot_tie"])
    c_broke = sum(1 for r in records if r["cot_r1_correct"]      and not r["cot_maj_correct"] and not r["cot_tie"])
    a_fixed = sum(1 for r in records if not r["astar_r1_correct"] and r["astar_maj_correct"] and not r["astar_tie"])
    a_broke = sum(1 for r in records if r["astar_r1_correct"]     and not r["astar_maj_correct"] and not r["astar_tie"])

    # ── Print ─────────────────────────────────────────────────────────
    print(); print(SEP)
    print("  Majority Vote Analysis — 6 Runs on 93 Disagreement Questions")
    print(SEP)
    print(f"\n  Dataset   : StrategyQA (yes/no)")
    print(f"  Questions : {n}  (CoT != A* disagreement set)")
    print(f"  Runs      : 3 x CoT  +  3 x A*  =  6 total")

    print(f"\n  {SEP2}")
    print(f"  INDIVIDUAL RUN ACCURACY")
    print(f"  {SEP2}")
    print(f"  CoT  Run-1 (original) : {c_r1:2d} / {n}  ({c_r1/n*100:.1f}%)")
    print(f"  CoT  Run-2 (re-run)   : {c_r2:2d} / {n}  ({c_r2/n*100:.1f}%)")
    print(f"  CoT  Run-3 (re-run)   : {c_r3:2d} / {n}  ({c_r3/n*100:.1f}%)")
    print(f"  A*   Run-1 (original) : {a_r1:2d} / {n}  ({a_r1/n*100:.1f}%)")
    print(f"  A*   Run-2 (re-run)   : {a_r2:2d} / {n}  ({a_r2/n*100:.1f}%)")
    print(f"  A*   Run-3 (re-run)   : {a_r3:2d} / {n}  ({a_r3/n*100:.1f}%)")

    print(f"\n  {SEP2}")
    print(f"  CONSISTENCY ACROSS 3 RUNS")
    print(f"  {SEP2}")
    print(f"  CoT  all-3 agree : {c_all3:2d} / {n}  ({c_all3/n*100:.1f}%)")
    print(f"  CoT  2-of-3      : {c_2of3:2d} / {n}  ({c_2of3/n*100:.1f}%)  <- majority decides")
    print(f"  CoT  3-way tie   : {c_ties:2d} / {n}  ({c_ties/n*100:.1f}%)  <- no clear answer")
    print(f"  A*   all-3 agree : {a_all3:2d} / {n}  ({a_all3/n*100:.1f}%)")
    print(f"  A*   2-of-3      : {a_2of3:2d} / {n}  ({a_2of3/n*100:.1f}%)  <- majority decides")
    print(f"  A*   3-way tie   : {a_ties:2d} / {n}  ({a_ties/n*100:.1f}%)  <- no clear answer")

    print(f"\n  {SEP2}")
    print(f"  MAJORITY VOTE ACCURACY  (best-of-3 per method)")
    print(f"  {SEP2}")
    print(f"  CoT  majority : {c_maj:2d} / {n}  ({c_maj/n*100:.1f}%)")
    print(f"  A*   majority : {a_maj:2d} / {n}  ({a_maj/n*100:.1f}%)")

    c_best = max(c_r1, c_r2, c_r3)
    a_best = max(a_r1, a_r2, a_r3)
    c_delta = c_maj - c_best
    a_delta = a_maj - a_best
    print(f"\n  CoT  majority vs best individual : {c_maj} vs {c_best}  "
          f"({'up +' if c_delta >= 0 else 'down '}{abs(c_delta)})")
    print(f"  A*   majority vs best individual : {a_maj} vs {a_best}  "
          f"({'up +' if a_delta >= 0 else 'down '}{abs(a_delta)})")

    print(f"\n  {SEP2}")
    print(f"  HEAD-TO-HEAD  (majority-CoT vs majority-A*)")
    print(f"  {SEP2}")
    winner = ("CoT" if c_maj > a_maj else "A*" if a_maj > c_maj else "TIE")
    print(f"  CoT majority : {c_maj:2d} / {n}  ({c_maj/n*100:.1f}%)")
    print(f"  A*  majority : {a_maj:2d} / {n}  ({a_maj/n*100:.1f}%)")
    if winner == "TIE":
        print(f"  RESULT: TIE")
    else:
        diff = abs(c_maj - a_maj)
        print(f"  WINNER: {winner}  (+{diff} questions, +{diff/n*100:.1f}%)")

    print(f"\n  On the majority-voted predictions:")
    print(f"  Both correct   : {both_ok}")
    print(f"  Both wrong     : {both_bad}  <- consistently hard")
    print(f"  Only CoT right : {only_cot}")
    print(f"  Only A*  right : {only_ast}")

    print(f"\n  {SEP2}")
    print(f"  MAJORITY VOTE EFFECT  (vs original Run-1)")
    print(f"  {SEP2}")
    print(f"  CoT : majority fixed {c_fixed}  previously-wrong,  broke {c_broke}  previously-right")
    print(f"  A*  : majority fixed {a_fixed} previously-wrong,  broke {a_broke} previously-right")

    print(); print(SEP)

    if args.save:
        out = os.path.join(DIR, "majority_vote_results.json")
        json.dump(records, open(out, "w"), indent=2)
        print(f"  Saved -> {out}")
        print(SEP)


if __name__ == "__main__":
    main()

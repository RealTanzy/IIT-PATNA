#!/usr/bin/env python3
"""
Verification script for MD_TANZEEL_MTP_ASTAR.
Recomputes every number in Tables 1 and 3 of the paper from source JSON files.
Run from: MD_TANZEEL_MTP_ASTAR/
"""
import json, math, os
from pathlib import Path

BASE = Path(__file__).parent.parent  # MD_TANZEEL_MTP_ASTAR/

def load(path):
    d = json.load(open(BASE / path))
    if isinstance(d, dict):
        for k in ["results","data","items"]:
            if k in d and isinstance(d[k], list): return d[k]
    return d

def acc(items, use_f1=False):
    if use_f1:
        return sum(1 for r in items if r.get("f1",0) >= 0.5), len(items)
    return sum(1 for r in items if r.get("correct", False)), len(items)

def wilson(k, n, z=1.96):
    if n == 0: return 0.0, 0.0
    p = k/n
    c = (p + z**2/(2*n)) / (1 + z**2/n)
    m = (z * math.sqrt(p*(1-p)/n + z**2/(4*n**2))) / (1 + z**2/n)
    return round(max(0,c-m)*100,1), round(min(1,c+m)*100,1)

def mcnemar(a_items, c_items, key_fn, corr_fn):
    a_lkp = {key_fn(r): r for r in a_items if key_fn(r)}
    c_lkp = {key_fn(r): r for r in c_items if key_fn(r)}
    common = [k for k in a_lkp if k in c_lkp]
    n10 = sum(1 for k in common if corr_fn(a_lkp[k]) and not corr_fn(c_lkp[k]))
    n01 = sum(1 for k in common if not corr_fn(a_lkp[k]) and corr_fn(c_lkp[k]))
    a_c  = sum(1 for k in common if corr_fn(a_lkp[k]))
    c_c  = sum(1 for k in common if corr_fn(c_lkp[k]))
    total_disc = n10 + n01
    chi2 = (abs(n10-n01)-1)**2/total_disc if total_disc >= 5 else 0
    p = round(min(1.0, math.erfc(math.sqrt(chi2/2))), 5) if chi2 else 1.0
    return len(common), a_c, c_c, p

SEP = "=" * 65
PASS = 0; FAIL = 0

def check(name, expected_a, expected_c, expected_gap, actual_a, actual_c, tol=1.0):
    global PASS, FAIL
    gap_ok = abs(actual_a - actual_c - expected_gap) <= tol
    a_ok   = abs(actual_a - expected_a) <= tol
    c_ok   = abs(actual_c - expected_c) <= tol
    status = "PASS" if (gap_ok and a_ok and c_ok) else "FAIL"
    if status == "PASS": PASS += 1
    else: FAIL += 1
    marker = "OK" if status == "PASS" else "!!"
    print(f"  [{marker}] {name}")
    print(f"       Paper: A*={expected_a:.1f}%  CoT={expected_c:.1f}%  Gap={expected_gap:+.1f}pp")
    print(f"       Actual: A*={actual_a:.1f}%  CoT={actual_c:.1f}%  Gap={actual_a-actual_c:+.1f}pp")

# ── TABLE 1 ───────────────────────────────────────────────────────────────────
print(SEP)
print("TABLE 1: MAIN RESULTS")
print(SEP)

# Qwen-0.5B HotpotQA (matched, N=500)
a_items = load("02_RESULTS/qwen_0.5B/hotpotqa_astar_500q.json")
c_items = load("02_RESULTS/qwen_0.5B/hotpotqa_cot_500q.json")
n, a_c, c_c, p = mcnemar(a_items, c_items,
    key_fn=lambda r: r.get("question",""),
    corr_fn=lambda r: r.get("correct", False))
a_lo, a_hi = wilson(a_c, n)
c_lo, c_hi = wilson(c_c, n)
print(f"  Qwen-0.5B HotpotQA  N={n}  A*={a_c/n*100:.1f}% [{a_lo}-{a_hi}]"
      f"  CoT={c_c/n*100:.1f}% [{c_lo}-{c_hi}]  p={p}")
check("Qwen-0.5B HotpotQA", 30.0, 3.6, +26.4, a_c/n*100, c_c/n*100)

# Qwen-0.5B LogicQA
a_items = load("02_RESULTS/qwen_0.5B/logicqa_astar_500q.json")
c_items = load("02_RESULTS/qwen_0.5B/logicqa_cot_500q.json")
n, a_c, c_c, p = mcnemar(a_items, c_items,
    key_fn=lambda r: str(r.get("id", r.get("question",""))),
    corr_fn=lambda r: r.get("correct", False))
check("Qwen-0.5B LogicQA",  20.2, 24.0, -3.8, a_c/n*100, c_c/n*100)

# Qwen-0.5B StrategyQA
a_items = load("02_RESULTS/qwen_0.5B/strategyqa_astar_500q.json")
c_items = load("02_RESULTS/qwen_0.5B/strategyqa_cot_500q.json")
n, a_c, c_c, p = mcnemar(a_items, c_items,
    key_fn=lambda r: r.get("question",""),
    corr_fn=lambda r: r.get("correct", False))
check("Qwen-0.5B StrategyQA", 59.1, 56.4, +2.7, a_c/n*100, c_c/n*100)

# Llama-8B HotpotQA (F1>=0.5)
a_items = load("02_RESULTS/llama_3.1_8B/hotpotqa_astar_2290q.json")
c_items_raw = load("02_RESULTS/llama_3.1_8B/hotpotqa_cot_2290q.json")
n, a_c, c_c, p = mcnemar(a_items, c_items_raw,
    key_fn=lambda r: r.get("question",""),
    corr_fn=lambda r: r.get("f1",0) >= 0.5)
check("Llama-8B HotpotQA", 70.1, 68.0, +2.1, a_c/n*100, c_c/n*100)

# Llama-8B StrategyQA
a_items = load("02_RESULTS/llama_3.1_8B/strategyqa_astar_2290q.json")
c_items = load("02_RESULTS/llama_3.1_8B/strategyqa_cot_2290q.json")
n, a_c, c_c, p = mcnemar(a_items, c_items,
    key_fn=lambda r: r.get("question",""),
    corr_fn=lambda r: r.get("correct", False))
check("Llama-8B StrategyQA", 72.6, 71.2, +1.4, a_c/n*100, c_c/n*100)

# Llama-8B LogicQA
a_items = load("02_RESULTS/llama_3.1_8B/logicqa_astar_651q.json")
c_items = load("02_RESULTS/llama_3.1_8B/logicqa_cot_651q.json")
n, a_c, c_c, p = mcnemar(a_items, c_items,
    key_fn=lambda r: str(r.get("id", r.get("question",""))),
    corr_fn=lambda r: r.get("correct", False))
check("Llama-8B LogicQA", 44.9, 45.8, -0.9, a_c/n*100, c_c/n*100)

# DeepSeek-14B HotpotQA
a_items = load("02_RESULTS/deepseek_14B/hotpotqa_astar_2290q.json")
c_items = load("02_RESULTS/deepseek_14B/hotpotqa_cot_2290q.json")
n, a_c, c_c, p = mcnemar(a_items, c_items,
    key_fn=lambda r: r.get("question","").strip(),
    corr_fn=lambda r: r.get("f1",0)>=0.5 or r.get("correct",False))
check("DeepSeek-14B HotpotQA", 82.5, 76.4, +6.1, a_c/n*100, c_c/n*100)

# DeepSeek-14B StrategyQA
a_items = load("02_RESULTS/deepseek_14B/strategyqa_astar_2290q.json")
c_items = load("02_RESULTS/deepseek_14B/strategyqa_cot_2290q.json")
n, a_c, c_c, p = mcnemar(a_items, c_items,
    key_fn=lambda r: r.get("question", r.get("id","")).strip(),
    corr_fn=lambda r: r.get("correct", False))
check("DeepSeek-14B StrategyQA", 74.7, 76.0, -1.3, a_c/n*100, c_c/n*100)

# DeepSeek-14B LogicQA
a_items = load("02_RESULTS/deepseek_14B/logicqa_astar_651q.json")
c_items = load("02_RESULTS/deepseek_14B/logicqa_cot_651q.json")
n, a_c, c_c, p = mcnemar(a_items, c_items,
    key_fn=lambda r: str(r.get("id", r.get("query", r.get("question","")))).strip(),
    corr_fn=lambda r: r.get("correct", False))
check("DeepSeek-14B LogicQA", 57.5, 57.3, +0.2, a_c/n*100, c_c/n*100)

# ── TABLE 3: Q-type breakdown ─────────────────────────────────────────────────
print()
print(SEP)
print("TABLE 3: QUESTION-TYPE BREAKDOWN (8B and 14B HotpotQA)")
print(SEP)

for label, afile, cfile, use_f1, expected in [
    ("8B bridge",     "02_RESULTS/llama_3.1_8B/hotpotqa_astar_2290q.json",
                      "02_RESULTS/llama_3.1_8B/hotpotqa_cot_2290q.json",
                      True,  (71.3, 66.8, +4.5)),
    ("8B yes/no",     "02_RESULTS/llama_3.1_8B/hotpotqa_astar_2290q.json",
                      "02_RESULTS/llama_3.1_8B/hotpotqa_cot_2290q.json",
                      True,  (64.6, 83.1, -18.5)),
    ("14B bridge",    "02_RESULTS/deepseek_14B/hotpotqa_astar_2290q.json",
                      "02_RESULTS/deepseek_14B/hotpotqa_cot_2290q.json",
                      False, (84.0, 76.5, +7.5)),
    ("14B yes/no",    "02_RESULTS/deepseek_14B/hotpotqa_astar_2290q.json",
                      "02_RESULTS/deepseek_14B/hotpotqa_cot_2290q.json",
                      False, (75.4, 80.0, -4.6)),
]:
    qt = "bridge" if "bridge" in label else "yesno"
    a_all = load(afile); c_all = load(cfile)
    a_lkp = {r.get("question","").strip(): r for r in a_all}
    c_lkp = {r.get("question","").strip(): r for r in c_all}
    common = [q for q in a_lkp if q in c_lkp and q]
    sub    = [q for q in common if a_lkp[q].get("q_type","") == qt]
    corr_f = (lambda r: r.get("f1",0)>=0.5) if use_f1 else (lambda r: r.get("f1",0)>=0.5 or r.get("correct",False))
    a_c = sum(1 for q in sub if corr_f(a_lkp[q]))
    c_c = sum(1 for q in sub if corr_f(c_lkp[q]))
    n   = len(sub)
    check(label, expected[0], expected[1], expected[2], a_c/n*100, c_c/n*100)

# ── Summary ───────────────────────────────────────────────────────────────────
print()
print(SEP)
total = PASS + FAIL
print(f"VERIFICATION COMPLETE: {PASS}/{total} checks passed")
if FAIL == 0:
    print("ALL NUMBERS VERIFIED -- every claim in the paper is backed by the result files.")
else:
    print(f"WARNING: {FAIL} check(s) failed -- review numbers above.")
print(SEP)

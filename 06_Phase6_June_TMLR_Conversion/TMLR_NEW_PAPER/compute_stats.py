#!/usr/bin/env python3
"""Compute all CIs and p-values for TMLR paper tables."""
import json, math, os
from collections import defaultdict

BASE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(BASE)

def get_items(path):
    d = json.load(open(os.path.join(REPO, path)))
    return d if isinstance(d, list) else d.get('results', [])

def wilson_ci(k, n, z=1.96):
    if n == 0: return 0.0, 0.0
    p = k / n
    center = (p + z**2/(2*n)) / (1 + z**2/n)
    margin = (z * math.sqrt(p*(1-p)/n + z**2/(4*n**2))) / (1 + z**2/n)
    return round(max(0, center - margin)*100, 1), round(min(1, center + margin)*100, 1)

def mcnemar_p(n01, n10):
    total = n01 + n10
    if total < 5: return 1.0
    chi2 = (abs(n01 - n10) - 1)**2 / total
    return round(min(1.0, math.erfc(math.sqrt(chi2 / 2))), 4)

def p_label(p):
    if p < 0.001: return "p<0.001"
    if p < 0.01:  return f"p={p:.3f}"
    if p < 0.05:  return f"p={p:.3f}"
    return f"p={p:.2f} (NS)"

def sig_star(p):
    if p < 0.001: return "***"
    if p < 0.01:  return "**"
    if p < 0.05:  return "*"
    return "NS"

def run_comparison(a_items, c_items, key_fn, corr_fn):
    a_lkp = {key_fn(r): r for r in a_items if key_fn(r)}
    c_lkp = {key_fn(r): r for r in c_items if key_fn(r)}
    common = [k for k in a_lkp if k in c_lkp]
    n10 = sum(1 for k in common if corr_fn(a_lkp[k]) and not corr_fn(c_lkp[k]))
    n01 = sum(1 for k in common if not corr_fn(a_lkp[k]) and corr_fn(c_lkp[k]))
    a_c = sum(1 for k in common if corr_fn(a_lkp[k]))
    c_c = sum(1 for k in common if corr_fn(c_lkp[k]))
    p = mcnemar_p(n01, n10)
    a_lo, a_hi = wilson_ci(a_c, len(common))
    c_lo, c_hi = wilson_ci(c_c, len(common))
    a_acc = round(a_c / len(common) * 100, 1) if common else 0
    c_acc = round(c_c / len(common) * 100, 1) if common else 0
    return {
        'n': len(common), 'a_acc': a_acc, 'c_acc': c_acc,
        'gap': round(a_acc - c_acc, 1),
        'a_ci': (a_lo, a_hi), 'c_ci': (c_lo, c_hi),
        'p': p, 'n10': n10, 'n01': n01
    }

# ─────────────────────────────────────────────────────────────
# TABLE 1: Main results
# ─────────────────────────────────────────────────────────────
print("\n" + "="*70)
print("TABLE 1: MAIN RESULTS WITH STATISTICS")
print("="*70)
print(f"{'Model':<10} {'Dataset':<12} {'N':>5} {'A*%':>7} {'CoT%':>7} {'Gap':>6} {'p-value':<12} {'Sig'}")
print("-"*70)

# Qwen-0.5B HotpotQA (40 paired)
qhc = get_items('paper_results/hotpotqa_8_cot_qwen0.5b.json')
qha = get_items('paper_results/hotpotqa_30_astar_qwen0.5b.json')
r = run_comparison(qha, qhc,
    key_fn=lambda x: x['question'],
    corr_fn=lambda x: x.get('correct', False))
print(f"{'Qwen-0.5B':<10} {'HotpotQA':<12} {r['n']:>5} {r['a_acc']:>7} {r['c_acc']:>7} {r['gap']:>+6} {p_label(r['p']):<12} {sig_star(r['p'])}")
print(f"           {'':12} {'':>5} {'['+str(r['a_ci'][0])+'–'+str(r['a_ci'][1])+']':>7} {'['+str(r['c_ci'][0])+'–'+str(r['c_ci'][1])+']':>7}   (95% CI)")

# Qwen-0.5B LogicQA (500 paired by id)
qlc = get_items('paper_results/logicqa_24_cot_qwen0.5b.json')
qla = get_items('paper_results/logicqa_20_astar_qwen0.5b.json')
r = run_comparison(qla, qlc,
    key_fn=lambda x: str(x.get('id', x.get('question', ''))),
    corr_fn=lambda x: x.get('correct', False))
print(f"{'Qwen-0.5B':<10} {'LogicQA':<12} {r['n']:>5} {r['a_acc']:>7} {r['c_acc']:>7} {r['gap']:>+6} {p_label(r['p']):<12} {sig_star(r['p'])}")
print(f"           {'':12} {'':>5} {'['+str(r['a_ci'][0])+'–'+str(r['a_ci'][1])+']':>7} {'['+str(r['c_ci'][0])+'–'+str(r['c_ci'][1])+']':>7}   (95% CI)")

# Qwen-0.5B StrategyQA (110 paired)
qsc = get_items('paper_results/strategyqa_54_cot_qwen0.5b.json')
qsa = get_items('paper_results/strategyqa_53_astar_qwen0.5b.json')
r = run_comparison(qsa, qsc,
    key_fn=lambda x: x['question'],
    corr_fn=lambda x: x.get('correct', False))
print(f"{'Qwen-0.5B':<10} {'StrategyQA':<12} {r['n']:>5} {r['a_acc']:>7} {r['c_acc']:>7} {r['gap']:>+6} {p_label(r['p']):<12} {sig_star(r['p'])}")
print(f"           {'':12} {'':>5} {'['+str(r['a_ci'][0])+'–'+str(r['a_ci'][1])+']':>7} {'['+str(r['c_ci'][0])+'–'+str(r['c_ci'][1])+']':>7}   (95% CI)")
print()

# Llama-8B HotpotQA
hqa_a = get_items('27th_APRIL_WHOLE_DATASET_RUN/hotpotqa/astar/results.json')
hqa_c = get_items('27th_APRIL_WHOLE_DATASET_RUN/hotpotqa/cot/results.json')
r = run_comparison(hqa_a, hqa_c,
    key_fn=lambda x: x['question'],
    corr_fn=lambda x: x.get('f1', 0) >= 0.5)
print(f"{'Llama-8B':<10} {'HotpotQA':<12} {r['n']:>5} {r['a_acc']:>7} {r['c_acc']:>7} {r['gap']:>+6} {p_label(r['p']):<12} {sig_star(r['p'])}")
print(f"           {'':12} {'':>5} {'['+str(r['a_ci'][0])+'–'+str(r['a_ci'][1])+']':>7} {'['+str(r['c_ci'][0])+'–'+str(r['c_ci'][1])+']':>7}   (95% CI)")

# Llama-8B StrategyQA
sqa_a = get_items('27th_APRIL_WHOLE_DATASET_RUN/strategyqa/astar/results.json')
sqa_c = get_items('27th_APRIL_WHOLE_DATASET_RUN/strategyqa/cot/results.json')
r = run_comparison(sqa_a, sqa_c,
    key_fn=lambda x: x['question'],
    corr_fn=lambda x: x.get('correct', False))
print(f"{'Llama-8B':<10} {'StrategyQA':<12} {r['n']:>5} {r['a_acc']:>7} {r['c_acc']:>7} {r['gap']:>+6} {p_label(r['p']):<12} {sig_star(r['p'])}")
print(f"           {'':12} {'':>5} {'['+str(r['a_ci'][0])+'–'+str(r['a_ci'][1])+']':>7} {'['+str(r['c_ci'][0])+'–'+str(r['c_ci'][1])+']':>7}   (95% CI)")

# Llama-8B LogicQA
lqa_a = get_items('paper_results/logicqa_45_astar_llama3.1_8b.json')
lqa_c = get_items('paper_results/logicqa_46_cot_llama3.1_8b.json')
r = run_comparison(lqa_a, lqa_c,
    key_fn=lambda x: str(x.get('id', x.get('question', ''))),
    corr_fn=lambda x: x.get('correct', False))
print(f"{'Llama-8B':<10} {'LogicQA':<12} {r['n']:>5} {r['a_acc']:>7} {r['c_acc']:>7} {r['gap']:>+6} {p_label(r['p']):<12} {sig_star(r['p'])}")
print(f"           {'':12} {'':>5} {'['+str(r['a_ci'][0])+'–'+str(r['a_ci'][1])+']':>7} {'['+str(r['c_ci'][0])+'–'+str(r['c_ci'][1])+']':>7}   (95% CI)")
print()

# DeepSeek-14B
B14 = '14b_dataset_check/results_14b'
for ds, ap, cp, use_f1, kf in [
    ('HotpotQA',   f'{B14}/hotpotqa/astar/results.json',   f'{B14}/hotpotqa/cot/results.json',   True,  lambda x: x.get('question','').strip()),
    ('StrategyQA', f'{B14}/strategyqa/astar/results.json', f'{B14}/strategyqa/cot/results.json', False, lambda x: x.get('question', x.get('id','')).strip()),
    ('LogicQA',    f'{B14}/logicqa/astar/results.json',    f'{B14}/logicqa/cot/results.json',    False, lambda x: str(x.get('id', x.get('query',''))).strip()),
]:
    a_items = get_items(ap); c_items = get_items(cp)
    corr_fn = (lambda x: x.get('f1',0)>=0.5 or x.get('correct',False)) if use_f1 else (lambda x: x.get('correct',False))
    r = run_comparison(a_items, c_items, key_fn=kf, corr_fn=corr_fn)
    print(f"{'DS-14B':<10} {ds:<12} {r['n']:>5} {r['a_acc']:>7} {r['c_acc']:>7} {r['gap']:>+6} {p_label(r['p']):<12} {sig_star(r['p'])}")
    print(f"           {'':12} {'':>5} {'['+str(r['a_ci'][0])+'–'+str(r['a_ci'][1])+']':>7} {'['+str(r['c_ci'][0])+'–'+str(r['c_ci'][1])+']':>7}   (95% CI)")

# ─────────────────────────────────────────────────────────────
# TABLE 3: Question-type breakdown
# ─────────────────────────────────────────────────────────────
print("\n" + "="*70)
print("TABLE 3: QUESTION-TYPE BREAKDOWN WITH STATISTICS")
print("="*70)
print(f"{'Model':<8} {'Type':<12} {'N':>5} {'A*%':>7} {'CoT%':>7} {'Gap':>6} {'p-value':<12} {'Sig'}")
print("-"*70)

for label, ap, cp, use_f1 in [
    ('8B',  '27th_APRIL_WHOLE_DATASET_RUN/hotpotqa/astar/results.json',
             '27th_APRIL_WHOLE_DATASET_RUN/hotpotqa/cot/results.json',  True),
    ('14B', '14b_dataset_check/results_14b/hotpotqa/astar/results.json',
             '14b_dataset_check/results_14b/hotpotqa/cot/results.json', False),
]:
    a_items = get_items(ap); c_items = get_items(cp)
    a_lkp = {r.get('question','').strip(): r for r in a_items}
    c_lkp = {r.get('question','').strip(): r for r in c_items}
    common = [q for q in a_lkp if q in c_lkp and q]
    corr_fn = (lambda x: x.get('f1',0)>=0.5) if use_f1 else (lambda x: x.get('f1',0)>=0.5 or x.get('correct',False))

    for qt, qt_label in [('bridge','bridge'), ('comparison','comparison'), ('yesno','yes/no')]:
        sub = [q for q in common if a_lkp[q].get('q_type','') == qt]
        if not sub: continue
        n10 = sum(1 for q in sub if corr_fn(a_lkp[q]) and not corr_fn(c_lkp[q]))
        n01 = sum(1 for q in sub if not corr_fn(a_lkp[q]) and corr_fn(c_lkp[q]))
        a_c = sum(1 for q in sub if corr_fn(a_lkp[q]))
        c_c = sum(1 for q in sub if corr_fn(c_lkp[q]))
        p = mcnemar_p(n01, n10)
        a_lo, a_hi = wilson_ci(a_c, len(sub))
        c_lo, c_hi = wilson_ci(c_c, len(sub))
        a_acc = round(a_c/len(sub)*100,1)
        c_acc = round(c_c/len(sub)*100,1)
        print(f"{label:<8} {qt_label:<12} {len(sub):>5} {a_acc:>7} {c_acc:>7} {a_acc-c_acc:>+6.1f} {p_label(p):<12} {sig_star(p)}")
        print(f"         {'':12} {'':>5} {'['+str(a_lo)+'–'+str(a_hi)+']':>7} {'['+str(c_lo)+'–'+str(c_hi)+']':>7}   (95% CI)")
    print()

print("Legend: *** p<0.001  ** p<0.01  * p<0.05  NS = not significant")

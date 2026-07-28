# Performance Experiment — Server Instructions

## What This Does

Takes the **93 questions** where CoT and A* originally gave **different answers** (the disagreement set) and runs **both methods fresh** on all of them to get a clean head-to-head comparison.

---

## Files in This Folder

| File | Role |
|------|------|
| `experiment.py` | Main script — fully self-contained |
| `disagreement.json` | 93 questions where CoT ≠ A* originally (input) |

**Output files created by the script:**

| File | Created by |
|------|-----------|
| `cot_results.json` | Step 1 |
| `astar_results.json` | Step 2 |

---

## Setup

```bash
ollama serve &
ollama pull llama3.1:8b
pip install openai tqdm
python3 experiment.py --check
```

---

## Running

### Full run

```bash
python3 experiment.py
```

Runs in 3 steps:
- **Step 1**: CoT on all 93 questions → `cot_results.json`
- **Step 2**: A\* on all 93 questions → `astar_results.json`
- **Step 3**: Head-to-head performance report

### Resume if interrupted

```bash
python3 experiment.py --from 2   # step 1 done, resume from A*
python3 experiment.py --from 3   # both done, just print report
python3 experiment.py --only 3   # print report from saved files
```

---

## Expected Runtime

| Step | Questions | Time |
|------|-----------|------|
| Step 1 — CoT | 93 | ~20–30 min |
| Step 2 — A\* | 93 | ~45–75 min |
| **Total** | | **~65–105 min** |

All results saved after every question — safe to interrupt and resume.

---

## What to Send Back

1. `cot_results.json`
2. `astar_results.json`
3. Copy of the Step 3 report printed to console

---

## Quick Verify

```bash
python3 -c "
import json
cot  = json.load(open('cot_results.json'))
astr = json.load(open('astar_results.json'))
cot_ok  = sum(1 for r in cot  if r['correct'])
astr_ok = sum(1 for r in astr if r['correct'])
n = len(cot)
print(f'CoT : {cot_ok}/{n} ({cot_ok/n*100:.1f}%)')
print(f'A*  : {astr_ok}/{n} ({astr_ok/n*100:.1f}%)')
"
```

---

*Model: llama3.1:8b | Dataset: StrategyQA (yes/no) | Questions: 93 (disagreement set)*

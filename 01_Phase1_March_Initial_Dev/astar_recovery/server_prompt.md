# A* Recovery Experiment — Server Instructions

## Purpose

This experiment investigates **why A\* fails on certain StrategyQA questions that CoT gets right**.

We have 45 questions where A\* predicted the wrong answer but CoT was correct  
(from the original 500-question comparison run).

**Key question**: are these failures *systematic* (A\* always gets them wrong) or *flukes* (just unlucky sampling)?

**Method**: run A\* **5 independent times** per question using different temperatures,  
then classify each question:

| Category | Meaning |
|----------|---------|
| `never`  (0/5) | Systematic failure — A\* always converges to the wrong answer |
| `rare`   (1–2/5) | Mostly fails, occasional lucky run |
| `often`  (3–4/5) | Mostly correct, occasionally unlucky |
| `always` (5/5) | Original failure was a fluke |

---

## Files in This Folder

| File | Role |
|------|------|
| `astar_recovery_experiment.py` | Main script — self-contained, no imports from parent folders |
| `astar_fail_cot_win.json` | Input — 45 questions (A\* failed, CoT won) |
| `multirun_results.json` | Output — created by the script |
| `analysis_report.txt` | Text report — created by the script |

---

## Setup

```bash
# 1. Make sure Ollama is running
ollama serve &

# 2. Make sure the model is downloaded
ollama pull llama3.1:8b

# 3. Install Python dependencies
pip install openai tqdm

# 4. Check everything is ready
python3 astar_recovery_experiment.py --check
```

---

## Running the Experiment

### Option A — Full run (recommended)

```bash
python3 astar_recovery_experiment.py
```

This runs both steps sequentially:
- **Step 1**: A\* × 5 on all 45 questions → saves `multirun_results.json`
- **Step 2**: Analysis report → saves `analysis_report.txt` + prints to console

### Option B — Run only step 1 (A\* runs)

```bash
python3 astar_recovery_experiment.py --only 1
```

### Option C — Run only step 2 (analysis from existing results)

```bash
python3 astar_recovery_experiment.py --only 2
```

### Option D — Resume interrupted run

```bash
python3 astar_recovery_experiment.py --from 1   # resume from step 1
python3 astar_recovery_experiment.py --from 2   # skip to analysis
```

### Other flags

```bash
--model llama3.1:8b     # change model (default: llama3.1:8b)
--no-check              # skip health check
--check                 # only run health check
```

---

## Expected Runtime

- ~10–15 seconds per question × 5 runs = ~50–75 s per question
- 45 questions → **~40–60 minutes total**

Progress is shown with a `tqdm` progress bar.

Results are **saved after every question** — if it crashes, just re-run and it resumes from where it left off.

---

## What to Send Back

After the experiment completes, please send:

1. **`multirun_results.json`** — full results (required)
2. **`analysis_report.txt`** — text summary (required)
3. Copy of the printed console output (optional but helpful)

---

## Quick Sanity Check

After step 1 is done, you can verify with:

```bash
python3 -c "
import json
r = json.load(open('multirun_results.json'))
from collections import Counter
cats = Counter(x['category'] for x in r)
print(f'Done: {len(r)}/45 questions')
print('Categories:', dict(cats))
rec = sum(1 for x in r if x['recovered_any'])
print(f'Recovered at least once: {rec}/{len(r)} ({rec/len(r)*100:.1f}%)')
"
```

---

## Interpretation Guide

When reading the results:

- A high `never` % → A\*'s search **systematically fails** on these knowledge-intensive questions  
  → CoT's linear chain-of-thought is fundamentally better suited for this question type
- A high `always` % → original failures were **random sampling noise**, not a systematic issue
- High `often`+`always` → A\* is **competitive but unlucky** — majority voting would fix it
- Mixed results → A\* is **borderline** — these questions sit at the edge of the model's capability

---

*Model: llama3.1:8b | Dataset: StrategyQA (yes/no) | Runs per question: 5*

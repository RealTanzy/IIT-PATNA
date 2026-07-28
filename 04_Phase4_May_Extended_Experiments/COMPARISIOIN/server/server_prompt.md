# Server Task Brief — CoT vs A* Comparison Experiment

## What You Are Doing

You are running a **one-file Python experiment** that evaluates and compares two reasoning methods — Chain-of-Thought (CoT) and A* search — on StrategyQA yes/no questions.

The experiment has already been partially run. We have:
- **500 questions** evaluated by both CoT and A* (using `llama3.1:8b` via Ollama)
- CoT accuracy: **74.0%** | A* accuracy: **74.8%**
- **49 questions** where CoT failed but A* got it right
- **45 questions** where A* failed but CoT got it right

Your job is to **re-run each method on the questions it originally failed**, record the results, and produce a final comparison report.

---

## Files in This Directory

```
experiment.py              ← the only script you need to run
cot_fail_astar_win.json    ← 49 questions  (CoT failed, A* won)
astar_fail_cot_win.json    ← 45 questions  (A* failed, CoT won)
```

---

## Requirements

- Python 3.8+
- Ollama running locally on port 11434
- Model pulled: `llama3.1:8b`
- Packages: `openai`, `tqdm`

```bash
pip install openai tqdm
ollama pull llama3.1:8b
```

---

## How to Run

### Step 1 — Verify everything is ready
```bash
python3 experiment.py --check
```
This checks: Ollama is up, model is available, input files exist, a test generation works.
**Do not proceed until this passes.**

---

### Step 2 — Re-run CoT on its 49 failures
```bash
python3 experiment.py --only 2
```
- Runs each of the 49 questions through CoT **twice** (temperatures 0.2 and 0.5)
- Takes ~5–10 minutes
- Saves results to `cot_rerun_on_cot_failures.json` after every question (crash-safe)

---

### Step 3 — Re-run A* on its 45 failures
```bash
python3 experiment.py --only 3
```
- Runs each of the 45 questions through full A* search (max 15 nodes, depth 8)
- Takes ~20–35 minutes
- Saves results to `astar_rerun_on_astar_failures.json` after every question (crash-safe)

---

### Step 4 — Print the final report
```bash
python3 experiment.py --only 4
```
Prints a table comparing baseline vs updated accuracy for both methods.

---

### Run everything at once
```bash
python3 experiment.py
```

### Resume after interruption
```bash
python3 experiment.py --from 3    # resume from step 3
```

---

## What Gets Saved

| File | When created | Contents |
|---|---|---|
| `cot_rerun_on_cot_failures.json` | After step 2 | CoT 2nd-attempt results on 49 questions |
| `astar_rerun_on_astar_failures.json` | After step 3 | A* 2nd-attempt results on 45 questions |

Both files save **incrementally** — one entry per question. Safe to kill and resume at any time.

---

## What the Final Report Shows

```
                          |    CoT    |    A*
--------------------------+-----------+-----------
Baseline accuracy         | 370/500   | 374/500
Failures recovered        |  +??      |  +??
Updated accuracy          | ???/500   | ???/500
Consistency (recovery %)  |  ??%      |  ??%
```

This tells us:
- How many of each method's failures were **flukes** (recoverable on 2nd attempt)
- Which method is more **consistent** under re-testing
- Whether the accuracy gap between CoT and A* holds after controlling for randomness

---

## If Something Goes Wrong

| Problem | Fix |
|---|---|
| `Ollama is NOT running` | Run `ollama serve` in a separate terminal |
| `Model not found` | Run `ollama pull llama3.1:8b` |
| `No module named openai` | Run `pip install openai tqdm` |
| Script crashes mid-run | Re-run the same step — it will resume from where it stopped |
| Want a different model | `python3 experiment.py --model llama3.2:3b` |

---

## Send Back

When done, send back these two output files:
```
cot_rerun_on_cot_failures.json
astar_rerun_on_astar_failures.json
```

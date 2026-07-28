#!/usr/bin/env python3
"""
Fix 1: Re-run Qwen-0.5B CoT on the exact 500 questions used in the A* run.

This resolves the N=40 overlap problem. The A* result file has 500 questions;
this script runs CoT on the same 500 questions using the same model (qwen:0.5b)
and saves results to TMLR_NEW_PAPER/qwen_matched_cot_results.json.

Run with:
    python run_qwen_matched_cot.py

Prerequisites:
    ollama pull qwen:0.5b
    (Ollama must be running: ollama serve)

Expected runtime: ~3-6 hours on CPU depending on machine.
Progress is saved incrementally every 10 questions.
Safe to interrupt and resume — already-done questions are skipped.
"""

import json, re, time, os, requests
from tqdm import tqdm

# ─── Config ────────────────────────────────────────────────────────────────
MODEL       = "qwen:0.5b"
OLLAMA_URL  = "http://localhost:11434/api/generate"
ASTAR_FILE  = os.path.join(os.path.dirname(__file__), "..", "paper_results", "hotpotqa_30_astar_qwen0.5b.json")
OUT_FILE    = os.path.join(os.path.dirname(__file__), "qwen_matched_cot_results.json")
SAVE_EVERY  = 10   # checkpoint every N questions

# Build HotpotQA context lookup from the dataset
print("Loading HotpotQA validation set for context...")
from datasets import load_dataset
_hqa_ds = load_dataset("hotpot_qa", "distractor", split="validation")
CONTEXT_LOOKUP = {}
for item in _hqa_ds:
    titles = item["context"]["title"]
    sents  = item["context"]["sentences"]
    ctx    = " ".join(t + ": " + " ".join(s) for t, s in zip(titles, sents))
    CONTEXT_LOOKUP[item["question"]] = ctx
print(f"  {len(CONTEXT_LOOKUP)} questions indexed")

# ─── Prompt ────────────────────────────────────────────────────────────────
COT_PROMPT = """Answer the following multi-hop question by reasoning step by step.
Use the provided context paragraphs to find the answer.
At the end, output the final answer as: "The answer is: [answer]"

Context:
{context}

Question: {question}
Answer: Let's think step by step."""

COT_PROMPT_NO_CTX = """Answer the following question by reasoning step by step.
At the end, output the final answer as: "The answer is: [answer]"

Question: {question}
Answer: Let's think step by step."""

# ─── Helpers ───────────────────────────────────────────────────────────────
def ollama_generate(prompt: str, model: str = MODEL, temperature: float = 0.1) -> str:
    data = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": temperature,
            "num_predict": 200,
            "stop": ["Question:", "\n\n"]
        }
    }
    try:
        r = requests.post(OLLAMA_URL, json=data, timeout=300)
        if r.status_code == 200:
            return r.json().get("response", "").strip()
        print(f"  HTTP {r.status_code}")
        return ""
    except Exception as e:
        print(f"  Error: {e}")
        return ""

def extract_answer(response: str) -> str:
    # Try "The answer is: X"
    m = re.search(r'[Tt]he answer is:?\s*["\']?([^"\'\n]+)["\']?', response)
    if m:
        return m.group(1).strip().rstrip('.')
    # Try last non-empty line
    lines = [l.strip() for l in response.split('\n') if l.strip()]
    return lines[-1] if lines else ""

def f1_score(pred: str, gold: str) -> float:
    pred_tokens = pred.lower().split()
    gold_tokens = gold.lower().split()
    common = set(pred_tokens) & set(gold_tokens)
    if not common:
        return 0.0
    p = len(common) / len(pred_tokens) if pred_tokens else 0
    r = len(common) / len(gold_tokens) if gold_tokens else 0
    return 2 * p * r / (p + r) if (p + r) > 0 else 0.0

def normalize(s: str) -> str:
    s = s.lower().strip()
    s = re.sub(r'\s+', ' ', s)
    return s.rstrip('.')

def is_correct(pred: str, gold: str) -> bool:
    return f1_score(normalize(pred), normalize(gold)) >= 0.5

# ─── Load A* questions ──────────────────────────────────────────────────────
print(f"Loading A* results from {ASTAR_FILE}")
with open(ASTAR_FILE) as f:
    astar_data = json.load(f)
astar_items = astar_data if isinstance(astar_data, list) else astar_data.get("results", [])
print(f"  {len(astar_items)} questions to run CoT on")

# ─── Load existing results (resume support) ──────────────────────────────
done = {}
if os.path.exists(OUT_FILE):
    with open(OUT_FILE) as f:
        existing = json.load(f)
    for r in existing:
        done[r["question"]] = r
    print(f"  Resuming: {len(done)} already done, {len(astar_items) - len(done)} remaining")
else:
    print("  Starting fresh")

# ─── Check Ollama ───────────────────────────────────────────────────────────
print(f"\nChecking Ollama ({MODEL})...")
test = ollama_generate("Say: ready", temperature=0.0)
if not test:
    print("ERROR: Ollama not responding. Run: ollama serve")
    print("Then pull model: ollama pull qwen:0.5b")
    exit(1)
print(f"  OK: {test[:50]}")

# ─── Main loop ──────────────────────────────────────────────────────────────
results = list(done.values())
new_since_save = 0

print(f"\nRunning CoT on {len(astar_items)} questions...\n")
for item in tqdm(astar_items, desc="CoT", unit="q"):
    q = item["question"]
    if q in done:
        continue

    gold = str(item.get("gold", ""))
    # Look up context from the dataset
    ctx = CONTEXT_LOOKUP.get(q, "")

    if ctx.strip():
        prompt = f"Answer the question. Context: {ctx[:600]} Question: {q} Answer:"
    else:
        prompt = f"Answer the question. Question: {q} Answer:"

    response = ollama_generate(prompt)
    pred = extract_answer(response)
    f1 = f1_score(normalize(pred), normalize(gold))
    correct = f1 >= 0.5

    result = {
        "question": q,
        "gold": gold,
        "pred": pred,
        "correct": correct,
        "f1": round(f1, 4),
        "cot_response": response[:800],
        "q_type": item.get("q_type", ""),
    }
    results.append(result)
    done[q] = result
    new_since_save += 1

    if new_since_save >= SAVE_EVERY:
        with open(OUT_FILE, "w") as f:
            json.dump(results, f, indent=2)
        new_since_save = 0

# Final save
with open(OUT_FILE, "w") as f:
    json.dump(results, f, indent=2)

# ─── Summary ────────────────────────────────────────────────────────────────
total = len(results)
correct = sum(1 for r in results if r["correct"])
print(f"\n{'='*50}")
print(f"DONE: {total} questions")
print(f"CoT accuracy (F1>=0.5): {correct}/{total} = {correct/total*100:.1f}%")

# Compare with A* on same questions
astar_lookup = {item["question"]: item for item in astar_items}
paired = [r for r in results if r["question"] in astar_lookup]
if paired:
    import math
    cot_corr = sum(1 for r in paired if r["correct"])
    ast_corr = sum(1 for r in paired if astar_lookup[r["question"]].get("correct", False))
    n10 = sum(1 for r in paired if astar_lookup[r["question"]].get("correct",False) and not r["correct"])
    n01 = sum(1 for r in paired if not astar_lookup[r["question"]].get("correct",False) and r["correct"])
    total_disc = n10 + n01
    chi2 = (abs(n10 - n01) - 1)**2 / total_disc if total_disc >= 5 else None
    p = min(1.0, math.erfc(math.sqrt(chi2/2))) if chi2 else 1.0
    print(f"\nMatched comparison (N={len(paired)}):")
    print(f"  CoT: {cot_corr/len(paired)*100:.1f}%")
    print(f"  A*:  {ast_corr/len(paired)*100:.1f}%")
    print(f"  Gap: {(ast_corr-cot_corr)/len(paired)*100:+.1f}pp")
    print(f"  McNemar p={p:.4f}")

print(f"\nResults saved to: {OUT_FILE}")

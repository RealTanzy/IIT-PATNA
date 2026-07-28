#!/usr/bin/env python3
"""Quick 3-question smoke test before the overnight run."""
import json, re, os, requests

MODEL      = "qwen:0.5b"
OLLAMA_URL = "http://localhost:11434/api/generate"
ASTAR_FILE = os.path.join(os.path.dirname(__file__), "..", "paper_results", "hotpotqa_30_astar_qwen0.5b.json")

print("Loading HotpotQA context index...")
from datasets import load_dataset
_ds = load_dataset("hotpot_qa", "distractor", split="validation")
CTX = {item["question"]: " ".join(t+": "+" ".join(s) for t,s in zip(item["context"]["title"], item["context"]["sentences"])) for item in _ds}
print(f"  {len(CTX)} questions indexed")

def ollama_generate(prompt, model=MODEL, temperature=0.1):
    data = {"model": model, "prompt": prompt, "stream": False,
            "options": {"temperature": temperature, "num_predict": 200, "stop": ["Question:", "\n\n"]}}
    r = requests.post(OLLAMA_URL, json=data, timeout=300)
    return r.json().get("response", "").strip() if r.status_code == 200 else ""

def extract_answer(response):
    m = re.search(r'[Tt]he answer is:?\s*["\']?([^"\'\n]+)["\']?', response)
    if m: return m.group(1).strip().rstrip('.')
    lines = [l.strip() for l in response.split('\n') if l.strip()]
    return lines[-1] if lines else ""

def f1(pred, gold):
    p_toks = pred.lower().split(); g_toks = gold.lower().split()
    common = set(p_toks) & set(g_toks)
    if not common: return 0.0
    pr = len(common)/len(p_toks) if p_toks else 0
    rc = len(common)/len(g_toks) if g_toks else 0
    return 2*pr*rc/(pr+rc) if (pr+rc) > 0 else 0.0

with open(ASTAR_FILE) as f:
    items = json.load(f)[:3]

print(f"Smoke test: 3 questions with {MODEL}\n")
for i, item in enumerate(items):
    q = item["question"]
    gold = str(item.get("gold",""))
    ctx = CTX.get(q, "")

    prompt = f"""Answer the question. Context: {ctx[:600]} Question: {q} Answer:"""

    response = ollama_generate(prompt)
    pred = extract_answer(response)
    score = f1(pred.lower().strip(), gold.lower().strip())
    print(f"Q{i+1}: {q[:70]}")
    print(f"     Gold: {gold}  |  Pred: {pred}  |  F1: {score:.2f}")
    print(f"     Response snippet: {response[:120]}")
    print()
print("Smoke test complete. If predictions look reasonable, run the full script.")

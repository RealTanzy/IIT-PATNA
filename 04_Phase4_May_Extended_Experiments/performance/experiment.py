#!/usr/bin/env python3
"""
Performance Experiment — Disagreement Set
==========================================
Takes the 93 questions where CoT and A* originally DISAGREED
and runs BOTH methods fresh on all of them.

Gives a clean head-to-head performance comparison on exactly
the questions that caused disagreement.

STEPS:
  1. Run CoT  on all 93 disagreement questions
  2. Run A*   on all 93 disagreement questions
  3. Print side-by-side performance report

REQUIRES:  pip install openai tqdm
FILES NEEDED (same directory):
  disagreement.json

USAGE:
  python3 experiment.py --check
  python3 experiment.py
  python3 experiment.py --from 2
  python3 experiment.py --only 3
"""

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────

OLLAMA_BASE_URL = "http://localhost:11434/v1"
OLLAMA_API_KEY  = "ollama"
MODEL           = "llama3.1:8b"

# A* hyper-parameters (same as original experiment)
BRANCH_K   = 2
MAX_DEPTH  = 8
MAX_NODES  = 15
MIN_GOAL_D = 2
MAJORITY   = 2
TEMPS      = [0.2, 0.6]
W_DEPTH    = 0.3
W_COVERAGE = 0.3

# ─────────────────────────────────────────────────────────────────────────────
# IMPORTS
# ─────────────────────────────────────────────────────────────────────────────

import os, re, sys, json, heapq, time, argparse, urllib.request
from collections import Counter
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Set

try:
    from openai import OpenAI
except ImportError:
    print("ERROR: openai not installed — run:  pip install openai tqdm"); sys.exit(1)
try:
    from tqdm import tqdm
except ImportError:
    def tqdm(it, **kw): return it

DIR  = os.path.dirname(os.path.abspath(__file__))
SEP  = "=" * 65
SEP2 = "─" * 65


# ─────────────────────────────────────────────────────────────────────────────
# A* SOLVER  (fully embedded)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ReasoningStep:
    content: str
    confidence: float = 0.8
    is_factual: bool  = False

@dataclass
class State:
    steps: List[ReasoningStep] = field(default_factory=list)
    depth: int = 0
    def get_trace(self) -> str:
        return "\n".join(f"Step {i+1}: {s.content}" for i, s in enumerate(self.steps))
    def signature(self) -> str:
        return " || ".join(s.content.strip() for s in self.steps)

@dataclass
class Node:
    state: State
    g_score: float; h_score: float; f_score: float; node_id: int
    parent_id: Optional[int] = None
    def __lt__(self, other: "Node") -> bool:
        if abs(self.f_score - other.f_score) > 1e-9:
            return self.f_score < other.f_score
        return self.node_id < other.node_id

_SW = frozenset({
    'a','an','the','is','was','were','are','of','in','on','at','for','and','or',
    'to','by','with','it','its','that','this','from','as','be','been','being',
    'have','has','had','do','does','did','will','would','could','should','may',
    'might','can','what','which','who','where','when','how','why','same','both',
    'also','not','no','yes','than','more','most','other','some','any','all',
    'many','much','few','several','first','last','new','old',
})

def _entities(q: str) -> Set[str]:
    ents: Set[str] = set()
    for m in re.finditer(r'"([^"]+)"', q):
        ents.add(m.group(1).lower())
    q2 = re.sub(r'^(What|Which|Who|Where|When|How|Are|Were|Is|Was|Did|Do|Does|'
                r'Could|Would|Can|Has|Have|Had|The)\s+', '', q, flags=re.IGNORECASE)
    for m in re.finditer(r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b', q2):
        ph = m.group(1).lower()
        if ph not in _SW and len(ph) > 2: ents.add(ph)
    for m in re.finditer(r'\b(\d{4}|\d+(?:,\d{3})*)\b', q):
        ents.add(m.group(1))
    _GEN = _SW | frozenset({'person','people','human','animal','thing','place','object',
        'type','kind','part','year','date','time','name','known','called','named',
        'located','founded','formed','able','capable','possible','likely',
        'usually','generally','often','typically'})
    for w in re.findall(r'\b[a-zA-Z]+\b', q.lower()):
        if w not in _GEN and len(w) > 4: ents.add(w)
    return ents

def _coverage(state: State, ents: Set[str]) -> float:
    if not ents: return 1.0
    tr = state.get_trace().lower()
    return sum(1 for e in ents if e in tr) / len(ents)

def _h(state: State, ents: Set[str]) -> float:
    return max(0, 2 - state.depth) * W_DEPTH + (1.0 - _coverage(state, ents)) * W_COVERAGE

def _jaccard(a: str, b: str) -> float:
    wa, wb = set(a.lower().split()), set(b.lower().split())
    if not wa or not wb: return 0.0
    return len(wa & wb) / len(wa | wb)

def _degen(c: str) -> bool:
    if len(c) < 10: return False
    s = c.strip()
    if len(set(s)) <= 3 and len(s) > 20: return True
    words = s.split()
    if len(words) > 10 and Counter(words).most_common(1)[0][1] / len(words) > 0.6: return True
    return False


class AStarSolver:
    def __init__(self, model: str = MODEL, cot_temp: float = 0.2):
        self.client    = OpenAI(base_url=OLLAMA_BASE_URL, api_key=OLLAMA_API_KEY)
        self.model     = model
        self.cot_temp  = cot_temp
        self.api_calls = 0

    def _llm(self, sys_p: str, usr_p: str,
             temp: float = 0.3, max_tok: int = 512) -> Tuple[Optional[str], float]:
        try:
            r = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role":"system","content":sys_p},{"role":"user","content":usr_p}],
                temperature=temp, max_tokens=max_tok, stop=["\n\n\n"],
            )
            self.api_calls += 1
            content = r.choices[0].message.content or ""
            if _degen(content): return None, 0.0
            return self._parse(content)
        except Exception:
            return None, 0.0

    def _parse(self, content: str) -> Tuple[Optional[str], float]:
        sm = re.search(r'STEP(?:\s*\d*)?:\s*(.+?)(?=\n(?:STEP|CONFIDENCE)|$)',
                       content, re.IGNORECASE | re.DOTALL)
        cm = re.search(r'CONFIDENCE:\s*([0-9.]+)', content, re.IGNORECASE)
        if sm:
            text = sm.group(1).strip()
            conf = max(0.5, min(0.99, float(cm.group(1)) if cm else 0.75))
            if len(text) > 3: return text, conf
        am = re.search(r'[Tt]he answer is[:\s]+(.+?)(?:\.|$)', content)
        if am:
            conf = max(0.5, min(0.99, float(cm.group(1)) if cm else 0.75))
            return f"The answer is: {am.group(1).strip()}", conf
        text = re.sub(r'^(?:Step\s*\d+[:\s]*|STEP[:\s]*)', '',
                      content.strip(), flags=re.IGNORECASE).strip()
        if 5 < len(text) < 500: return text, 0.6
        return None, 0.0

    def _step(self, question: str, state: State, step_num: int):
        cands = []
        sf = ("\n".join(f"  {i+1}. {s.content}" for i, s in enumerate(state.steps))
              if state.steps else "")
        if step_num == 1:
            for sp, up, t in [
                ("You break down yes/no questions. Recall key facts. Do NOT answer yet.",
                 f"Question: {question}\n\nTASK: What facts do you need?\n\nSTEP: [key facts]\nCONFIDENCE: [0.6-0.95]",
                 self.cot_temp),
                ("You recall factual information from world knowledge. Be specific. No yes/no yet.",
                 f"Question: {question}\n\nTASK: Recall relevant fact for each entity.\n\nSTEP: [Entity]: [fact].\nCONFIDENCE: [0.6-0.95]",
                 0.6),
            ]:
                text, conf = self._llm(sp, up, t, 300)
                if text and 'the answer is' not in text.lower():
                    cands.append((text, conf, True))
                elif text:
                    cands.append((text, conf, False))
        else:
            sp = "You answer yes/no questions using previously gathered facts."
            up = (f"Question: {question}\n\nFacts:\n{sf}\n\n"
                  f"STEP: The answer is: [yes or no]\nCONFIDENCE: [0.7-1.0]")
            for i in range(min(BRANCH_K, 2)):
                text, conf = self._llm(sp, up, TEMPS[i] if i < len(TEMPS) else 0.4, 128)
                if text: cands.append((text, conf, False))
        return cands

    def _conclude(self, question: str, state: State):
        sf = ("\n".join(f"  {i+1}. {s.content}" for i, s in enumerate(state.steps))
              if state.steps else "  (none)")
        text, conf = self._llm(
            "You give concise yes/no answers based on reasoning.",
            f"Question: {question}\n\nReasoning:\n{sf}\n\nSTEP: The answer is: [yes or no]\nCONFIDENCE: [0.7-1.0]",
            0.1, 128
        )
        if text:
            if 'the answer is' not in text.lower():
                text = f"The answer is: {text}"
            return [(text, conf, False)]
        return [("The answer is: no", 0.5, False)]

    def candidates(self, question: str, state: State, force: bool = False):
        if force: return self._conclude(question, state)
        return self._step(question, state, len(state.steps) + 1)

    @staticmethod
    def extract(trace: str) -> Optional[str]:
        matches = re.findall(r'[Tt]he answer is[:\s]+([^\n]+)', trace)
        if matches:
            raw = re.sub(r'\s*\([^)]*\)\s*$', '',
                   re.sub(r'^(?:Answer|Response|Result)[:\s]+', '',
                          matches[-1].strip().lower().rstrip('.'))).strip()
            if re.search(r'\byes\b', raw): return 'yes'
            if re.search(r'\bno\b',  raw): return 'no'
        lines = [l.strip() for l in trace.strip().split('\n') if l.strip()]
        if lines:
            last = lines[-1].lower()
            if re.search(r'\byes\b', last): return 'yes'
            if re.search(r'\bno\b',  last): return 'no'
        return None


def _pick_goal(goals: List[Node], target: str, solver: AStarSolver) -> Node:
    t = target.lower().strip()
    matching = [n for n in goals
                if (solver.extract(n.state.get_trace()) or "").lower().strip() == t]
    if matching: return max(matching, key=lambda n: n.state.steps[-1].confidence)
    return min(goals, key=lambda n: n.f_score)


def run_astar(question: str, solver: AStarSolver) -> Tuple[Optional[str], int]:
    ents   = _entities(question)
    open_s: List[Node] = []
    nc = ne = 0
    visited: Set[str]    = set()
    goal_nodes: List[Node] = []
    goal_votes: Counter  = Counter()
    best: Optional[Node] = None

    root = Node(State(), 0.0, 0.0, 0.0, 0)
    root.h_score = root.f_score = _h(root.state, ents)
    heapq.heappush(open_s, root)

    cot_txt, cot_conf = solver._llm(
        "You answer yes/no questions. Reason step by step.",
        f"Question: {question}\n\nThink step by step.\nEnd with: The answer is: [yes or no]\n",
        solver.cot_temp, 512
    )
    if cot_txt:
        nc += 1
        has_ans   = 'the answer is' in cot_txt.lower()
        cot_state = State(steps=[ReasoningStep(cot_txt, cot_conf, not has_ans)],
                          depth=2 if has_ans else 1)
        cot_g = 1.0 + (1.0 - cot_conf)
        cot_h = _h(cot_state, ents)
        heapq.heappush(open_s, Node(cot_state, cot_g, cot_h, cot_g + cot_h, nc, 0))

    while open_s and ne < MAX_NODES:
        cur = heapq.heappop(open_s)
        sig = cur.state.signature()
        if sig in visited: continue
        visited.add(sig); ne += 1

        if cur.state.steps:
            ls      = cur.state.steps[-1]
            is_goal = bool(re.search(r'the answer is[:\s]+(yes|no)\b', ls.content.lower()))
            if is_goal and cur.state.depth < MIN_GOAL_D and ls.confidence < 0.75:
                is_goal = False
            if is_goal:
                goal_nodes.append(cur)
                ans = solver.extract(cur.state.get_trace())
                if ans:
                    goal_votes[ans.lower()] += 1
                    if goal_votes[ans.lower()] >= MAJORITY:
                        bg = _pick_goal(goal_nodes, ans, solver)
                        return solver.extract(bg.state.get_trace()), ne
                continue

        if best is None or cur.state.depth > best.state.depth:
            best = cur
        if cur.state.depth >= MAX_DEPTH: continue

        force = (cur.state.depth >= MAX_DEPTH - 1)
        cands = solver.candidates(question, cur.state, force)
        existing = [s.content for s in cur.state.steps]
        seen: Set[str] = set(); uniq = []
        for txt, conf, isf in cands:
            key = txt.strip().lower()
            if key in seen: continue
            if any(_jaccard(txt, p) > 0.7 for p in existing): continue
            if any(_jaccard(txt, u[0]) > 0.7 for u in uniq): continue
            seen.add(key); uniq.append((txt, conf, isf))
        if not uniq and cur.state.steps:
            uniq = solver._conclude(question, cur.state)
        for txt, conf, isf in uniq:
            nc += 1
            ns = State(steps=cur.state.steps + [ReasoningStep(txt, conf, isf)],
                       depth=cur.state.depth + 1)
            ng = cur.g_score + 1.0 + (1.0 - conf)
            nh = _h(ns, ents)
            heapq.heappush(open_s, Node(ns, ng, nh, ng + nh, nc, cur.node_id))

    if goal_nodes:
        top, cnt = goal_votes.most_common(1)[0] if goal_votes else ("", 0)
        if cnt >= 2:
            bg = _pick_goal(goal_nodes, top, solver)
            return solver.extract(bg.state.get_trace()), ne
        bg = min(goal_nodes, key=lambda n: n.f_score)
        return solver.extract(bg.state.get_trace()), ne

    fb = best or root
    forced = solver._conclude(question, fb.state)
    if forced:
        txt, conf, _ = forced[0]
        nc += 1
        fs = State(steps=fb.state.steps + [ReasoningStep(txt, conf, False)],
                   depth=fb.state.depth + 1)
        fn = Node(fs, fb.g_score + 1.0, 0.0, fb.g_score + 1.0, nc, fb.node_id)
        return solver.extract(fn.state.get_trace()), ne
    return solver.extract(fb.state.get_trace()), ne


# ─────────────────────────────────────────────────────────────────────────────
# CoT RUNNER
# ─────────────────────────────────────────────────────────────────────────────

def run_cot(question: str, client: OpenAI) -> Optional[str]:
    try:
        resp = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role":"system","content":(
                    "You answer yes/no questions using your world knowledge. "
                    "Think step by step, then end with: The answer is: [yes or no]")},
                {"role":"user","content":(
                    f"Question: {question}\n\nThink step by step.\n"
                    f"End with: The answer is: [yes or no]")},
            ],
            temperature=0.3, max_tokens=512,
        )
        content = resp.choices[0].message.content or ""
        matches = re.findall(r'[Tt]he answer is[:\s]+([^\n.]+)', content)
        if matches:
            raw = matches[-1].strip().lower()
            if re.search(r'\byes\b', raw): return 'yes'
            if re.search(r'\bno\b',  raw): return 'no'
        lines = [l.strip() for l in content.strip().split('\n') if l.strip()]
        if lines:
            last = lines[-1].lower()
            if re.search(r'\byes\b', last): return 'yes'
            if re.search(r'\bno\b',  last): return 'no'
    except Exception:
        pass
    return None


def em(pred: Optional[str], gold: str) -> bool:
    return bool(pred) and pred.lower().strip() == gold.lower().strip()


# ─────────────────────────────────────────────────────────────────────────────
# HEALTH CHECK
# ─────────────────────────────────────────────────────────────────────────────

def check() -> bool:
    ok = True
    print(SEP)
    print("  Performance Experiment — Health Check")
    print(SEP)

    print("\n[1] Ollama …", end=" ")
    try:
        urllib.request.urlopen("http://localhost:11434", timeout=5)
        print("✓  UP")
    except Exception:
        print("✗  NOT RUNNING  →  ollama serve"); ok = False

    print(f"\n[2] Model '{MODEL}' …", end=" ")
    try:
        r     = urllib.request.urlopen("http://localhost:11434/api/tags", timeout=5)
        names = [m["name"] for m in json.loads(r.read()).get("models", [])]
        if any(MODEL.split(":")[0] in n for n in names):
            print("✓  found")
        else:
            print(f"✗  NOT FOUND  →  ollama pull {MODEL}"); ok = False
    except Exception as e:
        print(f"✗  {e}"); ok = False

    print("\n[3] Live generation …", end=" ", flush=True)
    try:
        cl = OpenAI(base_url=OLLAMA_BASE_URL, api_key=OLLAMA_API_KEY)
        t0 = time.time()
        r  = cl.chat.completions.create(
            model=MODEL,
            messages=[{"role":"user","content":"Reply with one word: ready"}],
            max_tokens=8, temperature=0.0
        )
        print(f"✓  '{r.choices[0].message.content.strip()}' ({time.time()-t0:.1f}s)")
    except Exception as e:
        print(f"✗  {e}"); ok = False

    print("\n[4] disagreement.json …", end=" ")
    dp = os.path.join(DIR, "disagreement.json")
    if os.path.exists(dp):
        n = len(json.load(open(dp)))
        print(f"✓  {n} questions")
    else:
        print("✗  MISSING — run _extract.py first or copy the file"); ok = False

    print()
    print(SEP)
    print("  ✓  ALL GOOD" if ok else "  ✗  Fix issues above first")
    print(SEP)
    return ok


# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 — RUN CoT ON DISAGREEMENT SET
# ─────────────────────────────────────────────────────────────────────────────

def step1_cot():
    inp = os.path.join(DIR, "disagreement.json")
    out = os.path.join(DIR, "cot_results.json")

    print(); print(SEP)
    print("  STEP 1 — CoT on all 93 disagreement questions")
    print(SEP)

    questions = json.load(open(inp))
    print(f"  Questions : {len(questions)}")
    print(f"  Model     : {MODEL}\n")

    done: dict = {}
    results: list = []
    if os.path.exists(out):
        results = json.load(open(out))
        done    = {r["question"]: r for r in results}
        print(f"  Resuming — {len(done)} already done\n")

    client    = OpenAI(base_url=OLLAMA_BASE_URL, api_key=OLLAMA_API_KEY)
    correct   = sum(1 for r in results if r.get("correct"))

    for item in tqdm(questions, desc="CoT"):
        q = item["question"]; gold = item["gold"]
        if q in done: continue
        pred = run_cot(q, client)
        c    = em(pred, gold)
        if c: correct += 1
        results.append({"question": q, "gold": gold, "pred": pred, "correct": c,
                        "original_cot_pred": item["cot_pred"],
                        "original_astar_pred": item["astar_pred"]})
        json.dump(results, open(out, "w"), indent=2)

    n = len(results)
    print(f"\n  {SEP2}")
    print(f"  CoT on Disagreement Set")
    print(f"  {SEP2}")
    print(f"  Correct : {correct} / {n}  ({correct/n*100:.1f}%)")
    print(f"  Wrong   : {n-correct} / {n}  ({(n-correct)/n*100:.1f}%)")
    print(f"  Saved → {out}")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 — RUN A* ON DISAGREEMENT SET
# ─────────────────────────────────────────────────────────────────────────────

def step2_astar():
    inp = os.path.join(DIR, "disagreement.json")
    out = os.path.join(DIR, "astar_results.json")

    print(); print(SEP)
    print("  STEP 2 — A* on all 93 disagreement questions")
    print(SEP)

    questions = json.load(open(inp))
    print(f"  Questions : {len(questions)}")
    print(f"  Model     : {MODEL}\n")

    done: dict = {}
    results: list = []
    if os.path.exists(out):
        results = json.load(open(out))
        done    = {r["question"]: r for r in results}
        print(f"  Resuming — {len(done)} already done\n")

    solver  = AStarSolver(model=MODEL, cot_temp=0.2)
    correct = sum(1 for r in results if r.get("correct"))

    for item in tqdm(questions, desc="A*"):
        q = item["question"]; gold = item["gold"]
        if q in done: continue
        try:
            pred, nodes = run_astar(q, solver)
        except Exception:
            pred, nodes = None, 0
        c = em(pred, gold)
        if c: correct += 1
        results.append({"question": q, "gold": gold, "pred": pred, "correct": c,
                        "nodes": nodes,
                        "original_cot_pred": item["cot_pred"],
                        "original_astar_pred": item["astar_pred"]})
        json.dump(results, open(out, "w"), indent=2)

    n         = len(results)
    avg_nodes = sum(r.get("nodes", 0) for r in results) / n if n else 0
    print(f"\n  {SEP2}")
    print(f"  A* on Disagreement Set")
    print(f"  {SEP2}")
    print(f"  Correct   : {correct} / {n}  ({correct/n*100:.1f}%)")
    print(f"  Wrong     : {n-correct} / {n}  ({(n-correct)/n*100:.1f}%)")
    print(f"  Avg nodes : {avg_nodes:.1f}")
    print(f"  Saved → {out}")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 — PERFORMANCE REPORT
# ─────────────────────────────────────────────────────────────────────────────

def step3_report():
    print(); print(SEP)
    print("  STEP 3 — Performance Report")
    print(SEP)

    d_path    = os.path.join(DIR, "disagreement.json")
    cot_path  = os.path.join(DIR, "cot_results.json")
    astr_path = os.path.join(DIR, "astar_results.json")

    if not os.path.exists(d_path):
        print("  ⚠  disagreement.json not found"); return

    disagreements = json.load(open(d_path))
    n = len(disagreements)

    # Original performance on this set
    orig_cot_ok  = sum(1 for r in disagreements if r["cot_correct"])
    orig_astr_ok = sum(1 for r in disagreements if r["astar_correct"])

    print(f"\n  Dataset   : StrategyQA (yes/no) — disagreement set only")
    print(f"  Model     : {MODEL}")
    print(f"  Questions : {n}  (questions where CoT and A* originally DISAGREED)")
    print(f"  (By definition: on the disagreement set, one method was right and the other wrong)")

    print(f"\n  {SEP2}")
    print(f"  ORIGINAL PREDICTIONS  (from first experiment)")
    print(f"  {SEP2}")
    print(f"  CoT correct  : {orig_cot_ok:2d} / {n}  ({orig_cot_ok/n*100:.1f}%)")
    print(f"  A*  correct  : {orig_astr_ok:2d} / {n}  ({orig_astr_ok/n*100:.1f}%)")
    print(f"  Note: these sum to {orig_cot_ok+orig_astr_ok} not {n} — one method was always right per question")

    # New results
    cot_new  = json.load(open(cot_path))  if os.path.exists(cot_path)  else None
    astr_new = json.load(open(astr_path)) if os.path.exists(astr_path) else None

    if cot_new:
        cot_ok = sum(1 for r in cot_new if r["correct"])
        nc     = len(cot_new)
        print(f"\n  {SEP2}")
        print(f"  CoT RE-RUN on all {n} questions")
        print(f"  {SEP2}")
        print(f"  Correct    : {cot_ok} / {nc}  ({cot_ok/nc*100:.1f}%)")
        print(f"  Wrong      : {nc-cot_ok} / {nc}  ({(nc-cot_ok)/nc*100:.1f}%)")
        print(f"  vs original: {orig_cot_ok}/{n} ({orig_cot_ok/n*100:.1f}%)  →  "
              f"{'improved' if cot_ok > orig_cot_ok else 'declined' if cot_ok < orig_cot_ok else 'same'} "
              f"by {abs(cot_ok - orig_cot_ok)} questions")

    if astr_new:
        astr_ok   = sum(1 for r in astr_new if r["correct"])
        na        = len(astr_new)
        avg_nodes = sum(r.get("nodes",0) for r in astr_new) / na if na else 0
        print(f"\n  {SEP2}")
        print(f"  A* RE-RUN on all {n} questions")
        print(f"  {SEP2}")
        print(f"  Correct    : {astr_ok} / {na}  ({astr_ok/na*100:.1f}%)")
        print(f"  Wrong      : {na-astr_ok} / {na}  ({(na-astr_ok)/na*100:.1f}%)")
        print(f"  Avg nodes  : {avg_nodes:.1f}")
        print(f"  vs original: {orig_astr_ok}/{n} ({orig_astr_ok/n*100:.1f}%)  →  "
              f"{'improved' if astr_ok > orig_astr_ok else 'declined' if astr_ok < orig_astr_ok else 'same'} "
              f"by {abs(astr_ok - orig_astr_ok)} questions")

    if cot_new and astr_new:
        cot_ok  = sum(1 for r in cot_new  if r["correct"])
        astr_ok = sum(1 for r in astr_new if r["correct"])
        winner  = "CoT" if cot_ok > astr_ok else "A*" if astr_ok > cot_ok else "TIE"
        print(f"\n  {SEP2}")
        print(f"  HEAD-TO-HEAD  (re-run, same {n} questions)")
        print(f"  {SEP2}")
        print(f"  CoT : {cot_ok:2d} / {n}  ({cot_ok/n*100:.1f}%)")
        print(f"  A*  : {astr_ok:2d} / {n}  ({astr_ok/n*100:.1f}%)")
        if winner == "TIE":
            print(f"  RESULT: TIE")
        else:
            diff = abs(cot_ok - astr_ok)
            print(f"  WINNER: {winner}  (+{diff} questions, +{diff/n*100:.1f}%)")

        # Per-question agreement breakdown on new run
        cot_map  = {r["question"]: r["correct"] for r in cot_new}
        astr_map = {r["question"]: r["correct"] for r in astr_new}
        both_now_ok  = sum(1 for q in cot_map if cot_map.get(q) and astr_map.get(q))
        both_now_bad = sum(1 for q in cot_map if not cot_map.get(q) and not astr_map.get(q))
        only_cot     = sum(1 for q in cot_map if cot_map.get(q) and not astr_map.get(q))
        only_astr    = sum(1 for q in cot_map if not cot_map.get(q) and astr_map.get(q))
        print(f"\n  On the re-run:")
        print(f"  Both correct now : {both_now_ok}")
        print(f"  Both wrong now   : {both_now_bad}  ← consistently hard questions")
        print(f"  Only CoT right   : {only_cot}")
        print(f"  Only A*  right   : {only_astr}")
    else:
        miss = []
        if not cot_new:  miss.append("step 1 (CoT re-run)")
        if not astr_new: miss.append("step 2 (A* re-run)")
        print(f"\n  ⚠  Partial report — still need: {', '.join(miss)}")

    print()


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    global MODEL
    parser = argparse.ArgumentParser(
        description="Run CoT and A* on the 93-question disagreement set",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Steps:
  1  Run CoT  on all 93 disagreement questions
  2  Run A*   on all 93 disagreement questions
  3  Print head-to-head performance report

Examples:
  python3 experiment.py --check
  python3 experiment.py                   run all 3 steps
  python3 experiment.py --only 3          print report (from saved results)
  python3 experiment.py --from 2          skip CoT, run A* then report
  python3 experiment.py --model llama3.1:8b
""")
    parser.add_argument("--check",    action="store_true")
    parser.add_argument("--from",     dest="from_step", type=int, default=1)
    parser.add_argument("--only",     dest="only_step", type=int, default=None)
    parser.add_argument("--no-check", action="store_true")
    parser.add_argument("--model",    default=MODEL)
    args  = parser.parse_args()
    MODEL = args.model

    if args.check:
        sys.exit(0 if check() else 1)

    print(); print(SEP)
    print("  Performance Experiment — Disagreement Set")
    print(f"  93 questions, CoT vs A*, head-to-head")
    print(f"  Model  : {MODEL}")
    print(f"  Server : {OLLAMA_BASE_URL}")
    print(SEP)

    if not args.no_check:
        print("\nHealth check …\n")
        if not check():
            print("\nFix issues above, then re-run."); sys.exit(1)

    steps    = [args.only_step] if args.only_step else list(range(args.from_step, 4))
    step_map = {1: step1_cot, 2: step2_astar, 3: step3_report}
    t0       = time.time()

    for s in steps:
        if s not in step_map:
            print(f"Unknown step {s}"); sys.exit(1)
        try:
            step_map[s]()
        except KeyboardInterrupt:
            print(f"\n\nInterrupted at step {s}. Progress saved.")
            print(f"Resume:  python3 experiment.py --from {s}")
            sys.exit(1)

    print(); print(SEP)
    print(f"  Done in {time.time()-t0:.0f}s")
    print(SEP)


if __name__ == "__main__":
    main()

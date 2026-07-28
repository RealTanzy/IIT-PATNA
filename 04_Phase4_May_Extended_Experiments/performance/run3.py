#!/usr/bin/env python3
"""
Run-3 — Third independent run of CoT and A* on the 93-question disagreement set.

Saves to:
  cot_results_run3.json
  astar_results_run3.json

(Does NOT overwrite cot_results.json / astar_results.json which are Run-2.)

USAGE:
  python3 run3.py                  # both steps
  python3 run3.py --from 2         # skip CoT, run A* only
  python3 run3.py --only 2         # only A*
"""

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────

OLLAMA_BASE_URL = "http://localhost:11434/v1"
OLLAMA_API_KEY  = "ollama"
MODEL           = "llama3.1:8b"

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

import os, re, sys, json, heapq, time, argparse
from collections import Counter
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Set

try:
    from openai import OpenAI
except ImportError:
    print("ERROR: pip install openai"); sys.exit(1)
try:
    from tqdm import tqdm
except ImportError:
    def tqdm(it, **kw): return it

DIR  = os.path.dirname(os.path.abspath(__file__))
SEP  = "=" * 65
SEP2 = "─" * 65

# ─────────────────────────────────────────────────────────────────────────────
# A* SOLVER  (identical to experiment.py)
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


def em(pred, gold):
    return bool(pred) and pred.lower().strip() == gold.lower().strip()


# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 — CoT Run-3
# ─────────────────────────────────────────────────────────────────────────────

def step1_cot():
    inp = os.path.join(DIR, "disagreement.json")
    out = os.path.join(DIR, "cot_results_run3.json")

    print(); print(SEP)
    print("  RUN-3  STEP 1 — CoT on all 93 disagreement questions")
    print(f"  Output → cot_results_run3.json  (Run-2 kept as cot_results.json)")
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

    client  = OpenAI(base_url=OLLAMA_BASE_URL, api_key=OLLAMA_API_KEY)
    correct = sum(1 for r in results if r.get("correct"))

    for item in tqdm(questions, desc="CoT run-3"):
        q = item["question"]; gold = item["gold"]
        if q in done: continue
        pred = run_cot(q, client)
        c    = em(pred, gold)
        if c: correct += 1
        results.append({"question": q, "gold": gold, "pred": pred, "correct": c})
        json.dump(results, open(out, "w"), indent=2)

    n = len(results)
    print(f"\n  {SEP2}")
    print(f"  CoT Run-3 on Disagreement Set")
    print(f"  {SEP2}")
    print(f"  Correct : {correct} / {n}  ({correct/n*100:.1f}%)")
    print(f"  Wrong   : {n-correct} / {n}  ({(n-correct)/n*100:.1f}%)")
    print(f"  Saved → {out}")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 — A* Run-3
# ─────────────────────────────────────────────────────────────────────────────

def step2_astar():
    inp = os.path.join(DIR, "disagreement.json")
    out = os.path.join(DIR, "astar_results_run3.json")

    print(); print(SEP)
    print("  RUN-3  STEP 2 — A* on all 93 disagreement questions")
    print(f"  Output → astar_results_run3.json  (Run-2 kept as astar_results.json)")
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

    for item in tqdm(questions, desc="A* run-3"):
        q = item["question"]; gold = item["gold"]
        if q in done: continue
        try:
            pred, nodes = run_astar(q, solver)
        except Exception:
            pred, nodes = None, 0
        c = em(pred, gold)
        if c: correct += 1
        results.append({"question": q, "gold": gold, "pred": pred, "correct": c,
                        "nodes": nodes})
        json.dump(results, open(out, "w"), indent=2)

    n         = len(results)
    avg_nodes = sum(r.get("nodes", 0) for r in results) / n if n else 0
    print(f"\n  {SEP2}")
    print(f"  A* Run-3 on Disagreement Set")
    print(f"  {SEP2}")
    print(f"  Correct   : {correct} / {n}  ({correct/n*100:.1f}%)")
    print(f"  Wrong     : {n-correct} / {n}  ({(n-correct)/n*100:.1f}%)")
    print(f"  Avg nodes : {avg_nodes:.1f}")
    print(f"  Saved → {out}")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Run-3: third independent CoT and A* run on 93-question disagreement set",
        epilog="""
Steps:
  1  CoT  Run-3  → cot_results_run3.json
  2  A*   Run-3  → astar_results_run3.json

Examples:
  python3 run3.py               run both steps
  python3 run3.py --from 2      skip CoT, run A* only
  python3 run3.py --only 1      run CoT only
""")
    parser.add_argument("--from",  dest="from_step", type=int, default=1)
    parser.add_argument("--only",  dest="only_step", type=int, default=None)
    args = parser.parse_args()

    print(); print(SEP)
    print("  Run-3 — CoT & A* (third independent run)")
    print(f"  Model  : {MODEL}")
    print(f"  Output : cot_results_run3.json  /  astar_results_run3.json")
    print(SEP)

    steps    = [args.only_step] if args.only_step else list(range(args.from_step, 3))
    step_map = {1: step1_cot, 2: step2_astar}
    t0       = time.time()

    for s in steps:
        if s not in step_map:
            print(f"Unknown step {s}"); sys.exit(1)
        try:
            step_map[s]()
        except KeyboardInterrupt:
            print(f"\n\nInterrupted at step {s}. Progress saved.")
            print(f"Resume:  python3 run3.py --from {s}")
            sys.exit(1)

    print(); print(SEP)
    print(f"  Done in {time.time()-t0:.0f}s")
    print(f"  Files written:")
    for fname in ["cot_results_run3.json", "astar_results_run3.json"]:
        p = os.path.join(DIR, fname)
        if os.path.exists(p):
            n = len(json.load(open(p)))
            print(f"    {fname}  ({n} questions)")
    print(SEP)


if __name__ == "__main__":
    main()

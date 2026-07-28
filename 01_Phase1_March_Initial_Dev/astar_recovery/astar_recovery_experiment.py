#!/usr/bin/env python3
"""
A* Recovery Experiment
=======================
Focused re-test: run A* multiple times on the 45 questions
where A* originally failed but CoT succeeded.

Goal: determine whether A*'s failures are FLUKES (recoverable
with another attempt) or SYSTEMATIC (model doesn't know the answer).

Each question is run N_RUNS=5 times independently (different temperatures).
Results are classified per question:
  - never    : 0/5 correct  → systematic failure
  - rare     : 1–2/5        → mostly failing, occasionally lucky
  - often    : 3–4/5        → mostly correct, occasionally unlucky
  - always   : 5/5 correct  → original failure was a fluke

REQUIRES:
    pip install openai tqdm

FILES NEEDED (same directory):
    astar_fail_cot_win.json   ← 45 questions (A* failed, CoT won)

USAGE:
    python3 astar_recovery_experiment.py --check
    python3 astar_recovery_experiment.py
    python3 astar_recovery_experiment.py --from 2
    python3 astar_recovery_experiment.py --only 3
"""

# ============================================================================
# CONFIG
# ============================================================================

OLLAMA_BASE_URL = "http://localhost:11434/v1"
OLLAMA_API_KEY  = "ollama"
MODEL           = "llama3.1:8b"

# 5 independent runs per question — each with a different temperature seed
N_RUNS      = 5
RUN_TEMPS   = [0.2, 0.4, 0.6, 0.3, 0.5]   # one temperature per run

# A* parameters (identical to original experiment for fair comparison)
BRANCH_K    = 2
MAX_DEPTH   = 8
MAX_NODES   = 15
MIN_GOAL_D  = 2
MAJORITY    = 2
TEMPS       = [0.2, 0.6]
W_DEPTH     = 0.3
W_COVERAGE  = 0.3

# ============================================================================
# IMPORTS
# ============================================================================

import os, re, sys, json, heapq, time, argparse, urllib.request
from collections import Counter
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Set

try:
    from openai import OpenAI
except ImportError:
    print("ERROR: openai not installed.  Run:  pip install openai tqdm")
    sys.exit(1)

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(it, **kw): return it

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SEP = "=" * 65


# ============================================================================
# A* SOLVER (embedded — no external imports needed)
# ============================================================================

@dataclass
class ReasoningStep:
    content: str
    confidence: float = 0.8
    is_factual: bool = False

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
    g_score: float
    h_score: float
    f_score: float
    node_id: int
    parent_id: Optional[int] = None

    def __lt__(self, other: "Node") -> bool:
        if abs(self.f_score - other.f_score) > 1e-9:
            return self.f_score < other.f_score
        return self.node_id < other.node_id

_STOPWORDS = frozenset({
    'a','an','the','is','was','were','are','of','in','on','at','for','and','or',
    'to','by','with','it','its','that','this','from','as','be','been','being',
    'have','has','had','do','does','did','will','would','could','should','may',
    'might','can','what','which','who','whom','where','when','how','why','same',
    'both','also','not','no','yes','than','more','most','other','some','any',
    'all','many','much','few','several','first','last','new','old','older','younger',
})

def _extract_entities(question: str) -> Set[str]:
    entities: Set[str] = set()
    for m in re.finditer(r'"([^"]+)"', question):
        entities.add(m.group(1).lower())
    q2 = re.sub(
        r'^(What|Which|Who|Where|When|How|Are|Were|Is|Was|Did|Do|Does|Could|Would|Can|Has|Have|Had|The)\s+',
        '', question, flags=re.IGNORECASE)
    for m in re.finditer(r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b', q2):
        ph = m.group(1).lower()
        if ph not in _STOPWORDS and len(ph) > 2:
            entities.add(ph)
    for m in re.finditer(r'\b(\d{4}|\d+(?:,\d{3})*)\b', question):
        entities.add(m.group(1))
    _GEN = _STOPWORDS | frozenset({'person','people','human','animal','thing','place',
        'object','type','kind','part','year','date','time','name','known','called',
        'named','located','founded','formed','able','capable','possible','likely',
        'usually','generally','often','typically'})
    for w in re.findall(r'\b[a-zA-Z]+\b', question.lower()):
        if w not in _GEN and len(w) > 4:
            entities.add(w)
    return entities

def _coverage(state: State, ents: Set[str]) -> float:
    if not ents: return 1.0
    tr = state.get_trace().lower()
    return sum(1 for e in ents if e in tr) / len(ents)

def _heuristic(state: State, ents: Set[str]) -> float:
    return max(0, 2 - state.depth) * W_DEPTH + (1.0 - _coverage(state, ents)) * W_COVERAGE

def _jaccard(a: str, b: str) -> float:
    wa, wb = set(a.lower().split()), set(b.lower().split())
    if not wa or not wb: return 0.0
    return len(wa & wb) / len(wa | wb)

def _degenerate(content: str) -> bool:
    if len(content) < 10: return False
    s = content.strip()
    if len(set(s)) <= 3 and len(s) > 20: return True
    words = s.split()
    if len(words) > 10 and Counter(words).most_common(1)[0][1] / len(words) > 0.6:
        return True
    return False


class Solver:
    """Self-contained A* solver — identical logic to original experiment."""

    def __init__(self, model: str = MODEL, cot_temp: float = 0.2):
        self.client    = OpenAI(base_url=OLLAMA_BASE_URL, api_key=OLLAMA_API_KEY)
        self.model     = model
        self.cot_temp  = cot_temp   # temperature for COT seed & step-1 strategy A
        self.api_calls = 0
        self.tokens    = 0

    def _llm(self, system: str, prompt: str,
             temp: float = 0.3, max_tokens: int = 512) -> Tuple[Optional[str], float]:
        try:
            r = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role":"system","content":system},
                          {"role":"user","content":prompt}],
                temperature=temp, max_tokens=max_tokens, stop=["\n\n\n"],
            )
            self.api_calls += 1
            self.tokens    += r.usage.total_tokens
            content = r.choices[0].message.content or ""
            if _degenerate(content): return None, 0.0
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

    def _fact_step(self, question: str, state: State, step_num: int) -> List[Tuple[str,float,bool]]:
        candidates = []
        steps_so_far = "\n".join(f"  {i+1}. {s.content}"
                                 for i, s in enumerate(state.steps)) if state.steps else ""
        if step_num == 1:
            strats = [
                {
                    "system": (
                        "You are a knowledgeable assistant that breaks down yes/no questions. "
                        "First identify the key sub-facts needed, then recall them from memory. "
                        "Do NOT give the yes/no answer yet — only gather facts."
                    ),
                    "prompt": (
                        f"Question: {question}\n\nTASK: What intermediate facts do you need to "
                        f"answer this question? Recall 1-2 specific facts from your knowledge.\n\n"
                        f"STEP: [The key facts relevant to this question]\nCONFIDENCE: [0.6-0.95]"
                    ),
                    "temp": self.cot_temp,
                },
                {
                    "system": (
                        "You recall factual information about entities from your world knowledge. "
                        "Be specific and concise. Do not answer yes or no yet."
                    ),
                    "prompt": (
                        f"Question: {question}\n\nTASK: Identify the entities or concepts and "
                        f"recall the relevant property or fact for each one.\n\n"
                        f"STEP: [Entity 1]: [fact]. [Entity 2]: [fact].\nCONFIDENCE: [0.6-0.95]"
                    ),
                    "temp": 0.6,
                },
            ]
            for strat in strats[:BRANCH_K]:
                text, conf = self._llm(strat["system"], strat["prompt"], strat["temp"], 300)
                if text and 'the answer is' not in text.lower():
                    candidates.append((text, conf, True))
                elif text:
                    candidates.append((text, conf, False))
        else:
            sys_p  = "You answer yes/no questions using previously gathered facts."
            usr_p  = (f"Question: {question}\n\nFacts so far:\n{steps_so_far}\n\n"
                      f"TASK: Based on the facts, answer yes or no.\n"
                      f"STEP: The answer is: [yes or no]\nCONFIDENCE: [0.7-1.0]")
            for i in range(min(BRANCH_K, 2)):
                temp_i = TEMPS[i] if i < len(TEMPS) else 0.4
                # Shift temperature by cot_temp offset to vary search per run
                varied_temp = min(0.9, temp_i + (self.cot_temp - 0.2) * 0.5)
                text, conf = self._llm(sys_p, usr_p, varied_temp, 128)
                if text: candidates.append((text, conf, False))
        return candidates

    def _conclude(self, question: str, state: State) -> List[Tuple[str,float,bool]]:
        steps = ("\n".join(f"  {i+1}. {s.content}"
                           for i, s in enumerate(state.steps)) if state.steps else "  (none)")
        text, conf = self._llm(
            "You give concise yes/no answers based on reasoning.",
            f"Question: {question}\n\nReasoning:\n{steps}\n\n"
            f"STEP: The answer is: [yes or no]\nCONFIDENCE: [0.7-1.0]",
            0.1, 128
        )
        if text:
            if 'the answer is' not in text.lower():
                text = f"The answer is: {text}"
            return [(text, conf, False)]
        return [("The answer is: no", 0.5, False)]

    def candidates(self, question: str, state: State, force: bool = False) -> List[Tuple[str,float,bool]]:
        if force: return self._conclude(question, state)
        return self._fact_step(question, state, len(state.steps) + 1)

    def extract(self, trace: str) -> Optional[str]:
        matches = re.findall(r'[Tt]he answer is[:\s]+([^\n]+)', trace)
        if matches:
            raw = re.sub(r'\s*\([^)]*\)\s*$', '',
                   re.sub(r'^(?:Answer|Response|Result)[:\s]+', '',
                          matches[-1].strip().lower().rstrip('.'),
                          flags=re.IGNORECASE)).strip()
            if re.search(r'\byes\b', raw): return 'yes'
            if re.search(r'\bno\b',  raw): return 'no'
            if re.search(r'\b(true|correct|indeed|affirmative)\b', raw): return 'yes'
            if re.search(r'\b(false|incorrect|negative)\b',  raw):       return 'no'
            return raw
        lines = [l.strip() for l in trace.strip().split('\n') if l.strip()]
        if lines:
            last = lines[-1].lower()
            if re.search(r'\byes\b', last): return 'yes'
            if re.search(r'\bno\b',  last): return 'no'
        return None


def run_astar(question: str, solver: Solver, ents: Set[str]) -> Tuple[Optional[str], int, str]:
    """Single A* run. Returns (predicted_answer, nodes_explored, trace)."""
    open_set: List[Node] = []
    nc = ne = 0
    visited: Set[str] = set()
    goal_nodes: List[Node] = []
    goal_votes: Counter = Counter()
    best: Optional[Node] = None

    root = Node(State(), 0.0, 0.0, 0.0, 0)
    root.h_score = root.f_score = _heuristic(root.state, ents)
    heapq.heappush(open_set, root)

    # COT seed
    cot_txt, cot_conf = solver._llm(
        "You answer yes/no questions using your world knowledge. Reason step by step.",
        f"Question: {question}\n\nThink through this step by step.\nEnd with: The answer is: [yes or no]\n",
        solver.cot_temp, 512
    )
    if cot_txt:
        nc += 1
        has_ans   = 'the answer is' in cot_txt.lower()
        cot_state = State(steps=[ReasoningStep(cot_txt, cot_conf, not has_ans)],
                          depth=2 if has_ans else 1)
        cot_g = 1.0 + (1.0 - cot_conf)
        cot_h = _heuristic(cot_state, ents)
        heapq.heappush(open_set, Node(cot_state, cot_g, cot_h, cot_g + cot_h, nc, 0))

    while open_set and ne < MAX_NODES:
        cur = heapq.heappop(open_set)
        sig = cur.state.signature()
        if sig in visited: continue
        visited.add(sig)
        ne += 1

        if cur.state.steps:
            ls = cur.state.steps[-1]
            is_goal = bool(re.search(r'the answer is[:\s]+(yes|no)\b', ls.content.lower()))
            if is_goal and cur.state.depth < MIN_GOAL_D and ls.confidence < 0.75:
                is_goal = False
            if is_goal:
                goal_nodes.append(cur)
                ans = solver.extract(cur.state.get_trace())
                if ans:
                    goal_votes[ans.lower()] += 1
                    if goal_votes[ans.lower()] >= MAJORITY:
                        best_g = _pick_goal(goal_nodes, ans, solver)
                        return solver.extract(best_g.state.get_trace()), ne, best_g.state.get_trace()
                continue

        if best is None or cur.state.depth > best.state.depth:
            best = cur
        if cur.state.depth >= MAX_DEPTH: continue

        force = (cur.state.depth >= MAX_DEPTH - 1)
        cands = solver.candidates(question, cur.state, force)

        existing = [s.content for s in cur.state.steps]
        seen: Set[str] = set()
        uniq: List[Tuple[str,float,bool]] = []
        for txt, conf, isf in cands:
            key = txt.strip().lower()
            if key in seen: continue
            if any(_jaccard(txt, p) > 0.7 for p in existing): continue
            if any(_jaccard(txt, u[0]) > 0.7 for u in uniq): continue
            seen.add(key)
            uniq.append((txt, conf, isf))
        if not uniq and cur.state.steps:
            uniq = solver._conclude(question, cur.state)

        for txt, conf, isf in uniq:
            nc += 1
            ns = State(steps=cur.state.steps + [ReasoningStep(txt, conf, isf)],
                       depth=cur.state.depth + 1)
            ng = cur.g_score + 1.0 + (1.0 - conf)
            nh = _heuristic(ns, ents)
            heapq.heappush(open_set, Node(ns, ng, nh, ng + nh, nc, cur.node_id))

    if goal_nodes:
        top, cnt = goal_votes.most_common(1)[0] if goal_votes else ("", 0)
        if cnt >= 2:
            bg = _pick_goal(goal_nodes, top, solver)
            return solver.extract(bg.state.get_trace()), ne, bg.state.get_trace()
        bg = min(goal_nodes, key=lambda n: n.f_score)
        return solver.extract(bg.state.get_trace()), ne, bg.state.get_trace()

    fb = best or root
    forced = solver._conclude(question, fb.state)
    if forced:
        txt, conf, _ = forced[0]
        nc += 1
        fs = State(steps=fb.state.steps + [ReasoningStep(txt, conf, False)],
                   depth=fb.state.depth + 1)
        fn = Node(fs, fb.g_score + 1.0, 0.0, fb.g_score + 1.0, nc, fb.node_id)
        return solver.extract(fn.state.get_trace()), ne, fn.state.get_trace()

    return solver.extract(fb.state.get_trace()), ne, fb.state.get_trace()


def _pick_goal(goals: List[Node], target: str, solver: Solver) -> Node:
    t = target.lower().strip()
    matching = [n for n in goals if (solver.extract(n.state.get_trace()) or "").lower().strip() == t]
    if matching: return max(matching, key=lambda n: n.state.steps[-1].confidence)
    return min(goals, key=lambda n: n.f_score)


def _em(pred: Optional[str], gold: str) -> bool:
    return bool(pred) and pred.lower().strip().rstrip('.') == gold.lower().strip().rstrip('.')


# ============================================================================
# HEALTH CHECK
# ============================================================================

def check() -> bool:
    all_ok = True
    print(SEP)
    print("  A* Recovery Experiment — Health Check")
    print(SEP)

    print("\n[1] Ollama reachable …", end=" ")
    try:
        urllib.request.urlopen("http://localhost:11434", timeout=5)
        print("✓  UP")
    except Exception:
        print("✗  NOT RUNNING  →  ollama serve")
        all_ok = False

    print(f"\n[2] Model '{MODEL}' available …", end=" ")
    try:
        req   = urllib.request.urlopen("http://localhost:11434/api/tags", timeout=5)
        names = [m["name"] for m in json.loads(req.read()).get("models", [])]
        if any(MODEL.split(":")[0] in n for n in names):
            print("✓  found")
        else:
            print(f"✗  NOT FOUND  →  ollama pull {MODEL}")
            all_ok = False
    except Exception as e:
        print(f"✗  {e}")
        all_ok = False

    print(f"\n[3] Live generation test …", end=" ", flush=True)
    try:
        client = OpenAI(base_url=OLLAMA_BASE_URL, api_key=OLLAMA_API_KEY)
        t0   = time.time()
        resp = client.chat.completions.create(
            model=MODEL,
            messages=[{"role":"user","content":"Reply with one word: ready"}],
            max_tokens=8, temperature=0.0
        )
        print(f"✓  '{resp.choices[0].message.content.strip()}' in {time.time()-t0:.1f}s")
    except Exception as e:
        print(f"✗  {e}")
        all_ok = False

    print(f"\n[4] Input file …", end=" ")
    inp = os.path.join(SCRIPT_DIR, "astar_fail_cot_win.json")
    if os.path.exists(inp):
        n = len(json.load(open(inp)))
        print(f"✓  astar_fail_cot_win.json  ({n} questions)")
    else:
        print("✗  astar_fail_cot_win.json  MISSING")
        print("   →  Copy it from strategyqa_Comparison/astar_fail_cot_win.json")
        all_ok = False

    print()
    print(SEP)
    if all_ok:
        print("  ✓  ALL CHECKS PASSED")
        print(f"\n  Expected runtime: ~{45 * N_RUNS * 10 // 60}–{45 * N_RUNS * 15 // 60} min  ({N_RUNS} runs × 45 questions)")
    else:
        print("  ✗  Issues found — fix above before running")
    print(SEP)
    return all_ok


# ============================================================================
# STEP 1 — RUN A* 5× PER QUESTION
# ============================================================================

def step1_multirun():
    inp_path = os.path.join(SCRIPT_DIR, "astar_fail_cot_win.json")
    out_path = os.path.join(SCRIPT_DIR, "multirun_results.json")

    print(SEP)
    print("  STEP 1 — A* Multi-Run (5× per question)")
    print(SEP)
    print(f"  Questions : 45  (A* original failures)")
    print(f"  Runs/Q    : {N_RUNS}  (temps: {RUN_TEMPS})")
    print(f"  Model     : {MODEL}")
    print(f"  Output    : multirun_results.json")
    print(f"  Safety    : saves after every question")
    print()

    questions = json.load(open(inp_path))

    # Resume support
    done: dict = {}
    results: list = []
    if os.path.exists(out_path):
        results = json.load(open(out_path))
        done    = {r["question"]: r for r in results}
        print(f"  Resuming — {len(done)} already done\n")

    total_recovered = sum(1 for r in results if r["recovered_any"])

    for item in tqdm(questions, desc="A* multi-run"):
        q    = item["question"]
        gold = item["gold"]

        if q in done:
            continue

        q_ents = _extract_entities(q)
        run_results = []
        correct_count = 0

        for run_idx in range(N_RUNS):
            cot_temp = RUN_TEMPS[run_idx]
            solver   = Solver(model=MODEL, cot_temp=cot_temp)
            try:
                pred, nodes, trace = run_astar(q, solver, q_ents)
                correct = _em(pred, gold)
                if correct: correct_count += 1
                run_results.append({
                    "run"       : run_idx + 1,
                    "cot_temp"  : cot_temp,
                    "pred"      : pred,
                    "correct"   : correct,
                    "nodes"     : nodes,
                    "api_calls" : solver.api_calls,
                })
            except Exception as e:
                run_results.append({
                    "run"       : run_idx + 1,
                    "cot_temp"  : cot_temp,
                    "pred"      : None,
                    "correct"   : False,
                    "nodes"     : 0,
                    "api_calls" : 0,
                    "error"     : str(e),
                })

        # Classify recovery pattern
        rate = correct_count / N_RUNS
        if   rate == 0.0:          category = "never"    # 0/5
        elif rate <= 2/N_RUNS:     category = "rare"     # 1-2/5
        elif rate <= 4/N_RUNS:     category = "often"    # 3-4/5
        else:                       category = "always"   # 5/5

        recovered_any = correct_count > 0

        # Majority vote answer across runs
        preds = [r["pred"] for r in run_results if r["pred"]]
        vote  = Counter(preds).most_common(1)[0][0] if preds else None

        if recovered_any:
            total_recovered += 1

        record = {
            "question"             : q,
            "gold"                 : gold,
            "original_astar_pred"  : item.get("astar_pred"),
            "cot_pred"             : item.get("cot_pred"),
            "correct_count"        : correct_count,
            "n_runs"               : N_RUNS,
            "recovery_rate"        : round(rate, 3),
            "category"             : category,
            "majority_vote_pred"   : vote,
            "majority_vote_correct": _em(vote, gold),
            "recovered_any"        : recovered_any,
            "runs"                 : run_results,
        }
        results.append(record)
        # Save without traces (already stripped above) incrementally
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)

    n = len(results)
    cats = Counter(r["category"] for r in results)
    avg_nodes = sum(
        sum(rr.get("nodes", 0) for rr in r["runs"]) / N_RUNS
        for r in results
    ) / n if n else 0

    print(f"\n{'─'*55}")
    print(f"  A* MULTI-RUN COMPLETE  ({n} questions, {N_RUNS} runs each)")
    print(f"{'─'*55}")
    print(f"  Recovered at least once : {total_recovered:3d} / {n}  ({total_recovered/n*100:.1f}%)")
    print(f"  never  (0/{N_RUNS})         : {cats['never']:3d}  ({cats['never']/n*100:.1f}%)  ← systematic failure")
    print(f"  rare   (1-2/{N_RUNS})       : {cats['rare']:3d}  ({cats['rare']/n*100:.1f}%)  ← mostly failing")
    print(f"  often  (3-4/{N_RUNS})       : {cats['often']:3d}  ({cats['often']/n*100:.1f}%)  ← mostly recovering")
    print(f"  always (5/{N_RUNS})         : {cats['always']:3d}  ({cats['always']/n*100:.1f}%)  ← original was a fluke")
    print(f"  Avg nodes/run           : {avg_nodes:.1f}")
    print(f"\n  Saved → {out_path}")


# ============================================================================
# STEP 2 — ANALYSIS REPORT
# ============================================================================

def step2_report():
    out_path = os.path.join(SCRIPT_DIR, "multirun_results.json")

    print(SEP)
    print("  STEP 2 — Analysis Report")
    print(SEP)

    if not os.path.exists(out_path):
        print("  ⚠  multirun_results.json not found — run step 1 first")
        return

    results = json.load(open(out_path))
    n       = len(results)
    cats    = Counter(r["category"] for r in results)
    gold_d  = Counter(r["gold"]     for r in results)

    # Overall recovery stats
    recovered_any  = sum(1 for r in results if r["recovered_any"])
    majority_right = sum(1 for r in results if r.get("majority_vote_correct"))
    avg_rate       = sum(r["recovery_rate"] for r in results) / n

    # Per-run accuracy
    per_run = [0] * N_RUNS
    for r in results:
        for run in r["runs"]:
            if run["correct"]:
                per_run[run["run"]-1] += 1

    # Avg nodes per run
    avg_nodes_per_run = []
    for ri in range(N_RUNS):
        nodes_this_run = [r["runs"][ri]["nodes"] for r in results if len(r["runs"]) > ri]
        avg_nodes_per_run.append(sum(nodes_this_run) / len(nodes_this_run) if nodes_this_run else 0)

    print(f"""
  Dataset   : StrategyQA (yes/no)
  Model     : {MODEL}
  Questions : {n}  (A* original failures — CoT got all of these right)
  Runs/Q    : {N_RUNS}
""")

    print("  ── Recovery Category Distribution ──────────────────────")
    print(f"  never  (0/{N_RUNS}) — systematic failure   : {cats['never']:3d} / {n}  ({cats['never']/n*100:.1f}%)")
    print(f"  rare   (1-2/{N_RUNS}) — mostly failing     : {cats['rare']:3d}  / {n}  ({cats['rare']/n*100:.1f}%)")
    print(f"  often  (3-4/{N_RUNS}) — mostly recovering  : {cats['often']:3d}  / {n}  ({cats['often']/n*100:.1f}%)")
    print(f"  always (5/{N_RUNS}) — fluke failure        : {cats['always']:3d}  / {n}  ({cats['always']/n*100:.1f}%)")
    print()

    print("  ── Overall Recovery ─────────────────────────────────────")
    print(f"  Recovered at least once   : {recovered_any} / {n}  ({recovered_any/n*100:.1f}%)")
    print(f"  Majority-vote correct     : {majority_right} / {n}  ({majority_right/n*100:.1f}%)")
    print(f"  Mean recovery rate        : {avg_rate*100:.1f}%  (avg correct runs per question)")
    print()

    print("  ── Per-Run Accuracy ─────────────────────────────────────")
    for i, (correct, nodes) in enumerate(zip(per_run, avg_nodes_per_run), 1):
        bar = "█" * int(correct / n * 30)
        print(f"  Run {i} (temp={RUN_TEMPS[i-1]})  {correct:3d}/{n} ({correct/n*100:.1f}%)  {bar}  avg_nodes={nodes:.1f}")
    print()

    print("  ── Gold Label Breakdown ─────────────────────────────────")
    for label in ["yes","no"]:
        subset = [r for r in results if r["gold"] == label]
        if not subset: continue
        rec    = sum(1 for r in subset if r["recovered_any"])
        s_cats = Counter(r["category"] for r in subset)
        print(f"  Gold={label}  ({len(subset)} questions)")
        print(f"    Recovered any   : {rec} / {len(subset)}  ({rec/len(subset)*100:.1f}%)")
        print(f"    never/rare/often/always : {s_cats['never']}/{s_cats['rare']}/{s_cats['often']}/{s_cats['always']}")
    print()

    print("  ── Systematic Failures (never recovered in 5 runs) ──────")
    never_list = [r for r in results if r["category"] == "never"]
    for i, r in enumerate(never_list[:15], 1):
        orig = r.get("original_astar_pred","?")
        print(f"  {i:2d}. [{r['gold']}] A*_orig={orig}  Q: {r['question'][:65]}")
    if len(never_list) > 15:
        print(f"  ... and {len(never_list)-15} more")
    print()

    print("  ── Fluke Failures (always correct on re-run) ────────────")
    always_list = [r for r in results if r["category"] == "always"]
    for i, r in enumerate(always_list, 1):
        print(f"  {i:2d}. [{r['gold']}]  Q: {r['question'][:65]}")
    print()

    print("  ── Interpretation ───────────────────────────────────────")
    systematic_pct = cats["never"] / n * 100
    fluke_pct      = cats["always"] / n * 100
    borderline_pct = (cats["rare"] + cats["often"]) / n * 100
    print(f"  {systematic_pct:.0f}% of A*'s failures are systematic — A* consistently")
    print(f"    converges on the wrong reasoning path for these questions.")
    print(f"  {fluke_pct:.0f}% were pure flukes — A* gets them right every subsequent time.")
    print(f"  {borderline_pct:.0f}% are borderline — the question sits at A*'s decision")
    print(f"    boundary and temperature variation tips the answer either way.")
    print()

    # Save text report
    report_path = os.path.join(SCRIPT_DIR, "analysis_report.txt")
    import io
    # Write a clean summary to file
    lines = [
        "A* Recovery Experiment — Analysis Report",
        f"Model: {MODEL} | Questions: {n} | Runs/Q: {N_RUNS}",
        "",
        "RECOVERY CATEGORY DISTRIBUTION",
        f"  never  (0/{N_RUNS}) — systematic failure   : {cats['never']:3d} / {n}  ({cats['never']/n*100:.1f}%)",
        f"  rare   (1-2/{N_RUNS}) — mostly failing     : {cats['rare']:3d}  / {n}  ({cats['rare']/n*100:.1f}%)",
        f"  often  (3-4/{N_RUNS}) — mostly recovering  : {cats['often']:3d}  / {n}  ({cats['often']/n*100:.1f}%)",
        f"  always (5/{N_RUNS}) — fluke failure        : {cats['always']:3d}  / {n}  ({cats['always']/n*100:.1f}%)",
        "",
        "OVERALL RECOVERY",
        f"  Recovered at least once : {recovered_any} / {n}  ({recovered_any/n*100:.1f}%)",
        f"  Majority-vote correct   : {majority_right} / {n}  ({majority_right/n*100:.1f}%)",
        f"  Mean recovery rate      : {avg_rate*100:.1f}%",
        "",
        "PER-RUN ACCURACY",
    ]
    for i, (correct, nodes) in enumerate(zip(per_run, avg_nodes_per_run), 1):
        lines.append(f"  Run {i} (temp={RUN_TEMPS[i-1]})  {correct}/{n} ({correct/n*100:.1f}%)  avg_nodes={nodes:.1f}")
    lines += [
        "",
        "SYSTEMATIC FAILURES (0/5 runs correct)",
    ]
    for r in never_list:
        lines.append(f"  [{r['gold']}] A*={r.get('original_astar_pred','?')}  {r['question']}")
    with open(report_path, "w") as f:
        f.write("\n".join(lines))
    print(f"  Full report saved → {report_path}")


# ============================================================================
# MAIN
# ============================================================================

def main():
    global MODEL
    parser = argparse.ArgumentParser(
        description="A* Recovery — multi-run experiment on A* failures",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 astar_recovery_experiment.py --check       verify server
  python3 astar_recovery_experiment.py               run all (step1 + step2)
  python3 astar_recovery_experiment.py --only 1      run A* 5x only
  python3 astar_recovery_experiment.py --only 2      print report only
  python3 astar_recovery_experiment.py --from 2      skip to report
  python3 astar_recovery_experiment.py --model llama3.1:8b
""")
    parser.add_argument("--check",    action="store_true")
    parser.add_argument("--from",     dest="from_step", type=int, default=1)
    parser.add_argument("--only",     dest="only_step", type=int, default=None)
    parser.add_argument("--no-check", action="store_true")
    parser.add_argument("--model",    default=MODEL)
    args = parser.parse_args()
    MODEL = args.model

    if args.check:
        sys.exit(0 if check() else 1)

    print()
    print(SEP)
    print("  A* Recovery Experiment")
    print(f"  Goal    : why does A* fail where CoT succeeds?")
    print(f"  Method  : run A* {N_RUNS}× per question (temps: {RUN_TEMPS})")
    print(f"  Model   : {MODEL}")
    print(f"  Server  : {OLLAMA_BASE_URL}")
    print(SEP)

    if not args.no_check:
        print("\nRunning health check …\n")
        if not check():
            print("\nFix issues above, then re-run.")
            sys.exit(1)

    steps = [args.only_step] if args.only_step else list(range(args.from_step, 3))
    t0 = time.time()
    step_map = {1: step1_multirun, 2: step2_report}

    for s in steps:
        if s not in step_map:
            print(f"Unknown step: {s}")
            sys.exit(1)
        try:
            step_map[s]()
        except KeyboardInterrupt:
            print(f"\n\nInterrupted at step {s}. Results saved incrementally.")
            print(f"Resume with:  python3 astar_recovery_experiment.py --from {s}")
            sys.exit(1)

    print()
    print(SEP)
    print(f"  Done in {time.time()-t0:.0f}s")
    print(SEP)


if __name__ == "__main__":
    main()

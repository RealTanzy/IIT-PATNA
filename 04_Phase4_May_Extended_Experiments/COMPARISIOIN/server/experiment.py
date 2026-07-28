#!/usr/bin/env python3
"""
CoT vs A* Comparison — Self-Contained Experiment
==================================================
Copy this file + the two input JSON files to any machine with Ollama running.
Everything is embedded — no other local imports needed.

REQUIRES:
    pip install openai tqdm

FILES NEEDED (same directory as this script):
    cot_fail_astar_win.json    ← 49 questions CoT failed / A* won
    astar_fail_cot_win.json    ← 45 questions A* failed / CoT won

USAGE:
    python3 experiment.py --check            # verify Ollama is up
    python3 experiment.py                    # run all steps
    python3 experiment.py --from 2           # resume from step 2
    python3 experiment.py --only 3           # run only step 3
    python3 experiment.py --only 4           # print final report
"""

# ============================================================================
# ██████████  CONFIG — edit this block only  █████████████████████████████████
# ============================================================================

OLLAMA_BASE_URL = "http://localhost:11434/v1"
OLLAMA_API_KEY  = "ollama"
MODEL           = "llama3.1:8b"

# CoT re-run: 2 independent attempts per question
COT_N_RUNS  = 2
COT_TEMPS   = [0.2, 0.5]

# A* search parameters (must match original run to be comparable)
ASTAR_BRANCH_K    = 2
ASTAR_MAX_DEPTH   = 8
ASTAR_MAX_NODES   = 15
ASTAR_MIN_GOAL_D  = 2
ASTAR_MAJORITY    = 2
ASTAR_TEMPS       = [0.2, 0.6]
ASTAR_W_DEPTH     = 0.3
ASTAR_W_COVERAGE  = 0.3

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
    print("ERROR: openai package not found.  Run:  pip install openai tqdm")
    sys.exit(1)

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(it, **kw): return it   # silent fallback

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SEP = "=" * 65

# ============================================================================
# ── A* DATA STRUCTURES ──────────────────────────────────────────────────────
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

# ============================================================================
# ── HEURISTIC HELPERS ───────────────────────────────────────────────────────
# ============================================================================

_STOPWORDS = frozenset({
    'a','an','the','is','was','were','are','of','in','on','at','for','and','or',
    'to','by','with','it','its','that','this','from','as','be','been','being',
    'have','has','had','do','does','did','will','would','could','should','may',
    'might','can','what','which','who','whom','where','when','how','why','same',
    'both','also','not','no','yes','than','more','most','other','some','any',
    'all','many','much','few','several','first','last','new','old','older','younger',
})

def extract_question_entities(question: str) -> Set[str]:
    entities: Set[str] = set()
    for m in re.finditer(r'"([^"]+)"', question):
        entities.add(m.group(1).lower())
    q_trimmed = re.sub(
        r'^(What|Which|Who|Where|When|How|Are|Were|Is|Was|Did|Do|Does|Could|Would|Can|Has|Have|Had|The)\s+',
        '', question, flags=re.IGNORECASE)
    for m in re.finditer(r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b', q_trimmed):
        ph = m.group(1).lower()
        if ph not in _STOPWORDS and len(ph) > 2:
            entities.add(ph)
    for m in re.finditer(r'\b(\d{4}|\d+(?:,\d{3})*)\b', question):
        entities.add(m.group(1))
    _GENERIC = _STOPWORDS | frozenset({
        'person','people','human','animal','thing','place','object','type','kind',
        'part','year','date','time','name','known','called','named','located',
        'founded','formed','able','capable','possible','likely','usually',
        'generally','often','typically',
    })
    for w in re.findall(r'\b[a-zA-Z]+\b', question.lower()):
        if w not in _GENERIC and len(w) > 4:
            entities.add(w)
    return entities

def compute_entity_coverage(state: State, q_entities: Set[str]) -> float:
    if not q_entities:
        return 1.0
    trace = state.get_trace().lower()
    return sum(1 for e in q_entities if e in trace) / len(q_entities)

def compute_heuristic(state: State, q_entities: Set[str]) -> float:
    h_depth    = max(0, 2 - state.depth) * ASTAR_W_DEPTH
    h_coverage = (1.0 - compute_entity_coverage(state, q_entities)) * ASTAR_W_COVERAGE
    return h_depth + h_coverage

def jaccard_sim(a: str, b: str) -> float:
    wa, wb = set(a.lower().split()), set(b.lower().split())
    if not wa or not wb: return 0.0
    return len(wa & wb) / len(wa | wb)

def _is_degenerate(content: str) -> bool:
    if len(content) < 10: return False
    s = content.strip()
    if len(set(s)) <= 3 and len(s) > 20: return True
    words = s.split()
    if len(words) > 10:
        if Counter(words).most_common(1)[0][1] / len(words) > 0.6: return True
    return False

# ============================================================================
# ── A* SOLVER ───────────────────────────────────────────────────────────────
# ============================================================================

class StrategyQASolver:
    def __init__(self, model: str = MODEL, verbose: bool = False,
                 base_url: str = OLLAMA_BASE_URL):
        self.client     = OpenAI(base_url=base_url, api_key=OLLAMA_API_KEY)
        self.model      = model
        self.verbose    = verbose
        self.api_calls  = 0
        self.total_tokens = 0

    def _call_llm(self, system: str, prompt: str,
                  temperature: float = 0.3, max_tokens: int = 512
                  ) -> Tuple[Optional[str], float]:
        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role":"system","content":system},
                          {"role":"user","content":prompt}],
                temperature=temperature, max_tokens=max_tokens,
                stop=["\n\n\n"],
            )
            self.api_calls    += 1
            self.total_tokens += resp.usage.total_tokens
            content = resp.choices[0].message.content or ""
            if _is_degenerate(content): return None, 0.0
            return self._parse_output(content)
        except Exception as e:
            if self.verbose: print(f"    LLM error: {e}")
            return None, 0.0

    def _parse_output(self, content: str) -> Tuple[Optional[str], float]:
        step_m = re.search(r'STEP(?:\s*\d*)?:\s*(.+?)(?=\n(?:STEP|CONFIDENCE)|$)',
                           content, re.IGNORECASE | re.DOTALL)
        conf_m = re.search(r'CONFIDENCE:\s*([0-9.]+)', content, re.IGNORECASE)
        if step_m:
            text = step_m.group(1).strip()
            conf = max(0.5, min(0.99, float(conf_m.group(1)) if conf_m else 0.75))
            if len(text) > 3: return text, conf
        ans_m = re.search(r'[Tt]he answer is[:\s]+(.+?)(?:\.|$)', content)
        if ans_m:
            conf = max(0.5, min(0.99, float(conf_m.group(1)) if conf_m else 0.75))
            return f"The answer is: {ans_m.group(1).strip()}", conf
        text = re.sub(r'^(?:Step\s*\d+[:\s]*|STEP[:\s]*)', '',
                      content.strip(), flags=re.IGNORECASE).strip()
        if 5 < len(text) < 500: return text, 0.6
        return None, 0.0

    def _generate_fact_step(self, question: str, state: State,
                             step_num: int) -> List[Tuple[str, float, bool]]:
        candidates = []
        steps_so_far = "\n".join(f"  {i+1}. {s.content}"
                                 for i, s in enumerate(state.steps)) if state.steps else ""
        if step_num == 1:
            strategies = [
                {
                    "system": (
                        "You are a knowledgeable assistant that breaks down yes/no questions. "
                        "First identify the key sub-facts needed, then recall them from memory. "
                        "Do NOT give the yes/no answer yet — only gather facts."
                    ),
                    "prompt": (
                        f"Question: {question}\n\nTASK: What intermediate facts do you need to answer "
                        f"this question? Recall 1-2 specific facts from your knowledge.\n\n"
                        f"Example: 'Could a penguin outrun a human?' → "
                        f"'Penguins waddle at ~2.5 mph. Average human walks at 3.5 mph.'\n\n"
                        f"STEP: [The key facts relevant to this question]\nCONFIDENCE: [0.6-0.95]"
                    ),
                },
                {
                    "system": (
                        "You recall factual information about entities from your world knowledge. "
                        "Be specific and concise. Do not answer yes or no yet."
                    ),
                    "prompt": (
                        f"Question: {question}\n\nTASK: Identify the entities or concepts in this "
                        f"question and recall the relevant property or fact for each one.\n\n"
                        f"STEP: [Entity 1]: [relevant fact]. [Entity 2]: [relevant fact].\n"
                        f"CONFIDENCE: [0.6-0.95]"
                    ),
                },
            ]
            for i, strat in enumerate(strategies[:ASTAR_BRANCH_K]):
                temp = ASTAR_TEMPS[i] if i < len(ASTAR_TEMPS) else 0.4
                text, conf = self._call_llm(strat["system"], strat["prompt"], temp, 300)
                if text and 'the answer is' not in text.lower():
                    candidates.append((text, conf, True))
                elif text:
                    candidates.append((text, conf, False))
        else:
            strat_prompt = (
                f"Question: {question}\n\nFacts gathered so far:\n{steps_so_far}\n\n"
                f"TASK: Based on the facts above, answer the question.\n"
                f"- If the facts support the claim → 'yes'\n"
                f"- If the facts contradict the claim → 'no'\n\n"
                f"STEP: The answer is: [yes or no]\nCONFIDENCE: [0.7-1.0]"
            )
            sys_prompt = "You answer yes/no questions using previously gathered facts. Answer ONLY 'yes' or 'no'."
            for i in range(min(ASTAR_BRANCH_K, 2)):
                text, conf = self._call_llm(sys_prompt, strat_prompt, ASTAR_TEMPS[i], 128)
                if text: candidates.append((text, conf, False))
        return candidates

    def generate_candidates(self, question: str, state: State,
                            force_conclusion: bool = False) -> List[Tuple[str, float, bool]]:
        if force_conclusion:
            return self._generate_conclusion(question, state)
        return self._generate_fact_step(question, state, len(state.steps) + 1)

    def _generate_conclusion(self, question: str,
                              state: State) -> List[Tuple[str, float, bool]]:
        steps_so_far = "\n".join(f"  {i+1}. {s.content}"
                                 for i, s in enumerate(state.steps)) if state.steps else "  (none)"
        text, conf = self._call_llm(
            "You give concise yes/no answers based on reasoning.",
            f"Question: {question}\n\nReasoning so far:\n{steps_so_far}\n\n"
            f"TASK: Based on your reasoning, answer yes or no.\n\n"
            f"STEP: The answer is: [yes or no]\nCONFIDENCE: [0.7-1.0]",
            0.1, 128
        )
        if text:
            if 'the answer is' not in text.lower():
                text = f"The answer is: {text}"
            return [(text, conf, False)]
        return [("The answer is: no", 0.5, False)]

    def extract_answer(self, trace: str) -> Optional[str]:
        matches = re.findall(r'[Tt]he answer is[:\s]+([^\n]+)', trace)
        if matches:
            raw = re.sub(r'\s*\([^)]*\)\s*$', '',
                   re.sub(r'^(?:Answer|Response|Result)[:\s]+', '',
                          matches[-1].strip().lower().rstrip('.'),
                          flags=re.IGNORECASE)).strip()
            if re.search(r'\byes\b', raw): return 'yes'
            if re.search(r'\bno\b',  raw): return 'no'
            if re.search(r'\b(true|correct|indeed|affirmative)\b', raw): return 'yes'
            if re.search(r'\b(false|incorrect|negative)\b', raw):        return 'no'
            return raw
        lines = [l.strip() for l in trace.strip().split('\n') if l.strip()]
        if lines:
            last = lines[-1].lower()
            if re.search(r'\byes\b', last): return 'yes'
            if re.search(r'\bno\b',  last): return 'no'
        return None


def astar_search(question: str, solver: StrategyQASolver,
                 q_entities: Set[str] = None,
                 verbose: bool = False) -> Tuple[Node, int]:
    if q_entities is None: q_entities = set()
    open_set: List[Node] = []
    node_counter = nodes_explored = 0
    visited: Set[str] = set()
    goal_nodes: List[Node] = []
    goal_votes: Counter = Counter()
    best_current: Optional[Node] = None

    root = Node(state=State(), g_score=0.0, h_score=0.0, f_score=0.0, node_id=0)
    root.h_score = root.f_score = compute_heuristic(root.state, q_entities)
    heapq.heappush(open_set, root)

    # COT seed
    cot_text, cot_conf = solver._call_llm(
        "You answer yes/no questions using your world knowledge. Reason step by step, then give a clear yes or no answer.",
        f"Question: {question}\n\nThink through this step by step using your world knowledge.\nEnd with: The answer is: [yes or no]\n",
        0.2, 512
    )
    if cot_text:
        node_counter += 1
        has_ans   = 'the answer is' in cot_text.lower()
        cot_state = State(steps=[ReasoningStep(cot_text, cot_conf, not has_ans)],
                          depth=2 if has_ans else 1)
        cot_g = 1.0 + (1.0 - cot_conf)
        cot_h = compute_heuristic(cot_state, q_entities)
        heapq.heappush(open_set, Node(cot_state, cot_g, cot_h, cot_g + cot_h, node_counter, 0))

    while open_set and nodes_explored < ASTAR_MAX_NODES:
        current = heapq.heappop(open_set)
        sig = current.state.signature()
        if sig in visited: continue
        visited.add(sig)
        nodes_explored += 1

        if current.state.steps:
            last_step = current.state.steps[-1]
            last_text = last_step.content.lower()
            is_goal = bool(re.search(r'the answer is[:\s]+(yes|no)\b', last_text))
            if is_goal and current.state.depth < ASTAR_MIN_GOAL_D and last_step.confidence < 0.75:
                is_goal = False
            if is_goal:
                goal_nodes.append(current)
                ans = solver.extract_answer(current.state.get_trace())
                if ans:
                    goal_votes[ans.lower()] += 1
                    if goal_votes[ans.lower()] >= ASTAR_MAJORITY:
                        return _best_goal(goal_nodes, ans, solver), nodes_explored
                continue

        if best_current is None or current.state.depth > best_current.state.depth:
            best_current = current
        if current.state.depth >= ASTAR_MAX_DEPTH:
            continue

        force = (current.state.depth >= ASTAR_MAX_DEPTH - 1)
        candidates = solver.generate_candidates(question, current.state, force)

        existing = [s.content for s in current.state.steps]
        seen_keys: Set[str] = set()
        unique: List[Tuple[str, float, bool]] = []
        for text, conf, is_factual in candidates:
            key = text.strip().lower()
            if key in seen_keys: continue
            if any(jaccard_sim(text, p) > 0.7 for p in existing): continue
            if any(jaccard_sim(text, u[0]) > 0.7 for u in unique): continue
            seen_keys.add(key)
            unique.append((text, conf, is_factual))
        if not unique and current.state.steps:
            unique = solver._generate_conclusion(question, current.state)

        for text, conf, is_factual in unique:
            node_counter += 1
            new_state = State(steps=current.state.steps + [ReasoningStep(text, conf, is_factual)],
                              depth=current.state.depth + 1)
            new_g = current.g_score + 1.0 + (1.0 - conf)
            new_h = compute_heuristic(new_state, q_entities)
            heapq.heappush(open_set, Node(new_state, new_g, new_h, new_g + new_h,
                                          node_counter, current.node_id))

    if goal_nodes:
        top_ans, top_cnt = goal_votes.most_common(1)[0] if goal_votes else ("", 0)
        if top_cnt >= 2:
            return _best_goal(goal_nodes, top_ans, solver), nodes_explored
        return min(goal_nodes, key=lambda n: n.f_score), nodes_explored

    fallback = best_current or root
    forced = solver._generate_conclusion(question, fallback.state)
    if forced:
        text, conf, _ = forced[0]
        node_counter += 1
        fs = State(steps=fallback.state.steps + [ReasoningStep(text, conf, False)],
                   depth=fallback.state.depth + 1)
        return Node(fs, fallback.g_score + 1.0, 0.0, fallback.g_score + 1.0, node_counter,
                    fallback.node_id), nodes_explored
    return fallback, nodes_explored


def _best_goal(goal_nodes: List[Node], target: str, solver: StrategyQASolver) -> Node:
    t = target.lower().strip()
    matching = [n for n in goal_nodes
                if (solver.extract_answer(n.state.get_trace()) or "").lower().strip() == t]
    if matching:
        return max(matching, key=lambda n: n.state.steps[-1].confidence)
    return min(goal_nodes, key=lambda n: n.f_score)


def _em(pred: Optional[str], gold: str) -> bool:
    return bool(pred) and pred.lower().strip().rstrip('.') == gold.lower().strip().rstrip('.')


# ============================================================================
# ── HEALTH CHECK ────────────────────────────────────────────────────────────
# ============================================================================

def check_server() -> bool:
    all_ok = True
    print(SEP)
    print("  Health Check")
    print(SEP)

    # 1. Ollama reachable
    print("\n[1] Ollama reachable at localhost:11434 …", end=" ")
    try:
        urllib.request.urlopen("http://localhost:11434", timeout=5)
        print("✓  UP")
    except Exception:
        print("✗  NOT RUNNING")
        print("   → Start with:  ollama serve")
        all_ok = False

    # 2. Model available
    print(f"\n[2] Model '{MODEL}' available …", end=" ")
    try:
        req  = urllib.request.urlopen("http://localhost:11434/api/tags", timeout=5)
        data = json.loads(req.read())
        names = [m["name"] for m in data.get("models", [])]
        if any(MODEL.split(":")[0] in n for n in names):
            print("✓  found")
        else:
            print(f"✗  NOT FOUND  (available: {names})")
            print(f"   → Pull with:  ollama pull {MODEL}")
            all_ok = False
    except Exception as e:
        print(f"✗  Error: {e}")
        all_ok = False

    # 3. Live generation
    print("\n[3] Live generation test …", end=" ", flush=True)
    try:
        client = OpenAI(base_url=OLLAMA_BASE_URL, api_key=OLLAMA_API_KEY)
        t0   = time.time()
        resp = client.chat.completions.create(
            model=MODEL,
            messages=[{"role":"user","content":"Reply with one word: ready"}],
            max_tokens=8, temperature=0.0
        )
        reply   = resp.choices[0].message.content.strip()
        elapsed = time.time() - t0
        print(f"✓  replied '{reply}' in {elapsed:.1f}s")
    except Exception as e:
        print(f"✗  {e}")
        all_ok = False

    # 4. Input files
    print("\n[4] Input files …")
    for fname in ["cot_fail_astar_win.json", "astar_fail_cot_win.json"]:
        path = os.path.join(SCRIPT_DIR, fname)
        if os.path.exists(path):
            n = len(json.load(open(path)))
            print(f"    ✓  {fname}  ({n} questions)")
        else:
            print(f"    ✗  {fname}  MISSING")
            print(f"       → Copy it from your local strategyqa_Comparison/ folder")
            all_ok = False

    print()
    print(SEP)
    if all_ok:
        print("  ✓  ALL CHECKS PASSED — run:  python3 experiment.py")
    else:
        print("  ✗  ISSUES FOUND — fix above then re-run --check")
    print(SEP)
    return all_ok


# ============================================================================
# ── STEP 1: FIND DISAGREEMENTS ──────────────────────────────────────────────
# ── (optional — only needed if you have the raw result files here too) ───────
# ============================================================================

def step1_find_disagreements():
    """
    Build cot_fail_astar_win.json and astar_fail_cot_win.json from raw result files.
    Requires: 42_cot_question_only_results.jsonl  and
              Astar_llama3.1_8b_42_strategyqa_astar_results_final.json
    Skip this step if you already copied the two JSON files directly.
    """
    cot_path   = os.path.join(SCRIPT_DIR, "42_cot_question_only_results.jsonl")
    astar_path = os.path.join(SCRIPT_DIR, "Astar_llama3.1_8b_42_strategyqa_astar_results_final.json")

    if not os.path.exists(cot_path) or not os.path.exists(astar_path):
        print("Step 1 skipped — raw result files not present.")
        print("Make sure cot_fail_astar_win.json and astar_fail_cot_win.json are here instead.")
        return

    print("Loading CoT results …")
    cot_by_q = {}
    with open(cot_path) as f:
        for line in f:
            d = json.loads(line.strip())
            gold = "yes" if d["gold"] else "no"
            pred = "yes" if d["predicted"] else "no"
            cot_by_q[d["question"].strip()] = {
                "question": d["question"].strip(), "gold": gold,
                "pred": pred, "correct": d["correct"],
            }

    print("Loading A* results …")
    astar_by_q = {}
    for d in json.load(open(astar_path)):
        astar_by_q[d["question"].strip()] = d

    common = set(cot_by_q) & set(astar_by_q)
    print(f"Matched: {len(common)}")

    cot_fail_win, astar_fail_win = [], []
    for q in sorted(common):
        cot, astar = cot_by_q[q], astar_by_q[q]
        rec = {
            "question": q, "gold": cot["gold"],
            "cot_pred": cot["pred"],   "cot_correct": cot["correct"],
            "astar_pred": astar.get("pred"), "astar_correct": astar["correct"],
            "astar_nodes": astar.get("nodes", 0),
        }
        if not cot["correct"] and astar["correct"]:
            cot_fail_win.append(rec)
        elif not astar["correct"] and cot["correct"]:
            astar_fail_win.append(rec)

    _save("cot_fail_astar_win.json",  cot_fail_win)
    _save("astar_fail_cot_win.json",  astar_fail_win)
    print(f"CoT-fail / A*-win : {len(cot_fail_win)}")
    print(f"A*-fail / CoT-win : {len(astar_fail_win)}")


# ============================================================================
# ── STEP 2: RE-RUN CoT on CoT failures ──────────────────────────────────────
# ============================================================================

COT_SYSTEM = (
    "You are a knowledgeable assistant. Answer yes/no questions by reasoning "
    "step by step from your world knowledge. End your response with exactly: "
    "'Final Answer: Yes' or 'Final Answer: No'."
)

def _cot_prompt(question: str) -> str:
    return (f"Question: {question}\n\n"
            f"Reason through this step by step. "
            f"End with: Final Answer: Yes  or  Final Answer: No")

def _extract_cot_answer(response: str) -> Optional[str]:
    m = re.search(r'[Ff]inal\s+[Aa]nswer[:\s*]+\**(yes|no)\**', response, re.IGNORECASE)
    if m: return m.group(1).lower()
    m = re.search(r'[Tt]he answer is[:\s]+(yes|no)\b', response, re.IGNORECASE)
    if m: return m.group(1).lower()
    lines = [l.strip() for l in response.strip().split('\n') if l.strip()]
    if lines:
        last = lines[-1].lower()
        if re.search(r'\byes\b', last): return 'yes'
        if re.search(r'\bno\b',  last): return 'no'
    return None

def step2_rerun_cot():
    input_path  = os.path.join(SCRIPT_DIR, "cot_fail_astar_win.json")
    output_path = os.path.join(SCRIPT_DIR, "cot_rerun_on_cot_failures.json")

    print(SEP)
    print("  STEP 2 — Re-run CoT on its 49 failures")
    print(SEP)

    questions = json.load(open(input_path))
    print(f"  Questions   : {len(questions)}")
    print(f"  Runs/Q      : {COT_N_RUNS}  (temps: {COT_TEMPS})")
    print(f"  Output      : {os.path.basename(output_path)}")
    print()

    # Resume if partially done
    done_qs = set()
    results = []
    if os.path.exists(output_path):
        results = json.load(open(output_path))
        done_qs = {r["question"] for r in results}
        print(f"  Resuming — {len(done_qs)} already done")

    client    = OpenAI(base_url=OLLAMA_BASE_URL, api_key=OLLAMA_API_KEY)
    recovered = sum(1 for r in results if r.get("rerun_cot_correct"))
    total     = len(results)

    for item in tqdm(questions, desc="CoT re-run"):
        if item["question"] in done_qs:
            continue

        question = item["question"]
        gold     = item["gold"]
        run_results = []

        for i in range(COT_N_RUNS):
            temp = COT_TEMPS[i] if i < len(COT_TEMPS) else 0.3
            try:
                resp = client.chat.completions.create(
                    model=MODEL,
                    messages=[{"role":"system","content":COT_SYSTEM},
                              {"role":"user","content":_cot_prompt(question)}],
                    temperature=temp, max_tokens=512,
                )
                response_text = resp.choices[0].message.content or ""
                pred    = _extract_cot_answer(response_text)
                correct = _em(pred, gold)
                run_results.append({"run": i+1, "temp": temp, "pred": pred,
                                    "correct": correct, "response": response_text})
            except Exception as e:
                run_results.append({"run": i+1, "temp": temp, "pred": None,
                                    "correct": False, "error": str(e)})

        preds        = [r["pred"] for r in run_results if r["pred"]]
        vote         = Counter(preds).most_common(1)[0][0] if preds else None
        vote_correct = _em(vote, gold)
        if vote_correct: recovered += 1
        total += 1

        results.append({
            "question"            : question,
            "gold"                : gold,
            "original_cot_pred"   : item.get("cot_pred"),
            "original_cot_correct": False,
            "astar_pred"          : item.get("astar_pred"),
            "astar_correct"       : True,
            "rerun_cot_vote"      : vote,
            "rerun_cot_correct"   : vote_correct,
            "runs"                : run_results,
        })
        _save(output_path, results, abs_path=True)

    n = len(results)
    print(f"\n  CoT recovered : {recovered} / {n}  ({recovered/n*100:.1f}%)")
    print(f"  Still failing : {n-recovered} / {n}  ({(n-recovered)/n*100:.1f}%)")
    print(f"  Saved → {output_path}")


# ============================================================================
# ── STEP 3: RE-RUN A* on A* failures ────────────────────────────────────────
# ============================================================================

def step3_rerun_astar():
    input_path  = os.path.join(SCRIPT_DIR, "astar_fail_cot_win.json")
    output_path = os.path.join(SCRIPT_DIR, "astar_rerun_on_astar_failures.json")

    print(SEP)
    print("  STEP 3 — Re-run A* on its 45 failures")
    print(SEP)

    questions = json.load(open(input_path))
    print(f"  Questions   : {len(questions)}")
    print(f"  Max nodes   : {ASTAR_MAX_NODES}  |  Max depth: {ASTAR_MAX_DEPTH}")
    print(f"  Output      : {os.path.basename(output_path)}")
    print(f"  Note        : Saves after every question — safe to interrupt")
    print()

    # Resume if partially done
    done_qs = set()
    results = []
    if os.path.exists(output_path):
        results = json.load(open(output_path))
        done_qs = {r["question"] for r in results}
        print(f"  Resuming — {len(done_qs)} already done")

    solver    = StrategyQASolver(model=MODEL)
    recovered = sum(1 for r in results if r.get("rerun_astar_correct"))

    for item in tqdm(questions, desc="A* re-run"):
        if item["question"] in done_qs:
            continue

        question = item["question"]
        gold     = item["gold"]
        q_entities = extract_question_entities(question)

        try:
            best_node, num_nodes = astar_search(question, solver, q_entities)
            trace      = best_node.state.get_trace()
            pred       = solver.extract_answer(trace)
            is_correct = _em(pred, gold)
        except Exception as e:
            pred, is_correct, num_nodes, trace = None, False, 0, f"ERROR: {e}"

        if is_correct: recovered += 1

        results.append({
            "question"               : question,
            "gold"                   : gold,
            "original_astar_pred"    : item.get("astar_pred"),
            "original_astar_correct" : False,
            "cot_pred"               : item.get("cot_pred"),
            "cot_correct"            : True,
            "rerun_astar_pred"       : pred,
            "rerun_astar_correct"    : is_correct,
            "nodes"                  : num_nodes,
        })
        _save(output_path, results, abs_path=True)

    n = len(results)
    print(f"\n  A* recovered  : {recovered} / {n}  ({recovered/n*100:.1f}%)")
    print(f"  Still failing : {n-recovered} / {n}  ({(n-recovered)/n*100:.1f}%)")
    print(f"  Saved → {output_path}")


# ============================================================================
# ── STEP 4: FINAL COMPARISON REPORT ─────────────────────────────────────────
# ============================================================================

def step4_report():
    print(SEP)
    print("  STEP 4 — Final Comparison Report")
    print(SEP)

    cot_fail_win  = _load("cot_fail_astar_win.json")   or []
    astar_fail_win = _load("astar_fail_cot_win.json")  or []
    both_correct  = _load("both_correct.json")          or []
    both_failed   = _load("both_failed.json")           or []
    cot_rerun     = _load("cot_rerun_on_cot_failures.json")
    astar_rerun   = _load("astar_rerun_on_astar_failures.json")

    total = len(cot_fail_win) + len(astar_fail_win) + len(both_correct) + len(both_failed)
    if total == 0:
        total = len(cot_fail_win) + len(astar_fail_win)  # minimal case
    if total == 0:
        print("  No data found. Run steps 1-3 first.")
        return

    cot_base  = len(both_correct) + len(astar_fail_win)
    astr_base = len(both_correct) + len(cot_fail_win)
    denom     = total if total > 0 else 1

    print(f"""
  ┌─────────────────────────────────────────────────────┐
  │  BASELINE (matched 500 questions from original run) │
  ├────────────────────────────┬────────────────────────┤
  │ Both correct               │ {len(both_correct):3d}  ({len(both_correct)/denom*100:.1f}%)           │
  │ Both failed                │ {len(both_failed):3d}  ({len(both_failed)/denom*100:.1f}%)           │
  │ CoT fail / A* win          │ {len(cot_fail_win):3d}  ({len(cot_fail_win)/denom*100:.1f}%)  ← re-run CoT  │
  │ A* fail  / CoT win         │ {len(astar_fail_win):3d}  ({len(astar_fail_win)/denom*100:.1f}%)  ← re-run A*   │
  ├────────────────────────────┼────────────────────────┤
  │ CoT  accuracy (baseline)   │ {cot_base:3d}/{total}  ({cot_base/denom*100:.1f}%)         │
  │ A*   accuracy (baseline)   │ {astr_base:3d}/{total}  ({astr_base/denom*100:.1f}%)         │
  └────────────────────────────┴────────────────────────┘""")

    # CoT re-run
    print(f"\n  {'─'*55}")
    print(f"  CoT Re-run  (on {len(cot_fail_win)} questions CoT originally failed)")
    print(f"  {'─'*55}")
    if cot_rerun is None:
        print("  ⚠  Not done yet — run step 2")
    else:
        n   = len(cot_rerun)
        rec = sum(1 for r in cot_rerun if r.get("rerun_cot_correct"))
        print(f"  Recovered     : {rec:3d} / {n}  ({rec/n*100:.1f}%)")
        print(f"  Still failing : {n-rec:3d} / {n}  ({(n-rec)/n*100:.1f}%)")

    # A* re-run
    print(f"\n  {'─'*55}")
    print(f"  A* Re-run  (on {len(astar_fail_win)} questions A* originally failed)")
    print(f"  {'─'*55}")
    if astar_rerun is None:
        print("  ⚠  Not done yet — run step 3")
    else:
        n   = len(astar_rerun)
        rec = sum(1 for r in astar_rerun if r.get("rerun_astar_correct"))
        avg_nodes = sum(r.get("nodes", 0) for r in astar_rerun) / n if n else 0
        print(f"  Recovered     : {rec:3d} / {n}  ({rec/n*100:.1f}%)")
        print(f"  Still failing : {n-rec:3d} / {n}  ({(n-rec)/n*100:.1f}%)")
        print(f"  Avg nodes     : {avg_nodes:.1f}")

    # Updated accuracy table
    if cot_rerun is not None and astar_rerun is not None:
        cot_rec   = sum(1 for r in cot_rerun   if r.get("rerun_cot_correct"))
        astr_rec  = sum(1 for r in astar_rerun if r.get("rerun_astar_correct"))
        cot_new   = cot_base  + cot_rec
        astr_new  = astr_base + astr_rec
        cot_acc   = cot_new  / denom * 100
        astr_acc  = astr_new / denom * 100

        winner = ("CoT" if cot_acc > astr_acc + 0.5
                  else "A*" if astr_acc > cot_acc + 0.5
                  else "Tie")

        print(f"""
  ┌──────────────────────────┬───────────┬───────────┐
  │                          │    CoT    │    A*     │
  ├──────────────────────────┼───────────┼───────────┤
  │ Baseline accuracy        │  {cot_base:3d}/{total} ({cot_base/denom*100:.1f}%) │  {astr_base:3d}/{total} ({astr_base/denom*100:.1f}%) │
  │ Failures recovered       │  +{cot_rec:2d}        │  +{astr_rec:2d}        │
  │ Updated accuracy         │  {cot_new:3d}/{total} ({cot_acc:.1f}%) │  {astr_new:3d}/{total} ({astr_acc:.1f}%) │
  ├──────────────────────────┼───────────┴───────────┤
  │ Consistency (recovery %) │  {cot_rec/len(cot_fail_win)*100:.0f}% of failures    {astr_rec/len(astar_fail_win)*100:.0f}% of failures │
  └──────────────────────────┴───────────────────────┘

  → Winner after re-runs: {winner}""")


# ============================================================================
# ── HELPERS ─────────────────────────────────────────────────────────────────
# ============================================================================

def _save(filename_or_path: str, data, abs_path: bool = False):
    path = filename_or_path if abs_path else os.path.join(SCRIPT_DIR, filename_or_path)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)

def _load(filename: str):
    path = os.path.join(SCRIPT_DIR, filename)
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)

def _already_done(filename: str, label: str) -> bool:
    data = _load(filename)
    if data is not None:
        print(f"  ✓  {label} already done ({len(data)} items) — skipping")
        return True
    return False


# ============================================================================
# ── MAIN ─────────────────────────────────────────────────────────────────────
# ============================================================================

STEP_MAP = {1: step1_find_disagreements,
            2: step2_rerun_cot,
            3: step3_rerun_astar,
            4: step4_report}

def main():
    global MODEL
    parser = argparse.ArgumentParser(
        description="CoT vs A* Comparison — self-contained runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 experiment.py --check         verify server + files
  python3 experiment.py                 run all steps (1→4)
  python3 experiment.py --from 2        resume from step 2
  python3 experiment.py --only 3        run only step 3 (A* re-run)
  python3 experiment.py --only 4        print final report
""")
    parser.add_argument("--check",     action="store_true", help="Health check only")
    parser.add_argument("--from",      dest="from_step", type=int, default=1)
    parser.add_argument("--only",      dest="only_step", type=int, default=None)
    parser.add_argument("--no-check",  action="store_true", help="Skip health check")
    parser.add_argument("--model",     default=MODEL, help=f"Ollama model (default: {MODEL})")
    args = parser.parse_args()

    # Allow overriding model via CLI
    MODEL = args.model

    if args.check:
        sys.exit(0 if check_server() else 1)

    print()
    print(SEP)
    print("  CoT vs A* Comparison Experiment")
    print(f"  Model  : {MODEL}")
    print(f"  Server : {OLLAMA_BASE_URL}")
    print(SEP)

    if not args.no_check:
        print("\nRunning health check …\n")
        if not check_server():
            print("\nFix the issues above, then re-run.")
            print("Or skip with:  python3 experiment.py --no-check")
            sys.exit(1)

    steps = [args.only_step] if args.only_step else list(range(args.from_step, 5))
    t0 = time.time()

    for s in steps:
        fn = STEP_MAP.get(s)
        if fn is None:
            print(f"Unknown step: {s}")
            sys.exit(1)
        try:
            fn()
        except KeyboardInterrupt:
            print(f"\n\nInterrupted at step {s}. Partial results saved.")
            print(f"Resume with:  python3 experiment.py --from {s}")
            sys.exit(1)

    elapsed = time.time() - t0
    print()
    print(SEP)
    print(f"  Done in {elapsed:.0f}s")
    print(SEP)


if __name__ == "__main__":
    main()

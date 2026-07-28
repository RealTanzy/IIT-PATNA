#!/usr/bin/env python3
"""
LogicQA Ablation Script — Rebuttal Experiments
================================================

Mirrors hotpotqa_ablation.py for the LogicQA dataset.
Uses Ollama on port 11434 (llama3.1:8b by default).

CLI flags:
  --heuristic-mode {astar, bfs, greedy}
  --budget N        (max nodes to expand)
  --branch-k K      (candidates per expansion)
  --sample-file     (JSON from prepare_samples.py)
  --model / --base-url
  --out
"""

import os
import re
import sys
import heapq
import json
import argparse
import time
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional, Set
from openai import OpenAI
from tqdm import tqdm

SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
SAMPLES_DIR = os.path.join(SCRIPT_DIR, "samples")
RESULTS_DIR = os.path.join(SCRIPT_DIR, "results")

DEFAULT_MODEL    = "llama3.1:8b"
DEFAULT_BASE_URL = "http://localhost:11434/v1"
DEFAULT_BUDGET   = 12
DEFAULT_BRANCH_K = 2
DEFAULT_MAX_DEPTH = 5

WEIGHT_DEPTH    = 0.3
WEIGHT_COVERAGE = 0.3
LABEL_MAP = {0: "A", 1: "B", 2: "C", 3: "D"}

_STOPWORDS = frozenset({
    'a','an','the','is','was','were','are','of','in','on','at','for','and','or',
    'to','by','with','it','its','that','this','from','as','be','been','being',
    'have','has','had','do','does','did','will','would','could','should','may',
    'might','can','what','which','who','whom','where','when','how','why','same',
    'both','also','not','no','yes','than','more','most','other','some','any',
    'all','many','much','few','several',
})


@dataclass
class ReasoningStep:
    content: str
    confidence: float = 0.8

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
        if self.f_score != other.f_score:
            return self.f_score < other.f_score
        return self.node_id < other.node_id


def extract_question_entities(query: str) -> Set[str]:
    entities: Set[str] = set()
    for m in re.finditer(r'"([^"]+)"', query):
        entities.add(m.group(1).lower())
    q_trimmed = re.sub(
        r'^(What|Which|Who|Where|When|How|Are|Were|Is|Was|Did|Do|Does|The)\s+',
        '', query, flags=re.IGNORECASE)
    for m in re.finditer(r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b', q_trimmed):
        phrase = m.group(1).lower()
        if phrase not in _STOPWORDS and len(phrase) > 2:
            entities.add(phrase)
    for m in re.finditer(r'\b(\d{4}|\d+(?:,\d{3})*)\b', query):
        entities.add(m.group(1))
    for w in re.findall(r'\b[a-zA-Z]+\b', query.lower()):
        if w not in _STOPWORDS and len(w) > 3:
            entities.add(w)
    return entities


def compute_entity_coverage(state: State, entities: Set[str]) -> float:
    if not entities:
        return 1.0
    trace = state.get_trace().lower()
    return sum(1 for e in entities if e in trace) / len(entities)


def compute_heuristic(state: State, entities: Set[str],
                      heuristic_mode: str = 'astar') -> float:
    if heuristic_mode == 'bfs':
        return 0.0
    remaining = max(0, 1 - state.depth)  # LogicQA: min 1 hop
    h_depth    = remaining * WEIGHT_DEPTH
    h_coverage = (1.0 - compute_entity_coverage(state, entities)) * WEIGHT_COVERAGE
    return h_depth + h_coverage


class LogicQASolver:
    def __init__(self, model: str, base_url: str, branch_k: int = 2, verbose: bool = False):
        self.client   = OpenAI(base_url=base_url, api_key='ollama')
        self.model    = model
        self.branch_k = branch_k
        self.verbose  = verbose
        self.total_tokens = 0
        self.api_calls    = 0

    def extract_answer(self, trace: str) -> Optional[str]:
        m = re.search(r'[Tt]he answer is\s*\[?([ABCD])\]?', trace)
        if m: return m.group(1).upper()
        m = re.search(r'answer[:\s]+\[?([ABCD])\]?', trace, re.IGNORECASE)
        if m: return m.group(1).upper()
        m = re.search(r'\b(?:option|choice)\s*\[?([ABCD])\]?\b', trace, re.IGNORECASE)
        if m: return m.group(1).upper()
        for line in reversed(trace.strip().split('\n')):
            m = re.search(r'\b(?:option|choice)\s*\[?([ABCD])\]?\b', line, re.IGNORECASE)
            if m: return m.group(1).upper()
            m = re.fullmatch(r'\[?([ABCD])\]?\.?', line.strip())
            if m: return m.group(1).upper()
        return None

    def _call_llm(self, prompt: str, temperature: float = 0.7,
                  max_tokens: int = 200) -> Tuple[Optional[str], float]:
        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a concise logical reasoning assistant. Follow the output format exactly."},
                    {"role": "user",   "content": prompt},
                ],
                temperature=temperature,
                max_tokens=max_tokens,
            )
            self.total_tokens += resp.usage.total_tokens
            self.api_calls    += 1
            content = resp.choices[0].message.content or ""
            step_m = re.search(r'STEP(?:\s+\d+)?:\s*(.+?)(?=\nCONFIDENCE:|$)',
                                content, re.IGNORECASE | re.DOTALL)
            conf_m = re.search(r'CONFIDENCE:\s*([0-9.]+)', content, re.IGNORECASE)
            if step_m:
                text = step_m.group(1).strip()[:300]
                conf = float(conf_m.group(1)) if conf_m else 0.8
                conf = max(0.5, min(0.99, conf))
                if len(text) > 5:
                    return text, conf
            return None, 0.0
        except Exception as e:
            if self.verbose: print(f"  LLM error: {e}")
            return None, 0.0

    def generate_candidates(self, context: str, query: str, options: List[str],
                            state: State, force: bool = False) -> List[Tuple[str, float]]:
        opts = "\n".join(f"{chr(65+i)}. {opt}" for i, opt in enumerate(options))
        steps_so_far = ("\nReasoning so far:\n" +
                        "\n".join(f"{i+1}. {s.content}" for i, s in enumerate(state.steps))
                        ) if state.steps else ""
        step_num = len(state.steps) + 1

        if force:
            prompt = (f"Passage: {context}\nQuestion: {query}\nOptions:\n{opts}\n"
                      f"{steps_so_far}\nCommit to ONE final answer now.\n"
                      f"STEP: The answer is [A/B/C/D] because [one sentence reason]\n"
                      f"CONFIDENCE: 0.95")
            text, conf = self._call_llm(prompt, temperature=0.05, max_tokens=150)
            if text: return [(text, conf)]
            return [(f"The answer is A (fallback)", 0.5)]

        prompt = (f"Passage: {context}\nQuestion: {query}\nOptions:\n{opts}\n"
                  f"{steps_so_far}\n"
                  f"Make ONE short logical deduction (max 2 sentences). "
                  f"If you know the answer write: 'The answer is [A/B/C/D] because ...' "
                  f"NEVER repeat a previous step.\n"
                  f"STEP: [your deduction]\nCONFIDENCE: [0.6-0.95]\nStep {step_num}:")

        candidates = []
        seen: set = set()
        for _ in range(self.branch_k):
            text, conf = self._call_llm(prompt, temperature=0.7)
            if text:
                key = text[:80].lower()
                if key not in seen:
                    seen.add(key)
                    candidates.append((text, conf))
        return candidates

    def force_final(self, context: str, query: str, options: List[str],
                    state: State) -> Tuple[str, float]:
        cands = self.generate_candidates(context, query, options, state, force=True)
        return cands[0] if cands else ("The answer is A (fallback)", 0.5)


def astar_search(
    context: str, query: str, options: List[str],
    solver: LogicQASolver, entities: Set[str],
    heuristic_mode: str = 'astar',
    max_nodes: int = DEFAULT_BUDGET,
    max_depth: int = DEFAULT_MAX_DEPTH,
    verbose: bool = False,
) -> Tuple[Node, int]:
    open_set: List[Node] = []
    node_counter = 0
    visited: set = set()
    goal_nodes: List[Node] = []
    best_current: Optional[Node] = None

    root = State()
    h0   = compute_heuristic(root, entities, heuristic_mode)
    root_node = Node(state=root, g_score=0.0, h_score=h0, f_score=h0, node_id=0)
    heapq.heappush(open_set, root_node)
    nodes_explored = 0

    while open_set and nodes_explored < max_nodes:
        current = heapq.heappop(open_set)
        sig = current.state.signature()
        if sig in visited: continue
        visited.add(sig)
        nodes_explored += 1

        if current.state.steps:
            last = current.state.steps[-1].content
            if re.search(r'[Tt]he answer is\s*\[?[ABCD]\]?', last):
                goal_nodes.append(current)
                if len(goal_nodes) >= 2: break
                continue

        if best_current is None or current.state.depth > best_current.state.depth:
            best_current = current

        if current.state.depth >= max_depth: continue

        force = (current.state.depth >= 4)
        cands = solver.generate_candidates(context, query, options, current.state, force)
        if not cands:
            step, conf = solver.force_final(context, query, options, current.state)
            cands = [(step, conf)]

        for text, conf in cands:
            node_counter += 1
            new_state = State(
                steps=current.state.steps + [ReasoningStep(text, conf)],
                depth=current.state.depth + 1,
            )
            if new_state.signature() in visited: continue
            new_g = current.g_score + 1.0 + (1.0 - conf)
            new_h = compute_heuristic(new_state, entities, heuristic_mode)
            new_f = new_h if heuristic_mode == 'greedy' else new_g + new_h
            heapq.heappush(open_set, Node(state=new_state, g_score=new_g,
                                          h_score=new_h, f_score=new_f,
                                          node_id=node_counter,
                                          parent_id=current.node_id))

    # Majority vote among goal nodes
    if goal_nodes:
        votes: Dict[str, float] = {}
        for gn in goal_nodes:
            ans = solver.extract_answer(gn.state.get_trace())
            if ans:
                w = 1.0 / (gn.g_score + 1e-6)
                votes[ans] = votes.get(ans, 0.0) + w
        if votes:
            best_ans = max(votes, key=lambda a: votes[a])
            for gn in goal_nodes:
                if solver.extract_answer(gn.state.get_trace()) == best_ans:
                    return gn, nodes_explored
        return goal_nodes[0], nodes_explored

    # Fallback: force answer on deepest node
    fallback = best_current or Node(state=State(), g_score=0, h_score=0, f_score=0, node_id=0)
    step, conf = solver.force_final(context, query, options, fallback.state)
    node_counter += 1
    final_state = State(
        steps=fallback.state.steps + [ReasoningStep(step, conf)],
        depth=fallback.state.depth + 1,
    )
    return Node(state=final_state, g_score=fallback.g_score + 1.0,
                h_score=0.0, f_score=fallback.g_score + 1.0,
                node_id=node_counter), nodes_explored


def main():
    parser = argparse.ArgumentParser(description="LogicQA rebuttal ablation")
    parser.add_argument("--sample-file", default=None)
    parser.add_argument("--heuristic-mode", choices=["astar", "bfs", "greedy"], default="astar")
    parser.add_argument("--budget",   type=int, default=DEFAULT_BUDGET)
    parser.add_argument("--branch-k", type=int, default=DEFAULT_BRANCH_K)
    parser.add_argument("--model",    default=DEFAULT_MODEL)
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--out",      default=None)
    parser.add_argument("--verbose",  action="store_true")
    args = parser.parse_args()

    os.makedirs(RESULTS_DIR, exist_ok=True)
    sample_file = args.sample_file or os.path.join(SAMPLES_DIR, "logicqa_200.json")
    if not os.path.exists(sample_file):
        print(f"ERROR: sample file not found: {sample_file}")
        print("Run prepare_samples.py first.")
        sys.exit(1)

    if args.out is None:
        tag = f"mode{args.heuristic_mode}_budget{args.budget}_k{args.branch_k}"
        args.out = os.path.join(RESULTS_DIR, f"logicqa_{tag}.json")

    print(f"Config: heuristic={args.heuristic_mode} | budget={args.budget} | branch_k={args.branch_k}")

    with open(sample_file) as f:
        items = json.load(f)
    print(f"Loaded {len(items)} LogicQA questions.")

    solver = LogicQASolver(model=args.model, base_url=args.base_url,
                           branch_k=args.branch_k, verbose=args.verbose)
    results = []
    correct = 0
    total   = 0
    total_nodes = 0
    start_time  = time.time()

    for item in tqdm(items, desc=f"LogicQA [{args.heuristic_mode}|B={args.budget}|K={args.branch_k}]"):
        try:
            context = item["context"]
            query   = item["query"]
            options = item["options"]
            gold    = item["gold"]
            entities = extract_question_entities(query)

            best_node, n_explored = astar_search(
                context=context, query=query, options=options,
                solver=solver, entities=entities,
                heuristic_mode=args.heuristic_mode,
                max_nodes=args.budget,
                max_depth=DEFAULT_MAX_DEPTH,
                verbose=args.verbose,
            )

            pred = solver.extract_answer(best_node.state.get_trace())
            is_correct = (pred == gold)

            total_nodes += n_explored
            if is_correct: correct += 1
            total += 1

            results.append({
                "id": item.get("id", total),
                "query": query, "gold": gold, "pred": pred,
                "correct": is_correct, "nodes": n_explored,
                "trace": best_node.state.get_trace(),
            })
        except KeyboardInterrupt:
            print("\nInterrupted — saving partial results...")
            break
        except Exception as e:
            print(f"  Error on item {total}: {e}")
            results.append({
                "id": item.get("id", total), "query": item.get("query",""),
                "gold": item.get("gold",""), "pred": None,
                "correct": False, "nodes": 0, "trace": "",
            })
            total += 1

    elapsed  = time.time() - start_time
    accuracy = 100 * correct / total if total > 0 else 0.0

    summary = {
        "config": {"heuristic_mode": args.heuristic_mode, "budget": args.budget,
                   "branch_k": args.branch_k, "model": args.model},
        "total": total, "correct": correct, "accuracy": round(accuracy, 4),
        "avg_nodes": round(total_nodes / total if total else 0, 2),
        "elapsed_s": round(elapsed, 1),
        "total_tokens": solver.total_tokens,
        "results": results,
    }

    with open(args.out, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nAccuracy: {accuracy:.2f}%  ({correct}/{total})")
    print(f"Saved: {args.out}")


if __name__ == "__main__":
    main()

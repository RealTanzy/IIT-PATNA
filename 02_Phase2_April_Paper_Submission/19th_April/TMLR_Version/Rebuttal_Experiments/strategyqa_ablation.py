#!/usr/bin/env python3
"""
StrategyQA Ablation Script — Rebuttal Experiments
==================================================

Mirrors hotpotqa_ablation.py for StrategyQA (all yes/no, no context).
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
from collections import Counter
from openai import OpenAI
from tqdm import tqdm

SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
SAMPLES_DIR = os.path.join(SCRIPT_DIR, "samples")
RESULTS_DIR = os.path.join(SCRIPT_DIR, "results")

DEFAULT_MODEL    = "llama3.1:8b"
DEFAULT_BASE_URL = "http://localhost:11434/v1"
DEFAULT_BUDGET   = 15
DEFAULT_BRANCH_K = 2
DEFAULT_MAX_DEPTH = 8
DEFAULT_MIN_GOAL_DEPTH = 2

WEIGHT_DEPTH    = 0.3
WEIGHT_COVERAGE = 0.3
TEMPERATURES    = [0.2, 0.5, 0.7, 0.9]

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
        if abs(self.f_score - other.f_score) > 1e-9:
            return self.f_score < other.f_score
        return self.node_id < other.node_id


def extract_question_entities(question: str) -> Set[str]:
    entities: Set[str] = set()
    for m in re.finditer(r'"([^"]+)"', question):
        entities.add(m.group(1).lower())
    q_trimmed = re.sub(
        r'^(Are|Were|Is|Was|Did|Do|Does|Can|Has|Have|Could|Would|Should)\s+',
        '', question, flags=re.IGNORECASE)
    for m in re.finditer(r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b', q_trimmed):
        phrase = m.group(1).lower()
        if phrase not in _STOPWORDS and len(phrase) > 2:
            entities.add(phrase)
    for m in re.finditer(r'\b(\d{4}|\d+)\b', question):
        entities.add(m.group(1))
    for w in re.findall(r'\b[a-zA-Z]+\b', question.lower()):
        if w not in _STOPWORDS and len(w) > 3:
            entities.add(w)
    return entities


def compute_entity_coverage(state: State, entities: Set[str]) -> float:
    if not entities: return 1.0
    trace = state.get_trace().lower()
    return sum(1 for e in entities if e in trace) / len(entities)


def compute_heuristic(state: State, entities: Set[str],
                      heuristic_mode: str = 'astar') -> float:
    if heuristic_mode == 'bfs':
        return 0.0
    remaining  = max(0, 2 - state.depth)  # StrategyQA: min 2 hops
    h_depth    = remaining * WEIGHT_DEPTH
    h_coverage = (1.0 - compute_entity_coverage(state, entities)) * WEIGHT_COVERAGE
    return h_depth + h_coverage


class StrategyQASolver:
    def __init__(self, model: str, base_url: str, branch_k: int = 2, verbose: bool = False):
        self.client   = OpenAI(base_url=base_url, api_key='ollama')
        self.model    = model
        self.branch_k = branch_k
        self.verbose  = verbose
        self.total_tokens = 0
        self.api_calls    = 0

    def _call_llm(self, system: str, prompt: str,
                  temperature: float = 0.3, max_tokens: int = 256) -> Tuple[Optional[str], float]:
        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user",   "content": prompt},
                ],
                temperature=temperature,
                max_tokens=max_tokens,
            )
            self.total_tokens += resp.usage.total_tokens
            self.api_calls    += 1
            content = resp.choices[0].message.content or ""
            step_m = re.search(r'STEP(?:\s*\d*)?:\s*(.+?)(?=\n(?:STEP|CONFIDENCE)|$)',
                                content, re.IGNORECASE | re.DOTALL)
            conf_m = re.search(r'CONFIDENCE:\s*([0-9.]+)', content, re.IGNORECASE)
            if step_m:
                text = step_m.group(1).strip()
                conf = float(conf_m.group(1)) if conf_m else 0.75
                conf = max(0.5, min(0.99, conf))
                if len(text) > 3:
                    return text, conf
            return None, 0.0
        except Exception as e:
            if self.verbose: print(f"  LLM error: {e}")
            return None, 0.0

    def extract_answer(self, trace: str) -> Optional[str]:
        matches = re.findall(r'[Tt]he answer is[:\s]+([^\n]+)', trace)
        if matches:
            raw = matches[-1].strip().lower()
            if re.search(r'\byes\b', raw): return 'yes'
            if re.search(r'\bno\b', raw):  return 'no'
        for line in reversed(trace.strip().split('\n')):
            low = line.strip().lower()
            if re.search(r'\byes\b', low): return 'yes'
            if re.search(r'\bno\b', low):  return 'no'
        return None

    def generate_candidates(self, question: str, state: State,
                            force: bool = False) -> List[Tuple[str, float]]:
        steps_so_far = ("\nPrevious steps:\n" +
                        "\n".join(f"  {i+1}. {s.content}" for i, s in enumerate(state.steps))
                        ) if state.steps else ""
        step_num = len(state.steps) + 1

        if force or step_num > 2:
            sys_p = "You answer yes/no questions from world knowledge. Be decisive."
            usr_p = (f"Question: {question}\n{steps_so_far}\n\n"
                     "Based on the reasoning so far, give the final yes/no answer.\n"
                     "STEP: The answer is: [yes/no]\nCONFIDENCE: [0.8-0.99]")
            text, conf = self._call_llm(sys_p, usr_p, temperature=0.1, max_tokens=100)
            if text: return [(text, conf)]
            return [("The answer is: yes", 0.5)]

        candidates = []
        strategies = [
            ("You reason step by step about factual questions. Don't give the final answer in step 1.",
             f"Question: {question}\n{steps_so_far}\n\n"
             f"Step {step_num}: Think about what fact(s) you need to know to answer this question. "
             f"State one intermediate fact from your world knowledge.\n"
             f"STEP: [intermediate fact]\nCONFIDENCE: [0.6-0.9]"),
            ("You reason about yes/no questions carefully by identifying relevant facts.",
             f"Question: {question}\n{steps_so_far}\n\n"
             f"Step {step_num}: What key knowledge is needed for this question? Identify the relevant entities or properties.\n"
             f"STEP: [relevant fact or entity relationship]\nCONFIDENCE: [0.6-0.9]"),
            ("You break down yes/no questions into sub-questions to reason systematically.",
             f"Question: {question}\n{steps_so_far}\n\n"
             f"Step {step_num}: What sub-question must be answered first? State the answer to that sub-question.\n"
             f"STEP: [sub-answer]\nCONFIDENCE: [0.6-0.9]"),
            ("You reason from general knowledge and identify implications carefully.",
             f"Question: {question}\n{steps_so_far}\n\n"
             f"Step {step_num}: Reason about the specific condition in the question from world knowledge.\n"
             f"STEP: [reasoning step]\nCONFIDENCE: [0.6-0.9]"),
        ]

        for i, (sys_p, usr_p) in enumerate(strategies[:self.branch_k]):
            temp = TEMPERATURES[i] if i < len(TEMPERATURES) else 0.4
            text, conf = self._call_llm(sys_p, usr_p, temperature=temp, max_tokens=200)
            if text and 'the answer is' not in text.lower():
                candidates.append((text, conf))
            elif text:
                candidates.append((text, conf))

        return candidates

    def force_final(self, question: str, state: State) -> Tuple[str, float]:
        cands = self.generate_candidates(question, state, force=True)
        return cands[0] if cands else ("The answer is: yes", 0.5)


def astar_search(
    question: str,
    solver: StrategyQASolver,
    entities: Set[str],
    heuristic_mode: str = 'astar',
    max_nodes: int = DEFAULT_BUDGET,
    max_depth: int = DEFAULT_MAX_DEPTH,
    min_goal_depth: int = DEFAULT_MIN_GOAL_DEPTH,
    verbose: bool = False,
) -> Tuple[Node, int]:
    open_set: List[Node] = []
    node_counter = 0
    visited: Set[str] = set()
    goal_nodes: List[Node] = []
    goal_votes: Counter = Counter()
    best_current: Optional[Node] = None

    root  = State()
    h0    = compute_heuristic(root, entities, heuristic_mode)
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
            last = current.state.steps[-1].content.lower()
            is_goal = bool(re.search(r'the answer is[:\s]+(yes|no)\b', last))

            if is_goal and current.state.depth < min_goal_depth:
                is_goal = False  # too shallow

            if is_goal:
                goal_nodes.append(current)
                ans = solver.extract_answer(current.state.get_trace())
                if ans:
                    goal_votes[ans] += 1
                    if goal_votes[ans] >= 2:
                        best = max(
                            [gn for gn in goal_nodes
                             if solver.extract_answer(gn.state.get_trace()) == ans],
                            key=lambda n: n.state.steps[-1].confidence if n.state.steps else 0
                        )
                        return best, nodes_explored
                continue

        if best_current is None or current.state.depth > best_current.state.depth:
            best_current = current
        if current.state.depth >= max_depth: continue

        force = (current.state.depth >= max_depth - 1)
        cands = solver.generate_candidates(question, current.state, force)
        if not cands:
            step, conf = solver.force_final(question, current.state)
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

    if goal_nodes:
        if goal_votes:
            best_ans = goal_votes.most_common(1)[0][0]
            for gn in goal_nodes:
                if solver.extract_answer(gn.state.get_trace()) == best_ans:
                    return gn, nodes_explored
        return goal_nodes[0], nodes_explored

    fallback = best_current or Node(state=State(), g_score=0, h_score=0, f_score=0, node_id=0)
    step, conf = solver.force_final(question, fallback.state)
    node_counter += 1
    final_state = State(
        steps=fallback.state.steps + [ReasoningStep(step, conf)],
        depth=fallback.state.depth + 1,
    )
    return Node(state=final_state, g_score=fallback.g_score + 1.0,
                h_score=0.0, f_score=fallback.g_score + 1.0,
                node_id=node_counter), nodes_explored


def main():
    parser = argparse.ArgumentParser(description="StrategyQA rebuttal ablation")
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
    sample_file = args.sample_file or os.path.join(SAMPLES_DIR, "strategyqa_200.json")
    if not os.path.exists(sample_file):
        print(f"ERROR: sample file not found: {sample_file}")
        print("Run prepare_samples.py first.")
        sys.exit(1)

    if args.out is None:
        tag = f"mode{args.heuristic_mode}_budget{args.budget}_k{args.branch_k}"
        args.out = os.path.join(RESULTS_DIR, f"strategyqa_{tag}.json")

    print(f"Config: heuristic={args.heuristic_mode} | budget={args.budget} | branch_k={args.branch_k}")

    with open(sample_file) as f:
        items = json.load(f)
    print(f"Loaded {len(items)} StrategyQA questions.")

    solver = StrategyQASolver(model=args.model, base_url=args.base_url,
                              branch_k=args.branch_k, verbose=args.verbose)
    results = []
    correct = 0
    total   = 0
    total_nodes = 0
    start_time  = time.time()

    for item in tqdm(items, desc=f"StrategyQA [{args.heuristic_mode}|B={args.budget}|K={args.branch_k}]"):
        try:
            question = item["question"]
            gold     = item["answer"]
            entities = extract_question_entities(question)

            best_node, n_explored = astar_search(
                question=question, solver=solver, entities=entities,
                heuristic_mode=args.heuristic_mode,
                max_nodes=args.budget, max_depth=DEFAULT_MAX_DEPTH,
                min_goal_depth=DEFAULT_MIN_GOAL_DEPTH, verbose=args.verbose,
            )

            pred = solver.extract_answer(best_node.state.get_trace())
            is_correct = (pred == gold)
            total_nodes += n_explored
            if is_correct: correct += 1
            total += 1

            results.append({
                "id": item.get("id", total),
                "question": question, "gold": gold, "pred": pred,
                "correct": is_correct, "nodes": n_explored,
                "trace": best_node.state.get_trace(),
            })
        except KeyboardInterrupt:
            print("\nInterrupted — saving partial results...")
            break
        except Exception as e:
            print(f"  Error on item {total}: {e}")
            results.append({
                "id": item.get("id", total), "question": item.get("question",""),
                "gold": item.get("answer",""), "pred": None,
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

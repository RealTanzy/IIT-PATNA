#!/usr/bin/env python3
"""
A* Search Solver for LogicQA
Adapted from GSM8K A* solver for multiple-choice logical reasoning.
"""

import re
import heapq
import json
import argparse
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional
from openai import OpenAI
from datasets import load_dataset
from tqdm import tqdm

# ============================================================================
# CONFIGURATION
# ============================================================================

LAMBDA = 0.12        # Weight for depth penalty (encourage efficiency)
CONF   = 0.99        # Confidence baseline

# Initialize Ollama client
client = OpenAI(base_url='http://localhost:11434/v1', api_key='ollama')

LABEL_MAP = {0: 'A', 1: 'B', 2: 'C', 3: 'D'}

# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class ReasoningStep:
    content: str
    confidence: float = 0.8

@dataclass
class State:
    steps: List[ReasoningStep] = field(default_factory=list)
    depth: int = 0

    def get_trace(self) -> str:
        return "\n".join(f"Step {i}: {s.content}" for i, s in enumerate(self.steps, 1))

    def signature(self) -> str:
        return " || ".join(s.content.strip() for s in self.steps)

@dataclass
class Node:
    state: State
    g_score: float
    h_score: float
    h_score_semantic: float
    f_score: float
    node_id: int
    parent_id: Optional[int] = None

    def __lt__(self, other: "Node") -> bool:
        if self.f_score != other.f_score:
            return self.f_score < other.f_score
        return self.node_id < other.node_id

# ============================================================================
# HEURISTICS FOR LOGICAL REASONING
# ============================================================================

def h_remaining_steps(state: State, max_depth: int = 8) -> float:
    """Heuristic based on remaining depth budget."""
    remaining = max(0, max_depth - state.depth)
    return LAMBDA * remaining

def h_logic_complexity(state: State, context: str, query: str) -> float:
    """
    Heuristic estimating remaining difficulty based on logical features.
    Considers passage length, logical connectives, and question type.
    """
    text = (context + " " + query).lower()

    # Logical connectives and inference markers
    connectives = ['if', 'then', 'therefore', 'however', 'but', 'unless',
                   'because', 'since', 'although', 'whereas', 'neither',
                   'either', 'not', 'all', 'some', 'none', 'only']
    connective_count = sum(1 for w in connectives if w in text.split())

    # Question type difficulty
    hard_types = ['weaken', 'strengthen', 'assumption', 'flaw', 'paradox',
                  'inference', 'conclusion', 'evaluate']
    has_hard_type = any(t in query.lower() for t in hard_types)

    # Sentence count as proxy for passage complexity
    sentence_count = len(re.split(r'[.!?]', context))

    complexity = 2
    if connective_count >= 3:  complexity += 1
    if connective_count >= 6:  complexity += 1
    if has_hard_type:          complexity += 1
    if sentence_count >= 5:    complexity += 1
    if sentence_count >= 10:   complexity += 1

    remaining_complexity = max(0, complexity - state.depth)
    return LAMBDA * remaining_complexity

def compute_heuristic(state: State, context: str, query: str, max_depth: int = 8) -> float:
    """Compute combined heuristic."""
    h1 = h_remaining_steps(state, max_depth)
    h2 = h_logic_complexity(state, context, query)
    return (h1 + h2) / 2

# ============================================================================
# SOLVER LOGIC
# ============================================================================

class LogicQASolver:
    def __init__(self, model: str = "qwen2.5:0.5b", verbose: bool = True):
        self.client = OpenAI(base_url='http://localhost:11434/v1', api_key='ollama')
        self.model        = model
        self.total_tokens = 0
        self.api_calls    = 0
        self.verbose      = verbose

    # ------------------------------------------------------------------ #
    def extract_answer(self, trace: str) -> Optional[str]:
        """Extract option letter (A/B/C/D) from reasoning trace."""
        match = re.search(r'[Tt]he answer is\s*\[?([ABCD])\]?', trace)
        if match:
            return match.group(1).upper()
        match = re.search(r'answer[:\s]+\[?([ABCD])\]?', trace, re.IGNORECASE)
        if match:
            return match.group(1).upper()
        # Lone letter on the last non-empty line
        for line in reversed(trace.strip().split('\n')):
            line = line.strip()
            m = re.fullmatch(r'\[?([ABCD])\]?\.?', line)
            if m:
                return m.group(1).upper()
        return None

    # ------------------------------------------------------------------ #
    def generate_next_steps(
        self,
        context: str,
        query: str,
        options: List[str],
        current_state: State,
        n_candidates: int = 3,
        force_conclusion: bool = False,
    ) -> List[Tuple[str, float]]:
        """Generate next logical reasoning steps."""

        options_text = "\n".join(f"{chr(65+i)}. {opt}" for i, opt in enumerate(options))

        if current_state.steps:
            recent = "\n".join(f"{i+1}. {s.content}"
                               for i, s in enumerate(current_state.steps))
            context_str = f"Reasoning so far:\n{recent}\n\n"
            step_num = len(current_state.steps) + 1
        else:
            context_str = ""
            step_num = 1

        if force_conclusion:
            prompt = f"""You are solving a logical reasoning question. Commit to ONE final answer now.

Passage: {context}

Question: {query}

Options:
{options_text}

{context_str}Choose the single best option. Be decisive.

Output EXACTLY (one line each):
STEP: The answer is [A/B/C/D] because [one sentence reason]
CONFIDENCE: 0.95"""

        else:
            prompt = f"""You are solving a multiple-choice logical reasoning question. Make ONE short logical deduction.

Passage: {context}

Question: {query}

Options:
{options_text}

{context_str}Rules:
- ONE deduction only, max 2 sentences.
- Eliminate or confirm specific options using the passage.
- If you know the answer write: "The answer is [A/B/C/D] because [one sentence]."
- NEVER repeat a previous step.

Output EXACTLY (one line each):
STEP: [your deduction]
CONFIDENCE: [0.6-0.95]

Step {step_num}:"""

        temp    = 0.05 if force_conclusion else 0.7
        n_calls = 1 if force_conclusion else n_candidates
        candidates: List[Tuple[str, float]] = []
        seen_steps: set = set()

        for _ in range(n_calls):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system",
                         "content": "You are a concise logical reasoning assistant. Follow the output format exactly."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=temp,
                    max_tokens=200,
                    n=1,
                )

                self.total_tokens += response.usage.total_tokens
                self.api_calls    += 1

                content = response.choices[0].message.content

                step_match = re.search(
                    r'STEP(?:\s+\d+)?:\s*(.+?)(?=\nCONFIDENCE:|$)',
                    content, re.IGNORECASE | re.DOTALL
                )
                conf_match = re.search(r'CONFIDENCE:\s*([0-9.]+)', content, re.IGNORECASE)

                if step_match:
                    step_text  = step_match.group(1).strip()
                    # Truncate to 300 chars to prevent context explosion
                    step_text  = step_text[:300]
                    confidence = float(conf_match.group(1)) if conf_match else 0.8
                    confidence = max(0.5, min(0.99, confidence))
                    key = step_text[:80].lower()
                    if len(step_text) > 5 and key not in seen_steps:
                        seen_steps.add(key)
                        candidates.append((step_text, confidence))

            except Exception as e:
                if self.verbose:
                    print(f"Warning: LLM generation error: {e}")

        return candidates

    # ------------------------------------------------------------------ #
    def force_final_answer(
        self,
        context: str,
        query: str,
        options: List[str],
        state: State,
    ) -> Tuple[str, float]:
        """Force the model to output a final answer."""
        candidates = self.generate_next_steps(
            context, query, options, state, n_candidates=1, force_conclusion=True
        )
        if candidates:
            return candidates[0]
        return ("The answer is A (fallback)", 0.5)

# ============================================================================
# A* SEARCH ALGORITHM
# ============================================================================

def astar_search(
    context: str,
    query: str,
    options: List[str],
    solver: LogicQASolver,
    max_depth: int = 5,
    max_nodes: int = 12,
    n_candidates: int = 2,
    verbose: bool = False,
):
    """
    A* search with self-consistency voting.
    Explores multiple reasoning paths and majority-votes the final answer.
    """
    open_set: List[Node] = []
    node_counter  = 0
    visited_sigs: set = set()

    root_state = State()
    h_init     = compute_heuristic(root_state, context, query, max_depth)
    root_node  = Node(
        state=root_state, g_score=0.0, h_score=h_init,
        h_score_semantic=0.0, f_score=h_init, node_id=node_counter,
    )
    heapq.heappush(open_set, root_node)

    nodes_explored    = 0
    best_current_node = root_node
    goal_nodes: List[Node] = []          # collect ALL goal nodes for voting

    while open_set and nodes_explored < max_nodes:
        current = heapq.heappop(open_set)

        sig = current.state.signature()
        if sig in visited_sigs:
            continue
        visited_sigs.add(sig)
        nodes_explored += 1

        if verbose:
            print(f"Explored Node {current.node_id} "
                  f"(Depth {current.state.depth}, F={current.f_score:.2f})")
            if current.state.steps:
                last_preview = current.state.steps[-1].content[:120].replace('\n', ' ')
                print(f"  Last step: {last_preview}")

        # Goal check — collect but DON'T stop; keep exploring for majority vote
        if current.state.steps:
            last = current.state.steps[-1].content
            if re.search(r'[Tt]he answer is\s*\[?[ABCD]\]?', last):
                if verbose:
                    print("  Goal found!")
                goal_nodes.append(current)
                # Stop when we have 2+ votes (enough for majority)
                if len(goal_nodes) >= 2:
                    break
                continue          # don't expand a goal node further

        if current.state.depth > best_current_node.state.depth:
            best_current_node = current

        if current.state.depth >= max_depth:
            continue

        # Force conclusion at depth >= 4 (enough reasoning before committing)
        force = (current.state.depth >= 4)
        candidates = solver.generate_next_steps(
            context, query, options, current.state, n_candidates, force
        )

        if not candidates:
            final_step, conf = solver.force_final_answer(
                context, query, options, current.state
            )
            candidates = [(final_step, conf)]

        for step_text, confidence in candidates:
            node_counter += 1
            new_step  = ReasoningStep(step_text, confidence)
            new_state = State(
                steps=current.state.steps + [new_step],
                depth=current.state.depth + 1,
            )
            new_sig = new_state.signature()
            if new_sig in visited_sigs:
                continue

            step_cost = 1.0 + (1.0 - confidence)
            new_g     = current.g_score + step_cost
            new_h     = compute_heuristic(new_state, context, query, max_depth)
            new_node  = Node(
                state=new_state, g_score=new_g, h_score=new_h,
                h_score_semantic=0.0, f_score=new_g + new_h,
                node_id=node_counter, parent_id=current.node_id,
            )
            heapq.heappush(open_set, new_node)

    # ---- Self-consistency majority vote over collected goal nodes ----------
    if goal_nodes:
        votes: Dict[str, float] = {}
        for gn in goal_nodes:
            ans = solver.extract_answer(gn.state.get_trace())
            if ans:
                # Weight by inverse g_score (shorter path = higher weight)
                weight = 1.0 / (gn.g_score + 1e-6)
                votes[ans] = votes.get(ans, 0.0) + weight
        if votes:
            best_ans = max(votes, key=lambda a: votes[a])
            if verbose:
                print(f"  Votes: {votes}  →  chosen: {best_ans}")
            # Return the goal node whose answer matches the majority vote
            for gn in goal_nodes:
                if solver.extract_answer(gn.state.get_trace()) == best_ans:
                    return gn, nodes_explored
        return goal_nodes[0], nodes_explored

    # Fallback: force answer on the deepest explored node
    if verbose:
        print("No goal found, forcing answer...")
    final_step, conf = solver.force_final_answer(
        context, query, options, best_current_node.state
    )
    final_state = State(
        steps=best_current_node.state.steps + [ReasoningStep(final_step, conf)],
        depth=best_current_node.state.depth + 1,
    )
    final_node = Node(
        state=final_state, g_score=best_current_node.g_score + 1.0,
        h_score=0.0, h_score_semantic=0.0,
        f_score=best_current_node.g_score + 1.0,
        node_id=node_counter + 1, parent_id=best_current_node.node_id,
    )
    return final_node, nodes_explored

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def parse_limit(value: str):
    """Accept an integer or 'all' for --limit."""
    if value.lower() == 'all':
        return None
    try:
        return int(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"--limit must be an integer or 'all', got: {value!r}")


# ============================================================================
# EVALUATION HELPERS
# ============================================================================

def normalize(text: str) -> str:
    """Lowercase, strip punctuation and extra whitespace."""
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    return re.sub(r'\s+', ' ', text).strip()

def f1_score(pred: str, gold: str) -> float:
    """Token-overlap F1 between predicted and gold option text."""
    pred_tokens = normalize(pred).split()
    gold_tokens = normalize(gold).split()
    if not pred_tokens or not gold_tokens:
        return 0.0
    common = set(pred_tokens) & set(gold_tokens)
    if not common:
        return 0.0
    precision = len(common) / len(pred_tokens)
    recall    = len(common) / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed",   type=int, default=42)
    parser.add_argument("--limit",  type=parse_limit, default=10,
                        help="Number of examples to evaluate (integer) or 'all' for the full split")
    parser.add_argument("--split",  default="test",
                        help="Dataset split: train / test / validation")
    parser.add_argument("--output", default="./results_shot/qwen0.5b_logicqa_astar_selfcons.json")
    parser.add_argument("--model",  default="qwen2.5:0.5b",
                        help="Model name tag to record in the output JSON")
    args = parser.parse_args()

    print("=" * 80)
    print("LogicQA A* SOLVER")
    print("=" * 80)

    print("Loading LogicQA dataset...")
    dataset = load_dataset("lucasmccabe/logiqa", split=args.split, trust_remote_code=True)
    print(f"Loaded {len(dataset)} examples")

    limit   = len(dataset) if args.limit is None else min(args.limit, len(dataset))
    solver  = LogicQASolver(model=args.model, verbose=False)

    correct = 0
    total   = 0
    results = []

    # Determine output path early for incremental saving
    import os
    out_dir = os.path.dirname(args.output)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    out_path = args.output

    def _save_incremental(results, correct, total):
        """Save results to disk after every example (crash-safe)."""
        acc = correct / total * 100 if total > 0 else 0.0
        avg_f1 = sum(r['f1'] for r in results) / total * 100 if total > 0 else 0.0
        with open(out_path, 'w') as f:
            json.dump({
                "model": args.model,
                "split": args.split,
                "accuracy": acc,
                "avg_f1": avg_f1,
                "correct": correct,
                "total": total,
                "results": results,
            }, f, indent=2)

    for i, item in tqdm(enumerate(dataset.select(range(limit)))):
        context  = item['context']
        query    = item['query']
        options  = item['options']          # list of 4 strings
        gold_idx = int(item['correct_option'])
        gold     = LABEL_MAP[gold_idx]

        print(f"\nProblem {i+1}: {query[:100]}...")

        try:
            best_node, nodes = astar_search(
                context, query, options,
                solver,
                max_depth=5,
                max_nodes=12,
                n_candidates=2,
            )

            trace = best_node.state.get_trace()
            pred  = solver.extract_answer(trace)

            is_correct = (pred == gold)

            # F1: token overlap between predicted option text and gold option text
            pred_idx = ord(pred) - ord('A') if pred else None
            pred_text = options[pred_idx] if pred_idx is not None and 0 <= pred_idx < len(options) else (pred or "")
            gold_text = options[gold_idx]
            f1 = f1_score(pred_text, gold_text)

            print(f"  Predicted: {pred}")
            print(f"  Gold:      {gold} ({gold_text})")
            print(f"  EM: {'✅ CORRECT' if is_correct else '❌ INCORRECT'}  |  F1: {f1:.2f}")

            if is_correct:
                correct += 1
            total += 1

            results.append({
                "id":      i,
                "context": context,
                "query":   query,
                "options": options,
                "gold":    gold,
                "pred":    pred,
                "correct": is_correct,
                "f1":      f1,
                "nodes":   nodes,
                "trace":   trace,
            })
            _save_incremental(results, correct, total)

        except Exception as e:
            import traceback
            print(f"  Error: {e}")
            traceback.print_exc()
            total += 1
            results.append({
                "id":      i,
                "context": context,
                "query":   query,
                "options": options,
                "gold":    gold,
                "pred":    None,
                "correct": False,
                "f1":      0.0,
                "nodes":   0,
                "trace":   "",
            })
            _save_incremental(results, correct, total)

    acc       = correct / total * 100 if total > 0 else 0.0
    total_f1  = sum(r['f1'] for r in results)
    avg_f1    = total_f1 / total * 100 if total > 0 else 0.0

    print(f"\nFinal EM: {correct}/{total} ({acc:.1f}%)  |  Avg F1: {avg_f1:.1f}%")
    print(f"Total API calls: {solver.api_calls} | Total tokens: {solver.total_tokens}")

    # Final save (already saved incrementally, this is the definitive version)
    merged_results = results
    merged_correct = correct
    merged_total   = total
    merged_acc     = acc
    merged_avg_f1  = avg_f1

    with open(out_path, "w") as f:
        json.dump({
            "model":    args.model,
            "split":    args.split,
            "accuracy": merged_acc,
            "avg_f1":   merged_avg_f1,
            "correct":  merged_correct,
            "total":    merged_total,
            "results":  merged_results,
        }, f, indent=2)
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Prompt templates for Phase 3 dual-trace synthesis."""

REWRITE_SYSTEM_PROMPT = (
    "You rewrite reasoning traces for clarity. Keep logic unchanged. "
    "Do not add or remove factual claims. Do not solve again."
)

REWRITE_USER_TEMPLATE = """Rewrite the A* reasoning trace into a clean linear chain.

Rules:
1) Preserve original logical content exactly.
2) Keep only reasoning steps in order.
3) Improve grammar and clarity.
4) Remove repetition and broken phrasing.
5) Keep final conclusion step if present.
6) Do not introduce any new facts.

Output format:
Step 1: ...
Step 2: ...
...

Original A* trace:
{astar_trace}
"""

SYNTHESIS_SYSTEM_PROMPT = (
    "You are a careful reasoning synthesizer. "
    "Given CoT and refined A* traces, build one improved chain and final answer. "
    "Return only plain text steps and final answer, no headings, no markdown."
)

SYNTHESIS_USER_TEMPLATE = """Task: Synthesize one improved reasoning chain using both sources.

Question:
{question}

Options:
{options_text}

Source 1 (CoT trace):
{cot_trace}

Source 2 (Refined A* trace):
{astar_refined_trace}

Answer format:
{answer_format}

Instructions:
1) Use the strongest consistent steps from both traces.
2) Remove contradictory or weak steps.
3) Keep reasoning concise and logically ordered.
4) End with exactly one final answer in the required format.
5) Do not use any other answer style.
6) Output only the required format; do not add commentary.

Output format:
Step 1: ...
Step 2: ...
...
Final Answer: <answer>
"""

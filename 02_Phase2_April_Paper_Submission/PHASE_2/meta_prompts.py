"""
Meta-prompt library for Phase 2 experiments.

This keeps prompting strategy separate from search logic so prompt ablations
and prompt-version tracking become easy.
"""

from dataclasses import dataclass
from typing import Dict


PROMPT_VERSION = "phase2_v1"


@dataclass(frozen=True)
class PromptSpec:
    system: str
    user: str


def _render(text: str, values: Dict[str, str]) -> str:
    return text.format(**values)


def thought_decompose(question: str) -> PromptSpec:
    return PromptSpec(
        system=(
            "You are a reasoning planner for yes/no questions. "
            "Generate useful intermediate facts and avoid giving final answers yet."
        ),
        user=_render(
            "Question: {question}\n\n"
            "Task:\n"
            "1) Identify the minimum facts needed to resolve this question.\n"
            "2) State those facts clearly using world knowledge.\n"
            "3) Do not output yes/no yet.\n\n"
            "Output format:\n"
            "STEP: [facts]\n"
            "CONFIDENCE: [0.60-0.95]",
            {"question": question},
        ),
    )


def thought_entity_lookup(question: str) -> PromptSpec:
    return PromptSpec(
        system=(
            "You are a factual recall assistant. "
            "Extract entities and provide concise, relevant properties."
        ),
        user=_render(
            "Question: {question}\n\n"
            "Task:\n"
            "List key entities/concepts and one relevant factual property for each.\n"
            "Do not provide a final yes/no answer.\n\n"
            "Output format:\n"
            "STEP: [Entity]: [fact]. [Entity]: [fact].\n"
            "CONFIDENCE: [0.60-0.95]",
            {"question": question},
        ),
    )


def thought_refine(question: str, trace: str) -> PromptSpec:
    return PromptSpec(
        system=(
            "You refine reasoning chains by fixing unsupported assumptions. "
            "Do not produce final yes/no unless explicitly asked."
        ),
        user=_render(
            "Question: {question}\n\n"
            "Current reasoning:\n{trace}\n\n"
            "Task:\n"
            "Identify one weak or missing link and repair it with a concrete fact.\n"
            "Output a single improved intermediate step only.\n\n"
            "Output format:\n"
            "STEP: [improved factual step]\n"
            "CONFIDENCE: [0.60-0.95]",
            {"question": question, "trace": trace or "(none)"},
        ),
    )


def conclude_yes_no(question: str, trace: str) -> PromptSpec:
    return PromptSpec(
        system=(
            "You are a strict yes/no decision module. "
            "Use only the given reasoning and concise world knowledge inference."
        ),
        user=_render(
            "Question: {question}\n\n"
            "Reasoning so far:\n{trace}\n\n"
            "Task:\n"
            "Return a final binary answer.\n\n"
            "Output format:\n"
            "STEP: The answer is: [yes or no]\n"
            "CONFIDENCE: [0.70-1.00]",
            {"question": question, "trace": trace or "(none)"},
        ),
    )


def goal_check(question: str, trace: str) -> PromptSpec:
    return PromptSpec(
        system="You verify if reasoning has reached a final answer.",
        user=_render(
            "Question: {question}\n\n"
            "Reasoning:\n{trace}\n\n"
            "Output exactly one line:\n"
            "GOAL: [YES or NO]",
            {"question": question, "trace": trace or "(none)"},
        ),
    )

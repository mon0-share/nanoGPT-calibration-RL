#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 14 16:07:23 2026

@author: christopher
"""


from dataclasses import dataclass
from typing import List, Sequence, Tuple

# Allowed outputs (3 actions)
CHOICES = ["1. 25%", "2. 25%", "1. 50%"]

# Interpret each output as p(A is correct)
P_A = {
    "1. 25%": 0.25,
    "2. 25%": 0.75,  # B has 25% => A has 75%
    "1. 50%": 0.50,
}

@dataclass(frozen=True)
class MCQExample:
    qid: int
    question: str
    A: str
    B: str
    correct: str  # "A" or "B"

def make_prompt(ex: MCQExample) -> str:
    return (
        f"{ex.qid}. {ex.question}\n"
        f"A) {ex.A}\n"
        f"B) {ex.B}\n"
        f"Reply with exactly one of: {', '.join(CHOICES)}\n"
        f"Prediction:"
    )

def brier(choice_str: str, correct: str) -> float:
    if choice_str not in P_A:
        raise ValueError(f"choice must be one of {CHOICES}, got {choice_str}")
    if correct not in ("A", "B"):
        raise ValueError("correct must be 'A' or 'B'")
    p = P_A[choice_str]
    y = 1.0 if correct == "A" else 0.0
    return (p - y) ** 2

def reward(choice_str: str, correct: str, mode: str = "neg_brier") -> float:
    b = brier(choice_str, correct)
    if mode == "neg_brier":
        return -b
    if mode == "one_minus":
        return 1.0 - b
    raise ValueError("mode must be 'neg_brier' or 'one_minus'")

def score_batch(
    choices: Sequence[str],
    batch: Sequence[MCQExample],
    mode: str = "neg_brier",
) -> Tuple[List[float], List[float], float, float]:
    briers = [brier(c, ex.correct) for c, ex in zip(choices, batch)]
    rewards = [reward(c, ex.correct, mode=mode) for c, ex in zip(choices, batch)]
    mean_brier = sum(briers) / len(briers)
    mean_reward = sum(rewards) / len(rewards)
    return briers, rewards, mean_brier, mean_reward

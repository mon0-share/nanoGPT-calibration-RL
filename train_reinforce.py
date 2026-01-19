#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 14 16:32:50 2026

@author: christopher
"""

# train_reinforce.py
import argparse
import os
import re
import json
import random
from typing import Dict, List, Tuple

import torch
import torch.nn as nn

from policy_choice import ChoicePolicy
from env_brier_min import MCQExample, make_prompt, brier, reward


# ----------------------------
# Parsing your text format
# ----------------------------
# Questions line example:
#   1. what is 2+2?  A) 4 B) 5
Q_RE = re.compile(r"^\s*(\d+)\.\s*(.*?)\s*A\)\s*(.*?)\s*B\)\s*(.*?)\s*$")

# Answers line example:
#   1. A
A_RE = re.compile(r"^\s*(\d+)\.\s*([AB])\s*$", re.IGNORECASE)


def load_questions(path: str, strict: bool = True) -> Dict[int, Tuple[str, str, str]]:
    out: Dict[int, Tuple[str, str, str]] = {}
    bad = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            m = Q_RE.match(s)
            if not m:
                bad.append(line.rstrip("\n"))
                continue
            qid = int(m.group(1))
            q = m.group(2).strip()
            A = m.group(3).strip()
            B = m.group(4).strip()
            out[qid] = (q, A, B)
    if strict and bad:
        preview = "\n".join(bad[:5])
        raise ValueError(
            "Some question lines didn't match:\n"
            "Expected: '<id>. <question>  A) <A> B) <B>' on one line.\n"
            f"First bad lines:\n{preview}"
        )
    return out


def load_answers(path: str, strict: bool = True) -> Dict[int, str]:
    out: Dict[int, str] = {}
    bad = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            m = A_RE.match(s)
            if not m:
                bad.append(line.rstrip("\n"))
                continue
            qid = int(m.group(1))
            ans = m.group(2).upper()
            out[qid] = ans
    if strict and bad:
        preview = "\n".join(bad[:5])
        raise ValueError(
            "Some answer lines didn't match:\n"
            "Expected: '<id>. A' or '<id>. B' on one line.\n"
            f"First bad lines:\n{preview}"
        )
    return out


def build_dataset(qpath: str, apath: str, strict: bool = True) -> List[MCQExample]:
    qmap = load_questions(qpath, strict=strict)
    amap = load_answers(apath, strict=strict)

    missing = []
    ds: List[MCQExample] = []
    for qid in sorted(qmap.keys()):
        if qid not in amap:
            missing.append(qid)
            continue
        q, A, B = qmap[qid]
        ds.append(MCQExample(qid=qid, question=q, A=A, B=B, correct=amap[qid]))

    if strict and missing:
        raise ValueError(f"Missing answers for qids: {missing[:20]} (showing up to 20)")
    if strict and not ds:
        raise ValueError("Parsed 0 examples; check file formats.")

    return ds


def split_train_val(ds: List[MCQExample], val_frac: float, seed: int) -> Tuple[List[MCQExample], List[MCQExample]]:
    rng = random.Random(seed)
    idxs = list(range(len(ds)))
    rng.shuffle(idxs)
    n_val = max(1, int(len(ds) * val_frac)) if len(ds) > 1 else 0
    val = [ds[i] for i in idxs[:n_val]]
    train = [ds[i] for i in idxs[n_val:]]
    return train, val


# ----------------------------
# Evaluation
# ----------------------------
@torch.no_grad()
def evaluate_mean_brier(policy: ChoicePolicy, ds: List[MCQExample]) -> float:
    policy.model.eval()
    total = 0.0
    for ex in ds:
        prompt = make_prompt(ex)
        _a, choice_str, _probs = policy.eval_choice(prompt)  # deterministic argmax
        total += brier(choice_str, ex.correct)
    policy.model.train()
    return total / max(1, len(ds))


# ----------------------------
# Training (REINFORCE)
# ----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--questions", required=True, help="Path to questions text file")
    ap.add_argument("--answers", required=True, help="Path to answers text file")

    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda", "mps"])
    ap.add_argument("--meta", default="data/shakespeare_char/meta.pkl", help="Char vocab meta.pkl used by policy")

    ap.add_argument("--out_dir", default="out-rl-brier")
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--val_frac", type=float, default=0.2)

    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--grad_clip", type=float, default=1.0)

    ap.add_argument("--reward_mode", default="neg_brier", choices=["neg_brier", "one_minus"])
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--eval_interval", type=int, default=200)
    ap.add_argument("--save_interval", type=int, default=200)

    # tiny GPT size (random init)
    ap.add_argument("--block_size", type=int, default=128)
    ap.add_argument("--n_layer", type=int, default=4)
    ap.add_argument("--n_head", type=int, default=4)
    ap.add_argument("--n_embd", type=int, default=128)

    args = ap.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    os.makedirs(args.out_dir, exist_ok=True)

    ds = build_dataset(args.questions, args.answers, strict=True)
    train_ds, val_ds = split_train_val(ds, val_frac=args.val_frac, seed=args.seed)

    # Policy (random init GPT + constrained 3-token actions)
    policy = ChoicePolicy(
        meta_path=args.meta,
        device=args.device,
        block_size=args.block_size,
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_embd=args.n_embd,
        seed=args.seed,
    )

    # Optimizer over GPT parameters (no separate head in this setup)
    optimizer = torch.optim.AdamW(policy.model.parameters(), lr=args.lr)

    # Save config
    with open(os.path.join(args.out_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2)

    # Initial eval
    init_val = evaluate_mean_brier(policy, val_ds) if val_ds else float("nan")
    print(f"dataset: train={len(train_ds)} val={len(val_ds)}")
    print(f"initial val mean brier: {init_val:.6f}")

    # Training loop
    policy.model.train()
    train_ptr = 0

    for step in range(1, args.steps + 1):
        # Sample a minibatch of size batch_size (cycle through shuffled train set)
        # For simplicity: reshuffle each epoch-like pass
        if train_ptr == 0:
            random.shuffle(train_ds)

        batch = train_ds[train_ptr : train_ptr + args.batch_size]
        train_ptr += args.batch_size
        if train_ptr >= len(train_ds):
            train_ptr = 0

        # Collect logprobs + rewards for REINFORCE
        logprobs: List[torch.Tensor] = []
        rewards: List[float] = []
        briers: List[float] = []

        for ex in batch:
            prompt = make_prompt(ex)
            act = policy.act(prompt, sample=True, temperature=args.temperature)  # sampling for RL
            r = reward(act.choice_str, ex.correct, mode=args.reward_mode)
            b = brier(act.choice_str, ex.correct)

            logprobs.append(act.logprob)  # tensor with grad
            rewards.append(float(r))
            briers.append(float(b))

        # Baseline: batch mean reward (variance reduction)
        baseline = sum(rewards) / len(rewards)
        advantages = [r - baseline for r in rewards]  # floats

        # REINFORCE loss: maximize E[r] => minimize -adv*logprob
        # loss = - mean_i (adv_i * logprob_i)
        loss = 0.0
        for adv, lp in zip(advantages, logprobs):
            loss = loss + (-adv) * lp
        loss = loss / len(logprobs)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if args.grad_clip is not None and args.grad_clip > 0:
            nn.utils.clip_grad_norm_(policy.model.parameters(), args.grad_clip)
        optimizer.step()

        if step % 50 == 0:
            mean_b = sum(briers) / len(briers)
            mean_r = sum(rewards) / len(rewards)
            print(f"step {step:5d} | train mean brier {mean_b:.6f} | train mean reward {mean_r:.6f} | loss {float(loss):.6f}")

        if step % args.eval_interval == 0 and val_ds:
            val_b = evaluate_mean_brier(policy, val_ds)
            print(f"== eval @ step {step} | val mean brier {val_b:.6f}")

        if step % args.save_interval == 0:
            ckpt = {
                "step": step,
                "model_state_dict": policy.model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "choice_token_ids": policy.choice_token_ids,
                "meta_path": args.meta,
            }
            torch.save(ckpt, os.path.join(args.out_dir, "ckpt.pt"))
            # lightweight rolling checkpoint too
            torch.save(ckpt, os.path.join(args.out_dir, f"ckpt_step_{step}.pt"))

    # Final save
    ckpt = {
        "step": args.steps,
        "model_state_dict": policy.model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "choice_token_ids": policy.choice_token_ids,
        "meta_path": args.meta,
    }
    torch.save(ckpt, os.path.join(args.out_dir, "ckpt.pt"))
    print("done.")


if __name__ == "__main__":
    main()

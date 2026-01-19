#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 14 16:20:54 2026

@author: christopher
"""

# policy_choice.py
import pickle
from dataclasses import dataclass
from typing import List, Tuple, Optional

import torch
import torch.nn.functional as F

from model import GPT, GPTConfig

# 3 allowed outputs
CHOICES = ["1. 25%", "2. 25%", "1. 50%"]


@dataclass
class ActResult:
    action_id: int          # 0..2
    choice_str: str         # one of CHOICES
    logprob: torch.Tensor   # scalar tensor (keeps grad)
    probs: torch.Tensor     # shape (3,) probs over the 3 actions (keeps grad)
    token_id: int           # actual vocab token id of the chosen special token


class ChoicePolicy:
    """
    Random-init nanoGPT policy constrained to 3 special output tokens.
    The input is still char-level encoding using the existing meta.pkl vocab.
    """

    def __init__(
        self,
        meta_path: str = "data/shakespeare_char/meta.pkl",
        device: str = "cpu",
        block_size: int = 128,
        n_layer: int = 4,
        n_head: int = 4,
        n_embd: int = 128,
        seed: int = 1337,
    ):
        self.device = torch.device(device)
        torch.manual_seed(seed)

        # Load tokenizer + extend vocab with 3 special "output tokens"
        meta = pickle.load(open(meta_path, "rb"))
        self.stoi = dict(meta["stoi"])
        self.itos = list(meta["itos"])

        self.choice_token_ids: List[int] = []
        for s in CHOICES:
            if s not in self.stoi:
                self.stoi[s] = len(self.itos)
                self.itos.append(s)
            self.choice_token_ids.append(self.stoi[s])

        self.vocab_size = len(self.itos)
        self.block_size = block_size

        # Random-init GPT
        cfg = GPTConfig(
            vocab_size=self.vocab_size,
            block_size=block_size,
            n_layer=n_layer,
            n_head=n_head,
            n_embd=n_embd,
            dropout=0.0,
            bias=False,
        )
        self.model = GPT(cfg).to(self.device)

    def encode(self, text: str) -> List[int]:
        """
        Char-level encode using *single-character* tokens only.
        (We intentionally do NOT emit the special CHOICES tokens from text.)
        """
        ids = [self.stoi[c] for c in text if (c in self.stoi and len(c) == 1)]
        if not ids:
            ids = [self.stoi.get(" ", 0)]
        return ids[: self.block_size]

    def _masked_logits_last(self, prompt_ids: List[int]) -> torch.Tensor:
        """
        Return masked logits over full vocab at the last position:
        only the 3 choice tokens are allowed, all others are -inf.
        Shape: (vocab,)
        """
        x = torch.tensor(prompt_ids, dtype=torch.long, device=self.device).unsqueeze(0)  # (1,T)

        # nanoGPT GPT.forward returns (logits, loss)
        # In inference mode (targets=None), logits is (B, 1, vocab) at last position.
        logits, _ = self.model(x)
        logits = logits[0, -1, :]  # (vocab,)

        masked = torch.full_like(logits, float("-inf"))
        masked[self.choice_token_ids] = logits[self.choice_token_ids]
        return masked

    def act(
        self,
        prompt_text: str,
        sample: bool = True,
        temperature: float = 1.0,
    ) -> ActResult:
        """
        Sample (or argmax) one of the 3 actions and return logprob for REINFORCE.
        - Sampling uses probs.detach() so sampling itself doesn't create weird grads,
          but logprob/probs DO keep gradients for the update step.
        """
        prompt_ids = self.encode(prompt_text)
        masked_logits = self._masked_logits_last(prompt_ids)  # (vocab,)

        # restrict to the 3 choices (still keeps grad)
        choice_logits = masked_logits[self.choice_token_ids]  # (3,)

        # temperature
        t = max(temperature, 1e-6)
        choice_logits_t = choice_logits / t

        probs = F.softmax(choice_logits_t, dim=-1)  # (3,)

        if sample:
            a = int(torch.multinomial(probs.detach(), num_samples=1).item())
        else:
            a = int(torch.argmax(probs, dim=-1).item())

        logprob = torch.log(probs[a] + 1e-12)  # scalar tensor
        tok_id = self.choice_token_ids[a]
        return ActResult(
            action_id=a,
            choice_str=CHOICES[a],
            logprob=logprob,
            probs=probs,
            token_id=tok_id,
        )

    @torch.no_grad()
    def eval_choice(self, prompt_text: str) -> Tuple[int, str, List[float]]:
        """
        Deterministic evaluation helper: returns argmax action and its probs (as python floats).
        """
        self.model.eval()
        prompt_ids = self.encode(prompt_text)
        masked_logits = self._masked_logits_last(prompt_ids)
        choice_logits = masked_logits[self.choice_token_ids]
        probs = torch.softmax(choice_logits, dim=-1)
        a = int(torch.argmax(probs).item())
        return a, CHOICES[a], [float(p) for p in probs.cpu()]

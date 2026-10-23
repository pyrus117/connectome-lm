"""The vocabulary-list task: memorise N random word pairings.

Cue and answer words are single tokens, so a pairing is one token predicting
another. Pairings are random, so English gives no hint; the model has to store
them. Within one sequence every cue appears once, so attention cannot copy an
answer from earlier in the same sequence: recall has to come from weights.
"""
import re

import numpy as np
import torch

WORD = re.compile(r"Ġ[a-z]{4,10}")
IGNORE = -100


def word_pool(tok):
    """Ids of single-token lowercase words (4-10 letters, leading space)."""
    return np.array([i for i in range(tok.get_vocab_size())
                     if WORD.fullmatch(tok.id_to_token(i) or "")], dtype=np.int64)


def make_pairs(pool, n, seed):
    """n (cue, answer) id pairs; cues and answers are disjoint."""
    if 2 * n > len(pool):
        raise ValueError(f"{n} pairs need {2 * n} words; the pool has {len(pool)}")
    words = np.random.default_rng(seed).permutation(pool)[:2 * n]
    return words[:n], words[n:]


def _pack(cues, answers, idx, eot):
    """eot, then cue/answer for each index in idx. Targets are the answer at
    each cue position and IGNORE everywhere else."""
    x = np.empty(1 + 2 * len(idx), np.int64)
    x[0] = eot
    x[1::2], x[2::2] = cues[idx], answers[idx]
    y = np.full_like(x, IGNORE)
    y[1::2] = answers[idx]
    return x, y


def _stack(rows):
    return (torch.from_numpy(np.stack([r[0] for r in rows])),
            torch.from_numpy(np.stack([r[1] for r in rows])))


def train_batch(cues, answers, batch, per_seq, seed, step, eot):
    rng = np.random.default_rng([seed, step])
    return _stack([_pack(cues, answers, rng.choice(len(cues), per_seq, replace=False), eot)
                   for _ in range(batch)])


def quiz_batches(cues, answers, per_seq, seed, eot):
    """Every cue exactly once, in a fixed order, per_seq cues per sequence."""
    if len(cues) % per_seq:
        raise ValueError(f"{len(cues)} pairs do not split into sequences of {per_seq}")
    # 2**31 keeps the quiz order apart from every training step's stream
    order = np.random.default_rng([seed, 2**31]).permutation(len(cues))
    return _stack([_pack(cues, answers, order[i:i + per_seq], eot)
                   for i in range(0, len(order), per_seq)])


def correct(logits, y):
    """Number of cue positions whose top guess is the answer."""
    m = y != IGNORE
    return int((logits.argmax(-1)[m] == y[m]).sum())

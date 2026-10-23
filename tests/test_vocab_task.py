# tests/test_vocab_task.py
from pathlib import Path

import numpy as np
import pytest
import torch
from tokenizers import Tokenizer

from clm.vocab_task import IGNORE, correct, make_pairs, quiz_batches, train_batch, word_pool

TOKENIZER = Path(__file__).resolve().parent.parent / "data" / "fineweb" / "tokenizer.json"
EOT = 0


def test_word_pool_is_single_token_lowercase_words_and_big_enough():
    tok = Tokenizer.from_file(str(TOKENIZER))
    pool = word_pool(tok)
    assert len(pool) >= 2 * 1536
    assert len(set(pool.tolist())) == len(pool)
    for i in pool[:200]:
        w = tok.decode([int(i)])
        assert w[0] == " " and w[1:].isalpha() and w[1:].islower() and 4 <= len(w) - 1 <= 10


def test_pairs_are_disjoint_and_fixed_by_seed():
    pool = np.arange(100, 400)
    c, a = make_pairs(pool, 128, seed=0)
    assert len(c) == len(a) == 128
    assert not set(c.tolist()) & set(a.tolist())
    c2, a2 = make_pairs(pool, 128, seed=0)
    assert np.array_equal(c, c2) and np.array_equal(a, a2)
    assert not np.array_equal(c, make_pairs(pool, 128, seed=1)[0])
    with pytest.raises(ValueError):
        make_pairs(pool, 151, seed=0)


def test_train_batch_targets_only_answers_and_never_repeats_a_cue_in_a_row():
    c, a = make_pairs(np.arange(100, 400), 128, seed=0)
    answer_of = dict(zip(c.tolist(), a.tolist()))
    x, y = train_batch(c, a, batch=8, per_seq=64, seed=0, step=5, eot=EOT)
    assert x.shape == y.shape == (8, 129)
    assert (x[:, 0] == EOT).all()
    for row_x, row_y in zip(x.tolist(), y.tolist()):
        cues = row_x[1::2]
        assert len(set(cues)) == 64
        assert row_y[1::2] == [answer_of[q] for q in cues]
        assert row_x[2::2] == row_y[1::2]
        assert set(row_y[0::2]) == {IGNORE}
    x2, _ = train_batch(c, a, batch=8, per_seq=64, seed=0, step=5, eot=EOT)
    assert torch.equal(x, x2)
    assert not torch.equal(x, train_batch(c, a, 8, 64, 0, 6, EOT)[0])


def test_quiz_asks_every_cue_exactly_once():
    c, a = make_pairs(np.arange(100, 400), 128, seed=0)
    x, y = quiz_batches(c, a, per_seq=64, seed=0, eot=EOT)
    assert x.shape == (2, 129)
    assert sorted(x[:, 1::2].reshape(-1).tolist()) == sorted(c.tolist())
    with pytest.raises(ValueError):
        quiz_batches(c[:100], a[:100], per_seq=64, seed=0, eot=EOT)


def test_correct_counts_top_guesses_at_cue_positions_only():
    y = torch.tensor([[IGNORE, 5, IGNORE, 7]])
    logits = torch.zeros(1, 4, 10)
    logits[0, 1, 5] = 1.0     # right
    logits[0, 3, 2] = 1.0     # wrong
    logits[0, 0, 9] = 1.0     # ignored position
    assert correct(logits, y) == 1

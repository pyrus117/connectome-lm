import numpy as np
import pytest

from clm import counts as C
from clm.countread import CountReader
from clm.ngram_lm import NgramLM, add_one_unigram
from test_counts import EOT

V = 30                                   # tokens 0..28 plus EOT 29


def build_lm(stream, tmp_path, c_min=(1, 1, 2, 2)):
    for k, c in zip(range(1, 5), c_min):
        C.save(C.build(stream, k, EOT, c, parts=2), tmp_path / f"order{k + 1}", {"c_min": c, "eot_id": EOT})
    return NgramLM(CountReader(tmp_path), add_one_unigram(stream, V))


@pytest.fixture
def stream():
    rng = np.random.default_rng(0)
    p = 1.0 / np.arange(1, 29) ** 1.1          # 28 live tokens: common histories have fanout above TOP
    p /= p.sum()
    out = []
    for _ in range(400):
        out += list(rng.choice(28, size=int(rng.integers(3, 40)), p=p)) + [EOT]
    return np.asarray(out, dtype=np.uint16)


def test_probabilities_sum_to_one_over_the_vocabulary_at_every_position(stream, tmp_path):
    lm = build_lm(stream, tmp_path)
    x = np.stack([stream[s:s + 20] for s in (0, 300, 900, 2000)]).astype(np.int64)
    every = np.broadcast_to(np.arange(V), (*x.shape, V))
    p = lm.prob(x, every)
    assert p.min() >= 0
    assert np.allclose(p.sum(-1), 1.0, atol=1e-9)


def test_a_stored_history_gives_discounted_ratios_and_a_new_one_gives_the_unigram(tmp_path):
    t = np.array([1, 2, 1, 2, 1, 3, EOT], dtype=np.uint16)
    d = tmp_path / "t"
    C.save(C.build(t, 1, EOT, 1, parts=1), d / "order2", {"c_min": 1, "eot_id": EOT})
    uni = add_one_unigram(t, V)
    lm = NgramLM(CountReader(d), uni)
    x = np.array([[1, 4]], dtype=np.int64)
    p = lm.prob(x, np.array([[[2, 3, 4], [0, 1, 4]]]))
    d = lm.d
    mass = 1 - ((2 - d) + (1 - d)) / 3                         # history (1): 2 twice, 3 once
    other = mass * uni[4] / (1 - uni[2] - uni[3])
    assert p[0, 0] == pytest.approx([(2 - d) / 3, (1 - d) / 3, other])
    assert p[0, 1] == pytest.approx(uni[[0, 1, 4]])            # history (4) was never seen: unigram


def test_a_truncated_history_hands_its_leftover_mass_to_the_shorter_history(stream, tmp_path):
    """Token 0's history has more than 16 continuations, so part of its mass is
    not in the stored entries; unstored tokens must share it, not get zero."""
    lm = build_lm(stream, tmp_path, c_min=(1, 1, 1, 1))
    table = lm.tables[0]
    key = C.pack_history((0,))
    row = int(np.searchsorted(table.keys, key))
    lo, hi = table.offsets[row], table.offsets[row + 1]
    assert hi - lo == C.TOP and table.hcount[row] > table.cnt[lo:hi].sum()
    stored = set(int(v) for v in table.tok[lo:hi])
    other = next(w for w in range(28) if w not in stored)
    x = np.array([[0]], dtype=np.int64)
    p = lm.prob(x, np.array([[[other]]]))[0, 0, 0]
    assert p > 0
    leftover = 1 - (table.cnt[lo:hi].sum() - lm.d * C.TOP) / table.hcount[row]
    assert 0 < p < leftover


def test_nll_is_minus_log_of_the_target_probability(stream, tmp_path):
    lm = build_lm(stream, tmp_path)
    x = stream[:40].astype(np.int64)[None]
    y = stream[1:41].astype(np.int64)[None]
    nll = lm.nll(x, y)
    assert nll.shape == x.shape and np.all(nll >= 0)       # 0 is legal: a history seen once always continues the same way
    assert np.allclose(np.exp(-nll), lm.prob(x, y[..., None])[..., 0])


def test_unigram_rejects_a_token_outside_the_vocabulary():
    with pytest.raises(ValueError, match="vocabulary"):
        add_one_unigram(np.array([1, 9], dtype=np.uint16), 5)

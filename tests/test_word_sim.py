import numpy as np

from clm.word_sim import context_similarity


def test_words_used_in_the_same_contexts_are_more_similar():
    # words 10 and 11 always sit between 5 and 6; word 12 sits between 7 and 8
    rng = np.random.default_rng(0)
    chunks = []
    for _ in range(300):
        k = int(rng.integers(3))
        w, a, b = [(10, 5, 6), (11, 5, 6), (12, 7, 8)][k]
        chunks += [a, w, b, int(rng.integers(20, 40))]
    tokens = np.array(chunks, dtype=np.uint16)
    pool = np.array([10, 11, 12])
    sim = context_similarity(tokens, pool, vocab=64, window=1)
    assert sim.shape == (3, 3)
    assert np.allclose(np.diag(sim), 1.0)
    assert np.allclose(sim, sim.T)
    assert sim[0, 1] > 0.9 and sim[0, 2] < 0.1


def test_a_word_that_never_occurs_has_zero_similarity():
    tokens = np.array([5, 10, 6, 5, 11, 6] * 20, dtype=np.uint16)
    sim = context_similarity(tokens, np.array([10, 11, 12]), vocab=64, window=1)
    assert sim[2, 0] == sim[2, 1] == 0.0

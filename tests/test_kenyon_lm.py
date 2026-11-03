import numpy as np
import pytest

from clm import counts as C
from clm import kenyon as K
from clm import kenyon_lm as L
from clm.ngram_lm import NgramLM, add_one_unigram

CLUSTERS, PER = 30, 8
V = CLUSTERS * PER
EOT = V                                              # an id outside the live tokens


def world(n_tokens, seed):
    """Next cluster = (cluster of the token before + cluster of the last token) mod CLUSTERS, 90% of the time:
    the answer needs both history tokens, and histories of unseen token pairs share clusters with seen ones."""
    rng = np.random.default_rng(seed)
    toks = [int(rng.integers(V)), int(rng.integers(V))]
    while len(toks) < n_tokens:
        c = (toks[-2] // PER + toks[-1] // PER) % CLUSTERS if rng.random() < 0.9 else int(rng.integers(CLUSTERS))
        toks.append(c * PER + int(rng.integers(PER)))
    return np.array(toks, dtype=np.uint16)


def cluster_codes():
    rng = np.random.default_rng(0)
    centres = rng.standard_normal((CLUSTERS, 24))
    return (np.repeat(centres, PER, axis=0) + 0.03 * rng.standard_normal((V, 24))).astype(np.float32)


@pytest.fixture(scope="module")
def setup():
    train, held, val = world(60_000, 1), world(12_000, 2), world(12_000, 3)
    t1, t2 = C.build(train, 1, EOT, 1, parts=2), C.build(train, 2, EOT, 1, parts=2)
    uni = add_one_unigram(train, V + 1)
    exact = NgramLM([t1, t2], uni)
    kh = K.KenyonHash(cluster_codes(), k=2, units=32, seed=0)
    kh.calibrate(train[:2002].astype(np.int64)[np.arange(2000)[:, None] + np.arange(2)])
    buckets = {"kenyon": lambda packed, toks: kh.bucket_keys(toks),
               "random": lambda packed, toks: K.hash_bucket(packed, 32 * 32)}
    lms = {}
    for name, fn in buckets.items():
        bt = K.bucket_table(train, 2, EOT, fn, c_min=1, parts=2)
        lms[name] = NgramLM([t1, bt], uni, keyfns=[None, fn])
    return exact, lms, t2, held, val


def run(setup, name):
    exact, lms, t2, held, val = setup
    out = {}
    for tag, stream in (("held", held), ("val", val)):
        out[tag] = L.score_windows(exact, lms[name], t2, stream, 64, len(stream) - 1, micro=4)
    lam = L.fit_lambdas(*out["held"])
    pe, pb, cls = out["val"]
    return L.class_loss(pe, cls), L.class_loss(L.mixture_probs(pe, pb, cls, lam), cls), lam


def test_kenyon_buckets_recover_what_exact_histories_cannot_see_and_random_buckets_do_not(setup):
    exact_loss, kenyon_loss, lam_k = run(setup, "kenyon")
    _, random_loss, lam_r = run(setup, "random")
    unseen = "not stored"
    assert exact_loss[unseen] - kenyon_loss[unseen] > 0.15                      # similar histories answer for unseen ones
    assert exact_loss[unseen] - random_loss[unseen] < 0.03                      # pooling unrelated histories does not
    assert lam_k[0] > 0.3 > lam_r[0]


def test_fit_lambdas_finds_the_likelihood_maximum_and_never_mixes_short_windows():
    rng = np.random.default_rng(0)
    n = 40_000
    a, b = rng.uniform(0.01, 0.5, n), rng.uniform(0.01, 0.5, n)
    cls = np.zeros(n, dtype=int)
    lam = L.fit_lambdas(a, b, cls)
    ll = lambda l: np.log((1 - l) * a + l * b).sum()
    assert ll(lam[0]) >= max(ll(0.0), ll(1.0), ll(lam[0] + 0.05), ll(lam[0] - 0.05)) - 1e-6
    only_b_helps = L.fit_lambdas(np.full(n, 0.01), np.full(n, 0.4), cls)
    assert only_b_helps[0] > 0.99
    assert lam[-1] == 0.0


def test_mixture_is_a_proper_blend_and_class_loss_reports_every_class():
    pe, pb = np.array([0.2, 0.4, 0.1]), np.array([0.6, 0.2, 0.1])
    cls = np.array([0, 1, 5])
    lam = np.array([0.5, 0.0, 0, 0, 0, 0.0])
    assert L.mixture_probs(pe, pb, cls, lam).tolist() == pytest.approx([0.4, 0.4, 0.1])
    loss = L.class_loss(np.array([0.5, 0.25, 0.1]), cls)
    assert loss["not stored"] == pytest.approx(np.log(2)) and loss["1-9"] == pytest.approx(np.log(4))
    assert loss["10-99"] is None and loss["short window"] == pytest.approx(np.log(10))
    assert L.rare_loss(np.array([0.5, 0.25, 0.1]), cls) == pytest.approx((np.log(2) + np.log(4)) / 2)


def test_history_class_follows_the_stored_count_and_marks_windows_too_short_for_the_history():
    stream = np.array([1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 4, 5, 6], dtype=np.uint16)
    table = C.build(stream, 2, EOT, 1, parts=1)                    # (1,2) seen 4 times, (2,3) 3 times, (2,4) once, ...
    x = np.array([[1, 2, 3, 6, 5]], dtype=np.int64)
    cls = L.history_class(table, x)[0]
    assert cls.tolist() == [5, 1, 1, 0, 0]                          # window start; (1,2) and (2,3) seen 1-9 times; (3,6), (6,5) never

from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pytest

from clm import counts as C
from clm import kenyon as K
from clm.countread import BUCKET_LABELS, CountReader, Prefetcher, bucket_of, make_step
from test_counts import EOT, brute, corpus

C_MIN = (1, 1, 2, 2)                  # orders 2..5, as in the real tables
K_TOP = 4


@pytest.fixture(scope="module")
def stream():
    return corpus(seed=7, n_docs=300, vocab=14)


@pytest.fixture(scope="module")
def reader(stream, tmp_path_factory):
    d = tmp_path_factory.mktemp("tables")
    for k, c_min in zip(range(1, 5), C_MIN):
        C.save(C.build(stream, k, EOT, c_min, parts=3), d / f"order{k + 1}", {"c_min": c_min, "eot_id": EOT})
    return CountReader(d, k_top=K_TOP)


def windows(stream, rng, batch=6, ctx=24):
    starts = rng.integers(0, len(stream) - ctx - 1, batch)
    x = np.stack([stream[s:s + ctx] for s in starts]).astype(np.int64)
    y = np.stack([stream[s + 1:s + ctx + 1] for s in starts]).astype(np.int64)
    return x, y


def expected(stream, x, y, loo):
    """Brute-force COUNT read: {(b, t, order index): {token: score}}."""
    out = {}
    for o, k in enumerate(range(1, 5)):
        rows = brute(stream, k)
        for b in range(x.shape[0]):
            for t in range(k - 1, x.shape[1]):
                h = tuple(int(v) for v in x[b, t - k + 1:t + 1])
                if EOT in h or h not in rows:
                    continue
                cnt = rows[h]
                total = sum(cnt.values())
                if total < C_MIN[o]:
                    continue
                top = sorted(cnt.items(), key=lambda kv: (-kv[1], kv[0]))[:C.TOP][:K_TOP]
                if loo:
                    w = int(y[b, t])
                    total -= 1
                    if total < C_MIN[o]:
                        continue
                    top = [(tok, c - (tok == w)) for tok, c in top]
                got = {tok: c / total / len(C_MIN) for tok, c in top if c > 0}
                if got:
                    out[(b, t, o)] = got
    return out


def read_back(reader, tok, score):
    out = {}
    for b, t, s in zip(*np.nonzero(score)):
        o = s // K_TOP
        out.setdefault((int(b), int(t), int(o)), {})[int(tok[b, t, s])] = float(score[b, t, s])
    return out


def close(a, b):
    assert a.keys() == b.keys()
    for key in a:
        assert a[key].keys() == b[key].keys(), key
        for tok in a[key]:
            assert a[key][tok] == pytest.approx(b[key][tok], rel=1e-5), (key, tok)


@pytest.mark.parametrize("loo", [False, True])
def test_read_matches_a_brute_force_count_with_and_without_leave_one_out(stream, reader, loo):
    x, y = windows(stream, np.random.default_rng(1))
    tok, score = reader.lookup(x, y if loo else None)
    assert tok.shape == score.shape == (6, 24, 4 * K_TOP)
    close(read_back(reader, tok, score), expected(stream, x, y, loo))


def test_leave_one_out_changes_the_read_only_where_the_position_s_own_count_mattered(stream, reader):
    x, y = windows(stream, np.random.default_rng(2))
    plain, loo = expected(stream, x, y, False), expected(stream, x, y, True)
    assert plain != loo and len(loo) < len(plain)          # singletons and c_min drops disappear
    assert all(sum(v.values()) <= 1 / len(C_MIN) + 1e-9 for v in plain.values())


def test_leave_one_out_removes_a_history_that_was_seen_only_once():
    """A 5-token stream seen once: with the position's own occurrence removed, nothing is left."""
    t = np.array([3, 4, 5, 6, EOT], dtype=np.uint16)
    import tempfile
    from pathlib import Path
    with tempfile.TemporaryDirectory() as d:
        C.save(C.build(t, 1, EOT, 1, parts=1), Path(d) / "order2", {"c_min": 1, "eot_id": EOT})
        r = CountReader(d, k_top=2)
        x, y = t[None, :4].astype(np.int64), t[None, 1:5].astype(np.int64)
        assert r.lookup(x)[1].max() == 1.0               # one order, so J = 1
        assert r.lookup(x, y)[1].max() == 0.0


def test_a_window_start_has_no_slot_for_a_longer_history_than_it_holds(stream, reader):
    x, y = windows(stream, np.random.default_rng(3))
    _, score = reader.lookup(x, y)
    for o, k in enumerate(range(1, 5)):
        assert not score[:, :k - 1, o * K_TOP:(o + 1) * K_TOP].any()


def test_a_history_holding_end_of_text_leaves_its_slot_empty(reader):
    x = np.array([[1, 2, EOT, 1, 2, 3, 1, 2]], dtype=np.int64)
    _, score = reader.lookup(x)
    for o, k in enumerate(range(1, 5)):
        empty = [t for t in range(x.shape[1]) if t >= k - 1 and EOT in x[0, t - k + 1:t + 1]]
        for t in empty:
            assert not score[0, t, o * K_TOP:(o + 1) * K_TOP].any(), (k, t)


def test_prefetcher_hands_over_each_step_it_was_asked_for_and_reads_ahead(stream, reader):
    train = stream
    make = lambda s: make_step(train, 16, 4, 0, s, reader, True)
    pf = Prefetcher(make, end=6)
    for s in range(2, 6):                                    # starting mid-run, as a resumed run does
        got = pf.get(s)
        want = make(s)
        assert got.step == s
        assert np.array_equal(got.x, want.x) and np.array_equal(got.score, want.score)
    assert pf.hits == 3 and pf.pending is None
    pf.close()


def test_step_data_depends_only_on_seed_and_step(stream, reader):
    a = make_step(stream, 16, 4, 0, 3, reader, True)
    b = make_step(stream, 16, 4, 0, 3, reader, True)
    c = make_step(stream, 16, 4, 1, 3, reader, True)
    assert np.array_equal(a.x, b.x) and np.array_equal(a.score, b.score)
    assert not np.array_equal(a.x, c.x)
    none = make_step(stream, 16, 4, 0, 3, None, True)
    assert none.tok is None and np.array_equal(none.x, a.x) and np.array_equal(none.y, a.y)


def test_reader_rejects_bad_setup(tmp_path):
    with pytest.raises(FileNotFoundError):
        CountReader(tmp_path)
    t = np.array([1, 2, 3, EOT], dtype=np.uint16)
    C.save(C.build(t, 1, EOT, 1, parts=1), tmp_path / "order2", {"c_min": 1, "eot_id": EOT})
    with pytest.raises(ValueError, match="k_top"):
        CountReader(tmp_path, k_top=C.TOP + 1)


def test_history_counts_report_the_stored_count_zero_for_unstored_and_minus_one_for_short_windows(stream, reader):
    x, _ = windows(stream, np.random.default_rng(4))
    for k in (3, 4):
        got = reader.history_counts(x, k)
        assert got.shape == x.shape and (got[:, :k - 1] == -1).all()
        rows = brute(stream, k)
        c_min = C_MIN[k - 1]
        for b in range(x.shape[0]):
            for t in range(k - 1, x.shape[1]):
                h = tuple(int(v) for v in x[b, t - k + 1:t + 1])
                total = sum(rows[h].values()) if h in rows and EOT not in h else 0
                assert got[b, t] == (total if total >= c_min else 0), (k, b, t)


def test_buckets_split_at_1_10_100_and_1000_and_skip_short_windows():
    c = np.array([-1, 0, 1, 9, 10, 99, 100, 999, 1000, 10**6])
    assert bucket_of(c).tolist() == [-1, 0, 1, 1, 2, 2, 3, 3, 4, 4]
    assert len(BUCKET_LABELS) == 5


def bucket_fn(packed, toks):
    return (toks.sum(1) % 6).astype(np.uint64)          # 6 buckets: pools unrelated histories, but exactly computable


BUCKET_KS = (2, 3)


@pytest.fixture(scope="module")
def bucket_reader(stream, tmp_path_factory):
    d = tmp_path_factory.mktemp("tables2")
    for k, c_min in zip(range(1, 5), C_MIN):
        C.save(C.build(stream, k, EOT, c_min, parts=3), d / f"order{k + 1}", {"c_min": c_min, "eot_id": EOT})
    fn = K.exclude_eot(bucket_fn, EOT)
    buckets = {k: (K.bucket_table(stream, k, EOT, fn, c_min=1, parts=3), fn) for k in BUCKET_KS}
    return CountReader(d, k_top=K_TOP, buckets=buckets)


def expected_with_buckets(stream, x, y, loo):
    """Brute force: the exact read, and for each bucketed order a missing exact history gets its bucket's pooled row."""
    out = expected(stream, x, y, loo)
    for k in BUCKET_KS:
        o = k - 1
        pooled = defaultdict(Counter)
        for h, cont in brute(stream, k).items():
            for w, c in cont.items():
                pooled[sum(h) % 6][w] += c
        for b in range(x.shape[0]):
            for t in range(k - 1, x.shape[1]):
                h = tuple(int(v) for v in x[b, t - k + 1:t + 1])
                if (b, t, o) in out or EOT in h:
                    continue
                cnt = pooled[sum(h) % 6]
                top = sorted(cnt.items(), key=lambda kv: (-kv[1], kv[0]))[:C.TOP][:K_TOP]
                total = sum(cnt.values())
                if loo:
                    w = int(y[b, t])
                    total -= 1
                    top = [(tok, c - (tok == w)) for tok, c in top]
                if total < 1:
                    continue
                got = {tok: c / total / len(C_MIN) for tok, c in top if c > 0}
                if got:
                    out[(b, t, o)] = got
    return out


@pytest.mark.parametrize("loo", [False, True])
def test_a_missing_exact_history_reads_its_pooled_bucket_instead(stream, bucket_reader, loo):
    x, y = windows(stream, np.random.default_rng(9))
    tok, score = bucket_reader.lookup(x, y if loo else None)
    want = expected_with_buckets(stream, x, y, loo)
    close(read_back(bucket_reader, tok, score), want)
    plain = expected(stream, x, y, loo)
    assert len(want) > len(plain)                                   # the fallback filled slots that were empty


def test_exact_histories_keep_their_own_counts_and_buckets_only_fill_gaps(stream, reader, bucket_reader):
    x, y = windows(stream, np.random.default_rng(10))
    for yy in (None, y):
        tok0, score0 = reader.lookup(x, yy)
        tok1, score1 = bucket_reader.lookup(x, yy)
        filled = score0 > 0
        assert np.array_equal(score0[filled], score1[filled]) and np.array_equal(tok0[filled], tok1[filled])
        unbucketed = [o for o in range(4) if o + 1 not in BUCKET_KS]
        for o in unbucketed:
            sl = slice(o * K_TOP, (o + 1) * K_TOP)
            assert np.array_equal(score0[..., sl], score1[..., sl])


def test_buckets_for_a_history_length_without_an_exact_table_are_refused(stream, tmp_path):
    C.save(C.build(stream, 1, EOT, 1, parts=1), tmp_path / "order2", {"c_min": 1, "eot_id": EOT})
    other = K.bucket_table(stream, 2, EOT, K.exclude_eot(bucket_fn, EOT), c_min=1, parts=1)
    with pytest.raises(ValueError, match="no exact table"):
        CountReader(tmp_path, k_top=K_TOP, buckets={2: (other, None)})

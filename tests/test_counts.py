from collections import Counter, defaultdict

import numpy as np
import pytest

from clm import counts as C

EOT = 29


def corpus(seed=0, n_docs=150, vocab=12):
    """Skewed random documents, each closed by EOT, so histories repeat unevenly."""
    rng = np.random.default_rng(seed)
    p = 1.0 / np.arange(1, vocab + 1) ** 1.2
    p /= p.sum()
    out = []
    for _ in range(n_docs):
        out += list(rng.choice(vocab, size=int(rng.integers(1, 30)), p=p)) + [EOT]
    return np.asarray(out, dtype=np.uint16)


def brute(tokens, k):
    rows = defaultdict(Counter)
    t = [int(x) for x in tokens]
    for i in range(len(t) - k):
        h = tuple(t[i:i + k])
        if EOT not in h:
            rows[h][t[i + k]] += 1
    return rows


def expected(tokens, k, c_min):
    out = {}
    for h, c in brute(tokens, k).items():
        total = sum(c.values())
        if total >= c_min:
            out[C.pack_history(h)] = (total, sorted(c.items(), key=lambda kv: (-kv[1], kv[0]))[:C.TOP])
    return out


def as_dict(table):
    out = {}
    for r, key in enumerate(table.keys):
        lo, hi = table.offsets[r], table.offsets[r + 1]
        out[int(key)] = (int(table.hcount[r]), [(int(t), int(c)) for t, c in zip(table.tok[lo:hi], table.cnt[lo:hi])])
    return out


@pytest.mark.parametrize("k", [1, 2, 3, 4])
@pytest.mark.parametrize("c_min", [1, 2, 5])
def test_table_matches_a_brute_force_count(k, c_min):
    t = corpus()
    table = C.build(t, k, EOT, c_min, parts=3, chunk=101)
    assert as_dict(table) == expected(t, k, c_min)
    assert np.all(np.diff(table.keys.astype(np.int64)) > 0)
    assert np.all(np.diff(table.offsets.astype(np.int64)) >= 1)


def test_no_history_spans_a_document_boundary_but_a_continuation_may_be_eot():
    t = np.array([1, 2, EOT, 3, 4, EOT], dtype=np.uint16)
    table = C.build(t, 2, EOT, 1, parts=1)
    assert as_dict(table) == {C.pack_history((1, 2)): (1, [(EOT, 1)]), C.pack_history((3, 4)): (1, [(EOT, 1)])}


def test_partitions_and_chunks_do_not_change_the_table():
    t = corpus(seed=1)
    ref = as_dict(C.build(t, 3, EOT, 1, parts=1, chunk=1 << 20))
    for parts, chunk in ((7, 1 << 20), (1, 13), (5, 64)):
        table = C.build(t, 3, EOT, 1, parts=parts, chunk=chunk)
        assert np.all(np.diff(table.keys.astype(np.int64)) > 0)      # a dict would hide duplicate rows
        assert as_dict(table) == ref


def test_partition_bounds_cover_the_vocabulary_in_order():
    b = C.partition_bounds(corpus(seed=2), 6, EOT, chunk=50)
    assert b[0] == 0 and b[-1] == 1 << C.VOCAB_BITS and b == sorted(b)


def test_only_the_top_16_continuations_are_kept_but_the_history_count_is_complete():
    body = []
    for x in range(6, 26):
        body += [5, x] * (x % 3 + 1) + [EOT]
    t = np.asarray(body, dtype=np.uint16)
    got = as_dict(C.build(t, 1, EOT, 1, parts=2))
    assert got == expected(t, 1, 1)                                   # every row, so a spill into the next shows
    row = got[C.pack_history((5,))]
    full = Counter({x: x % 3 + 1 for x in range(6, 26)})
    assert row[0] == sum(full.values())
    assert row[1] == sorted(full.items(), key=lambda kv: (-kv[1], kv[0]))[:C.TOP]
    assert len(row[1]) == C.TOP


def test_c_min_drops_rarer_histories_and_keeps_those_at_the_threshold():
    t = corpus(seed=3)
    counts = {h: sum(c.values()) for h, c in brute(t, 2).items()}
    c_min = sorted(set(counts.values()))[len(set(counts.values())) // 2]
    got = {int(k) for k in C.build(t, 2, EOT, c_min, parts=2).keys}
    assert got == {C.pack_history(h) for h, n in counts.items() if n >= c_min}
    assert any(n == c_min for n in counts.values())


@pytest.mark.parametrize("k, vocab", [(3, 12), (1, 28)])      # vocab 28 gives fanout above TOP
def test_profile_predicts_the_size_of_every_threshold(k, vocab):
    t = corpus(seed=4, n_docs=400, vocab=vocab)
    rows, entries = C.profile(t, k, EOT, parts=3, chunk=97)
    kept_rows, kept_entries = C.sizes_by_threshold(rows, entries)
    for c_min in (1, 2, 3, 6):
        table = C.build(t, k, EOT, c_min, parts=3, chunk=97)
        assert kept_rows[c_min] == table.keys.size
        assert kept_entries[c_min] == table.tok.size


def test_pick_cmin_takes_the_smallest_threshold_that_fits():
    rows, entries = np.zeros(C.CAP + 1, np.int64), np.zeros(C.CAP + 1, np.int64)
    rows[1], rows[5], entries[1], entries[5] = 100, 10, 300, 30
    assert C.pick_cmin(rows, entries, C.table_bytes(110, 330)) == 1
    assert C.pick_cmin(rows, entries, C.table_bytes(110, 330) - 1) == 2
    assert C.pick_cmin(rows, entries, C.table_bytes(10, 30)) == 2
    with pytest.raises(ValueError, match="non-empty"):
        C.pick_cmin(rows, entries, C.table_bytes(10, 30) - 1)


def test_save_and_load_round_trip(tmp_path):
    table = C.build(corpus(seed=5), 2, EOT, 2, parts=2)
    C.save(table, tmp_path / "o3", {"c_min": 2})
    back, meta = C.load(tmp_path / "o3")
    assert meta == {"k": 2, "c_min": 2}
    assert as_dict(back) == as_dict(table)


def test_bad_inputs_raise():
    t = corpus()
    with pytest.raises(ValueError, match="1..4"):
        C.build(t, 5, EOT, 1)
    with pytest.raises(ValueError, match="c_min"):
        C.build(t, 2, EOT, 0)
    with pytest.raises(ValueError, match="no history"):
        C.build(t, 2, EOT, 10**9)
    with pytest.raises(ValueError, match="13 bits"):
        C.build(np.array([1, 2, 9000, 3], dtype=np.uint16), 1, EOT, 1)

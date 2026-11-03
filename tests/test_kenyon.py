from collections import Counter, defaultdict

import numpy as np
import pytest

from clm import counts as C
from clm import kenyon as K
from test_counts import EOT, as_dict, brute, corpus


def test_cooccurrence_matches_a_brute_force_count_and_skips_end_of_text():
    t = corpus(seed=1, n_docs=60, vocab=10)
    got = K.cooccurrence(t, 30, window=2, eot=EOT, chunk=37)
    want = np.zeros((30, 30), dtype=np.int64)
    for i in range(len(t)):
        for off in (1, 2):
            if i + off < len(t) and EOT not in (t[i], t[i + off]):
                want[t[i], t[i + off]] += 1
                want[t[i + off], t[i]] += 1
    assert np.array_equal(got, want) and np.array_equal(got, got.T)


def test_ppmi_codes_put_tokens_with_shared_contexts_close_together():
    rng = np.random.default_rng(0)
    seq = []
    for _ in range(4000):                                    # "x {a|b} y" and "u c v": a and b share contexts, c does not
        if rng.random() < 0.5:
            seq += [10, int(rng.choice([0, 1])), 11]
        else:
            seq += [12, 2, 13]
        seq += [int(rng.integers(14, 20))]
    t = np.array(seq, dtype=np.uint16)
    codes = K.ppmi_codes(K.cooccurrence(t, 24, window=2), dim=6)
    cos = lambda a, b: float(codes[a] @ codes[b] / (np.linalg.norm(codes[a]) * np.linalg.norm(codes[b])))
    assert codes.shape == (24, 6) and not codes[21].any()    # a token never seen has no code
    assert cos(0, 1) > 0.9 > cos(0, 2) + 0.3


def clustered_codes(n_clusters=40, per=4, dim=16, noise=0.05, seed=0):
    rng = np.random.default_rng(seed)
    centres = rng.standard_normal((n_clusters, dim))
    return (np.repeat(centres, per, axis=0) + noise * rng.standard_normal((n_clusters * per, dim))).astype(np.float32)


def test_activation_is_the_projection_of_the_centred_normalised_concatenated_odour():
    codes = clustered_codes()
    h = K.KenyonHash(codes, k=3, units=32, seed=5)
    tokens = np.random.default_rng(0).integers(0, len(codes), (50, 3))
    centred = codes - codes.mean(0)
    odour = centred[tokens].reshape(50, -1)
    odour = odour / np.linalg.norm(odour, axis=1, keepdims=True)
    w = np.random.default_rng((5, 3, 0xFEED)).standard_normal((64, 3 * codes.shape[1])).astype(np.float32)
    assert np.allclose(h.activation(tokens), odour @ w.T, atol=1e-4)


def test_similar_histories_share_a_bucket_far_more_often_than_unrelated_ones():
    codes = clustered_codes(per=4)
    h = K.KenyonHash(codes, k=2, units=64, seed=1)
    rng = np.random.default_rng(1)
    base = rng.integers(0, 40, (3000, 2)) * 4                    # first token of each cluster
    twin = base + rng.integers(0, 4, (3000, 2))                  # a token of the same cluster, per position
    other = rng.integers(0, 160, (3000, 2))
    same = (h.bucket_keys(base) == h.bucket_keys(twin)).mean()
    diff = (h.bucket_keys(base) == h.bucket_keys(other)).mean()
    assert same > 0.2 and same > 8 * max(diff, 1e-3)


def test_calibration_balances_how_often_each_cell_wins():
    codes = clustered_codes()
    h = K.KenyonHash(codes, k=2, units=64, seed=2)
    freq = np.exp(-np.arange(len(codes)) / 40.0)                 # skewed, like word frequencies
    rng = np.random.default_rng(0)
    sample = rng.choice(len(codes), (4000, 2), p=freq / freq.sum())
    before = h.unit_usage(sample)
    h.calibrate(sample)
    after = h.unit_usage(sample)
    for b, a in zip(before, after):
        assert a["entropy_fraction"] > b["entropy_fraction"] + 0.03
        assert a["max_share"] < b["max_share"] and a["max_share"] < 4 / 64
    off = K.KenyonHash(codes, k=2, units=64, thresholds=False, seed=2)
    off.calibrate(sample)
    assert not off.theta.any()


def test_winner_counts_and_sparse_inputs_change_the_buckets_and_are_validated():
    codes = clustered_codes()
    tokens = np.random.default_rng(0).integers(0, len(codes), (200, 2))
    one = K.KenyonHash(codes, 2, units=32, seed=0).bucket_keys(tokens)
    two = K.KenyonHash(codes, 2, units=32, winners=2, seed=0).bucket_keys(tokens)
    sparse = K.KenyonHash(codes, 2, units=32, in_degree=3, seed=0).bucket_keys(tokens)
    assert len(set(one.tolist())) < len(set(two.tolist()))       # more winners, finer buckets
    assert not np.array_equal(one, sparse)
    assert np.array_equal(one, K.KenyonHash(codes, 2, units=32, seed=0).bucket_keys(tokens))
    with pytest.raises(ValueError, match="in_degree"):
        K.KenyonHash(codes, 2, units=32, in_degree=10_000)


def test_hash_bucket_is_deterministic_in_range_and_spread():
    packed = np.arange(50_000, dtype=np.uint64) * np.uint64(7919)
    b = K.hash_bucket(packed, 1000)
    assert np.array_equal(b, K.hash_bucket(packed, 1000)) and b.max() < 1000
    assert np.bincount(b.astype(np.int64), minlength=1000).min() > 20


def test_unpack_inverts_pack_history():
    toks = [(5, 9, 100, 8191), (0, 1, 2, 3)]
    packed = np.array([C.pack_history(t) for t in toks], dtype=np.uint64)
    assert K.unpack(packed, 4).tolist() == [list(t) for t in toks]


def pooled(stream, k, bucket_of, c_min):
    rows = defaultdict(Counter)
    for h, cont in brute(stream, k).items():
        for w, c in cont.items():
            rows[bucket_of(h)][w] += c
    out = {}
    for b, cont in rows.items():
        total = sum(cont.values())
        if total >= c_min:
            out[b] = (total, sorted(cont.items(), key=lambda kv: (-kv[1], kv[0]))[:C.TOP])
    return out


@pytest.mark.parametrize("k", [1, 2, 3])
def test_bucket_table_pools_continuation_counts_across_the_histories_in_a_bucket(k):
    t = corpus(seed=4, n_docs=200, vocab=14)
    keyfn = lambda packed, toks: (toks.sum(1) % 5).astype(np.uint64)
    table = K.bucket_table(t, k, EOT, keyfn, c_min=1, parts=3, chunk=89, shards=4)
    assert as_dict(table) == pooled(t, k, lambda h: sum(h) % 5, 1)
    assert np.all(table.keys[1:] > table.keys[:-1])


def test_bucket_table_with_the_identity_key_is_the_exact_table():
    t = corpus(seed=5, n_docs=100, vocab=12)
    keyfn = lambda packed, toks: packed
    assert as_dict(K.bucket_table(t, 3, EOT, keyfn, c_min=2, parts=4, chunk=64)) == as_dict(C.build(t, 3, EOT, 2, parts=2))


def test_a_history_holding_end_of_text_gets_no_bucket():
    toks = np.array([[1, 2], [EOT, 2], [1, EOT]])
    keyfn = K.exclude_eot(lambda packed, t: np.zeros(len(t), dtype=np.uint64), EOT)
    got = keyfn(np.zeros(3, dtype=np.uint64), toks)
    assert got.tolist() == [0, int(K.NO_BUCKET), int(K.NO_BUCKET)]


def test_a_saved_hash_reloads_to_the_same_buckets(tmp_path):
    codes = clustered_codes()
    freq = np.exp(-np.arange(len(codes)) / 50.0)
    h = K.KenyonHash(codes, k=2, units=32, winners=2, in_degree=5, token_freq=freq, seed=3)
    sample = np.random.default_rng(0).integers(0, len(codes), (1500, 2))
    h.calibrate(sample)
    h.save(tmp_path / "h.npz")
    back = K.KenyonHash.load(tmp_path / "h.npz", codes, freq)
    tokens = np.random.default_rng(1).integers(0, len(codes), (300, 2))
    assert h.theta.any() and np.array_equal(h.bucket_keys(tokens), back.bucket_keys(tokens))


def test_distinct_winner_pairs_get_distinct_buckets():
    codes = clustered_codes()
    h = K.KenyonHash(codes, k=2, units=32, seed=4)
    tokens = np.random.default_rng(2).integers(0, len(codes), (4000, 2))
    win = h._winners(h._drive(tokens))[:, :, 0]
    pairs = {tuple(p) for p in win.tolist()}
    assert len(pairs) > 100 and len(set(h.bucket_keys(tokens).tolist())) == len(pairs)


@pytest.mark.parametrize("kind", ["kenyon", "random"])
def test_saved_buckets_reload_to_the_same_tables_and_keys(tmp_path, kind):
    stream = corpus(seed=6, n_docs=150, vocab=12)
    codes = clustered_codes(n_clusters=6, per=2)
    freq = np.ones(len(codes))
    hashes, tables, fns = {}, {}, {}
    for k in (2, 3):
        if kind == "kenyon":
            hashes[k] = K.KenyonHash(codes, k, units=16, token_freq=freq, seed=1)
            fns[k] = K.exclude_eot(lambda packed, toks, h=hashes[k]: h.bucket_keys(toks), EOT)
        else:
            fns[k] = K.exclude_eot(lambda packed, toks: K.hash_bucket(packed, 64), EOT)
        tables[k] = K.bucket_table(stream, k, EOT, fns[k], c_min=1, parts=2)
    K.save_buckets(tmp_path, tables, kind, EOT, hashes=hashes, codes=codes, freq=freq, n_buckets=64)
    loaded = K.load_buckets(tmp_path)
    assert sorted(loaded) == [2, 3]
    probe = np.random.default_rng(0).integers(0, 12, (40, 3)).astype(np.int64)
    for k in (2, 3):
        assert as_dict(loaded[k][0]) == as_dict(tables[k])
        packed = np.array([C.pack_history(r[:k]) for r in probe], dtype=np.uint64)
        assert np.array_equal(loaded[k][1](packed, probe[:, :k]), fns[k](packed, probe[:, :k]))


def test_unknown_bucket_kind_is_refused(tmp_path):
    with pytest.raises(ValueError, match="kind"):
        K.save_buckets(tmp_path, {}, "mystery", EOT)


def test_the_fused_gather_gives_the_drive_and_buckets_of_the_plain_sum():
    """The fused embedding_bag path was a speed-up only: same drive to float rounding, same buckets."""
    codes = clustered_codes()
    h = K.KenyonHash(codes, k=3, units=32, seed=7)
    h.calibrate(np.random.default_rng(1).integers(0, len(codes), (2000, 3)))
    tokens = np.random.default_rng(3).integers(0, len(codes), (500, 3))
    plain = sum(h.table[j][tokens[:, j]] for j in range(3)) / np.sqrt(h.norm2[tokens].sum(1)).clip(min=1e-6)[:, None]
    assert np.allclose(h.activation(tokens), plain, rtol=1e-5, atol=1e-6)
    win = (plain - h.theta).reshape(500, 2, 32).argmax(2).astype(np.uint64)
    assert np.array_equal(h.bucket_keys(tokens), win[:, 0] * np.uint64(32) + win[:, 1])


def test_the_number_of_shards_changes_nothing_even_for_skewed_or_hashed_keys():
    t = corpus(seed=7, n_docs=250, vocab=14)
    skewed = lambda packed, toks: np.where(toks[:, 0] < 3, 0, toks.sum(1)).astype(np.uint64)         # one huge bucket
    hashed = lambda packed, toks: K.hash_bucket(packed, 2**40) * np.uint64(1 << 20)                  # spread over 64 bits
    for fn in (skewed, hashed):
        ref = K.bucket_table(t, 3, EOT, fn, c_min=1, parts=2, shards=1)
        for shards in (2, 7, 40):
            got = K.bucket_table(t, 3, EOT, fn, c_min=1, parts=3, chunk=71, shards=shards)
            assert as_dict(got) == as_dict(ref) and np.all(got.keys[1:] > got.keys[:-1])

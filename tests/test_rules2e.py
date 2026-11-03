import pytest

from clm import rules2e as R
from clm.countread import BUCKET_LABELS


def classes(**shift):
    """Per-class losses for every arm: COUNT at 3.0 in every class, each other arm shifted by shift[arm][label]."""
    out = {"count": {lab: [3.0, 3.0, 3.0] for lab in BUCKET_LABELS}}
    for arm in ("gated", "kenyon", "random"):
        d = shift.get(arm, {})
        base = out["gated"] if arm != "gated" else out["count"]
        out[arm] = {lab: [v + d.get(lab, 0.0) for v in base[lab]] for lab in BUCKET_LABELS}
    return out


def losses(**d):
    out = {"count": {0: 3.32, 1: 3.32, 2: 3.33}}
    for arm, ds in d.items():
        ref = out["gated"] if arm in ("kenyon", "random") else out["count"]
        out[arm] = {s: ref[s] + v for s, v in zip((0, 1, 2), ds)}
    return out


def test_p10a_needs_a_not_stored_gain_and_no_cost_in_any_stored_class():
    assert R.p10a(classes(gated={"not stored": -0.012}))
    assert not R.p10a(classes(gated={"not stored": -0.008}))
    assert not R.p10a(classes(gated={"not stored": +0.012}))
    assert not R.p10a(classes(gated={"not stored": -0.02, "10-99": +0.011}))
    assert R.p10a(classes(gated={"not stored": -0.02, "10-99": +0.009, ">=1000": +0.05}))   # >=1000 is P10b's


def test_p10b_reads_only_the_largest_count_class():
    assert R.p10b(classes(gated={">=1000": -0.011}))
    assert not R.p10b(classes(gated={">=1000": -0.009, "not stored": -0.05}))


def test_p11_compares_kenyon_with_the_gated_read_not_with_count():
    assert R.p11(classes(gated={"not stored": -0.05}, kenyon={"not stored": -0.011}))
    assert not R.p11(classes(gated={"not stored": -0.05}, kenyon={"not stored": +0.03}))     # better than COUNT only
    assert not R.p11(classes(kenyon={"10-99": -0.05}))                                     # gain outside 'not stored'


def test_the_random_control_runs_only_when_there_is_a_kenyon_gain_to_attribute():
    flat = classes()
    assert not R.run_random(losses(gated=(-0.01, -0.01, -0.01), kenyon=(0.001, -0.002, 0.0)), flat)
    assert R.run_random(losses(gated=(0, 0, 0), kenyon=(-0.012, -0.011, -0.015)), flat)
    assert R.run_random(losses(gated=(0, 0, 0), kenyon=(0, 0, 0)), classes(kenyon={"not stored": -0.02}))


def test_similarity_needs_kenyon_to_beat_random_buckets_in_not_stored():
    c = classes(kenyon={"not stored": -0.03}, random={"not stored": -0.03})
    assert not R.p11_similarity(c)                    # random equal to Kenyon: pooling, not similarity
    c["random"]["not stored"] = [v + 0.02 for v in c["random"]["not stored"]]
    assert R.p11_similarity(c)


@pytest.mark.parametrize("d, want", [((-0.025, 0.002, -0.010), True),    # INCONCLUSIVE, |mean| < 0.02
                                     ((-0.03, 0.005, -0.04), False),     # INCONCLUSIVE, |mean| >= 0.02
                                     ((-0.012, -0.011, -0.013), False),  # X_BETTER
                                     ((0.001, -0.002, 0.003), False)])   # NO_MEANINGFUL_DIFFERENCE
def test_five_seed_rule(d, want):
    assert R.needs_seeds(losses(gated=d), "gated", "count") is want

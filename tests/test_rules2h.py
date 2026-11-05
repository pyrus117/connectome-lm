from clm import rules2h as R


def summ(*by_depth):
    return R.summary(list(by_depth))


def test_summary_finds_the_best_depth_with_ties_to_the_shallower_and_the_marginal_gains():
    s = summ(3.42, 3.38, 3.37, 3.37, 3.375, 3.38)
    assert (s["best_depth"], s["best_loss"], s["depth1"], s["deepest"]) == (3, 3.37, 3.42, 3.38)
    assert s["gains"] == [3.42 - 3.38, 3.38 - 3.37, 0.0, 3.37 - 3.375, 3.375 - 3.38]
    assert summ(3.4, 3.3, 3.5)["best_depth"] == 2


NEW = {0: summ(3.40, 3.36, 3.35), 1: summ(3.41, 3.37, 3.36), 2: summ(3.40, 3.36, 3.35)}
REF = {0: summ(3.42, 3.38, 3.37), 1: summ(3.43, 3.39, 3.38), 2: summ(3.42, 3.38, 3.37)}


def test_p17_needs_an_x_better_best_depth_loss_over_the_reference_on_every_seed():
    assert R.contrast(NEW, REF, "best_loss") == "X_BETTER" and R.p17(NEW, REF)       # 0.02 lower on all three seeds
    worse = {**NEW, 1: summ(3.45, 3.41, 3.40)}                                        # one seed 0.02 higher: mean -0.0067
    assert not R.p17(worse, REF)
    mixed = {0: summ(3.34), 1: summ(3.38), 2: summ(3.35)}                              # mean -0.0167, but seed 1 is level
    assert R.contrast(mixed, REF, "best_loss") == "INCONCLUSIVE" and not R.p17(mixed, REF)
    small = {0: summ(3.365), 1: summ(3.375), 2: summ(3.365)}                           # 0.005 lower on every seed
    assert R.contrast(small, REF, "best_loss") == "NO_MEANINGFUL_DIFFERENCE" and not R.p17(small, REF)
    assert R.contrast(REF, NEW, "best_loss") == "X_WORSE" and not R.p17(REF, NEW)
    late = {s: summ(3.40, 3.30, 3.45) for s in NEW}                                    # deepest readout worse, best depth better
    assert R.p17(late, REF) and R.contrast(late, REF, "deepest") == "X_WORSE"


def test_the_contrast_reads_the_entry_it_is_asked_for():
    assert R.contrast(NEW, REF, "depth1") == "X_BETTER" and R.contrast(REF, NEW, "deepest") == "X_WORSE"
    deep = {s: summ(3.40, 3.36, 3.351) for s in NEW}
    assert R.contrast(deep, NEW, "best_loss") == "NO_MEANINGFUL_DIFFERENCE"          # best depth equal at 3.35, deepest differs


def test_a_seed_zero_difference_of_zero_or_more_stops_the_remaining_seeds():
    assert R.futile(summ(3.371, 3.372), summ(3.37, 3.38))                              # 3.371 - 3.37 > 0
    assert R.futile(summ(3.37, 3.5), summ(3.37, 3.4))                                  # equal best loss: X_BETTER impossible
    assert not R.futile(summ(3.369, 3.5), summ(3.37, 3.4))
    assert not R.futile(summ(3.30), summ(3.37))

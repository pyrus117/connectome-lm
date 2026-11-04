from clm import rules2g as R

BASE = {8e-3: 3.6649, 32e-3: 3.66, 64e-3: 3.6409, 128e-3: 3.70}


def test_the_best_rate_is_the_lowest_loss_and_a_diverged_rate_is_skipped():
    assert R.best(BASE) == 64e-3
    assert R.best({8e-3: 3.66, 32e-3: None}) == 8e-3
    assert R.best({8e-3: None}) is None


def test_a_winner_on_the_top_edge_gets_one_higher_rate_once():
    assert R.extension(128e-3, set()) == [256e-3]
    assert R.extension(128e-3, {256e-3}) == []
    assert R.extension(64e-3, set()) == []


def test_b0_is_rerun_only_when_a_higher_adam_rate_beats_the_shared_rate_by_the_sesoi():
    assert R.separate(BASE)                                         # 3.6649 - 3.6409 = 0.024
    assert not R.separate(BASE | {64e-3: 3.656})                    # 0.0089, under the SESOI
    assert not R.separate(BASE | {32e-3: 3.70, 64e-3: 3.70, 128e-3: 3.70})   # the shared rate wins
    assert not R.separate({32e-3: 3.60})                            # no shared-rate pilot to compare with

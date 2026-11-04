from clm import rules2f as R


def scores(over=None):
    """Pilot losses: every grid cell at 3.50, the shared-rate cells as Stage 2A measured them, then over."""
    s = {cell: 3.50 for cell in R.grid()}
    s |= {(4e-3, 4e-3): 3.565, (8e-3, 8e-3): 3.525, (16e-3, 16e-3): 3.476, (32e-3, 32e-3): 3.464}
    return s | (over or {})


def test_the_grid_is_three_muon_rates_by_four_adam_rates():
    assert len(R.grid()) == 12 and (8e-3, 64e-3) in R.grid() and (32e-3, 8e-3) in R.grid()


def test_the_best_cell_is_the_lowest_loss_and_a_diverged_cell_is_skipped():
    assert R.best({(8e-3, 32e-3): 3.45, (32e-3, 32e-3): 3.46, (16e-3, 8e-3): None}) == (8e-3, 32e-3)
    assert R.best({(16e-3, 8e-3): None}) is None


def test_a_winner_on_an_edge_gets_one_step_past_that_edge_unless_already_tried():
    assert R.extensions((32e-3, 16e-3), set()) == [(64e-3, 16e-3)]
    assert R.extensions((8e-3, 16e-3), set()) == [(4e-3, 16e-3)]
    assert R.extensions((16e-3, 64e-3), set()) == [(16e-3, 128e-3)]
    assert R.extensions((16e-3, 8e-3), set()) == [(16e-3, 4e-3)]
    assert sorted(R.extensions((32e-3, 64e-3), set())) == [(32e-3, 128e-3), (64e-3, 64e-3)]    # a corner gets both
    assert R.extensions((16e-3, 32e-3), set()) == []
    assert R.extensions((32e-3, 16e-3), {(64e-3, 16e-3)}) == []


def test_confirmation_needs_a_separate_rate_cell_to_beat_every_shared_rate_cell_by_the_sesoi():
    assert R.confirm(scores({(8e-3, 32e-3): 3.452}))                    # 3.464 - 3.452 = 0.012
    assert not R.confirm(scores({(8e-3, 32e-3): 3.456}))                # 0.008, under the SESOI
    assert not R.confirm(scores())                                      # nothing separate beats the shared rate
    assert not R.confirm(scores({(32e-3, 32e-3): 3.44}))                # the shared rate wins
    assert not R.confirm(scores({(8e-3, 32e-3): 3.40, (16e-3, 16e-3): 3.39}))   # the best shared cell is the bar
    assert not R.confirm({(8e-3, 32e-3): 3.40})                         # no shared-rate cell to compare with

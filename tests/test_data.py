import numpy as np
import pytest
import torch

from clm.data import batch_starts, eval_windows, get_batch


def test_one_epoch_sampler_never_repeats_a_window():
    n, ctx, batch, steps = 10_000, 16, 8, 70          # 624 windows, 560 used
    starts = np.concatenate([batch_starts(n, ctx, batch, 0, s) for s in range(steps)])
    assert starts.size == np.unique(starts).size == steps * batch
    assert np.all(starts % ctx == 0) and starts.max() + ctx + 1 <= n


def test_sampler_depends_only_on_seed_and_step():
    a = batch_starts(10_000, 16, 8, 3, 5)
    b = batch_starts(10_000, 16, 8, 3, 5)
    c = batch_starts(10_000, 16, 8, 4, 5)
    np.testing.assert_array_equal(a, b)
    assert not np.array_equal(a, c)


def test_sampler_refuses_to_run_past_one_epoch():
    with pytest.raises(ValueError, match="epoch"):
        batch_starts(1000, 16, 8, 0, 100)


def test_targets_are_the_inputs_shifted_by_one():
    tokens = np.arange(1000, dtype=np.uint16)
    x, y = get_batch(tokens, np.array([0, 32]), 16, torch.device("cpu"))
    assert x.shape == (2, 16) and x.dtype == torch.int64
    assert x[1, 0].item() == 32 and torch.equal(y, x + 1)


def test_eval_windows_fit_inside_the_file_and_do_not_overlap():
    starts = eval_windows(1000, 16, 10_000)
    assert starts.size == 62                          # (1000 - 1) // 16
    assert starts[-1] + 16 + 1 <= 1000 and np.all(np.diff(starts) == 16)


def test_eval_windows_stop_at_the_requested_token_count():
    assert eval_windows(100_000, 16, 1600).size == 100

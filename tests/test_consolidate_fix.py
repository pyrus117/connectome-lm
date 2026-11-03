import copy

import numpy as np
import pytest
import torch

from clm.consolidate_fix import run_consolidate_variant
from clm.fastwrite_arms import ArmConfig, run_consolidate, run_write
from test_fastwrite import CPU, EOT, VOCAB, model, setup  # noqa: F401  (setup is a fixture)

CFG = ArmConfig(per_seq=8, batch=4, lr=5e-3, warmup=2, steps=40, anchor_batch=2)
LM, ANCHOR = np.arange(0, 3000, 64)[:6], np.arange(3000, 5900, 64)[:6]


def variant(m, setup, lists=2, weight=1.0, **kw):
    reader, val, ls = setup
    return run_consolidate_variant(m, reader, ls[:lists], weight, VOCAB, EOT, CFG, CPU, val, LM, ANCHOR, ctx=64, **kw)


def test_the_default_variant_reproduces_the_registered_consolidation_exactly(setup):
    reader, val, lists = setup
    a, b = model(), model()
    old = run_consolidate(a, reader, lists[:2], 1.0, VOCAB, EOT, CFG, CPU, val, LM, ANCHOR, ctx=64)
    new = variant(b, setup)
    for key in ("acc_matrix", "acc_matrix_with_memory", "lm_loss", "final_mean", "backward_transfer"):
        assert new[key] == old[key], key
    assert all(torch.equal(x, y) for x, y in zip(a.state_dict().values(), b.state_dict().values()))


def test_the_original_teacher_stays_clean_while_the_rolling_teacher_follows_the_student(setup):
    rolling = variant(model(), setup, teacher_mode="rolling")
    fixed = variant(model(), setup, teacher_mode="original")
    assert fixed["teacher_matrix"][1][0] == fixed["teacher_matrix"][0][0]       # list 1 reads the same in every phase
    assert rolling["teacher_matrix"][1][0] != rolling["teacher_matrix"][0][0]   # the moved backbone reads it differently
    assert rolling["teacher_matrix"][0] == fixed["teacher_matrix"][0]           # phase 1 starts from the same model


def test_the_teacher_matrix_is_the_original_model_reading_the_memory(setup):
    reader, val, lists = setup
    write = run_write(model(), reader, lists[:2], 1000.0, VOCAB, EOT, CFG, CPU, val, LM, ctx=64)
    fixed = variant(model(), setup, weight=1000.0, teacher_mode="original")
    assert fixed["teacher_matrix"][1][:2] == write["acc_matrix"][1][:2]
    assert write["acc_matrix"][1][0] > 0.1                                      # chance is 1/250, so the memory read is not a no-op


def test_the_student_is_trained_toward_the_clean_teacher_not_the_moved_one(setup):
    reader, val, lists = setup
    fixed_model, rolling_model = model(), model()
    variant(fixed_model, setup, teacher_mode="original")
    variant(rolling_model, setup, teacher_mode="rolling")
    assert not all(torch.equal(x, y) for x, y in zip(fixed_model.state_dict().values(), rolling_model.state_dict().values()))


def test_freezing_the_head_leaves_it_bit_identical_and_trains_everything_else(setup):
    m = model()
    before = copy.deepcopy(m.state_dict())
    variant(m, setup, lists=1, freeze_head=True)
    after = m.state_dict()
    assert torch.equal(before["head.weight"], after["head.weight"])
    assert any(not torch.equal(before[k], after[k]) for k in before if k != "head.weight")
    assert m.head.weight.requires_grad


def test_an_unfrozen_head_does_train(setup):
    m = model()
    before = m.head.weight.detach().clone()
    variant(m, setup, lists=1, freeze_head=False)
    assert not torch.equal(before, m.head.weight)


def test_a_bad_teacher_mode_is_refused_and_a_crash_leaves_the_head_trainable(setup):
    reader, val, lists = setup
    with pytest.raises(ValueError, match="teacher_mode"):
        variant(model(), setup, teacher_mode="student")
    m = model()
    with pytest.raises(ValueError):                                              # no anchor windows to draw from
        run_consolidate_variant(m, reader, lists[:1], 1.0, VOCAB, EOT, CFG, CPU, val, LM, ANCHOR[:1], ctx=64,
                                freeze_head=True)
    assert m.head.weight.requires_grad

import copy

import numpy as np
import pytest
import torch

from clm.consolidate_fix import GatedOverlay, run_consolidate_variant, run_write_gated
from clm.fastwrite import Overlay, count_input, cue_kl, quiz
from clm.fastwrite_arms import ArmConfig, run_consolidate, run_write
from clm.vocab_task import quiz_batches
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
    with pytest.raises(ValueError, match="memory_kl_weight"):
        variant(model(), setup, memory_kl_weight=-0.1)
    m = model()
    with pytest.raises(ValueError):                                              # no anchor windows to draw from
        run_consolidate_variant(m, reader, lists[:1], 1.0, VOCAB, EOT, CFG, CPU, val, LM, ANCHOR[:1], ctx=64,
                                freeze_head=True)
    assert m.head.weight.requires_grad


def test_gated_overlay_suppresses_cues_in_stored_phrases_and_keeps_unseen_or_short_histories(setup):
    reader, _, _ = setup
    table2 = next(t for t in reader.tables if t.k == 2)
    stored_key = int(table2.keys[int(np.argmax(table2.hcount))])
    w0, w1 = stored_key >> 13, stored_key & ((1 << 13) - 1)
    max_hc = int(table2.hcount.max())
    assert reader.history_counts(np.array([[w0, w1]]), 2)[0, 1] == max_hc and max_hc >= 1
    g = GatedOverlay(VOCAB, 2.5, reader, gate_k=2, max_hcount=0)
    g.write([w1], [99])
    # Position 0 (t < k-1, count -1), position 2 (after EOT, count 0), position 4 (stored bigram w0, w1, count >= 1)
    x = np.array([[w1, EOT, w1, w0, w1]], dtype=np.int64)
    tok, score = g.slot(x)
    assert tok[0, :, 0].tolist() == [99, 0, 99, 0, 0]
    assert score[0, :, 0].tolist() == pytest.approx([2.5, 0.0, 2.5, 0.0, 0.0])
    g_loose = GatedOverlay(VOCAB, 2.5, reader, gate_k=2, max_hcount=max_hc)
    g_loose.write([w1], [99])
    assert g_loose.slot(x)[0][0, 4, 0] == 99 and g_loose.slot(x)[1][0, 4, 0] == pytest.approx(2.5)
    with pytest.raises(ValueError, match="gate_k"):
        GatedOverlay(VOCAB, 1.0, reader, gate_k=1)
    with pytest.raises(ValueError, match="max_hcount"):
        GatedOverlay(VOCAB, 1.0, reader, gate_k=2, max_hcount=-1)


def test_run_write_gated_matches_run_write_when_ungated_and_cuts_phrase_injections_when_gated(setup):
    reader, val, lists = setup
    m = model()
    plain = run_write(m, reader, lists[:2], 50.0, VOCAB, EOT, CFG, CPU, val, LM, ctx=64)
    ungated = run_write_gated(m, reader, lists[:2], 50.0, VOCAB, EOT, CFG, CPU, val, LM, gate_k=None, ctx=64)
    gated = run_write_gated(m, reader, lists[:2], 50.0, VOCAB, EOT, CFG, CPU, val, LM, gate_k=2, max_hcount=0, ctx=64)
    for k in ("acc_matrix", "lm_loss_memory_on", "lm_loss_memory_off", "final_mean", "backward_transfer"):
        assert ungated[k] == plain[k], k
    assert gated["lm_loss_memory_on"] != plain["lm_loss_memory_on"]


def test_memory_kl_weight_trains_the_student_with_the_overlay_on_and_records_unseen_list_recall(setup):
    reader, val, lists = setup
    cues, answers = lists[0]
    unseen = lists[2]
    qx, qy = quiz_batches(cues, answers, 8, 0, EOT)
    o = Overlay(VOCAB, 100.0)
    o.write(cues, answers)
    m_off, m_on = model(), model()
    original = copy.deepcopy(m_off).eval()
    r_off = run_consolidate_variant(m_off, reader, lists[:1], 100.0, VOCAB, EOT, CFG, CPU, val, LM, ANCHOR, ctx=64,
                                    teacher_mode="original", memory_kl_weight=0.0, unseen_list=unseen)
    r_on = run_consolidate_variant(m_on, reader, lists[:1], 100.0, VOCAB, EOT, CFG, CPU, val, LM, ANCHOR, ctx=64,
                                   teacher_mode="original", memory_kl_weight=2.0, unseen_list=unseen)

    def mem_kl(student):
        student.eval()
        with torch.no_grad():
            s, _ = student(qx[0:1], count=count_input(reader, o, qx[0:1].numpy(), CPU))
            t, _ = original(qx[0:1], count=count_input(reader, o, qx[0:1].numpy(), CPU))
        return cue_kl(s, t, qy[0:1]).item()
    assert mem_kl(m_on) < mem_kl(m_off)
    assert r_on["with_memory"]["final_mean"] == pytest.approx(r_on["acc_matrix_with_memory"][-1][0])
    u_ov = Overlay(VOCAB, 100.0)
    u_ov.write(*unseen)
    u_qx, u_qy = quiz_batches(unseen[0], unseen[1], 8, 0, EOT)
    assert r_off["unseen_start"] == pytest.approx(quiz(original, reader, u_ov, u_qx, u_qy, 4, CPU))
    assert r_off["unseen_start"] > 0.1
    assert r_on["unseen_with_memory"][0] == pytest.approx(quiz(m_on, reader, u_ov, u_qx, u_qy, 4, CPU))
    assert r_on["unseen_with_memory"][0] != pytest.approx(quiz(m_on, reader, None, u_qx, u_qy, 4, CPU))

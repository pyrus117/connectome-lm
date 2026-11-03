import copy

import numpy as np
import pytest
import torch

from clm import counts as C
from clm import fastwrite as W
from clm.countread import CountReader
from clm.fastwrite_arms import ArmConfig, run_consolidate, run_gradient, run_write
from clm.s2model import B0Config, build_b0
from clm.vocab_task import IGNORE, make_lists, quiz_batches

CPU = torch.device("cpu")
VOCAB, EOT = 256, 255
CFG = B0Config(vocab=VOCAB, ctx=64, d_model=24, n_layer=3, n_head=2)
ARM = ArmConfig(per_seq=8, batch=4, lr=5e-3, warmup=2, steps=40, anchor_batch=2)


@pytest.fixture(scope="module")
def setup(tmp_path_factory):
    d = tmp_path_factory.mktemp("fw")
    rng = np.random.default_rng(0)
    stream = rng.integers(0, 250, 30_000).astype(np.uint16)
    for k, c in zip(range(1, 5), (1, 1, 1, 1)):
        C.save(C.build(stream, k, EOT, c, parts=2), d / f"order{k + 1}", {"c_min": c, "eot_id": EOT})
    reader = CountReader(d, k_top=4)
    val = rng.integers(0, 250, 6_000).astype(np.uint16)
    lists = make_lists(np.arange(1, 250), 3, 24, seed=0)
    return reader, val, lists


def model(seed=0):
    m = build_b0(CFG, seed)
    torch.manual_seed(seed)
    for blk in m.blocks:
        torch.nn.init.normal_(blk.proj.weight, std=0.2)
        torch.nn.init.normal_(blk.down.weight, std=0.2)
    torch.nn.init.normal_(m.head.weight, std=0.3)
    return m


def test_overlay_reads_a_written_cue_exactly_and_forgets_it_when_removed():
    o = W.Overlay(VOCAB, weight=0.7)
    o.write([5, 9], [20, 30])
    x = np.array([[5, 6, 9, 5]])
    tok, score = o.slot(x)
    assert tok.shape == score.shape == (1, 4, 1)
    assert tok[0, :, 0].tolist() == [20, 0, 30, 20] and score[0, :, 0].tolist() == pytest.approx([0.7, 0, 0.7, 0.7])
    o.write([5], [44])                                           # a new write for the same cue replaces its entry
    assert o.slot(x)[0][0, 0, 0] == 44
    o.remove([5])
    assert o.slot(x)[1][0, 0, 0] == 0 and o.slot(x)[1][0, 2, 0] == pytest.approx(0.7)
    o.clear()
    assert not o.slot(x)[1].any()


def test_a_bucket_overlay_lets_a_look_alike_cue_reach_the_stored_answer_but_exact_entries_win():
    keyfn = lambda t: np.asarray(t) // 2                          # cues 10 and 11 share a bucket
    o = W.Overlay(VOCAB, 1.0, keyfn)
    o.write([10], [77])
    tok, score = o.slot(np.array([[10, 11, 12]]))
    assert tok[0, :, 0].tolist() == [77, 77, 0] and score[0, :, 0].tolist() == [1.0, 1.0, 0.0]
    o.write([11], [88])                                           # the twin is written with its own answer
    assert o.slot(np.array([[10, 11]]))[0][0, :, 0].tolist() == [77, 88]
    plain = W.Overlay(VOCAB, 1.0)
    plain.write([10], [77])
    assert plain.slot(np.array([[11]]))[1][0, 0, 0] == 0


def test_count_input_gains_one_slot_that_carries_the_answer_at_cue_positions(setup):
    reader, _, lists = setup
    cues, answers = lists[0]
    o = W.Overlay(VOCAB, 0.5)
    o.write(cues, answers)
    x = np.array([[EOT, cues[0], answers[0], cues[1], answers[1]]])
    base_tok, base_score = reader.lookup(x)
    tok, score = W.count_input(reader, o, x, CPU)
    assert tok.shape[-1] == base_tok.shape[-1] + 1
    assert tok[0, 1, -1].item() == answers[0] and tok[0, 3, -1].item() == answers[1]
    assert score[0, :, -1].tolist() == pytest.approx([0, 0.5, 0, 0.5, 0])
    assert np.array_equal(tok[..., :-1].numpy(), base_tok) and np.array_equal(score[..., :-1].numpy(), base_score)


def test_a_write_changes_nothing_before_the_first_cue_and_something_after(setup):
    reader, _, lists = setup
    cues, answers = lists[0]
    m = model().eval()
    x = torch.from_numpy(np.array([[EOT, 3, 4, 5, 6, int(cues[0]), int(answers[0]), 7]], dtype=np.int64))
    o = W.Overlay(VOCAB, 1.0)
    off, _ = m(x, count=W.count_input(reader, None, x.numpy(), CPU))
    o.write(cues, answers)
    on, _ = m(x, count=W.count_input(reader, o, x.numpy(), CPU))
    assert torch.allclose(off[0, :5], on[0, :5], atol=1e-6)
    assert not torch.allclose(off[0, 5:], on[0, 5:])


def test_quiz_counts_the_cue_positions_whose_top_guess_is_the_answer(setup):
    reader, _, lists = setup
    cues, answers = lists[0]
    qx, qy = quiz_batches(cues, answers, 8, 0, EOT)
    m = model().eval()
    acc = W.quiz(m, reader, None, qx, qy, 4, CPU)
    hits = total = 0
    for i in range(len(qx)):
        logits, _ = m(qx[i:i + 1], count=W.count_input(reader, None, qx[i:i + 1].numpy(), CPU))
        mask = qy[i] != IGNORE
        hits += int((logits[0].argmax(-1)[mask] == qy[i][mask]).sum())
        total += int(mask.sum())
    assert acc == pytest.approx(hits / total) and total == len(cues)


def test_kl_helpers_are_zero_for_equal_logits_positive_otherwise_and_respect_the_cue_mask():
    a = torch.randn(2, 5, 7)
    y = torch.full((2, 5), IGNORE)
    y[0, 1] = 3
    assert W.cue_kl(a, a, y).item() == pytest.approx(0, abs=1e-6)
    b = a.clone()
    b[1] += 5                                                      # differs only where y is ignored
    assert W.cue_kl(a, b, y).item() == pytest.approx(0, abs=1e-6)
    b[0, 1, 0] += 3
    assert W.cue_kl(a, b, y).item() > 0.01
    assert W.lm_kl(a, a).item() == pytest.approx(0, abs=1e-6) and W.lm_kl(b, a).item() > 0
    p, q = torch.log(torch.tensor([[0.5, 0.5]])), torch.log(torch.tensor([[0.9, 0.1]]))
    expected = 0.5 * np.log(0.5 / 0.9) + 0.5 * np.log(0.5 / 0.1)                  # KL(p || q), which differs from KL(q || p)
    assert W.lm_kl(q, p).item() == pytest.approx(expected, rel=1e-5)             # lm_kl(student, anchor) = KL(anchor || student)
    assert W.cue_kl(q, p, torch.tensor([3])).item() == pytest.approx(expected, rel=1e-5)


def test_sequence_summary_is_final_mean_and_backward_transfer():
    s = W.sequence_summary([[1.0, 0, 0], [0.5, 1.0, 0], [0.25, 0.5, 1.0]])
    assert s["final_mean"] == pytest.approx((0.25 + 0.5 + 1.0) / 3)
    assert s["backward_transfer"] == pytest.approx(((0.25 - 1.0) + (0.5 - 1.0)) / 2)


def test_the_write_arm_trains_nothing_and_never_disturbs_an_earlier_list(setup):
    reader, val, lists = setup
    m = model()
    before = copy.deepcopy(m.state_dict())
    out = run_write(m, reader, lists, 1.0, VOCAB, EOT, ARM, CPU, val, np.arange(0, 3000, 64)[:6], ctx=64)
    assert all(torch.equal(before[k], v) for k, v in m.state_dict().items())
    matrix = out["acc_matrix"]
    for j in range(len(lists)):                                    # list j's accuracy is the same from its phase on
        assert matrix[j][j] == matrix[-1][j]
    assert out["backward_transfer"] == pytest.approx(0.0, abs=1e-12)
    assert len(out["lm_loss_memory_on"]) == 3 and out["lm_loss_memory_off"] > 0


def test_the_gradient_arm_learns_a_list_and_loses_it_when_the_next_is_learned(setup):
    reader, val, lists = setup
    cfg = ArmConfig(per_seq=8, batch=8, lr=1e-2, warmup=2, steps=80)
    out = run_gradient(model(), reader, lists[:2], EOT, cfg, CPU, val, np.arange(0, 3000, 64)[:6], ctx=64)
    m = out["acc_matrix"]
    assert m[0][0] > 0.6 and m[1][1] > 0.6
    assert m[1][0] < m[0][0] - 0.2 and out["backward_transfer"] < -0.2


def test_consolidation_moves_the_backbone_toward_backbone_plus_memory_without_touching_the_original(setup):
    reader, val, lists = setup
    m = model()
    original = copy.deepcopy(m).eval()
    cues, answers = lists[0]
    qx, qy = quiz_batches(cues, answers, 8, 0, EOT)
    o = W.Overlay(VOCAB, 1.0)
    o.write(cues, answers)

    def gap(student):
        s = student.eval()
        total = 0.0
        for i in range(len(qx)):
            with torch.no_grad():
                a, _ = s(qx[i:i + 1], count=W.count_input(reader, None, qx[i:i + 1].numpy(), CPU))
                t, _ = original(qx[i:i + 1], count=W.count_input(reader, o, qx[i:i + 1].numpy(), CPU))
            total += W.cue_kl(a, t, qy[i:i + 1]).item()
        return total
    start = gap(m)
    cfg = ArmConfig(per_seq=8, batch=4, lr=5e-3, warmup=2, steps=60, anchor_batch=2)
    out = run_consolidate(m, reader, lists[:1], 1.0, VOCAB, EOT, cfg, CPU, val, np.arange(0, 3000, 64)[:6],
                          np.arange(3000, 5900, 64)[:6], ctx=64)
    assert gap(m) < 0.5 * start
    assert len(out["acc_matrix"]) == 1 and len(out["acc_matrix_with_memory"]) == 1 and np.isfinite(out["lm_loss"]).all()


def test_language_loss_reads_the_memory_when_asked(setup):
    reader, val, lists = setup
    m = model()
    cues, answers = lists[0]
    o = W.Overlay(VOCAB, 50.0)
    o.write(np.arange(1, 250), np.roll(np.arange(1, 250), 1))        # cues that occur all through the validation text
    windows = np.arange(0, 3000, 64)[:6]
    off = W.language_loss(m, reader, None, val, windows, 64, CPU)
    on = W.language_loss(m, reader, o, val, windows, 64, CPU)
    again = W.language_loss(m, reader, None, val, windows, 64, CPU)
    assert off == pytest.approx(again) and abs(on - off) > 1e-3

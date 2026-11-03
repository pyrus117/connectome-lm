"""Stage 2C-2: two repairs to the consolidation recipe of clm.fastwrite_arms.run_consolidate, which that module keeps
unchanged because PREREG_stage2c.md froze it.

teacher_mode "rolling" copies the backbone at the start of every phase, as the registered run did. After phase 1 that
copy has already moved its unembedding, so the overlay (which adds the answer's head row to the embedding) no longer
reads cleanly through it. "original" keeps one untouched copy as the teacher for every phase.

freeze_head leaves head.weight out of training. The overlay and the COUNT read use the head's rows as the channel into
the embedding, so a head that moves breaks what the rest of the network learned to read from that channel.

With teacher_mode="rolling" and freeze_head=False this reproduces run_consolidate exactly."""
import copy
import math

import numpy as np
import torch

from clm.fastwrite import Overlay, cue_kl, language_loss, lm_kl, quiz, sequence_summary
from clm.fastwrite_arms import _forward, _quizzes, _step_lr
from clm.vocab_task import train_batch

TEACHER_MODES = ("rolling", "original")


def _frozen_copy(model):
    m = copy.deepcopy(model).eval()
    for p in m.parameters():
        p.requires_grad_(False)
    return m


def run_consolidate_variant(model, reader, lists, weight, vocab, eot, cfg, device, val, lm_windows, anchor_windows,
                            ctx=512, keyfn=None, teacher_mode="rolling", freeze_head=False):
    """As run_consolidate, plus `teacher_matrix[i][j]`: the teacher's own recall (memory on) of list j at the start of
    phase i, which is the ceiling the student can reach. `model` is trained in place and has its head trainable again
    on return."""
    if teacher_mode not in TEACHER_MODES:
        raise ValueError(f"teacher_mode must be one of {TEACHER_MODES}, got {teacher_mode!r}")
    overlay = Overlay(vocab, weight, keyfn)
    original = _frozen_copy(model)
    model.head.weight.requires_grad_(not freeze_head)
    try:
        params = [p for p in model.parameters() if p.requires_grad]
        opt = torch.optim.AdamW(params, lr=cfg.lr, betas=(0.9, 0.95), weight_decay=0.0)
        quizzes = _quizzes(lists, cfg, eot)
        rng = np.random.default_rng([cfg.task_seed, 7])
        matrix, matrix_with_memory, teacher_matrix, lm, step = [], [], [], [], 0
        seen_c, seen_a = [], []
        base = language_loss(model, reader, None, val, lm_windows, ctx, device)
        for cues, answers in lists:
            overlay.write(cues, answers)
            seen_c.append(cues)
            seen_a.append(answers)
            all_c, all_a = np.concatenate(seen_c), np.concatenate(seen_a)
            teacher = original if teacher_mode == "original" else _frozen_copy(model)
            teacher_matrix.append([quiz(teacher, reader, overlay, qx, qy, cfg.batch, device) for qx, qy in quizzes])
            teacher.eval()                                              # quiz leaves a model in train mode
            for _ in range(cfg.steps):
                _step_lr(opt, cfg, step)
                x, y = train_batch(all_c, all_a, cfg.batch, cfg.per_seq, cfg.task_seed, step, eot)
                with torch.no_grad():
                    t_logits = _forward(teacher, reader, overlay, x, device)
                s_logits = _forward(model, reader, None, x, device)
                loss = cue_kl(s_logits, t_logits, y.to(device))
                if cfg.anchor_weight > 0:
                    starts = rng.choice(anchor_windows, cfg.anchor_batch, replace=False)
                    ax = torch.from_numpy(np.stack([val[s:s + ctx] for s in starts]).astype(np.int64))
                    with torch.no_grad():
                        a_logits = _forward(original, reader, None, ax, device)
                    loss = loss + cfg.anchor_weight * lm_kl(_forward(model, reader, None, ax, device), a_logits)
                loss.backward()
                gnorm = torch.nn.utils.clip_grad_norm_(params, cfg.clip).item()
                if not (math.isfinite(loss.item()) and math.isfinite(gnorm)):
                    raise RuntimeError(f"non-finite loss {loss.item()} or grad norm {gnorm} at step {step}")
                opt.step()
                opt.zero_grad(set_to_none=True)
                step += 1
            matrix.append([quiz(model, reader, None, qx, qy, cfg.batch, device) for qx, qy in quizzes])
            matrix_with_memory.append([quiz(model, reader, overlay, qx, qy, cfg.batch, device) for qx, qy in quizzes])
            lm.append(language_loss(model, reader, None, val, lm_windows, ctx, device))
    finally:
        model.head.weight.requires_grad_(True)
    return {"arm": "consolidate", "teacher_mode": teacher_mode, "freeze_head": freeze_head, "weight": weight,
            "acc_matrix": matrix, "acc_matrix_with_memory": matrix_with_memory, "teacher_matrix": teacher_matrix,
            "lm_loss": lm, "lm_loss_start": base, **sequence_summary(matrix)}

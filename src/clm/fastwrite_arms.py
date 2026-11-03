"""Stage 2C arms on a trained B0 or COUNT model: fine-tune the backbone list by list (gradient), write the lists
into an overlay memory with no training (write), or write and then consolidate the memory into the backbone by
replay (consolidate). Each returns the accuracy matrix of lists after every phase and the language loss after
every phase. See clm.fastwrite for the memory itself."""
import copy
import dataclasses
import math

import numpy as np
import torch
import torch.nn.functional as F

from clm.fastwrite import (Overlay, count_input, cue_kl, language_loss, lm_kl, quiz,
                           sequence_summary)
from clm.train import _autocast
from clm.vocab_task import IGNORE, quiz_batches, train_batch


@dataclasses.dataclass(frozen=True)
class ArmConfig:
    per_seq: int = 64
    batch: int = 16
    lr: float = 3e-4
    warmup: int = 20
    steps: int = 300                 # per list
    clip: float = 1.0
    anchor_batch: int = 4            # ordinary-text windows per consolidation step
    anchor_weight: float = 1.0       # KL anchor to the original model on ordinary text
    task_seed: int = 0


def _quizzes(lists, cfg, eot):
    return [quiz_batches(c, a, cfg.per_seq, cfg.task_seed, eot) for c, a in lists]


def _step_lr(opt, cfg, step):
    for g in opt.param_groups:
        g["lr"] = cfg.lr * min(1.0, (step + 1) / cfg.warmup)


def _forward(model, reader, overlay, x, device):
    kw = {"count": count_input(reader, overlay, x.numpy(), device)} if reader is not None else {}
    with _autocast(device):
        logits, _ = model(x.to(device), **kw)
    return logits


def run_write(model, reader, lists, weight, vocab, eot, cfg, device, val=None, lm_windows=None, ctx=512, keyfn=None):
    """No training: after each list is written, quiz every list with the memory on."""
    overlay = Overlay(vocab, weight, keyfn)
    quizzes = _quizzes(lists, cfg, eot)
    matrix, lm = [], []
    base = language_loss(model, reader, None, val, lm_windows, ctx, device) if val is not None else None
    for cues, answers in lists:
        overlay.write(cues, answers)
        matrix.append([quiz(model, reader, overlay, qx, qy, cfg.batch, device) for qx, qy in quizzes])
        if val is not None:
            lm.append(language_loss(model, reader, overlay, val, lm_windows, ctx, device))
    return {"arm": "write", "weight": weight, "acc_matrix": matrix, "lm_loss_memory_on": lm, "lm_loss_memory_off": base,
            **sequence_summary(matrix)}


def run_gradient(model, reader, lists, eot, cfg, device, val=None, lm_windows=None, ctx=512):
    """Fine-tune every parameter on each list in turn; no memory."""
    params = [p for p in model.parameters()]
    opt = torch.optim.AdamW(params, lr=cfg.lr, betas=(0.9, 0.95), weight_decay=0.0)
    quizzes = _quizzes(lists, cfg, eot)
    matrix, lm, step = [], [], 0
    base = language_loss(model, reader, None, val, lm_windows, ctx, device) if val is not None else None
    for cues, answers in lists:
        for _ in range(cfg.steps):
            _step_lr(opt, cfg, step)
            x, y = train_batch(cues, answers, cfg.batch, cfg.per_seq, cfg.task_seed, step, eot)
            logits = _forward(model, reader, None, x, device)
            loss = F.cross_entropy(logits.float().reshape(-1, logits.shape[-1]), y.to(device).reshape(-1),
                                   ignore_index=IGNORE)
            loss.backward()
            gnorm = torch.nn.utils.clip_grad_norm_(params, cfg.clip).item()
            if not (math.isfinite(loss.item()) and math.isfinite(gnorm)):
                raise RuntimeError(f"non-finite loss {loss.item()} or grad norm {gnorm} at step {step}")
            opt.step()
            opt.zero_grad(set_to_none=True)
            step += 1
        matrix.append([quiz(model, reader, None, qx, qy, cfg.batch, device) for qx, qy in quizzes])
        if val is not None:
            lm.append(language_loss(model, reader, None, val, lm_windows, ctx, device))
    return {"arm": "gradient", "acc_matrix": matrix, "lm_loss": lm, "lm_loss_start": base, **sequence_summary(matrix)}


def run_consolidate(model, reader, lists, weight, vocab, eot, cfg, device, val, lm_windows, anchor_windows,
                    ctx=512, keyfn=None):
    """Write each list, then train the backbone (memory off) to match backbone plus memory on every pair
    written so far (replay), with a KL anchor to the original model on ordinary text. The memory keeps every
    list, so the replay set grows; the quiz after each phase reads the backbone alone."""
    overlay = Overlay(vocab, weight, keyfn)
    original = copy.deepcopy(model).eval()
    for p in original.parameters():
        p.requires_grad_(False)
    params = list(model.parameters())
    opt = torch.optim.AdamW(params, lr=cfg.lr, betas=(0.9, 0.95), weight_decay=0.0)
    quizzes = _quizzes(lists, cfg, eot)
    rng = np.random.default_rng([cfg.task_seed, 7])
    matrix, matrix_with_memory, lm, step = [], [], [], 0
    seen_c, seen_a = [], []
    base = language_loss(model, reader, None, val, lm_windows, ctx, device)
    for cues, answers in lists:
        overlay.write(cues, answers)
        seen_c.append(cues)
        seen_a.append(answers)
        all_c, all_a = np.concatenate(seen_c), np.concatenate(seen_a)
        teacher = copy.deepcopy(model).eval()
        for p in teacher.parameters():
            p.requires_grad_(False)
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
    return {"arm": "consolidate", "weight": weight, "acc_matrix": matrix, "acc_matrix_with_memory": matrix_with_memory,
            "lm_loss": lm, "lm_loss_start": base, **sequence_summary(matrix)}

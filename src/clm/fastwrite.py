"""Stage 2C: fast count writes and consolidation (complementary learning systems, McClelland et al. 1995).

A COUNT model reads, per position, weighted rows of its own unembedding (clm.countread). A write adds one
more entry to that read: when the text token is a written cue, the answer's head row is added with weight w.
Nothing is trained and no gradient flows into the weights, so a new pair cannot overwrite an old one; the
cost is that the cue word now injects its answer wherever it appears (measured by the language-model loss
with the memory on). A bucket overlay keys entries by a similarity bucket of the cue as well, so a look-alike
cue can reach the stored answer. Consolidation trains the backbone, memory off, to reproduce what backbone plus
memory says on every pair written so far, with a KL anchor to the original model on ordinary text.
"""
import numpy as np
import torch
import torch.nn.functional as F

from clm.train import _autocast
from clm.vocab_task import IGNORE, correct


class Overlay:
    """Written (cue token -> answer token) entries read as one extra COUNT slot of weight `weight`.
    With `keyfn` (cue token ids -> bucket ids) an unwritten cue falls back to the entry of its bucket."""

    def __init__(self, vocab, weight, keyfn=None):
        self.vocab, self.weight, self.keyfn = vocab, weight, keyfn
        self.exact = np.full(vocab, -1, dtype=np.int64)
        self.by_bucket = {}

    def write(self, cues, answers):
        cues, answers = np.asarray(cues), np.asarray(answers)
        self.exact[cues] = answers
        if self.keyfn is not None:
            for b, a in zip(self.keyfn(cues).tolist(), answers.tolist()):
                self.by_bucket[b] = a

    def remove(self, cues):
        self.exact[np.asarray(cues)] = -1

    def clear(self):
        self.exact[:] = -1
        self.by_bucket.clear()

    def _answers(self, x):
        ans = self.exact[x]
        if self.keyfn is not None and self.by_bucket:
            missing = ans < 0
            if missing.any():
                buckets = self.keyfn(x[missing])
                ans = ans.copy()
                ans[missing] = [self.by_bucket.get(b, -1) for b in buckets.tolist()]
        return ans

    def slot(self, x):
        """(tokens, scores), each (batch, time, 1), for the text tokens x."""
        ans = self._answers(np.asarray(x))
        hit = ans >= 0
        return np.where(hit, ans, 0)[..., None], np.where(hit, self.weight, 0.0).astype(np.float32)[..., None]


def count_input(reader, overlay, x, device):
    """The model's `count` argument for text x (numpy, batch x time): the COUNT read plus the overlay slot."""
    tok, score = reader.lookup(x)
    if overlay is not None:
        otok, oscore = overlay.slot(x)
        tok, score = np.concatenate([tok, otok], axis=-1), np.concatenate([score, oscore], axis=-1)
    return torch.from_numpy(tok).to(device), torch.from_numpy(score).to(device)


@torch.no_grad()
def quiz(model, reader, overlay, qx, qy, micro, device):
    """Fraction of cue positions whose top guess is the answer."""
    model.eval()
    hits = 0
    for i in range(0, len(qx), micro):
        x, y = qx[i:i + micro], qy[i:i + micro]
        kw = {"count": count_input(reader, overlay, x.numpy(), device)} if reader is not None else {}
        with _autocast(device):
            logits, _ = model(x.to(device), **kw)
        hits += correct(logits.cpu(), y)
    model.train()
    return hits / int((qy != IGNORE).sum())


@torch.no_grad()
def language_loss(model, reader, overlay, val, windows, ctx, device):
    """Mean next-token loss on the given validation windows (start offsets), memory on or off."""
    model.eval()
    total, n = 0.0, 0
    for i in range(0, len(windows), 8):
        starts = windows[i:i + 8]
        x = np.stack([val[s:s + ctx] for s in starts]).astype(np.int64)
        y = np.stack([val[s + 1:s + ctx + 1] for s in starts]).astype(np.int64)
        kw = {"count": count_input(reader, overlay, x, device)} if reader is not None else {}
        with _autocast(device):
            logits, _ = model(torch.from_numpy(x).to(device), **kw)
        total += float(F.cross_entropy(logits.reshape(-1, logits.shape[-1]), torch.from_numpy(y).to(device).reshape(-1),
                                       reduction="sum"))
        n += y.size
    model.train()
    return total / n


def cue_kl(student_logits, teacher_logits, y):
    """Mean KL(teacher || student) over the cue positions (y != IGNORE)."""
    m = y != IGNORE
    s = F.log_softmax(student_logits[m].float(), dim=-1)
    t = F.log_softmax(teacher_logits[m].float(), dim=-1)
    return (t.exp() * (t - s)).sum(-1).mean()


def lm_kl(student_logits, anchor_logits):
    """Mean KL(anchor || student) over every position of ordinary text."""
    s = F.log_softmax(student_logits.float(), dim=-1)
    a = F.log_softmax(anchor_logits.float(), dim=-1)
    return (a.exp() * (a - s)).sum(-1).mean()


def sequence_summary(acc_matrix):
    """final_mean and backward_transfer as in Lopez-Paz & Ranzato (2017); acc_matrix[i][j] is list j's
    accuracy after phase i. Negative backward transfer means forgetting."""
    last = acc_matrix[-1]
    n = len(acc_matrix)
    return {"final_mean": sum(last) / len(last),
            "backward_transfer": sum(last[j] - acc_matrix[j][j] for j in range(n - 1)) / max(1, n - 1)}

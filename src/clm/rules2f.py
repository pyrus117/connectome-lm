"""Stage 2F decision rules (PREREG_stage2f.md section 4), apart from file handling so tests can plant numbers.

A cell is (muon lr, adam lr): the rate of the block matrices and the rate of embeddings, head and gates.
scores: {cell: pilot validation loss, or None if the pilot diverged}. A coupled cell has muon == adam, the shared-rate
recipe of Stages 2A to 2E."""
from clm.analyze import SESOI

MUON = (8e-3, 16e-3, 32e-3)
ADAM = (8e-3, 16e-3, 32e-3, 64e-3)


def grid():
    return [(m, a) for m in MUON for a in ADAM]


def coupled(cell):
    return cell[0] == cell[1]


def best(scores):
    finite = {c: v for c, v in scores.items() if v is not None}
    return min(finite, key=finite.get) if finite else None


def extensions(win, tried):
    """One more rate past each grid edge the winner sits on, not already tried (applied once, as in Stage 2A)."""
    m, a = win
    out = []
    if m == MUON[0]:
        out.append((m / 2, a))
    if m == MUON[-1]:
        out.append((m * 2, a))
    if a == ADAM[0]:
        out.append((m, a / 2))
    if a == ADAM[-1]:
        out.append((m, a * 2))
    return [c for c in out if c not in tried]


def confirm(scores):
    """Run the 200M-token seeds only when the best decoupled cell beats the best coupled cell by the SESOI."""
    win, shared = best(scores), best({c: v for c, v in scores.items() if coupled(c)})
    return win is not None and shared is not None and scores[shared] - scores[win] >= SESOI

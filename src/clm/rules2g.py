"""Stage 2G decision rules (PREREG_stage2g.md section 4), apart from file handling so tests can plant numbers.

B0-M keeps Muon at its own 8e-3 and varies the AdamW rate; a cell is the AdamW rate. scores: {adam rate: pilot
validation loss, or None if diverged}. SHARED is the Stage 2.0 recipe, AdamW at the Muon rate."""
from clm.analyze import SESOI

MUON = 8e-3
SHARED = 8e-3
GRID = (8e-3, 32e-3, 64e-3, 128e-3)


def best(scores):
    finite = {a: v for a, v in scores.items() if v is not None}
    return min(finite, key=finite.get) if finite else None


def extension(win, tried):
    """One more rate above the grid if the winner sits on its top edge, once."""
    return [win * 2] if win == GRID[-1] and win * 2 not in tried else []


def separate(scores):
    """B0 is rerun at the best AdamW rate only if it beats the shared rate by the SESOI at the pilot;
    otherwise the Stage 2.0 B0 runs stay the reference."""
    win, shared = best(scores), scores.get(SHARED)
    return win is not None and shared is not None and shared - scores[win] >= SESOI

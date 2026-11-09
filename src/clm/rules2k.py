"""Stage 2K decision rules (PREREG_stage2k.md section 5), apart from file handling so tests can plant numbers.

loss: {arm: {seed: final validation loss}} for the arms "count" (plain COUNT on separate Muon and AdamW rates),
"gated" (the Stage 2E gated read, same rates) and "scaled" (COUNT with one learned scalar on the whole read)."""
from clm.analyze import verdict
from clm.rules2e import gap, mean_d, needs_seeds  # noqa: F401  (re-exported for the analysis)


def p19(loss):
    """The gated read still beats plain COUNT on separate rates: the Stage 1 verdict is X_BETTER."""
    return verdict(loss["gated"], loss["count"]) == "X_BETTER"


def reading(loss):
    """What the three paired verdicts say about the Stage 2E gain.
    NOT_HOLDING: the gate does not beat COUNT at these rates.
    STRUCTURE: the gate beats COUNT and also beats the scalar control, so the split by order, source and count bin matters.
    SCALE: the gate beats COUNT, the scalar control beats COUNT too, and the gate does not beat the scalar:
           the gain is a larger overall memory signal.
    UNRESOLVED: the gate beats COUNT but the control separates neither way."""
    if not p19(loss):
        return "NOT_HOLDING"
    if verdict(loss["gated"], loss["scaled"]) == "X_BETTER":
        return "STRUCTURE"
    if verdict(loss["scaled"], loss["count"]) == "X_BETTER":
        return "SCALE"
    return "UNRESOLVED"


def wants_seeds(loss):
    """The five-seed rule, decided once on three seeds: either contrast that matters is INCONCLUSIVE with a mean gap under 2 x SESOI."""
    return needs_seeds(loss, "gated", "count") or needs_seeds(loss, "gated", "scaled")

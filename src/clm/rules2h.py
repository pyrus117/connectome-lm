"""Stage 2H decision rules (PREREG_stage2h.md section 5), apart from file handling so tests can plant numbers.

A run is summarised by its validation loss at every depth 1..R at the last step. The new arm is the looped model with a
per-pass gain and the loss on every depth; the reference is Stage 2D's looped model."""
from clm.analyze import SESOI, verdict  # noqa: F401  (SESOI is re-exported for the analysis)


def summary(by_depth):
    """Best depth (a tie goes to the shallower one), its loss, the first and last depth's loss, and the marginal gain
    of each extra pass (loss at depth r minus loss at depth r + 1)."""
    best = min(range(len(by_depth)), key=by_depth.__getitem__)
    return {"best_depth": best + 1, "best_loss": by_depth[best], "depth1": by_depth[0], "deepest": by_depth[-1],
            "gains": [by_depth[i] - by_depth[i + 1] for i in range(len(by_depth) - 1)]}


def contrast(new, ref, key):
    """The Stage 1 paired verdict, new minus reference, on one summary entry. new, ref: {seed: summary}."""
    return verdict({s: v[key] for s, v in new.items()}, {s: v[key] for s, v in ref.items()})


def p17(new, ref):
    """P17 holds if and only if the new arm's best-depth loss is X_BETTER than the reference's."""
    return contrast(new, ref, "best_loss") == "X_BETTER"


def futile(new0, ref0):
    """After seed 0 of both arms. X_BETTER needs every seed's difference below zero, so a seed-0 difference of zero or
    more makes P17 impossible and the other seeds are not run."""
    return new0["best_loss"] - ref0["best_loss"] >= 0

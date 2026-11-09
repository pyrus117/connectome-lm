"""Stage 2K2 decision rules (PREREG_stage2k2.md section 3), apart from file handling so tests can plant numbers.

loss: {arm: {seed: final validation loss}} for "count" (plain COUNT) and the three candidate reads trained with the
gate parameters on a slower rate: "gated_low", "scaled_low" (one scalar) and "oscaled_low" (one scalar per order)."""
import statistics

from clm.analyze import verdict

CANDIDATES = ("gated_low", "scaled_low", "oscaled_low")


def versus_count(loss):
    """The Stage 1 paired verdict of each candidate read against plain COUNT (candidate minus COUNT)."""
    return {a: verdict(loss[a], loss["count"]) for a in CANDIDATES}


def p20(loss):
    """P20: at least one candidate read beats plain COUNT, X_BETTER on every seed."""
    return any(v == "X_BETTER" for v in versus_count(loss).values())


def carry_forward(loss):
    """The COUNT read later stages use: among the candidates that are X_BETTER than COUNT, the one with the lowest
    mean loss; plain COUNT if none is."""
    wins = [a for a, v in versus_count(loss).items() if v == "X_BETTER"]
    return min(wins, key=lambda a: statistics.fmean(loss[a].values())) if wins else "count"

"""Stage 2M decision rules (PREREG_stage2m.md section 3), apart from file handling so tests can plant numbers.

A run is summarised by clm.rules2h.summary. The loop's benefit is what the extra passes bought inside one trained
model: its depth-1 loss minus its best-depth loss. Stage 2H's loop earned about 0.05; Stage 2L's earned under 0.0001."""
from clm.analyze import SESOI
from clm.rules2h import summary  # noqa: F401  (re-exported for the analysis)


def benefit(s):
    """Nats the extra passes bought in one run. s: a rules2h.summary."""
    return s["depth1"] - s["best_loss"]


def alive(arm):
    """The loop is alive in an arm if and only if every seed's benefit reaches the SESOI. arm: {seed: summary}."""
    return all(benefit(s) >= SESOI for s in arm.values())


def futile(a0, b0):
    """After seed 0 of both arms. If neither loop earned the SESOI, P22 and P23 are both impossible (each needs
    every seed), so seeds 1 and 2 are not run."""
    return benefit(a0) < SESOI and benefit(b0) < SESOI

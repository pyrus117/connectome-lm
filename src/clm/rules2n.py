"""Stage 2N decision rules (PREREG_stage2n.md section 3): which 64e-3 AdamW group kills the loop?

Reuses Stage 2M's benefit and alive rules. Seed-0 futility is evaluated per arm: an arm whose seed-0 loop benefit is
below SESOI cannot pass `alive` (which requires all three seeds >= SESOI), so seeds 1 and 2 are run only for arms whose
seed-0 loop is alive."""
from clm.analyze import SESOI
from clm.rules2m import alive, benefit, summary  # noqa: F401


def keep_arm(s0):
    """Whether an arm runs seeds 1 and 2 after seed 0. s0: rules2h.summary of seed 0."""
    return benefit(s0) >= SESOI


def status(arm_summaries):
    """Decision status of an arm given its completed seed summaries ({seed: summary}).
    If only seed 0 ran, the arm stopped for futility; otherwise all three seeds decide."""
    if set(arm_summaries) == {0}:
        return "NOT_SUPPORTED_FUTILITY"
    return "SUPPORTED" if alive(arm_summaries) else "NOT_SUPPORTED"

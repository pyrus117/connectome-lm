"""Stage 2E decision rules (PREREG_stage2e.md section 5), apart from file handling so tests can plant numbers.

loss: {arm: {seed: final validation loss}}. classes: {arm: {class label: [loss per seed]}}, the 4-token-history
classes of scripts/bucket_eval.py. Arms: "count" (Stage 2A), "gated", "kenyon" (Kenyon buckets, gated) and,
when it was run, "random" (random buckets, gated)."""
import statistics

from clm.analyze import SESOI, verdict

CLASS_EFFECT = 0.01                     # a class gap counts at the SESOI
STORED = ("1-9", "10-99", "100-999")


def gap(classes, x, y, label):
    """Mean over seeds of arm x's loss in one class minus arm y's."""
    return statistics.fmean(classes[x][label]) - statistics.fmean(classes[y][label])


def mean_d(loss, x, y):
    return statistics.fmean(loss[x][s] - loss[y][s] for s in loss[x])


def needs_seeds(loss, x, y):
    """The five-seed rule: an INCONCLUSIVE contrast on three seeds whose mean gap is under 2 x SESOI."""
    return (len(loss[x]) == 3 and verdict(loss[x], loss[y]) == "INCONCLUSIVE"
            and abs(mean_d(loss, x, y)) < 2 * SESOI)


def p10a(classes):
    """The absent vector: the gate gains in 'not stored' and costs no stored class."""
    return (gap(classes, "gated", "count", "not stored") <= -CLASS_EFFECT
            and all(gap(classes, "gated", "count", lab) < CLASS_EFFECT for lab in STORED))


def p10b(classes):
    """The count bins: the gate gains in '>=1000', where COUNT was worse than B0."""
    return gap(classes, "gated", "count", ">=1000") <= -CLASS_EFFECT


def p11(classes):
    """Kenyon buckets help once gated: a gain over the gated read in 'not stored', the only class a fill reaches."""
    return gap(classes, "kenyon", "gated", "not stored") <= -CLASS_EFFECT


def run_random(loss, classes):
    """The gated random-bucket control runs only when there is a Kenyon gain to attribute."""
    return verdict(loss["kenyon"], loss["gated"]) == "X_BETTER" or p11(classes)


def p11_similarity(classes):
    """The Kenyon gain comes from similarity, not pooling: Kenyon beats random buckets in 'not stored'."""
    return gap(classes, "kenyon", "random", "not stored") <= -CLASS_EFFECT

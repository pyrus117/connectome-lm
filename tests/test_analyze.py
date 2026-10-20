import json
import math

import pytest

from clm.analyze import analyze, check_run, verdict
from clm.graphs import make_wiring

Y = {0: 3.50, 1: 3.52, 2: 3.48}


@pytest.mark.parametrize("diffs, expected", [
    ([-0.03, -0.02, -0.04], "X_BETTER"),
    ([0.03, 0.02, 0.04], "X_WORSE"),
    ([0.001, -0.002, 0.0015], "NO_MEANINGFUL_DIFFERENCE"),
    ([-0.05, 0.01, -0.02], "INCONCLUSIVE"),          # mean -0.02 but signs disagree
    ([-0.0101, -0.0101, -0.0101], "X_BETTER"),       # just past the boundary
    ([-0.0099, -0.0099, -0.0099], "NO_MEANINGFUL_DIFFERENCE"),   # just inside it
    ([0.0, 0.0, -0.025], "INCONCLUSIVE"),            # small mean, one large difference
])
def test_verdict_rules(diffs, expected):
    x = {s: Y[s] + d for s, d in enumerate(diffs)}
    assert verdict(x, Y) == expected


def test_fewer_than_three_seeds_or_unpaired_seeds_are_invalid():
    assert verdict({0: 3.4, 1: 3.4}, {0: 3.5, 1: 3.5}) == "INVALID"
    assert verdict({0: 3.4, 1: 3.4, 3: 3.4}, Y) == "INVALID"


GOOD = {"status": "complete", "final_val_loss": 3.5, "tokens_seen": 100, "fingerprints": ["f"]}


@pytest.mark.parametrize("change, fragment", [
    ({"status": "diverged"}, "status"),
    ({"final_val_loss": float("nan")}, "non-finite"),
    ({"final_val_loss": None}, "non-finite"),
    ({"tokens_seen": 99}, "tokens"),
    ({"fingerprints": ["g"]}, "fingerprint"),
])
def test_check_run_names_each_defect(change, fragment):
    problems = check_run({**GOOD, **change}, 100, "f")
    assert len(problems) == 1 and fragment in problems[0]


def test_check_run_passes_a_good_run_and_flags_a_missing_one():
    assert check_run(GOOD, 100, "f") == []
    assert check_run(None, 100, "f") == ["missing results.json"]
    assert "fingerprint" in check_run({**GOOD, "fingerprints": []}, 100, "f")[0]


def write_runs(tmp_path, region, loss_of):
    for arm in ("real", "shuf", "er", "real_randio", "dense"):
        for seed in range(3):
            fp = [make_wiring(region, arm, seed, 8).fingerprint()]
            d = tmp_path / "runs" / f"{arm}-{seed}"
            d.mkdir(parents=True)
            (d / "results.json").write_text(json.dumps(
                {"status": "complete", "arm": arm, "seed": seed, "fingerprints": fp,
                 "tokens_seen": 100, "final_val_loss": loss_of(arm, seed)}))


def test_end_to_end_report_on_a_known_positive(tmp_path, region):
    write_runs(tmp_path, region, lambda arm, s: 3.5 + 0.01 * s - (0.05 if arm == "real" else 0))
    r = analyze(tmp_path / "runs", region, 100, (0, 1, 2), k_io=8)
    assert r["contrasts"]["primary"]["verdict"] == "X_BETTER"
    assert r["contrasts"]["degree_stats"]["verdict"] == "NO_MEANINGFUL_DIFFERENCE"
    assert r["invalid_runs"] == {}


def test_one_bad_run_makes_every_contrast_that_uses_it_invalid(tmp_path, region):
    write_runs(tmp_path, region, lambda arm, s: math.nan if (arm, s) == ("shuf", 1) else 3.5)
    r = analyze(tmp_path / "runs", region, 100, (0, 1, 2), k_io=8)
    assert r["contrasts"]["primary"]["verdict"] == "INVALID"
    assert r["contrasts"]["degree_stats"]["verdict"] == "INVALID"
    assert r["contrasts"]["io_placement"]["verdict"] != "INVALID"
    assert "shuf-1" in r["invalid_runs"]


def test_a_run_trained_on_the_wrong_wiring_is_caught(tmp_path, region):
    write_runs(tmp_path, region, lambda arm, s: 3.5)
    p = tmp_path / "runs" / "shuf-2" / "results.json"
    r = json.loads(p.read_text())
    r["fingerprints"] = [make_wiring(region, "shuf", 0, 8).fingerprint()]   # seed 0's shuffle
    p.write_text(json.dumps(r))
    out = analyze(tmp_path / "runs", region, 100, (0, 1, 2), k_io=8)
    assert "shuf-2" in out["invalid_runs"]

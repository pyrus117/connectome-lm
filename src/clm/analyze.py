"""Turn finished runs into the pre-registered verdicts (PREREG.md section 5).

A run that is missing, incomplete, non-finite, short of the token budget, or
whose wiring fingerprint does not match the wiring its arm and seed should
have, is listed as invalid and never dropped silently: any invalid run makes
every contrast that uses its arm INVALID.
"""
import argparse
import json
import math
from pathlib import Path

from clm.graphs import make_wiring

SESOI = 0.01
CONTRASTS = (("primary", "real", "shuf"),          # does the real wiring matter?
             ("degree_stats", "shuf", "er"),       # do degree statistics matter?
             ("io_placement", "real", "real_randio"),
             ("vs_dense", "real", "dense"))


def check_run(result, tokens, fingerprint):
    if result is None:
        return ["missing results.json"]
    problems = []
    if result.get("status") != "complete":
        problems.append(f"status is {result.get('status')!r}")
    loss = result.get("final_val_loss")
    if not isinstance(loss, (int, float)) or not math.isfinite(loss):
        problems.append(f"non-finite final_val_loss {loss!r}")
    if result.get("tokens_seen") != tokens:
        problems.append(f"tokens_seen {result.get('tokens_seen')} != {tokens}")
    if result.get("fingerprints") != [fingerprint]:
        problems.append("wiring fingerprint does not match this arm and seed")
    return problems


def verdict(x, y, sesoi=SESOI):
    """Paired by seed: is loss x lower than loss y? (dicts seed -> loss)"""
    seeds = sorted(x)
    if sorted(y) != seeds or len(seeds) < 3:
        return "INVALID"
    d = [x[s] - y[s] for s in seeds]
    mean = sum(d) / len(d)
    if mean <= -sesoi and all(v < 0 for v in d):
        return "X_BETTER"
    if mean >= sesoi and all(v > 0 for v in d):
        return "X_WORSE"
    if abs(mean) < sesoi and max(abs(v) for v in d) < 2 * sesoi:
        return "NO_MEANINGFUL_DIFFERENCE"
    return "INCONCLUSIVE"


def analyze(runs_dir, region, tokens, seeds, k_io=256):
    runs_dir = Path(runs_dir)
    arms = sorted({a for _, x, y in CONTRASTS for a in (x, y)})
    loss, invalid = {a: {} for a in arms}, {}
    for arm in arms:
        for seed in seeds:
            name = f"{arm}-{seed}"
            path = runs_dir / name / "results.json"
            result = json.loads(path.read_text()) if path.exists() else None
            fp = make_wiring(region, arm, seed, k_io).fingerprint()
            problems = check_run(result, tokens, fp)
            if problems:
                invalid[name] = problems
            else:
                loss[arm][seed] = result["final_val_loss"]
    contrasts = {}
    for name, x, y in CONTRASTS:
        bad = any(n.rsplit("-", 1)[0] in (x, y) for n in invalid)
        diffs = {s: loss[x][s] - loss[y][s] for s in seeds if s in loss[x] and s in loss[y]}
        contrasts[name] = {"x": x, "y": y, "diffs": diffs,
                           "verdict": "INVALID" if bad else verdict(loss[x], loss[y])}
    return {"sesoi": SESOI, "losses": loss, "invalid_runs": invalid, "contrasts": contrasts}


def main():
    from clm.train import ROOT, load_config
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True)
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    mcfg, tcfg = load_config(args.config)
    tokens = (tcfg.tokens // (tcfg.batch * mcfg.ctx)) * tcfg.batch * mcfg.ctx
    report = analyze(args.runs, ROOT / tcfg.region, tokens, (0, 1, 2), mcfg.k_io)
    (Path(args.runs) / "analysis.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["contrasts"], indent=2))
    if report["invalid_runs"]:
        print("INVALID RUNS:", json.dumps(report["invalid_runs"], indent=2))


if __name__ == "__main__":
    main()

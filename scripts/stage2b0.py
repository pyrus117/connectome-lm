"""Stage 2B-0: offline pre-test of the Kenyon backoff (PREREG_stage2b0.md). CPU only.

    .venv/bin/python scripts/stage2b0.py sweep      # settings chosen on held-out training text (no validation read)
    .venv/bin/python scripts/stage2b0.py final      # the chosen settings and controls, scored on validation, gate 2

An exact n-gram model, a bucket n-gram model (histories pooled by Kenyon bucket) and their mixture are scored
by the exact 4-token history's stored count. Counts come from a universe of training documents; mixing
weights and settings come from a held-out slice of the count corpus that no count includes.
"""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

from clm import counts as C
from clm import kenyon as K
from clm import kenyon_lm as L
from clm.countread import BUCKET_LABELS
from clm.data import open_tokens
from clm.ngram_lm import NgramLM, add_one_unigram

ROOT = Path(__file__).resolve().parent.parent
VOCAB, UNITS, DIM = 8192, 2048, 64
C_MIN = (1, 1, 2, 2)                          # exact tables for histories of 1..4 tokens, as in Stage 2A
ORDERS = (2, 3, 4)                            # history lengths that get buckets (a 1-token history is its own bucket)
CTX, N_EVAL, HELD_TOKENS, CALIB = 512, 2_097_152, 10_000_000, 40_000
OUT = ROOT / "results"
PREREG = ROOT / "PREREG_stage2b0.md"


def prereg_sha():
    if not PREREG.exists():
        sys.exit(f"{PREREG.name} is missing: register the stage before running it")
    return hashlib.sha256(PREREG.read_bytes()).hexdigest()
SWEEP = [("default", {}), *[(f"K{k}", {"in_degree": k}) for k in (1, 3, 7, 15, 31)],
         ("winners2", {"winners": 2}), ("winners4", {"winners": 4}),
         ("thresholds_off", {"thresholds": False}), ("random_codes", {"codes": "random"})]
DEFAULT = {"codes": "ppmi", "in_degree": None, "winners": 1, "thresholds": True}
GATE = 0.02                                   # nats: Kenyon must beat random buckets by this in both rare classes


def load_selection():
    """(name, config) of the Kenyon configuration the final stage and the Stage 2B builder use: the override
    file if there is one (results/stage2b0_selection.json), else the sweep's own choice."""
    override = OUT / "stage2b0_selection.json"
    src = json.loads((override if override.exists() else OUT / "stage2b0_sweep.json").read_text())
    if "selected_config" not in src:
        sys.exit("run the sweep to completion first")
    return src["selected"], src["selected_config"]


def doc_start_at_or_after(tokens, eot, pos):
    window = np.asarray(tokens[pos:pos + 1_000_000])
    hit = np.flatnonzero(window == eot)
    if not hit.size:
        raise ValueError("no document boundary found")
    return pos + int(hit[0]) + 1


def split(corpus, eot, universe_tokens):
    """Document-aligned universe (the first documents) and held-out slice (the last HELD_TOKENS tokens)."""
    held_start = doc_start_at_or_after(corpus, eot, len(corpus) - HELD_TOKENS)
    if universe_tokens is None:
        end = held_start
    else:
        end = doc_start_at_or_after(corpus, eot, universe_tokens)
    if end > held_start:
        raise ValueError("universe overlaps the held-out slice")
    return corpus[:end], np.asarray(corpus[held_start:])


def exact_tables(universe, eot, parts, cache):
    tables = []
    for k, c in zip((1, 2, 3, 4), C_MIN):
        d = cache / f"order{k + 1}"
        if (d / "meta.json").exists():
            tables.append(C.load(d)[0])
            continue
        t0 = time.time()
        table = C.build(universe, k, eot, c, parts)
        C.save(table, d, {"c_min": c, "eot_id": eot, "corpus_tokens": int(len(universe))})
        print(f"exact table k={k}: {table.keys.size:,} rows, {time.time() - t0:.0f}s", flush=True)
        tables.append(C.load(d)[0])
    return tables


def token_codes(universe, eot, parts, cache):
    path = cache / "codes.npz"
    if path.exists():
        z = np.load(path)
        return {"ppmi": z["ppmi"], "random": z["random"]}, z["freq"]
    t0 = time.time()
    counts = K.cooccurrence(universe, VOCAB, window=2, eot=eot)
    freq = np.zeros(VOCAB)
    for s in range(0, len(universe), 1 << 26):
        freq += np.bincount(np.asarray(universe[s:s + (1 << 26)]), minlength=VOCAB)
    codes = {"ppmi": K.ppmi_codes(counts, DIM), "random": K.random_codes(VOCAB, DIM)}
    np.savez(path, ppmi=codes["ppmi"], random=codes["random"], freq=freq)
    print(f"codes: {time.time() - t0:.0f}s", flush=True)
    return codes, freq


def calibration_sample(universe, eot, k, rng):
    """Histories of k tokens at random positions of the universe (so frequent tokens are as common as in text)."""
    pos = rng.integers(0, len(universe) - k, int(CALIB * 1.3))
    toks = np.stack([np.asarray(universe[pos + j]) for j in range(k)], axis=1).astype(np.int64)
    return toks[(toks != eot).all(axis=1)][:CALIB]


def bucket_tables(universe, eot, cfg, codes, freq, parts, calib, kind):
    """Bucket tables and key functions for histories of ORDERS tokens. kind: 'kenyon' or 'random'."""
    tables, fns, usage = {}, {}, {}
    for k in ORDERS:
        t0 = time.time()
        if kind == "random":
            fn = K.exclude_eot(lambda packed, toks: K.hash_bucket(packed, UNITS * UNITS), eot)
        else:
            kh = K.KenyonHash(codes[cfg["codes"]], k, UNITS, cfg["winners"], cfg["in_degree"], cfg["thresholds"],
                              freq, seed=0)
            kh.calibrate(calib[k])
            usage[k] = kh.unit_usage(calib[k])
            fn = K.exclude_eot(lambda packed, toks, kh=kh: kh.bucket_keys(toks), eot)
        fns[k] = fn
        t1 = time.time()
        tables[k] = K.bucket_table(universe, k, eot, fn, c_min=1, parts=parts)
        print(f"  {kind} buckets k={k}: calibrate {t1 - t0:.0f}s, table {time.time() - t1:.0f}s, "
              f"{tables[k].keys.size:,} buckets", flush=True)
    return tables, fns, usage


def bucket_use(table):
    p = table.hcount.astype(np.float64)
    p /= p.sum()
    return {"buckets_used": int(table.keys.size), "max_share": float(p.max()),
            "entropy_bits": float(-(p * np.log2(p)).sum())}


def run_config(name, cfg, kind, ctx_data):
    t0 = time.time()
    d = ctx_data
    tables, fns, usage = bucket_tables(d["universe"], d["eot"], cfg, d["codes"], d["freq"], d["parts"], d["calib"], kind)
    exact = d["tables"]
    lm = NgramLM([exact[0], *(tables[k] for k in ORDERS)], d["uni"], keyfns=[None, *(fns[k] for k in ORDERS)])
    row = {"name": name, "kind": kind, "config": cfg, "unit_usage": {str(k): v for k, v in usage.items()},
           "bucket_use": {str(k): bucket_use(tables[k]) for k in ORDERS}}
    probs = {}
    for tag in d["streams"]:
        t1 = time.time()
        probs[tag] = L.score_model(lm, d["streams"][tag]["tokens"], CTX, N_EVAL)
        print(f"  scored {tag}: {time.time() - t1:.0f}s", flush=True)
    row["probs"] = probs
    row["seconds"] = round(time.time() - t0)
    return row


def summarise(row, d):
    """Fit the mixing weights on the held-out slice and report losses by class (and on validation if scored)."""
    lam = L.fit_lambdas(d["streams"]["held"]["exact"], row["probs"]["held"], d["streams"]["held"]["cls"])
    out = {"name": row["name"], "kind": row["kind"], "config": row["config"], "lambdas": lam.tolist(),
           "unit_usage": row["unit_usage"], "bucket_use": row["bucket_use"], "seconds": row["seconds"]}
    for tag, s in d["streams"].items():
        mix = L.mixture_probs(s["exact"], row["probs"][tag], s["cls"], lam)
        out[f"{tag}_exact"] = L.class_loss(s["exact"], s["cls"])
        out[f"{tag}_mixture"] = L.class_loss(mix, s["cls"])
        out[f"{tag}_rare_exact"], out[f"{tag}_rare_mixture"] = L.rare_loss(s["exact"], s["cls"]), L.rare_loss(mix, s["cls"])
    return out


def prepare(universe_tokens, parts, streams, cache_name):
    corpus = open_tokens(ROOT / "data/fineweb/count_corpus.bin")
    info = json.loads((ROOT / "data/fineweb/count_corpus.json").read_text())
    eot = info["eot_id"]
    universe, held = split(corpus, eot, universe_tokens)
    print(f"universe {len(universe):,} tokens, held-out {len(held):,}", flush=True)
    cache = ROOT / "data" / "count_2b0" / f"{cache_name}-{len(universe)}"
    cache.mkdir(parents=True, exist_ok=True)
    tables = exact_tables(universe, eot, parts, cache)
    codes, freq = token_codes(universe, eot, parts, cache)
    rng = np.random.default_rng(0)
    calib = {k: calibration_sample(universe, eot, k, rng) for k in ORDERS}
    uni = add_one_unigram(universe, VOCAB)
    exact_lm = NgramLM(tables, uni)
    data = {"universe": universe, "eot": eot, "parts": parts, "tables": tables, "codes": codes, "freq": freq,
            "calib": calib, "uni": uni, "streams": {}, "universe_tokens": len(universe)}
    toks = {"held": held, "val": open_tokens(ROOT / "data/fineweb/val.bin")}
    for tag in streams:
        t0 = time.time()
        data["streams"][tag] = {"tokens": toks[tag], "exact": L.score_model(exact_lm, toks[tag], CTX, N_EVAL),
                                "cls": L.score_classes(tables[3], toks[tag], CTX, N_EVAL)}
        print(f"exact model on {tag}: loss {-np.log(data['streams'][tag]['exact']).mean():.4f} "
              f"({time.time() - t0:.0f}s)", flush=True)
    return data


def sweep(args):
    prereg_sha()
    d = prepare(args.universe_tokens or 100_000_000, args.parts, ("held",), "sweep")
    path = OUT / "stage2b0_sweep.json"
    rows = json.loads(path.read_text())["rows"] if args.resume and path.exists() else []
    rows = [r for r in rows if r["kind"] == "kenyon"]
    for name, over in SWEEP:
        if args.only and name not in args.only:
            continue
        if name in {r["name"] for r in rows}:
            continue
        cfg = {**DEFAULT, **over}
        row = summarise(run_config(name, cfg, "kenyon", d), d)
        rows.append(row)
        print(f"{name}: held rare loss exact {row['held_rare_exact']:.4f} mixture {row['held_rare_mixture']:.4f} "
              f"lambdas {np.round(row['lambdas'][:2], 2).tolist()} ({row['seconds']}s)", flush=True)
        (OUT / "stage2b0_sweep.json").write_text(json.dumps({"rows": rows}, indent=2) + "\n")
    raw = run_config("random_buckets", DEFAULT, "random", d)
    rows.append(summarise(raw, d))
    chosen = min((r for r in rows if r["kind"] == "kenyon" and r["config"]["codes"] == "ppmi"),
                 key=lambda r: r["held_rare_mixture"])
    out = {"prereg_sha256": prereg_sha(), "universe_tokens": d["universe_tokens"], "rows": rows, "selected": chosen["name"],
           "selected_config": chosen["config"], "rule": "lowest held-out mixture loss on 4-token histories not stored or seen 1-9 times, "
                                                         "among Kenyon configs with PPMI codes"}
    (OUT / "stage2b0_sweep.json").write_text(json.dumps(out, indent=2) + "\n")
    print("SELECTED", chosen["name"], chosen["config"], flush=True)


def final(args):
    selected, cfg = load_selection()
    d = prepare(args.universe_tokens, args.parts, ("held", "val"), "final")
    runs = [("kenyon", cfg, "kenyon"), ("random_buckets", cfg, "random"),
            ("random_codes", {**cfg, "codes": "random"}, "kenyon")]
    rows = []
    for name, c, kind in runs:
        row = summarise(run_config(name, c, kind, d), d)
        rows.append(row)
        print(f"{name}: val rare loss exact {row['val_rare_exact']:.4f} mixture {row['val_rare_mixture']:.4f}", flush=True)
    by = {r["name"]: r for r in rows}
    gate = {}
    for lab in ("not stored", "1-9"):
        gain_k = by["kenyon"]["val_exact"][lab] - by["kenyon"]["val_mixture"][lab]
        gain_r = by["random_buckets"]["val_exact"][lab] - by["random_buckets"]["val_mixture"][lab]
        gate[lab] = {"kenyon_gain": gain_k, "random_bucket_gain": gain_r, "difference": gain_k - gain_r}
    passed = all(g["difference"] >= GATE for g in gate.values())
    out = {"prereg_sha256": prereg_sha(), "universe_tokens": d["universe_tokens"], "selected": selected,
           "config": cfg, "rows": rows,
           "gate2": {"threshold": GATE, "classes": gate, "result": "PASS" if passed else "FAIL"}}
    (OUT / "stage2b0_final.json").write_text(json.dumps(out, indent=2) + "\n")
    print("GATE 2", out["gate2"]["result"], json.dumps(gate), flush=True)


def main():
    global N_EVAL, OUT
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=("sweep", "final"))
    ap.add_argument("--universe-tokens", type=int, default=None,
                    help="sweep default 100M; final default every token before the held-out slice")
    ap.add_argument("--parts", type=int, default=None)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--eval-tokens", type=int, default=N_EVAL, help="tokens scored per stream (smoke tests only)")
    ap.add_argument("--only", nargs="*", help="run only these sweep configurations (smoke tests only)")
    ap.add_argument("--resume", action="store_true", help="sweep: keep the rows already in the result file")
    ap.add_argument("--out", default="results", help="directory for the result files")
    args = ap.parse_args()
    N_EVAL, OUT = args.eval_tokens, ROOT / args.out
    torch.set_num_threads(args.threads)
    if args.parts is None:
        args.parts = 8 if args.stage == "sweep" else 24
    {"sweep": sweep, "final": final}[args.stage](args)


if __name__ == "__main__":
    main()

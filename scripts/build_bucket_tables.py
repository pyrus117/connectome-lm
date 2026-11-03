"""Build the Stage 2B bucket tables over the whole count corpus.

    .venv/bin/python scripts/build_bucket_tables.py --kind kenyon --out data/count_kenyon
    .venv/bin/python scripts/build_bucket_tables.py --kind random --out data/count_randbucket

`kenyon` rebuilds the hash that Stage 2B-0's final run selected (its codes, settings and calibration
sample are deterministic) and pools every history of the corpus by its bucket; `random` pools by a plain
hash of the token ids into the same number of buckets. The output holds what CountReader needs:
the tables, the calibrated hashes and codes (kenyon) or the bucket count (random).
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

import stage2b0 as B
from clm import kenyon as K
from clm.data import open_tokens

ROOT = B.ROOT


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", choices=("kenyon", "random"), required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--parts", type=int, default=24)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--max-tokens", type=int, help="use only the first N tokens (smoke tests)")
    ap.add_argument("--sweep-json", type=Path, default=ROOT / "results" / "stage2b0_sweep.json")
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    info = json.loads((ROOT / "data/fineweb/count_corpus.json").read_text())
    corpus = open_tokens(ROOT / "data/fineweb/count_corpus.bin")
    if corpus.size != info["tokens"]:
        sys.exit("count corpus does not match its manifest")
    eot = info["eot_id"]
    corpus = corpus[:args.max_tokens] if args.max_tokens else corpus

    hashes, codes, freq, n_buckets = {}, None, None, None
    if args.kind == "random":
        n_buckets = B.UNITS * B.UNITS
        fns = {k: K.exclude_eot(lambda packed, toks: K.hash_bucket(packed, n_buckets), eot) for k in B.ORDERS}
    else:
        cfg = json.loads(args.sweep_json.read_text())["selected_config"]
        universe, _ = B.split(open_tokens(ROOT / "data/fineweb/count_corpus.bin"), eot, None)
        cache = ROOT / "data" / "count_2b0" / f"final-{len(universe)}"
        if not (cache / "codes.npz").exists():
            sys.exit(f"{cache}/codes.npz is missing: run Stage 2B-0's final stage first")
        z = np.load(cache / "codes.npz")
        codes, freq = z[cfg["codes"]], z["freq"]
        rng = np.random.default_rng(0)
        calib = {k: B.calibration_sample(universe, eot, k, rng) for k in B.ORDERS}
        fns = {}
        for k in B.ORDERS:
            h = K.KenyonHash(codes, k, B.UNITS, cfg["winners"], cfg["in_degree"], cfg["thresholds"], freq, seed=0)
            h.calibrate(calib[k])
            hashes[k] = h
            fns[k] = K.exclude_eot(lambda packed, toks, h=h: h.bucket_keys(toks), eot)
            print(f"hash k={k} calibrated: {h.unit_usage(calib[k])}", flush=True)
    tables = {}
    for k in B.ORDERS:
        t0 = time.time()
        tables[k] = K.bucket_table(corpus, k, eot, fns[k], c_min=1, parts=args.parts)
        print(f"buckets k={k}: {tables[k].keys.size:,} buckets, {tables[k].tok.size:,} entries, "
              f"{time.time() - t0:.0f}s", flush=True)
    K.save_buckets(args.out, tables, args.kind, eot, hashes=hashes, codes=codes, freq=freq, n_buckets=n_buckets)
    print(f"saved to {args.out}", flush=True)


if __name__ == "__main__":
    main()

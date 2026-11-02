"""Build the counted n-gram tables (orders 2..5) from the count corpus.

For each order in turn: profile every minimum-history-count threshold, take the
smallest one whose table fits that order's share of the byte budget (the
unspent part of the budget rolls over to the later, larger orders), build the
table, save it. Validation data is never read. Output: <out>/order<n>/ with
the arrays and meta.json.

    .venv/bin/python scripts/build_count_table.py [--max-tokens N] [--out DIR]
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

from clm import counts as C

ROOT = Path(__file__).resolve().parent.parent
FINEWEB = ROOT / "data" / "fineweb"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", type=Path, default=FINEWEB / "count_corpus.bin")
    ap.add_argument("--out", type=Path, default=ROOT / "data" / "count")
    ap.add_argument("--orders", type=int, nargs="+", default=[2, 3, 4, 5])
    ap.add_argument("--budget-gib", type=float, default=6.0)
    ap.add_argument("--parts", type=int, default=16)
    ap.add_argument("--max-tokens", type=int, default=None, help="use only the first N tokens (dry runs)")
    args = ap.parse_args()

    info_path = args.corpus.with_suffix(".json")
    if not args.corpus.exists() or not info_path.exists():
        sys.exit(f"{args.corpus} or its manifest {info_path.name} is missing; run build_count_corpus.py")
    info = json.loads(info_path.read_text())
    tokens = np.memmap(args.corpus, dtype=np.uint16, mode="r")
    if tokens.size != info["tokens"]:
        sys.exit(f"corpus has {tokens.size} tokens, manifest says {info['tokens']}")
    if args.max_tokens:
        tokens = tokens[:args.max_tokens]
    eot = info["eot_id"]
    remaining = int(args.budget_gib * 2**30)
    print(f"{tokens.size:,} tokens, budget {args.budget_gib} GiB, orders {args.orders}", flush=True)

    for i, n in enumerate(args.orders):
        k, share = n - 1, remaining // (len(args.orders) - i)
        t0 = time.time()
        rows, entries = C.profile(tokens, k, eot, args.parts)
        t_profile = time.time() - t0
        c_min = C.pick_cmin(rows, entries, share)
        kept_rows, kept_entries = C.sizes_by_threshold(rows, entries)
        print(f"order {n}: profile {t_profile:.0f}s; share {share / 2**30:.2f} GiB; c_min {c_min} keeps "
              f"{kept_rows[c_min]:,} of {kept_rows[1]:,} histories", flush=True)
        t0 = time.time()
        table = C.build(tokens, k, eot, c_min, args.parts)
        used = C.table_bytes(table.keys.size, table.tok.size)
        if table.keys.size != kept_rows[c_min] or table.tok.size != kept_entries[c_min]:
            sys.exit(f"order {n}: built table differs from its profile")
        meta = {"order": n, "c_min": c_min, "rows": int(table.keys.size), "entries": int(table.tok.size),
                "bytes": used, "share_bytes": share, "histories_seen": int(kept_rows[1]),
                "corpus_tokens": int(tokens.size), "corpus_sha256": info["sha256"], "eot_id": eot,
                "parts": args.parts, "top": C.TOP, "build_seconds": round(time.time() - t0),
                "profile_seconds": round(t_profile),
                "rows_at_threshold": {str(c): int(kept_rows[c]) for c in (1, 2, 3, 5, 10, 100, 1000)}}
        C.save(table, args.out / f"order{n}", meta)
        remaining -= used
        print(f"order {n}: built {used / 2**30:.2f} GiB in {meta['build_seconds']}s; "
              f"{remaining / 2**30:.2f} GiB left", flush=True)
        del table


if __name__ == "__main__":
    main()

"""Validation loss of the pure n-gram model over the count tables (the model-free
reference for Stage 2A), overall and by history-count bucket. Reads validation
tokens, never trains. Windows are the same 4,096 x 512 the networks are scored on.

    .venv/bin/python scripts/ngram_val_loss.py
Writes results/stage2a_ngram.json.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

from clm.countread import BUCKET_LABELS, CountReader, bucket_of
from clm.data import eval_windows, open_tokens
from clm.ngram_lm import NgramLM, add_one_unigram

ROOT = Path(__file__).resolve().parent.parent
CTX, N_EVAL, CHUNK = 512, 2_097_152, 8


def main():
    info = json.loads((ROOT / "data/fineweb/count_corpus.json").read_text())
    corpus = open_tokens(ROOT / "data/fineweb/count_corpus.bin")
    if corpus.size != info["tokens"]:
        sys.exit("count corpus does not match its manifest")
    val = open_tokens(ROOT / "data/fineweb/val.bin")
    reader = CountReader(ROOT / "data/count")
    lm = NgramLM(reader, add_one_unigram(corpus, 8192))
    starts = eval_windows(len(val), CTX, N_EVAL)
    total, n = 0.0, 0
    by_bucket = {k: np.zeros((len(BUCKET_LABELS), 2)) for k in (3, 4)}
    t0 = time.time()
    for i in range(0, len(starts), CHUNK):
        s = starts[i:i + CHUNK]
        x = np.stack([val[a:a + CTX] for a in s]).astype(np.int64)
        y = np.stack([val[a + 1:a + CTX + 1] for a in s]).astype(np.int64)
        nll = lm.nll(x, y, chunk=CHUNK)
        total, n = total + nll.sum(), n + nll.size
        for k, acc in by_bucket.items():
            b = bucket_of(reader.history_counts(x, k))
            for j in range(len(BUCKET_LABELS)):
                acc[j] += (nll[b == j].sum(), (b == j).sum())
        if (i // CHUNK) % 64 == 0:
            print(f"{i + CHUNK}/{len(starts)} windows, running loss {total / n:.4f}, {time.time() - t0:.0f}s",
                  flush=True)
    out = {"val_tokens": n, "windows": len(starts), "val_loss": total / n,
           "by_history_bucket": {f"history_{k}_tokens": {
               lab: {"share": float(acc[j, 1] / acc[:, 1].sum()),
                     "loss": float(acc[j, 0] / acc[j, 1]) if acc[j, 1] else None}
               for j, lab in enumerate(BUCKET_LABELS)} for k, acc in by_bucket.items()},
           "tables": {f"order{t.k + 1}": {"c_min": c} for t, c in zip(reader.tables, reader.c_min)},
           "corpus_sha256": info["sha256"], "seconds": round(time.time() - t0)}
    (ROOT / "results" / "stage2a_ngram.json").write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()

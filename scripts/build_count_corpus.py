"""Tokenise every training-split document of shard 000 into one count corpus.

Writes data/fineweb/count_corpus.bin (uint16, end-of-text after each document)
and count_corpus.json. The corpus is the same split rule, tokenizer and
document filter as prepare_data.py, run over the whole shard, so its first
tokens are exactly data/fineweb/train.bin; that is checked before the manifest
is written. Validation documents are never included.

    .venv/bin/python scripts/build_count_corpus.py
"""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from tokenizers import Tokenizer

from clm.prep import EOT, encode_docs, split_of

ROOT = Path(__file__).resolve().parent.parent
FINEWEB = ROOT / "data" / "fineweb"
SHARD = ROOT / "data" / "raw" / "000_00000.parquet"
BATCH = 1024


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def train_docs():
    pf = pq.ParquetFile(SHARD)
    missing = {"id", "text"} - set(pf.schema_arrow.names)
    if missing:
        sys.exit(f"shard lacks columns {missing}")
    for b in pf.iter_batches(batch_size=BATCH, columns=["id", "text"]):
        for doc_id, text in zip(b.column("id").to_pylist(), b.column("text").to_pylist()):
            if split_of(doc_id) == "train":
                yield text


def check_prefix(corpus, train, chunk=1 << 24):
    if corpus.size < train.size:
        sys.exit(f"count corpus ({corpus.size} tokens) is shorter than train.bin ({train.size})")
    for start in range(0, train.size, chunk):
        want = train[start:start + chunk]
        if not np.array_equal(corpus[start:start + want.size], want):
            sys.exit(f"count corpus differs from train.bin near token {start}")


def main():
    manifest = json.loads((FINEWEB / "manifest.json").read_text())
    if sha256(FINEWEB / "tokenizer.json") != manifest["tokenizer_sha256"]:
        sys.exit("tokenizer.json does not match the data manifest")
    tok = Tokenizer.from_file(str(FINEWEB / "tokenizer.json"))
    out_path = FINEWEB / "count_corpus.bin"
    tmp = out_path.with_suffix(".part")
    docs = skipped = tokens = 0
    pending = []

    def flush(f):
        nonlocal docs, tokens
        for arr in encode_docs(tok, pending):
            arr.tofile(f)
            tokens += arr.size
            docs += 1
        pending.clear()

    with open(tmp, "wb") as f:
        for text in train_docs():
            if EOT in text:
                skipped += 1
                continue
            pending.append(text)
            if len(pending) == BATCH:
                flush(f)
                if docs % (50 * BATCH) == 0:
                    print(f"docs {docs:,}  tokens {tokens:,}", flush=True)
        flush(f)
    corpus = np.memmap(tmp, dtype=np.uint16, mode="r")
    train = np.memmap(FINEWEB / "train.bin", dtype=np.uint16, mode="r")
    check_prefix(corpus, train)
    if int(corpus.max()) >= 1 << 13:
        sys.exit("token id above the 13-bit vocabulary")
    del corpus, train
    tmp.replace(out_path)
    info = {"tokens": tokens, "docs": docs, "skipped_docs_with_eot": skipped,
            "eot_id": tok.token_to_id(EOT), "sha256": sha256(out_path),
            "tokenizer_sha256": manifest["tokenizer_sha256"], "shard_sha256": manifest["shard_sha256"],
            "split": "train documents only (val iff int(sha256(id)[:8], 16) % 100 == 0)",
            "prefix_equals_train_bin_tokens": manifest["tokens"]["train"]}
    (FINEWEB / "count_corpus.json").write_text(json.dumps(info, indent=2) + "\n")
    print(json.dumps(info, indent=2))


if __name__ == "__main__":
    main()

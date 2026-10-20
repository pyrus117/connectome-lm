"""Download FineWeb-Edu sample-10BT shard 000 and build the token files.

Writes data/fineweb/{train,val}.bin (uint16), tokenizer.json,
{train,val}_ids.txt and manifest.json. The shard sits in data/raw/, its own
directory: it is downloaded data and is only ever read as parquet.
"""
import hashlib
import json
import sys
import urllib.request
from pathlib import Path

import pyarrow.parquet as pq

from clm.prep import EOT, encode_docs, split_of, train_tokenizer

ROOT = Path(__file__).resolve().parent.parent
REPO = "HuggingFaceFW/fineweb-edu"
SHARD = "sample/10BT/000_00000.parquet"
URL = f"https://huggingface.co/datasets/{REPO}/resolve/main/{SHARD}"
TRAIN_TOKENS = 200_000_001        # the run reads 199,999,488 + 1 for the last target
VAL_TOKENS = 2_097_153
TOK_DOCS = 50_000
VOCAB = 8192


def lfs_sha256(path):
    api = f"https://huggingface.co/api/datasets/{REPO}/tree/main/{path.rsplit('/', 1)[0]}"
    with urllib.request.urlopen(api, timeout=60) as r:
        for f in json.load(r):
            if f["path"] == path:
                return f["lfs"]["oid"]
    sys.exit(f"{path} not listed by the Hugging Face API")


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download(url, dest, expected):
    if dest.exists() and sha256(dest) == expected:
        print(f"{dest.name} already present, hash ok", flush=True)
        return
    tmp = dest.with_suffix(".part")
    done = 0
    with urllib.request.urlopen(url, timeout=60) as r, open(tmp, "wb") as f:
        while chunk := r.read(1 << 22):
            f.write(chunk)
            done += len(chunk)
            if done % (1 << 28) < len(chunk):
                print(f"downloaded {done / 1e9:.2f} GB", flush=True)
    got = sha256(tmp)
    if got != expected:
        tmp.unlink()
        sys.exit(f"download hash {got} != expected {expected}")
    tmp.replace(dest)


def docs(path):
    pf = pq.ParquetFile(path)
    missing = {"id", "text"} - set(pf.schema_arrow.names)
    if missing:
        sys.exit(f"shard lacks columns {missing}")
    for b in pf.iter_batches(batch_size=1024, columns=["id", "text"]):
        yield from zip(b.column("id").to_pylist(), b.column("text").to_pylist())


def main():
    raw, out = ROOT / "data" / "raw", ROOT / "data" / "fineweb"
    raw.mkdir(parents=True, exist_ok=True)
    out.mkdir(parents=True, exist_ok=True)
    shard = raw / "000_00000.parquet"
    expected = lfs_sha256(SHARD)
    download(URL, shard, expected)

    tok_texts = []
    for doc_id, text in docs(shard):
        if split_of(doc_id) == "train" and EOT not in text:
            tok_texts.append(text)
            if len(tok_texts) == TOK_DOCS:
                break
    tok = train_tokenizer(iter(tok_texts), VOCAB)
    tok.save(str(out / "tokenizer.json"))
    print("tokenizer trained", flush=True)

    counts, skipped = {"train": 0, "val": 0}, 0
    targets = {"train": TRAIN_TOKENS, "val": VAL_TOKENS}
    files = {s: open(out / f"{s}.bin", "wb") for s in counts}
    ids = {s: open(out / f"{s}_ids.txt", "w") for s in counts}
    pending = []

    def flush():
        enc = encode_docs(tok, [t for _, _, t in pending])
        for (split, doc_id, _), arr in zip(pending, enc):
            if counts[split] < targets[split]:
                arr.tofile(files[split])
                ids[split].write(doc_id + "\n")
                counts[split] += arr.size
        pending.clear()

    for doc_id, text in docs(shard):
        if all(counts[s] >= targets[s] for s in counts):
            break
        if EOT in text:
            skipped += 1
            continue
        pending.append((split_of(doc_id), doc_id, text))
        if len(pending) == 1024:
            flush()
            if counts["train"] % 20_000_000 < 1_000_000:
                print(f"tokens so far {counts}", flush=True)
    flush()
    for f in (*files.values(), *ids.values()):
        f.close()
    short = {s: (counts[s], targets[s]) for s in counts if counts[s] < targets[s]}
    if short:
        sys.exit(f"shard ran out before the targets were met: {short}")

    manifest = {"shard": URL, "shard_sha256": expected, "vocab": VOCAB, "tokenizer_docs": TOK_DOCS,
                "tokenizer_sha256": sha256(out / "tokenizer.json"),
                "split_rule": "val iff int(sha256(id)[:8], 16) % 100 == 0",
                "tokens": counts, "skipped_docs_with_eot": skipped,
                "licence": "ODC-By 1.0 (FineWeb-Edu)"}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()

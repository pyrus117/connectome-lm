"""Cut named FlyWire v783 regions out of the whole-brain connectome.

Reads the CSR cache and annotations already built by fly-criticality, so no
download is needed. Each region is saved as a local edge list (indices 0..k-1)
with the neurons' root ids and cell classes, plus a manifest recording input
hashes, so a later result can be traced to the exact wiring it used.

Usage: python scripts/extract_subgraph.py [--src DIR] [--out DIR]
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent

# Anatomical input and output classes are recorded per neuron, so the model
# can later choose between degree-based and anatomy-based input/readout sets.
REGIONS = {
    # mushroom body plus its olfactory projection-neuron input
    "mb_alpn": ["ALPN", "Kenyon_Cell", "MBON", "DAN", "MBIN"],
    # central complex: a second region, so a result can be checked for region-specificity
    "cx": ["CX"],
}

LICENCE = {
    "connectivity": "CC-BY-4.0, FlyWire Whole-brain Connectome Connectivity Data v783, "
                    "zenodo.org/records/10676866",
    "annotations": "CC-BY-4.0, Schlegel et al. 2024 supplemental files, zenodo.org/records/10877326",
    "cite": ["Dorkenwald et al. 2024, Nature, Neuronal wiring diagram of an adult brain",
             "Schlegel et al. 2024, Nature, Whole-brain annotation and multi-connectome cell typing",
             "Codex: doi 10.13140/RG.2.2.35928.67844"],
}


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load(src):
    files = {"csr": src / "csr_real.npz", "completeness": src / "Completeness_783.csv",
             "annotations": src / "annotations" / "neuron_annotations.tsv"}
    missing = [str(p) for p in files.values() if not p.exists()]
    if missing:
        sys.exit(f"missing input files: {missing}")

    z = np.load(files["csr"])
    indptr, post, w, n = z["indptr"], z["post"], z["w"], int(z["n"])
    ids = pd.read_csv(files["completeness"], index_col=0).index.to_numpy(np.int64)
    if len(ids) != n:
        sys.exit(f"CSR has {n} neurons but Completeness_783.csv has {len(ids)}; version mismatch")

    a = pd.read_csv(files["annotations"], sep="\t", dtype={"root_id": np.int64},
                    usecols=["root_id", "super_class", "cell_class"], low_memory=False)
    a = a.drop_duplicates("root_id").set_index("root_id").reindex(ids)
    # cell_class is legitimately blank for many neurons; super_class is the coverage test
    if a["super_class"].notna().sum() < 0.99 * n:
        sys.exit(f"only {a['super_class'].notna().sum()} of {n} neurons annotated; "
                 "wrong annotation file?")
    cls = a["cell_class"]

    pre = np.repeat(np.arange(n, dtype=np.int32), np.diff(indptr))
    return files, ids, cls.astype(str).to_numpy(), pre, post, w


def cut(classes, ids, cls, pre, post, w):
    mask = np.isin(cls, classes)
    k = int(mask.sum())
    if k == 0:
        raise ValueError(f"no neurons match {classes}")
    local = np.full(mask.size, -1, np.int32)
    local[mask] = np.arange(k, dtype=np.int32)
    keep = mask[pre] & mask[post]
    e_pre, e_post, e_w = local[pre[keep]], local[post[keep]], w[keep]
    if (e_pre < 0).any() or (e_post < 0).any():
        raise RuntimeError("edge endpoint outside the region; index mapping is wrong")
    return {"root_id": ids[mask], "cell_class": cls[mask].astype("U16"),
            "pre": e_pre, "post": e_post, "w": e_w.astype(np.float32)}


def summary(r):
    k = len(r["root_id"])
    ind = np.bincount(r["post"], minlength=k)
    outd = np.bincount(r["pre"], minlength=k)
    classes, counts = np.unique(r["cell_class"], return_counts=True)
    return {"neurons": k, "edges": int(r["pre"].size), "density": r["pre"].size / k / k,
            "self_loops": int((r["pre"] == r["post"]).sum()),
            "inhibitory_edge_fraction": float((r["w"] < 0).mean()),
            "in_degree_mean_max": [float(ind.mean()), int(ind.max())],
            "zero_in_degree": int((ind == 0).sum()), "zero_out_degree": int((outd == 0).sum()),
            "classes": {str(c): int(n) for c, n in zip(classes, counts)}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, default=Path.home() / "Projects/fly-criticality/data")
    ap.add_argument("--out", type=Path, default=ROOT / "data")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    files, ids, cls, pre, post, w = load(args.src)
    manifest = {"source": "FlyWire v783 via fly-criticality cache",
                "inputs": {k: {"path": str(p), "sha256": sha256(p)} for k, p in files.items()},
                "weights": "signed synapse count (sign from presynaptic neurotransmitter)",
                "licence": LICENCE, "regions": {}}
    for name, classes in REGIONS.items():
        r = cut(classes, ids, cls, pre, post, w)
        path = args.out / f"{name}_783.npz"
        np.savez_compressed(path, **r)
        manifest["regions"][name] = {"cell_classes": classes, "file": path.name,
                                     "sha256": sha256(path), **summary(r)}
        print(f"{name}: {manifest['regions'][name]['neurons']} neurons, "
              f"{manifest['regions'][name]['edges']} edges -> {path}")
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()

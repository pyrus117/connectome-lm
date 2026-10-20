import json
from pathlib import Path

import numpy as np

from clm.prep import EOT, split_of

DATA = Path(__file__).resolve().parent.parent / "data" / "fineweb"


def load(name):
    return np.memmap(DATA / name, dtype=np.uint16, mode="r")


def test_token_files_meet_the_budget_and_fit_the_vocabulary():
    m = json.loads((DATA / "manifest.json").read_text())
    train, val = load("train.bin"), load("val.bin")
    assert train.size >= 200_000_001 and val.size >= 2_097_153
    assert int(train.max()) < m["vocab"] and int(val.max()) < m["vocab"]


def test_train_and_validation_documents_are_disjoint_and_follow_the_rule():
    train_ids = (DATA / "train_ids.txt").read_text().split()
    val_ids = (DATA / "val_ids.txt").read_text().split()
    assert not set(train_ids) & set(val_ids)
    assert all(split_of(i) == "val" for i in val_ids)
    assert all(split_of(i) == "train" for i in train_ids[:20_000])


def test_every_document_ends_with_exactly_one_end_marker():
    from tokenizers import Tokenizer
    eot = Tokenizer.from_file(str(DATA / "tokenizer.json")).token_to_id(EOT)
    for name, ids_file in (("val.bin", "val_ids.txt"), ("train.bin", "train_ids.txt")):
        n_docs = len((DATA / ids_file).read_text().split())
        assert int((load(name) == eot).sum()) == n_docs, name

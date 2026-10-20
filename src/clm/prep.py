"""Split rule, tokenizer and encoding for the FineWeb-Edu token files.

scripts/prepare_data.py drives these over a downloaded shard.
"""
import hashlib

import numpy as np

EOT = "<|eot|>"


def split_of(doc_id):
    """1 in 100 documents is validation, chosen by a hash of its id, so the
    split never depends on file order."""
    return "val" if int(hashlib.sha256(doc_id.encode()).hexdigest()[:8], 16) % 100 == 0 else "train"


def train_tokenizer(texts, vocab):
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(vocab_size=vocab, special_tokens=[EOT],
                                  initial_alphabet=pre_tokenizers.ByteLevel.alphabet())
    tok.train_from_iterator(texts, trainer)
    return tok


def encode_docs(tok, texts):
    """Token ids for each text with the end-of-text id appended, as uint16."""
    if any(EOT in t for t in texts):
        raise ValueError("a document contains the end-of-text marker; filter it first")
    eot = tok.token_to_id(EOT)
    out = []
    for enc in tok.encode_batch(texts):
        ids = np.asarray(enc.ids + [eot], dtype=np.int64)
        if ids.max() >= 2**16:
            raise ValueError(f"token id {ids.max()} does not fit in uint16")
        out.append(ids.astype(np.uint16))
    return out

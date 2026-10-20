import numpy as np
import pytest

from clm.prep import EOT, encode_docs, split_of, train_tokenizer

TEXTS = ["Hello world.", " leading space", "Ünïcödé — and emoji 🙂", "tabs\tand\nnewlines"]


def test_split_sends_about_one_percent_to_validation_and_is_stable():
    ids = [f"<urn:uuid:{i}>" for i in range(20_000)]
    val = [i for i in ids if split_of(i) == "val"]
    assert 140 <= len(val) <= 260          # binomial(20000, 0.01): mean 200, sd 14
    assert val == [i for i in ids if split_of(i) == "val"]


@pytest.fixture(scope="module")
def tok():
    return train_tokenizer(iter(TEXTS * 50), 300)


def test_tokenizer_round_trips_each_document_and_marks_its_end(tok):
    eot = tok.token_to_id(EOT)
    for text, ids in zip(TEXTS, encode_docs(tok, TEXTS)):
        assert ids.dtype == np.uint16
        assert ids[-1] == eot and eot not in ids[:-1]
        assert tok.decode(ids[:-1].tolist()) == text


def test_a_document_containing_the_end_marker_is_refused(tok):
    with pytest.raises(ValueError, match="end-of-text"):
        encode_docs(tok, [f"a {EOT} b"])

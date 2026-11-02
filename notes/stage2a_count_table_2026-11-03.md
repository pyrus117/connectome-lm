# Stage 2A step 1: count corpus and count tables (2026-11-03)

Plan: `docs/superpowers/plans/2026-11-02-stage2-kenyon-memory.md` §7.2. Code: `src/clm/counts.py`, `scripts/build_count_corpus.py`, `scripts/build_count_table.py`, tests `tests/test_counts.py`, mutants M27–M36. Nothing here reads a validation token or any validation loss; the pure n-gram validation loss needs the COUNT read and is step 2.

## Conditions

Laptop CPU (16 threads), host RAM shared with the PolyStrata server (about 15 GB resident). Both jobs ran as systemd user units: corpus `MemoryMax=10G`, tables `MemoryMax=14G`, 24 partitions.

## Count corpus (`data/fineweb/count_corpus.bin`, manifest `count_corpus.json`)

- Every training-split document of shard 000 (same split rule, tokenizer and EOT filter as `prepare_data.py`): **881,048,239 tokens, 718,756 documents, 0 skipped**, EOT id 0 closing each document. That is 4.405 times the 200,000,154-token training stream. Validation documents are excluded.
- The first 200,000,154 tokens equal `data/fineweb/train.bin` exactly (checked before the manifest is written), so every training position's own occurrence is in the counts once. Leave-one-out subtraction of 1 is therefore exact.
- sha256 `11bd9c1f…c8ecaa8c07f` (full value in the manifest). Tokenising took 5 min wall.
- Correction to my own first run: the prefix check wrongly failed on the last chunk (it compared a corpus slice with a shorter `train.bin` slice). The data matched; the check was fixed and the corpus rebuilt.

## Tables (`data/count/order{2,3,4,5}/`, git-ignored, 5.2 GB)

A row is a history of k = n − 1 tokens: packed key, total count, up to 16 continuations with counts (highest count first, ties by lower token id). Histories never contain EOT; a continuation may. Budget 6 GiB in total; each order takes the smallest minimum history count `c_min` that fits its share (remaining budget ÷ orders left), so the choice uses size only.

| Order | History | c_min | Rows kept | Distinct histories | Entries | Size |
|---|---|---|---|---|---|---|
| 2 | 1 | 1 | 8,142 | 8,142 | 129,693 | 0.00 GiB |
| 3 | 2 | 1 | 12,973,613 | 12,973,613 | 53,711,310 | 0.49 GiB |
| 4 | 3 | 2 | 53,590,439 | 145,206,882 | 208,108,808 | 1.96 GiB |
| 5 | 4 | 2 | 85,100,508 | 403,107,781 | 254,928,682 | 2.69 GiB |

Total 5.15 GiB of the 6 GiB budget (0.85 GiB unspent; order 5 could not use it because c_min 1 would need far more). Build 24 min wall, 1 h 04 min CPU, peak 7.3 GB host RAM.

## Verification

- 22 unit tests against a brute-force counter (orders 2–5, several c_min, partitions and chunk sizes, EOT handling, top-16 truncation, profile vs build sizes, size-budget rule).
- Mutation audit: the first run left M28 (partition upper bound inclusive) and M30 (top-16 cut at 17) alive because the tests compared rows through a dict (hiding duplicate rows) and checked one row only. Both tests were strengthened; see the audit result in the commit message.
- Four random rows per order checked against a direct scan of all 881M tokens: 0 mismatches. Also asserted: keys strictly ascending, offsets end at the entry count, no history contains EOT, every `hcount` ≥ `c_min`.

## For step 2 (the COUNT read)

- Orders 4 and 5 drop histories seen once. With leave-one-out, a training position with C(h) = c_min is dropped too, as the plan says; for c_min = 2 that is every history seen twice or fewer.
- Truncation: after subtracting the position's own occurrence, a 17th continuation that would have entered the top 16 is not stored. Ignored, small.
- A position fewer than k tokens after a document start has no history of that order (the history would hold EOT); the read must back off to a lower order there.
- Whether a training window's first positions may use tokens before the window start is a read-side decision (the model cannot see them); COUNT uses token ids only.

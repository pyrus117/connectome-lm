# Laptop throughput for a standard small GPT and a host-RAM table, 2026-11-02

Planning measurement for Stage 2 (`docs/superpowers/plans/2026-11-02-stage2-kenyon-memory.md`). Measured with a scratch copy of `scripts/throughput_probe.py` (same model, settings and tables); rerun with `.venv/bin/python scripts/throughput_probe.py gpu` and `... cpu`.

Conditions: RTX 3050 Laptop GPU 4 GB (driver 615.78.08), AMD Ryzen 7 6800HS (8 cores, 16 threads), torch 2.15.0 (CUDA 13.4), numpy 2.5.4. `torch.cuda.set_per_process_memory_fraction(0.75, 0)`, `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`. GPU otherwise idle (EmbeddingGemma held 158 MiB). Model: pre-norm GPT, RMSNorm, fused-QKV causal SDPA, ReLU² MLP (4x), no biases, tied 8,192-token embedding, context 512, bf16 autocast, fused AdamW, no `torch.compile`. 5 warm-up steps, then the mean of 15 steps. Random tokens (throughput does not depend on content).

| d_model | layers | heads | non-embedding params | micro-batch | ms/step | tokens/s | 6N TFLOP/s | peak MiB | 200M tokens |
|---|---|---|---|---|---|---|---|---|---|
| 256 | 6 | 4 | 4.7M | 12 x 512 | 58 | 105,800 | 3.00 | 1,145 | 0.5 h |
| 384 | 6 | 6 | 10.6M | 8 x 512 | 64 | 64,100 | 4.09 | 1,046 | 0.9 h |
| 512 | 8 | 8 | 25.2M | 8 x 512 | 111 | 37,100 | 5.60 | 1,621 | 1.5 h |

The first attempt at micro-batch 32 x 512 (d 256) ran out of memory under the 0.75 cap: fp32 logits over 8,192 tokens cost 512 MiB per 16k tokens. Keep micro-batches at 8–12 x 512 and accumulate.

For comparison, the Stage 1 shared-brain model (5.11M params) trained at about 2,400 tokens/s on this laptop (6.7 s per 16,384-token step, `HANDOVER.md`), about 44 times slower than the 4.7M standard GPT above.

## Host-RAM table (CPU only)

Table 16,777,216 rows x 64 dims, fp16 (2.0 GiB). 16,384 tokens x 32 rows per token (524,288 row reads), mean of 3 calls.

| Threads | gather (`F.embedding_bag`, sum) | scatter-add (`index_add_`, fp16) |
|---|---|---|
| 8 | 4 ms (123M rows/s) | 57 ms (9M rows/s) |
| 16 | 4 ms (136M rows/s) | 58 ms (9M rows/s) |

Repeat with `scripts/throughput_probe.py`, minutes later: GPU rows within 0.2% (105.6k, 64.1k, 37.1k tokens/s; peaks 1,241, 1,110, 1,685 MiB, higher because the loss now runs outside autocast). Table: 8 threads gather 4 ms, scatter 36 ms; 16 threads gather 8 ms, scatter 97 ms. The CPU numbers are noisy under background load (the heretic-harness and EmbeddingGemma services were running); treat scatter as 36–97 ms.

The scatter-add did not speed up with threads; it is the cost to beat (sorting indices or an fp32 segment-sum may help). At 16,384 tokens per step the table adds about 60 ms to a 25M-model step of about 440 ms, before any overlap with GPU work. Host RAM at the time: 30 GiB total, 13 GiB available.

## Fly-hash address cost (added the same day)

`scripts/throughput_probe.py addr`: one n-gram order, 16,384 tokens, a 192-dim odour, a fixed binary expansion into 4,096 + 2,048 units (7 ones per unit), then top-16 per sub-codebook. Mean of 3 calls after 3 warm-up calls on the GPU. A first scratch run gave 180 ms (8 threads), 213 ms (16 threads) and 19.1 ms (GPU).

| Where | Time per micro-batch |
|---|---|
| CPU fp32, 8 threads | 174 ms |
| CPU fp32, 16 threads | 189 ms |
| GPU fp16, indices copied to host | 19.4 ms |

Two orders on the CPU (about 350 ms) would exceed a size-M step for the same tokens (about 256 ms), so runtime addresses belong on the GPU (about 39 ms for two orders, overlappable on a side stream). Offline table builds can stay on the CPU.

## Shard 000 size (added the same day)

`pyarrow` on `data/raw/000_00000.parquet` (2,152,819,114 bytes): 726,000 documents, 754,872,720 GPT-2 tokens (the shard's `token_count` column). Stage 1's training stream is 200,000,154 tokens from 163,015 documents (`data/fineweb/train_ids.txt`), 1,226.9 of our tokens per document. If the rest of the shard matches that rate, the shard holds about 0.89B of our tokens; this is an estimate, to be replaced by the real count when the count corpus is tokenised. Validation documents (1,702 in `val_ids.txt`) must never enter a count table.

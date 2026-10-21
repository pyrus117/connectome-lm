# Stage 1 pre-registration
Committed before any production run; its sha256 is recorded in every results.json.

## 1. Question
Does a transformer whose six layers share one recurrent brain wired like the Drosophila mushroom body (FlyWire v783, ALPN + KC + MBON + DAN + MBIN, 6,293 neurons, 570,682 synapses) reach lower validation loss than the same model with a degree-preserving shuffle of that wiring, at equal parameters, tokens, data order and initialisation?

## 2. Arms (seeds 0, 1, 2)
real; shuf (fresh degree-preserving shuffle per seed); er (Erdős–Rényi, same edge count); real_randio (real edges, random input/readout neurons); dense (fully connected brain, 760 neurons, matched parameters).

## 3. Fixed design
configs/stage1.json (micro 64, so no gradient accumulation at batch 64). Shared brain: one leaky step per layer (leak 0.5), state persists across layers within a token, RMS normalisation over all neurons after every step, no readout normalisation, 256 input and 256 readout neurons chosen by in-degree. Normalisation was chosen from a pilot on the er arm only (notes/measurements_2026-10-21.md). Reach gate: results/reach.json (100% in every arm and seed). 199,999,488 training tokens, one epoch (12,207 steps of 64 x 256 tokens). Primary metric: mean cross-entropy (nats/token) on the first 2,097,152 tokens of val.bin at the final step.

Data: FineWeb-Edu sample-10BT shard 000, own 8,192-token BPE. sha256: train.bin 7472ed04032886cfa0ea26d53268f6d103b964ec743b2166d42ddad258429fb0; val.bin 5d732f6c26e17befba64cdfea95ff4050c039b241227af543eb5899a80045ff0; train_ids.txt 830532a47b5a8313e18061e548dc3a3bbb68b5e01140a48ef0be438247f0086c; val_ids.txt dbe992d87fa575ae44bc537c8136dd2d7f927630e92f139b20af54d1c41c273d.

Hardware and software: one rented machine with two RTX 3090 (24 GB), one run per GPU at a time, runs drawn in seed-major order from a shared queue (scripts/run_matrix.py --gpus 2). Python 3.13.16, torch 2.14.1+cu130, numpy 2.5.4, scipy 1.18.1, tokenizers 0.23.3, pyarrow 26.0.0. Analysis runs in this same environment.

## 4. Predictions
fly-criticality E5/E6 predicts real < shuf. arXiv 2604.04033, conn2res (fruit-fly exception) and FlyGPT (snippet: no detectable difference) predict no meaningful difference. Both outcomes are reported.

## 5. Decision rules (src/clm/analyze.py, SESOI = 0.01 nats/token)
Paired by seed, d_s = loss_x − loss_y. X_BETTER: mean(d) ≤ −0.01 and every d_s < 0. X_WORSE: mean(d) ≥ +0.01 and every d_s > 0. NO_MEANINGFUL_DIFFERENCE: |mean(d)| < 0.01 and every |d_s| < 0.02. INCONCLUSIVE: anything else. INVALID: any run of either arm missing, diverged, non-finite, short of tokens, or with a wrong wiring fingerprint. Contrasts: primary real vs shuf; degree_stats shuf vs er; io_placement real vs real_randio; vs_dense real vs dense. Only the primary contrast decides the next stage.

## 6. What happens next
primary X_BETTER: proceed to larger models with the same rules. NO_MEANINGFUL_DIFFERENCE or X_WORSE: write up the null; optionally repeat once on data/cx_783.npz. INCONCLUSIVE: one extension only, seeds 3 and 4 for real and shuf, same rules on 5 pairs.

## 7. Conduct
No validation loss is read before all 15 runs finish. A diverged run is rerun once from scratch with the same seed; a second divergence makes it INVALID. No change to code, config or data after this file is committed, except crash fixes, each logged in notes/deviations.md with the commit hash. A run interrupted by a machine fault resumes from its last checkpoint; this is not a deviation.

## 8. Limits stated in advance
Three seeds give a screening result (best sign test on 3 pairs: p = 0.125). The brain arms cost about 40x the step time of a standard FFN; comparisons are at equal parameters, not equal compute. GPU sparse backward may not be bitwise reproducible.

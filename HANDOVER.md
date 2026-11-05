# HANDOVER — connectome-lm, Stage 1

Read this first. Updated 2026-10-21 (end of the first build session). Branch `stage1`; no git remote.

> **Status 2026-10-23: Stage 1 is finished.** All 15 runs completed on a rented 2x RTX 3090 (Vast, about US$23, 43 h; instance destroyed). Primary verdict **X_WORSE**: real − shuf = +0.0126, +0.0147, +0.0160 nats/token (`runs/analysis.json`). Secondary contrasts INCONCLUSIVE because er-1, real_randio-1 and dense-1 had finite gradient spikes (max grad norm 83, 38, 283); real-1 and shuf-1 were clean. Checkpoints and results for all 15 runs are in `runs/` (git-ignored, sha256-verified against the pod). Run log: `notes/live_run.md`. "Waiting on Nate" below is history.
>
> **Next: vocabulary-list pilot**, a brain-only memory task on `er-0`. Plan: `docs/superpowers/plans/2026-10-23-vocab-list-pilot.md`, not started. Agreed split: Haiku subagent for Task 1, Sonnet for Task 2, main session for Tasks 3–5 (GPU smoke run and pilot decisions).
>
> **Status 2026-11-02 (evening): Stage 2 approved and under way on branch `stage2`.** Plan: `docs/superpowers/plans/2026-11-02-stage2-kenyon-memory.md` (Nate answered §14 the same day), a counted host-RAM n-gram memory (COUNT) with a fly-hash similarity backoff for unseen histories, plus tests of Litwin-Kumar's dimension theory and DeepSeek's effort law. Measurements behind it: `notes/laptop_throughput_2026-11-02.md`, `notes/ngram_stats_2026-11-02.md`, `notes/lacuna_check_2026-11-02.md`. Literature: neural vault, `research-logs/2026-11-02 *`.
> - **Stage 2T done** (`notes/stage2t_2026-11-02.md`): P1 refuted (activity dimension real ÷ shuf about 1.0 despite half the rank), P2 not supported. Unregistered: only 19.5% of the real wiring's readout neurons are one hop from an input (shuffles about 95%), so its readout is one brain step late.
> - **Stage 2.0 done** 2026-11-03 06:51 (`notes/stage2_0_2026-11-03.md`, `results/stage2_0.json`, runs in `runs2/`): pilot chose Muon lr 8e-3. Final loss S 3.520, M 3.404, L 3.261 (3 seeds each, all clean). **SESOI 0.01 nats** (2 x SD of B0-M = 0.0054, so the floor binds). Ladder L = 8.031 x N^-0.0524; a 0.05-nat gain at M is worth about 1.33x model size. B0-L seed SD is 0.0099, so comparisons at L need more seeds.
> - **Stage 2A step 1 done** 2026-11-03 (`notes/stage2a_count_table_2026-11-03.md`): count corpus of 881,048,239 train-split tokens (4.405x the training stream, its first 200,000,154 tokens equal `train.bin`) and order 2–5 tables in `data/count/` (5.15 GiB, c_min 1, 1, 2, 2; git-ignored). Code `src/clm/counts.py`, mutants M27–M36 (36/36 killed). No validation loss read.
> - **Stage 2A done** 2026-11-03 (`notes/stage2a_result_2026-11-03.md`, `results/stage2a.json`): COUNT-M beats B0-M at every seed, mean gap -0.0798 nats (X_BETTER), 4.4% slower per step, about a 1.57x model-size equivalent. Gate 1 passes. The COUNT learning rate (0.032) sat at the edge of its pilot grid. Leave-one-out is essential (without it validation is 0.84 nats worse).
> - **Stage 2B-0 done** (`notes/stage2b0_result_2026-11-03.md`): the Kenyon backoff helps offline (0.0116 nats overall, random buckets 0.0001) but gate 2 FAILED narrowly (0.0151 and 0.0130 against 0.02). The selected setting is K31 (the sweep's winner does not fit in RAM; `notes/deviations.md`).
> - **Stage 2D done** (`notes/stage2d_result_2026-11-04.md`, `results/stage2d.json`): looped model, best fixed depth 4 (3.3705 against B0-M 3.4041), P9 supported, RL gate OPEN. Nate paused 2D on 2026-11-03: no RL work started.
> - **Stage 2C done** (`notes/stage2c_result_2026-11-04.md`): gradient-free writes forget nothing (P7 supported), consolidation fails as registered (0.061 against 0.760; undiagnosed, includes an unexplained memory-on score), Kenyon twin transfer 0.030 (P8 supported by the rule, small). The consolidation failure is now diagnosed: teacher drift, not head drift (`notes/stage2c2_result_2026-11-04.md`).
> - **Stage 2B done** 2026-11-04 08:21 (run at Nate's request despite the failed gate 2; `PREREG_stage2b.md` amendment, `notes/deviations.md`; `notes/stage2b_result_2026-11-04.md`, `results/stage2b.json`): NO_MEANINGFUL_DIFFERENCE. Mean loss against COUNT: Kenyon +0.0046, random buckets +0.0040 (both slightly worse), Kenyon against random +0.0007. P4 not supported: in the "not stored" class both bucket arms are worse than COUNT (+0.0085, +0.0073). The offline gain of 2B-0 did not carry into the network. Tables in `data/count_kenyon/` and `data/count_randbucket/` (git-ignored).
> - **Gated COUNT read built and pre-registered as Stage 2E** 2026-11-04 (`PREREG_stage2e.md`; branch `count-gate`): a gate per order, per source (exact or bucket) and per count bin, plus a learned absent vector per order, starting bit-identical to plain COUNT. Run and finished, see the next entry. Design and the Avenue B (multi-probe cells) assessment, not built: `notes/count_gate_design_2026-11-04.md`.
> - **Stage 2E done** 2026-11-04 20:35 (`notes/stage2e_result_2026-11-04.md`, `results/stage2e.json`): the gated COUNT read beats COUNT, X_BETTER, mean -0.0111 nats (P10 supported; clears the 0.01 bar by 0.0011, seed 0 alone is -0.0075). P10a and P10b not supported (gains in "not stored" -0.0083 and ">=1000" -0.0084, short of -0.01); the gain is in every class, largest in 10-99 and 100-999. P11 not supported: Kenyon-gated is +0.0035 worse than gated, so the bucket line is closed at this size and Avenue B is not built. Random-gated arm and seeds 3-4 not needed. Exact gates grew to 13-90, so scale is confounded with structure (limit 1). One run (gated seed 2) resumed from its step-9000 checkpoint after a forced restart (`notes/deviations.md`). **Next per PREREG_stage2e section 6: the scale-only control** (one global scalar gate, no per-order or per-bin split, no absent vector), needs its own pre-registration; then Nate's decision on gate 3.
> - **Stage 2F done** 2026-11-05 05:43 (`PREREG_stage2f.md`, `notes/stage2f_result_2026-11-05.md`, `results/stage2f.json`; audit follow-up 3A): separate rates for Muon and AdamW, **P12 supported, X_BETTER, mean -0.0614 nats** (seeds -0.086, -0.036, -0.062) at Muon 8e-3 and AdamW 64e-3, against COUNT at the shared 0.032. The AdamW rate drives it. A B0 pilot at the same cell gained 0.0240, so **gate 1's COUNT-over-B0 gap (-0.0798) needs rechecking at 200M tokens with B0 on separate rates**, and the Stage 2E gate result and the scale-only control should be rerun or repeated at the new recipe (all of 2A-2E used the shared rate). New setting `--adam-mult`; analysis crash fix 45c17a5 (`notes/deviations.md`).
> - **Stage 2G done** 2026-11-05 13:05 (`PREREG_stage2g.md`, `notes/stage2g_result_2026-11-05.md`, `results/stage2g.json`): gate 1 again with B0 on separate rates too (pilot chose the same cell, Muon 8e-3, AdamW 64e-3). **COUNT-M beats B0-M by -0.1224 nats, X_BETTER, every seed** (Stage 2A shared-rate gap was -0.0798). B0 gained -0.0188 from separate rates, COUNT -0.0614. Gate 1 holds; the memory track stands. COUNT-M separate (mean 3.2629) is slightly below B0-L shared-rate (3.261), not like-for-like.
> - Not started: the gated read and scale-only control at the separate-rate recipe (each needs a pre-registration; the 2E gated arm trains its gates under AdamW), size-L check (about 12-19 GPU-hours; needs Nate's go-ahead and a cost estimate), the RL exit policy (2D gate open, paused by Nate).
> - Measured B0 speed with the full recipe: M 43.7k tokens/s (0.375 s/step), L 26.6k; peak VRAM 2.4 GB at L.

## The question

Does a transformer whose layers share one recurrent "brain" wired like the real *Drosophila* mushroom body (FlyWire v783: ALPN + KC + MBON + DAN + MBIN, 6,293 neurons, 570,682 synapses) reach lower validation loss than the same model with a **degree-preserving shuffle** of that wiring, at equal parameters, tokens, data order and initialisation?

## The design as built (supersedes the per-layer-block design in the plan)

- 6-layer pre-norm GPT, d_model 256, 4 heads, ctx 256, vocab 8192 (own BPE), tied embeddings. About 5.11M parameters.
- **No FFNs.** One `SharedBrain` (`src/clm/block.py`) is used by every layer. Per token the brain state starts empty. Layer *l*: attention, then `w_in` writes into 256 input neurons, the brain takes **one** leaky step (leak 0.5, GELU, RMS-normalised over all neurons, no learnable gain), `w_out` reads 256 readout neurons back into the residual stream. The state persists across layers within a token; tokens never share brain state. The synapse strengths are shared by all layers; `w_in` and `w_out` are per layer.
- Input neurons = 256 lowest in-degree; readout = 256 highest in-degree. On the real graph these are 249 ALPN (olfactory input) and mostly DAN/MBON (reward/output).
- **Arms** (`src/clm/graphs.py`): `real`; `shuf` (degree-preserving shuffle, fresh per seed, Maslov–Sneppen repair); `er` (Erdős–Rényi, same edge count); `real_randio` (real edges, random input/readout); `dense` (fully connected brain of 760 neurons, 577,600 params vs 576,975). Seeds 0, 1, 2. 15 runs.
- Fairness: every tensor shared between arms starts bit-identical (name-keyed generators); one fan-in init rule for the brain; same data order per seed (one epoch, no repeats); reach gate shows 100% of edges get a gradient at the first step in every arm and seed (`results/reach.json`).

## Decisions and why

| Decision | Why | Evidence |
|---|---|---|
| One shared, persistent brain (Nate, 2026-10-21) | Every layer touches the brain at the cost of the old cheap 2-layer layout; closer to a real fly | design discussion |
| Per-step normalisation ON (Nate) | Fair start (100% edges active) and much faster learning | pilot on the ER arm: val 6.641 on vs 7.002 off after 80 steps |
| No readout normalisation | It cancels the last step's norm and idles about 38% of real edges at the start | `notes/measurements_2026-10-21.md`, design C |
| Pilots only on the `er` arm | Design choices must not be tuned toward real vs shuffled | — |
| Dense control = fully connected brain | Same code path, matched parameters | `tests/test_graphs.py` |

## State

- **114 tests pass** (`.venv/bin/python -m pytest tests`). **Mutation audit passes**: 16/16 planted bugs killed by assertions, both harness controls as required (`scripts/mutation_audit.py`, `notes/mutation_audit.md`).
- Data built: `data/fineweb/{train,val}.bin` (200,000,154 / 2,098,135 tokens, FineWeb-Edu sample-10BT shard 000). Token files are git-ignored. sha256: train.bin `7472ed04032886cfa0ea26d53268f6d103b964ec743b2166d42ddad258429fb0`, val.bin `5d732f6c26e17befba64cdfea95ff4050c039b241227af543eb5899a80045ff0`, train_ids.txt `830532a4…0086c`, val_ids.txt `dbe992d8…c273d`, tokenizer.json `6c93afdc…c1cc46` (committed).
- Laptop speed (RTX 3050 4 GB, micro 4): 6.7 s/step for sparse arms, 4.6 s/step for `dense`; about 23 h per run, 318 h for all 15. `results/speed.json` holds this laptop result with `fits: false`.
- **Not done:** `configs/stage1.json` (frozen config), `PREREG.md`, the 15 production runs, the analysis.
- Pilot runs live in `runs/` (git-ignored): `pilot-norm-on`, `pilot-norm-off`, `pilot-standard-1024`, `pilot-standard-442`. Table in `notes/measurements_2026-10-21.md`.

## Waiting on Nate

1. **GPU rental and budget.** Recommended: one RunPod Community Cloud RTX 3090 (about US$0.22–0.50/h; estimate 4–6x the laptop, about 58–86 h and US$13–43 for all 15 runs). Spending needs his explicit approval.
2. **Confirm or change the assumed values** before PREREG.md is committed: SESOI 0.01 nats/token, leak 0.5, k_io 256, the 90% reach gate, LR 1e-3 / warmup 500, three seeds (a screening result: best possible sign test on 3 pairs is p = 0.125).

## Runbook for a rented GPU

Pods usually have no systemd user session and an older Python. Use `tmux` or `nohup`, and a fresh uv venv (not `--system-site-packages`).

```bash
# laptop: package the repo and the token files
git -C /home/nate/Projects/connectome-lm bundle create /tmp/clm.bundle --all
scp /tmp/clm.bundle <pod>:~ && scp /home/nate/Projects/connectome-lm/data/fineweb/{train.bin,val.bin,train_ids.txt,val_ids.txt} <pod>:~/

# pod
git clone -b stage1 ~/clm.bundle ~/connectome-lm && cd ~/connectome-lm
mkdir -p data/fineweb && mv ~/{train.bin,val.bin,train_ids.txt,val_ids.txt} data/fineweb/
sha256sum data/fineweb/*.bin          # must match the hashes above
uv venv --python 3.13 .venv && uv pip install --python .venv/bin/python torch numpy scipy pytest tokenizers pyarrow && uv pip install --python .venv/bin/python -e .
.venv/bin/python -c "import torch, numpy; print(torch.__version__, torch.cuda.is_available(), numpy.__version__)"
.venv/bin/python -m pytest tests                 # all must pass on the pod
.venv/bin/python scripts/mutation_audit.py       # optional, about 5 min
.venv/bin/python scripts/check_reach.py          # must print pass
.venv/bin/python scripts/speed_test.py --budget-hours <Nate's budget / hourly rate>
.venv/bin/python scripts/freeze_config.py        # writes configs/stage1.json
# write PREREG.md (draft below), commit, then:
tmux new -s stage1 '.venv/bin/python scripts/run_matrix.py 2>&1 | tee runs/matrix.log'
# after all 15 finish, once:
.venv/bin/python -m clm.analyze --runs runs --config configs/stage1.json
```

Run the analysis in the **same environment** as training: `analyze` recomputes each wiring fingerprint, and numpy does not promise identical random streams across versions. Record the pod's library versions in the commit that adds the results (laptop: python 3.14.7, numpy 2.5.3, torch 2.14.0, scipy 1.18.1, tokenizers 0.23.3, pyarrow 26.0.0). Copy `runs/*/results.json`, `runs/*/log.jsonl` and `runs/analysis.json` back; checkpoints can stay.

## PREREG.md draft (commit before the first production run; fill `<micro>` and the hashes)

```markdown
# Stage 1 pre-registration
Committed before any production run; its sha256 is recorded in every results.json.

## 1. Question
Does a transformer whose six layers share one recurrent brain wired like the Drosophila mushroom body (FlyWire v783, ALPN + KC + MBON + DAN + MBIN, 6,293 neurons, 570,682 synapses) reach lower validation loss than the same model with a degree-preserving shuffle of that wiring, at equal parameters, tokens, data order and initialisation?

## 2. Arms (seeds 0, 1, 2)
real; shuf (fresh degree-preserving shuffle per seed); er (Erdős–Rényi, same edge count); real_randio (real edges, random input/readout neurons); dense (fully connected brain, 760 neurons, matched parameters).

## 3. Fixed design
configs/stage1.json (micro <micro>). Shared brain: one leaky step per layer (leak 0.5), state persists across layers within a token, RMS normalisation over all neurons after every step, no readout normalisation, 256 input and 256 readout neurons chosen by in-degree. Normalisation was chosen from a pilot on the er arm only (notes/measurements_2026-10-21.md). Reach gate: results/reach.json (100% in every arm and seed). 199,999,488 training tokens, one epoch. Primary metric: mean cross-entropy (nats/token) on the first 2,097,152 tokens of val.bin at the final step.

## 4. Predictions
fly-criticality E5/E6 predicts real < shuf. arXiv 2604.04033, conn2res (fruit-fly exception) and FlyGPT (snippet: no detectable difference) predict no meaningful difference. Both outcomes are reported.

## 5. Decision rules (src/clm/analyze.py, SESOI = 0.01 nats/token)
Paired by seed, d_s = loss_x − loss_y. X_BETTER: mean(d) ≤ −0.01 and every d_s < 0. X_WORSE: mean(d) ≥ +0.01 and every d_s > 0. NO_MEANINGFUL_DIFFERENCE: |mean(d)| < 0.01 and every |d_s| < 0.02. INCONCLUSIVE: anything else. INVALID: any run of either arm missing, diverged, non-finite, short of tokens, or with a wrong wiring fingerprint. Contrasts: primary real vs shuf; degree_stats shuf vs er; io_placement real vs real_randio; vs_dense real vs dense. Only the primary contrast decides the next stage.

## 6. What happens next
primary X_BETTER: proceed to larger models with the same rules. NO_MEANINGFUL_DIFFERENCE or X_WORSE: write up the null; optionally repeat once on data/cx_783.npz. INCONCLUSIVE: one extension only, seeds 3 and 4 for real and shuf, same rules on 5 pairs.

## 7. Conduct
No validation loss is read before all 15 runs finish. A diverged run is rerun once from scratch with the same seed; a second divergence makes it INVALID. No change to code, config or data after this file is committed, except crash fixes, each logged in notes/deviations.md with the commit hash.

## 8. Limits stated in advance
Three seeds give a screening result (best sign test on 3 pairs: p = 0.125). The brain arms cost about 40x the step time of a standard FFN; comparisons are at equal parameters, not equal compute. GPU sparse backward may not be bitwise reproducible.
```

## Traps already hit (do not rediscover)

- uv picks its own Python unless told: on the laptop use `--python /usr/bin/python3` (system torch is CUDA 3.14).
- torch sparse matmul has no bf16 kernel; CSR backward densifies (606 MiB); `set_per_process_memory_fraction` needs an index (`0`), not `"cuda"`.
- Random re-permutation never converges as a shuffle repair on this graph (hubs take input from half of all neurons); targeted swaps converge in 0.9 s.
- float32 GELU is exactly 0 below about z = −5.3, so a reached neuron can read 0 on some tokens.
- The laptop's internal panel link failed once while a test filled all 4 GB of the GPU (machine power-cycled). Cap memory (`--mem-fraction 0.75`) and start at micro 4 on the laptop.
- The harness kills background jobs after 30 min: use `systemd-run --user` on the laptop.
- EmbeddingGemma 2 (`embeddinggemma.service`, port 8090) holds about 310 MB of VRAM; Nate allows stopping it temporarily. Restart it afterwards and check it answers.

## Where things are

`docs/superpowers/plans/2026-10-21-stage1-10m.md` (original plan; see its status note) · `notes/` (measurements, both novelty checks, mutation audit) · `results/` (reach and speed) · `scripts/` (data, gates, speed, freeze, matrix, audit, standard-GPT pilot) · `src/clm/` (graphs, block, model, data, prep, train, analyze).

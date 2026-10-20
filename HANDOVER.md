# HANDOVER — connectome-lm, Stage 1

Read this first. Updated 2026-10-21 (end of the first build session). Branch `stage1`; no git remote.

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

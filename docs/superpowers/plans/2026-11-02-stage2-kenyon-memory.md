# Stage 2 research plan: a counted host-RAM memory with a mushroom-body similarity backoff

> **Status 2026-11-02: proposal.** Nothing is built or run. Stage 2T can start at once (it trains nothing); everything else waits on Nate's answers in §14. Stage 1 code, config, data and results stay untouched (its `PREREG.md` still governs them). Each stage gets its own task-by-task implementation plan and its own pre-registration before any production run.

## 1. The bet

On this laptop VRAM is the hard wall (4 GB) and host RAM is plentiful (31 GB). The strongest efficiency lever found for exactly that situation is **COUNT** (Magnusson, Smith & Koh, arXiv 2610.20456, Oct 2026): store n-gram continuation counts in host RAM and feed the model, as an extra input, its own unembedding rows weighted by those counts. COUNT adds no learned parameters, so it needs no GPU memory and no table updates. Across the authors' 60M–1B ladder a standard transformer needed 46–89% more compute to match COUNT's language-modelling loss and 6–56% more to match its downstream task loss (their §1). DeepSeek-V4.1-Flash makes the same bet at frontier scale with Engram, a learned n-gram memory of 196B parameters prefetched from host memory.

Both systems index the **exact** n-gram. An exact history with no counts gives them nothing but a fallback to a shorter history. In our data that is the common case: **52% of validation 4-token histories and 20% of 3-token histories never occur in the 200M training tokens**, and a further 25% and 23% occur fewer than 10 times (`notes/ngram_stats_2026-11-02.md`). (A history of h tokens predicts the next token, so it belongs to an n-gram of order h + 1; this plan counts histories in tokens.)

The fruit-fly mushroom body solves that problem for odours. Fixed receptor codes, a random expansion into Kenyon cells and k-winners-take-all under global inhibition form a **similarity-preserving hash** (Dasgupta et al. 2017): a new odour lands on cells that similar odours trained. **Stage 2 tests a "Kenyon backoff":** when a history has no counts, read the pooled counts of histories that share its fly-hash bucket, built over fixed distributional token codes. It is the 1990s idea of similarity-based smoothing (Dagan, Pereira & Lee 1994; Dagan, Lee & Pereira 1998), implemented with the fly's hash inside a modern counted memory, and it gets a cheap CPU-only test before any GPU run.

Around that core, the plan tests three theories cheaply: Litwin-Kumar's dimension theory (Stage 1's loss order, and the hash design), DeepSeek's effort law on latent depth (no RL needed to test it), and complementary learning systems (fast count writes plus replay).

## 2. What "more efficient" means here

- **Headline:** validation loss at equal wall-clock time and equal peak VRAM on this laptop, against a strong standard transformer. Reported with it: loss at equal tokens, and the **compute multiplier** (how much more training compute the standard model needs to match, read off a three-size ladder).
- **What a win shows:** that the lever pays at 5–25M backbone parameters on this hardware. It cannot show that it beats frontier systems; the ladder's slope (does the multiplier grow with size?) is the only scale evidence this machine can produce.

## 3. Measured budget on this machine (2026-11-02)

From `notes/laptop_throughput_2026-11-02.md` (bf16, no `torch.compile`, GPU memory cap 0.75, context 512, vocabulary 8,192):

| Backbone (non-embedding) | tokens/s | 200M tokens |
|---|---|---|
| S: d 256, 6 layers (4.7M) | 105,800 | 0.5 h |
| M: d 384, 6 layers (10.6M) | 64,100 | 0.9 h |
| L: d 512, 8 layers (25.2M) | 37,100 | 1.5 h |

- Host RAM serves sparse row gathers fast (524,288 reads from a 2 GiB table in 4–8 ms). Scatter-adds (learned tables) are the slow part (36–97 ms); a counted memory has none.
- Fly-hash addresses for 16,384 tokens: 19 ms on the GPU against 174–189 ms on the CPU (one n-gram order). **Runtime addresses are computed on the GPU; offline table builds run on the CPU.**
- The Stage 1 shared-brain model trained at about 2,400 tokens/s, about 44 times slower than a standard model of its size.
- **Data on disk:** shard 000 holds 726,000 documents and 754.9M GPT-2 tokens, about 0.89B tokens in our tokenizer; Stage 1 used 200M of them for training. The count corpus can be every training-split document of the shard (about 4.4 times the training stream) with no download.

## 4. What our own research says

Sources are neural-vault findings (`/home/nate/Projects/neural-vault/findings/`) unless a repo path is given.

| # | Lesson | Evidence |
|---|---|---|
| L1 | The fly's **wiring diagram** does not help. | connectome-lm Stage 1: real minus shuffled +0.0126, +0.0147, +0.0160 nats (`runs/analysis.json`); fly-criticality E10. Externally confirmed: Senge 2026 (arXiv 2610.09114) finds four connectomes' measured wiring gives no retrieval advantage over degree-preserving rewiring in the fly hash (median −1.6%). |
| L2 | **k-winners-take-all sparse coding** is the fly ingredient that pays, as a code; sparse activations do not lower training loss. | tinyworld X1: k-WTA +0.089 over dense. Literature: Q-Sparse, Turbo Sparse, Spark Transformer show parity or inference gains only. |
| L3 | Concentrated fan-in collapses dimension; lower rank went with worse loss. Theory predicts Stage 1's order. | critters S2: participation ratio 1.2 vs 17.0. connectome-lm: real wiring has half the shuffles' effective rank. Litwin-Kumar 2017: supervised plasticity favours dense wiring. Stage 1 losses (seeds 0 and 2): dense 3.894 < er 3.921 < real_randio 3.930 ≈ shuf 3.933 < real 3.947. |
| L4 | Local learning rules lose to backprop as training lengthens. | tinyworld X12 (PPO 0.839 vs MB agent 0.696 at 400k steps); critters S3. Literature: direct feedback alignment on a 6-layer transformer, WikiText-103 perplexity 52.0 vs 29.8–34.4 for backprop (Launay 2020). |
| L5 | A raw Hebbian write fails on shared inputs; an error-correcting write works. | tinyworld, critters, Bennett et al. 2021. |
| L6 | Sparsity savings count only in wall-clock. | tinyworld X9; §3. |
| L7 | A compute price can shrink compute without improving the result. | tinyworld "Pricing compute never buys efficiency". |
| L8 | Replay consolidates; a lead must survive longer training. | critters M4b (+4.82 pushes/min, n 20); tinyworld X12. |

**Design rule:** use the fly's algorithm (fixed codes, random expansion, k-WTA, a readout) as an addition to a backprop-trained backbone; never its literal wiring; never a local rule as the main learner.

## 5. What the literature says (2026-11-02 pass)

Three search agents (Lacuna plus web) and the main session. Read levels: **full** = full text; **part** = named sections; **abs** = abstract; **Lac** = Lacuna's machine summary only. Agent reports are in the vault (`research-logs/2026-11-02 Stage 2 literature *.md`).

**The memory axis at small scale**

| Source | Read | Result | Use here |
|---|---|---|---|
| COUNT, Magnusson et al. 2026, arXiv 2610.20456 | abs + §1–3 (main session) | Counted top-K continuations per history (orders ≤ 6, stupid backoff), read as count-weighted unembedding rows, added to the input with "embedding amplification" (√d LayerNorm). A standard transformer needs 46–89% more compute to match its LM loss and 6–56% to match its task loss (60M–1B). 60M–190M backbones used a 17.1 GB table. Beats learned n-gram tables when GPU memory is limited. | The base memory (Stage 2A). |
| Engram, Cheng et al. 2026, arXiv 2601.07372 | full (agent) | U-shaped allocation (20–25% of sparse budget to memory). Smallest model 0.56B active. Best single layer 2. 100B host table costs −2.8% tokens/s. | Confirms host tables work; no evidence below 0.5B. |
| modded-nanogpt PR 201; nanochat log 2026-01-27/28 | full (agent) | Learned bigram hash table: −9% steps, −5.3% time at 124M (bundled with two small fixes). At d25 the gain vanished on wall-clock and cost VRAM. Engram's gate and trigrams did not help at this scale. | A learned table is a weak lever here; counts avoid its costs. |
| Hash Layers, Roller et al. 2021, arXiv 2106.04426 | abs + table (agent) | Balanced random hash 23.16 ppl, token clustering 23.90 (222M). | Evidence **against** similarity addressing for learned parameters. |
| HashFormers, Xue & Aletras 2022, arXiv 2210.07904 | abs + results (agent) | With full tables random hashing matched or beat LSH; LSH won only when the table was compressed. | Predicts our niche: similarity helps where data are thin. |
| N-Grammer, Roy et al. 2022, arXiv 2207.06366 | part (agent) | Learned (k-means) similarity codes address a bigram table; beats Transformer and Primer on C4. | The closest positive precedent. |
| Dagan, Lee & Pereira 1998 (arXiv cs/9809110); Dagan, Pereira & Lee 1994 (ACL) | abs (1998, main session); title (1994) | Similarity-based estimates for unseen bigrams inside a back-off language model: 20% lower perplexity on unseen bigrams, and fewer speech-recognition errors. | The classical idea we re-implement, and the effect size to beat offline. |

**Adaptive depth and RL**

| Source | Read | Result | Use here |
|---|---|---|---|
| Schwethelm 2026 (2604.21106); Xiong 2026 (2610.26030); Ekbote 2026 (2610.40325) | abs + parts (agent) | Looped blocks lose loss per FLOP (recurrence-equivalence 0.46) unless over-trained (2 loops at 80 tokens/param match dense at 160). | Looping is not a training-efficiency lever at 20 tokens/param. |
| LoopRPT, Tang 2026 (2603.19714); Ouro, Zhu 2025 (2510.25741); LoopFormer, Jeddi 2026 (2602.11451) | part (agent) | Plain output GRPO collapses a looped LM (GSM8K 78.17 → 4.17); exit-policy RL with a time penalty cut steps 3.51 → 2.28 at +2.89 accuracy (2.6B). Loss-only depth weights collapse to the last step; an entropy term fixes it. Budget-conditioned schedules without RL. | Closest prior art for the effort track. |
| Clark et al. 2022 (2202.01169); Jolicoeur-Martineau 2025 TRM (2510.04871) | part (agent) | REINFORCE routers lose to balanced-assignment routing (S-BASE) at every size; RL (Q-learning) halting was dropped for a supervised one at 5–7M. | RL does not buy training efficiency here. |
| DeepSeek-V4.1-Flash, §5.1.4 and App. C | full (main session) | Effort b, groups per (prompt, b), price k(b) = k0·exp(−(b − b_min)/τ); affine length-in-b derived from exponentially decaying marginal benefit, explicitly "a local reward-level approximation". | The law Stage 2D tests on latent depth. |

**Functions and theory**

| Source | Read | Result | Use here |
|---|---|---|---|
| Litwin-Kumar et al. 2017, Neuron | full (vault) | Dimension of a random expansion vs in-degree K; global inhibition removes the peak (95% of maximum near 3% connectivity at N = 1,000); expansion beyond 10–50x adds little; supervised input plasticity favours dense wiring. | Predicts the hash design and Stage 1's order. |
| Clark et al. 2024 (2409.01969) | full (agent) | Activity dimension scales with connectivity effective rank. | Stage 2T prediction. |
| Wen et al. 2025 (2509.02046); Semenov et al. 2025 (2509.01440); modded-nanogpt README | full (agent) | LR tuning alone gave 2x at 130M; matrix optimizers 1.4x at 0.1B; at 16k tokens/step basic Muon lost to AdamW, Muon with weight decay ("D-Muon") was best at 1B tokens. Speedrun architecture bundle: QK-norm, ReLU², untied head, value embeddings, softcap. | The baseline recipe. |
| Khalfaoui-Hassani & Kesselheim 2026 (2502.01247); Zhuo et al. 2025 PolyCom (2411.03884) | full (agent) | Hermite-3 −0.029 nats vs GELU at 124M (5 seeds); PolyReLU −0.02 vs SwiGLU at 1.3B. Neither tested against ReLU² under Muon. | Optional activation test. |
| Marek 2026 (2605.26097); Ibrahim et al. 2024 (2403.08763) | abs / Lac (agent) | Self-generated replay with a KL penalty stops forgetting (sweeps to 46M); re-warm, re-decay and replay match retraining. | Consolidation method for Stage 2C. |

## 6. What we take from DeepSeek-V4.1-Flash, and what we leave

| Component | Decision | Why |
|---|---|---|
| Engram's principle: memory addressed by token ids, so it can live in host RAM and be prefetched | **Adopt**, in COUNT's counted form | It moves parameters out of VRAM. Counts avoid the learned table's scatter cost and optimizer state. |
| Auxiliary-loss-free load balancing (a bias nudged toward equal load) | **Adopt** as the Kenyon cells' homeostatic thresholds | The same rule as intrinsic plasticity toward a target firing rate. |
| Effort-conditioned price k(b) and its marginal-utility law (App. C) | **Adopt** for latent depth (Stage 2D) | A falsifiable law nobody has tested on hidden compute. |
| Muon with weight decay; momentum-only updates | **Adopt** in the baseline sweep | Optimizer choice is the largest free gain at this scale. |
| Engram's context gate, CED, CSA2, sparse indexers, SWA replay, FP4 KV cache, DSpark, MoE, mHC, model merging | Leave | The gate failed at 124M in two independent logs; the rest cuts long-context serving cost or needs depth and scale we do not have. |

## 7. The architecture

### 7.1 Backbone (B0)

Pre-norm decoder, context 512, the Stage 1 tokenizer (8,192 tokens; a small vocabulary keeps the head cheap). From the speedrun: RMSNorm without gain, RoPE, QK-norm, ReLU² MLP, zero-initialised output projections, no biases, untied head, value embeddings, logit softcap. bf16, micro-batch 8 x 512 with accumulation to about 16k tokens per step. Optimizer chosen in Stage 2.0 between tuned AdamW and Muon with weight decay; learning rate decayed to zero. Sizes S, M, L as in §3.

### 7.2 Counted memory (COUNT, Stage 2A)

- **Table build (CPU, offline):** count n-grams of orders 2–5 (histories of 1–4 tokens) over every **training-split** document of shard 000 (validation documents are never counted). Keep histories with count ≥ c_min (set per order so the whole table stays under 6 GiB of host RAM); store the top 16 continuations with their counts and the history count.
- **Read:** COUNT's Equation 1: average over orders of the stupid-backoff-scored top continuations, each weighting its row of the untied unembedding head; added to the token embedding with embedding amplification.
- **Leave-one-out (our addition):** COUNT counted a corpus far larger than its training stream, so a position's own occurrence barely moved its counts (our inference; the paper does not discuss it). Our count corpus is only 4.4 times our training stream, and a rare history's counts would leak the answer. During training each position subtracts its own occurrence (C(h, w) − 1, C(h) − 1, and drops the history if it falls below c_min). Validation needs no correction.
- **Prefetch:** lookups depend only on token ids; a CPU thread builds the next micro-batch's sparse score lists while the GPU trains. The GPU turns them into vectors with one sparse-dense product against the head.

### 7.3 Kenyon backoff (Stage 2B, the new part)

| Fly | Model |
|---|---|
| Receptor neurons with fixed tuning | **Token codes:** 64-dim PPMI-SVD vectors from the count corpus (window ±2, as in `src/clm/word_sim.py`, extended to all 8,192 tokens), frozen. |
| Projection neurons, gain-controlled | **The odour of a history:** its tokens' codes, concatenated, centred, L2-normalised. |
| PN → Kenyon-cell expansion | **Fixed random projection** into two sub-codebooks of 2,048 units. Dense Gaussian by default (Senge 2026: dense matches or beats sparse at equal arithmetic, and arithmetic is free on the GPU); in-degree K is a Stage 2B-0 variable. |
| APL inhibition, k-winners-take-all | **The winner in each sub-codebook**; the pair names one of 4,194,304 buckets per order. |
| Homeostatic excitability | **Per-unit thresholds** calibrated offline so every unit wins equally often (the auxiliary-loss-free bias rule), then frozen. |
| KC → MBON readout | **Bucket counts:** the summed continuation counts of every training history in the bucket, top 16 kept (about 0.4 GiB per order with 32-bit counts). |

When a history of 2–4 tokens has no stored counts, its bucket's scores join the COUNT average in place of nothing; exact histories keep their own counts. Leave-one-out applies to buckets too. **Random-bucket control:** identical, except the bucket is a plain hash of the history's token ids (it pools unrelated histories). **KM-rand codes:** Kenyon buckets built from random token codes (keeps the hash, drops the similarity).

### 7.4 Fast count writes and consolidation (Stage 2C)

A counted memory can learn at test time without gradients: a new association is one count increment. Complementary learning systems theory (McClelland et al. 1995) says such a fast, sparse store should take new pairs without overwriting old ones, and that replay can later move them into the slow weights.

### 7.5 Effort-conditioned depth (Stage 2D)

A looped variant: 2 prelude layers, a 2-layer shared block run r ≤ 6 times with a readout after every pass, 2 coda layers. Trained at random depths with an entropy term against collapse onto the last pass (Ouro). Because every pass is read out, one forward pass gives each token's loss at every depth, so the best depth for any price is computable exactly, without RL, while the other tokens run every pass. What this misses is the cross-token effect (a token that stops early changes the keys and values later tokens read); that effect is the only thing RL could add (§10).

## 8. Uncharted functions in this design

| Function | Origin | Role | Status in the literature |
|---|---|---|---|
| A memory with no learned parameters, read through the model's own unembedding | COUNT | VRAM-free capacity | New in 2026; untested at 5–25M or on a 4 GB GPU |
| Homeostatic k-winners-take-all as a hash | APL inhibition + intrinsic plasticity | Balanced similarity buckets | Not found for LM memories |
| Fixed random expansion in product form | PN → Kenyon cells | Key-free address for 4M buckets | Learned keys (product-key memory) are the norm |
| Fixed distributional receptor codes | Receptor neurons | Similarity without learned, drifting addresses | Learned codes in N-Grammer |
| One-shot count writes (Hebbian counting) | KC → MBON plasticity | Test-time memory without gradients | Gradient writes published (FwPKM 2601.00671); counted writes plus replay not found |
| Exponentially priced latent compute | DeepSeek effort control | One checkpoint spans a compute–loss frontier | Output tokens only (DeepSeek); budget schedules without a price (LoopFormer) |

**Ruled out by the evidence:** sparse activation functions as a training-efficiency lever (parity at best); bio-plausible learning rules as the main learner (L4); RL routers or RL halting for training efficiency (lose to balanced-assignment or supervised versions); looping as a training-efficiency lever at 20 tokens/param.

## 9. Theories and the predictions they make

Each is written before any run, with its direction.

| # | Theory | Prediction | Stage |
|---|---|---|---|
| P1 | Clark 2024 (activity dimension tracks effective rank) | Brain-activity participation ratio, real ÷ shuf, is near the effective-rank ratio (about 0.5); a ratio near 1 refutes the theory for this driven, 6-step brain. | 2T |
| P2 | Liu 2024 (low rank forces rich learning) | Relative weight change ‖W_T − W_0‖/‖W_0‖ is larger for real than for shuf, er and dense. | 2T |
| P3 | Similarity-based smoothing / LSH (Dagan 1998; Dasgupta 2017; HashFormers) | In an offline n-gram model, adding Kenyon buckets lowers validation cross-entropy at unseen and rare (1–9) histories more than random buckets do. | 2B-0 |
| P4 | Same, in the network | COUNT+Kenyon minus COUNT is negative in the unseen and 1–9 history buckets and near zero at 100+; random buckets show no such gain. | 2B |
| P5 | Litwin-Kumar 2017 with global inhibition; Senge 2026 | Bucket quality (held-out divergence between a history's continuations and its bucket's) improves with in-degree K and saturates; dense is at least as good as K = 7. | 2B-0 |
| P6 | Homeostatic plasticity | Without thresholds bucket use is uneven (lower entropy) and bucket quality is worse. | 2B-0 |
| P7 | Complementary learning systems | One-shot count writes store new pairs with backward transfer near 0, against −0.84 for gradient updates of the er-0 brain (`notes/pilot_vocab2_rules.md`). | 2C |
| P8 | The mushroom body's generalisation–discrimination trade-off | With the Kenyon backoff, a stored pair is recalled from its cue's look-alike twin (transfer); twins stored with different answers stay distinct because exact entries win. | 2C |
| P9 | DeepSeek App. C | The exact optimal depth per token, averaged, is affine in effort b over the middle of the range (R² ≥ 0.9); the 1–6 depth ceiling bends the ends. | 2D |

## 10. Reinforcement learning: where it enters, and where it does not

- **Evidence first:** at our scale RL buys inference compute, not training loss per FLOP (§5). The plan therefore tests DeepSeek's effort law on latent depth **exactly, without RL** (P9), and compares the optimal-depth frontier with fixed depth and with a free exit rule (stop when the hidden state stops changing, which Ouro found within 1–2% of a trained gate).
- **RL enters only if that comparison leaves room:** a per-token halting policy (input: state and effort b) trained by group-relative REINFORCE with reward −loss − k(b)·depth, groups per (sequence, b), as in DeepSeek. Its only advantage over supervised halting on the exact labels is cross-token effects (an early exit changes later tokens' keys and values). Gate: run it only if the exact optimum beats the free exit by at least the SESOI at equal mean depth.
- **L7 sets the rule:** a lower average depth is not a win; only a better frontier is.
- **RL does not replace backprop anywhere in this plan** (L4, and §5's evidence).

## 11. Experiments, gates and budget

Seeds 0, 1, 2 unless noted. SESOI is fixed after Stage 2.0 as twice the B0-M seed standard deviation, floor 0.01 nats/token; effects below 0.02 nats get 5 seeds. Decisions reuse Stage 1's paired rules (`src/clm/analyze.py`): X_BETTER needs mean(d) ≤ −SESOI and every d_s < 0. Only end-of-training losses are compared.

| Stage | What | Arms | Cost |
|---|---|---|---|
| **2T** | Stage 1 mechanism, no training: brain-activity participation ratio on validation tokens (all five arms, from the readout neurons and the full brain), relative weight change from regenerated initial weights. Tests P1, P2. | 15 checkpoints | about 1 GPU-h |
| **2.0** | Baseline: optimizer and learning-rate sweep on B0-M (short runs), then the size ladder; seed noise sets the SESOI. | B0-S, B0-M, B0-L x 3 seeds, 200M tokens | about 10 GPU-h |
| **2A** | Counted memory: build tables (CPU), report the pure n-gram model's validation loss, then train. One seed without leave-one-out as a leakage diagnostic. | COUNT x 3 seeds at M (B0-M reused) | 1 CPU-h + about 4 GPU-h |
| **2B-0** | **Offline pre-test of the Kenyon backoff (CPU only).** Interpolated n-gram models with and without bucket components (mixing weights fit on a slice of training documents held out of the counts, never on validation), scored on validation by history-count bucket. Sweeps: codes (PPMI vs random), K (1, 3, 7, 15, 31, dense), winners per sub-codebook (1, 2, 4), thresholds on/off. Tests P3, P5, P6. | none (n-gram models) | about 2 CPU-h |
| **2B** | Kenyon backoff in the network, settings fixed by 2B-0's pre-registered rule (the best held-out bucket quality), never by network loss. Tests P4. | COUNT+Kenyon, COUNT+random-bucket x 3 seeds at M | about 6 GPU-h |
| **2C** | Fast writes on the 2A/2B checkpoints: sequential lists (4 x 384) and twins. Arms: one-shot count writes; gradient updates of the backbone; count writes followed by replay into the backbone with a KL penalty, then recall with the written entries removed. Tests P7, P8. | B0, COUNT, COUNT+Kenyon | about 2 GPU-h |
| **2D** | Looped model at M; exact optimal-depth analysis (P9); frontier vs fixed depth and the free exit. RL halting only through its gate. | looped x 1 seed (+ RL: about 6 GPU-h) | about 4 GPU-h |
| 2.0-ext (optional) | Activation test: ReLU² vs PolyReLU (initialised at ReLU²) vs Hermite-3. | 3 arms x 5 seeds at S | about 7 GPU-h |

**Gates.**
1. **After 2A:** if COUNT does not beat B0 (X_BETTER), the memory track stops and the null is written up. 2T and 2D still run.
2. **After 2B-0:** the Kenyon GPU arm (2B) runs only if the bucket component lowers offline cross-entropy in the unseen and 1–9 history buckets by at least 0.02 nats more than random buckets do. Otherwise 2B is replaced by a write-up of the offline result.
3. **If COUNT or COUNT+Kenyon wins:** a size-L check (3 arms x 3 seeds, about 12 GPU-h), then table scaling: count more FineWeb-Edu shards (§14, question 4) and check whether the gain grows with table size, as COUNT reports.

**Core total: about 30 GPU-hours and 3 CPU-hours** (2T, 2.0, 2A, 2B-0, 2B, 2C, 2D without RL); with every contingent and optional run, about 60 GPU-hours. At 8–10 hours a night the core fits in four nights.

**Measured in every memory run:** loss overall and per history-count bucket (histories of 3 and 4 tokens), wall-clock per 200M tokens, peak VRAM, peak host RAM, CPU time per micro-batch for lookups, bucket-use entropy.

## 12. Prior art and novelty

- **Kenyon backoff (2B):** not found. The closest: N-Grammer (learned similarity codes into a learned bigram table; closeness 2), HashFormers (LSH into embedding rows; 2), Dagan et al. (similarity smoothing for n-gram models; 2), COUNT (exact counted memory; 2). What differs: similarity buckets as a backoff inside a counted host-RAM memory; a fixed fly hash with homeostatic balance over frozen codes, so addresses stay prefetchable; the hash design predicted from theory and tested offline first. Main risk: Hash Layers and HashFormers found random hashing beats similarity hashing for learned tables at full capacity. Our use is the thin-data niche where HashFormers found the reverse. Confidence that it is unpublished: about 60% (agent searches and the main session; Lacuna lags on late-2026 papers).
- **Effort law on latent depth (2D):** not found. The closest: LoopRPT (RL exit with a time penalty; closeness 3 on that half), DeepSeek (effort conditioning on output tokens; 3 on that half), LoopFormer (budget conditioning without RL; 2). No test of the affine law on hidden compute was found. About 65%.
- **Count writes with replay consolidation (2C):** sparse test-time writes are published (FwPKM, arXiv 2601.00671; Sparse Delta Memory, 2607.07386; cache language models before them); FwPKM names consolidation as future work. Counted writes consolidated by replay: not found (closeness 1–2).
- **2T:** a test of Clark's rank–dimension relation on a trained, connectome-constrained LM: not found.

## 13. Considered and set aside: the ETH refutations (October 2026)

On 26 October 2026 OpenAI and Anthropic each posted a model-written paper claiming a subexponential algorithm for 3-SAT, refuting the Exponential Time Hypothesis. Read for this plan: both abstracts and introductions, both technique overviews (OpenAI §2.1–2.4), Anthropic's README and `Challenge.lean`.

- **OpenAI, "The Exponential Time Hypothesis is False"** (67 pages, dated 17 Oct, revised 25 Oct; `github.com/openai/math`). Deterministic exact counting for k-CNF in (n+m)^O_k(1) · 2^O_k(n log log n / √log n) time and polynomial space. Method: replace each variable occurrence by a low-degree linear form in a parameter T, so that at chosen nonzero T the variable groups interact only through a few "shared linear summaries"; recover the count at T = 0 by interpolation.
- **Anthropic, "A 2^o(n) Time Algorithm for 3-SAT"** (49 pages, by "Claude"; `github.com/anthropics/formal-math`, folder `eth`). Deterministic #k-SAT in 2^O(min{k² n log log n/√log n, km/√log m}). Method: write the count as a tensor network on a small dense graph, merge nodes recursively, and cut the bits crossing each cut by expanding a crossing into a signed sum of low-rank terms (from the eight-square identity), each contracted in its cheaper order. The core algorithm came from 14M output tokens with no human input; the Lean proof was written and reviewed by models, and "no person is known to have checked it".
- **Status:** both proofs pass Lean; the expert review (Williams, Impagliazzo, Paturi) had no published verdict in the coverage found on 2 November.

**Why it does not enter the plan.**
1. **No practical algorithm.** Anthropic's bound beats exhaustive search only for n > 10^10 (any formula) and n > 10^10000 at m = 5n (its §1). Nothing we train or evaluate is near that size.
2. **The plan uses no ETH-based result.** The best-known attention lower bound (Alman & Song 2023, arXiv 2302.13214) reduces from Orthogonal Vectors (search summary), and Anthropic states its result does not refute the Orthogonal Vectors hypothesis.
3. **The transferable lesson is about method:** a model, an exact verifier and a very long search found what people had not. DeepSeek reports the same for RL (verifiable tasks and environments account for "essentially all" of its post-training gains, §5.1). Stage 2D therefore uses exact losses as its only reward.
4. **Not planned:** small #SAT instances as a difficulty-graded, exactly checkable task for the depth policy. The new papers change nothing about that experiment.

## 14. Questions for Nate (Stage 2.0 onward waits on these; 2T does not)

**Answered 2026-11-02:** 1, the recommendation (equal wall-clock and VRAM is the headline); 2, yes (memory track first, RL depth last); 3, yes (up to about 8 GiB host RAM and the whole GPU overnight, services stopped and restarted); 4, yes (more FineWeb-Edu shards if gate 3 fires).

1. **Headline metric:** loss at equal wall-clock and VRAM on this laptop (recommended), or loss at equal tokens?
2. **Order:** 2T, 2.0, 2A, 2B-0, 2B, 2C, then 2D (recommended: the memory track has the strongest evidence; the RL track is the most speculative).
3. **Resources during runs:** may runs hold up to about 8 GiB of host RAM and the whole GPU overnight (EmbeddingGemma and heretic-harness stopped for the night, then restarted and checked)?
4. **Data:** the count corpus starts as shard 000 (no download). If gate 3 fires, may we download more FineWeb-Edu shards (2.15 GB each; 64 GB of disk free) to scale the tables?

## 15. Discipline and housekeeping

- New branch `stage2` from `stage1`. New modules only (baseline, count tables, Kenyon hash, evaluation buckets); Stage 1 modules stay byte-identical so its runs still reproduce.
- After any change to `src/`: the suite and `scripts/mutation_audit.py`, extended with planted bugs in the new modules (a leaked count, a validation document counted, a stale prefetch, a bucket built from the wrong order).
- A `PREREG_stage2<x>.md` per stage, committed before its first production run. Pilots touch only B0 and COUNT; the Kenyon settings come from 2B-0's pre-registered rule, never from network loss.
- Long runs under `systemd-run --user`; GPU memory cap 0.75 (the panel-link fault); `nvidia-smi` before every job.
- Every measurement goes to `notes/` with its conditions; every result to the neural vault in the same session.

## 16. Sources

- DeepSeek-AI, "DeepSeek-V4.1-Flash: Pushing the Limits of KV Cache Compression", Sep 2026 (`DeepSeek_V41_Tech_Report.pdf`, read in full).
- Magnusson, Smith & Koh 2026, COUNT, arXiv 2610.20456 (abstract and §1–3 read). Senge 2026, arXiv 2610.09114 (abstract read).
- OpenAI 2026, "The Exponential Time Hypothesis is False"; Claude (Anthropic) 2026, "A 2^o(n) Time Algorithm for 3-SAT" (read in part, §13).
- Dasgupta, Stevens & Navlakha 2017, Science, doi 10.1126/science.aam9868. Litwin-Kumar et al. 2017, Neuron, doi 10.1016/j.neuron.2017.01.030. McClelland, McNaughton & O'Reilly 1995, doi 10.1037/0033-295X.102.3.419. Bennett et al. 2021, doi 10.1038/s41467-021-22592-4. Dagan, Lee & Pereira 1998, arXiv cs/9809110 (abstract read); Dagan, Pereira & Lee 1994, doi 10.3115/981732.981770 (title only).
- Every other source in §5 and §12, with its read level: the three agent reports in the vault, `research-logs/2026-11-02 Stage 2 literature *.md`.
- Own measurements: `notes/laptop_throughput_2026-11-02.md`, `notes/ngram_stats_2026-11-02.md`, `notes/lacuna_check_2026-11-02.md`.

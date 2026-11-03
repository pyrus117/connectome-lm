# Stage 2C pre-registration: fast count writes and consolidation

Dated 2026-11-03. Committed before any Stage 2C run. Plan: `docs/superpowers/plans/2026-11-02-stage2-kenyon-memory.md` (§7.4, §9 P7 P8, §11). Models: the finished COUNT-M seed 0 run (`runs2/count-M-0`, Stage 2A) and the B0-M seed 0 run (`runs2/b0-M-0`, Stage 2.0). Code: `src/clm/fastwrite.py`, `src/clm/fastwrite_arms.py`, `scripts/stage2c.py` (branch `stage2b0`). GPU time: about 30 minutes in total.

## 1. Questions

Complementary learning systems theory (McClelland, McNaughton & O'Reilly 1995) says brains use two stores. A fast, sparse store learns new pairs at once without overwriting old ones. Later, **replay** (showing the old items to the slow store again) moves them into the slow weights, which is called **consolidation**. A counted memory suggests a fast store with no gradients, so we test it.

Two predictions from the plan:

- **P7.** One-shot writes into the memory store new pairs with backward transfer near 0. **Backward transfer** is the average change in an earlier list's accuracy from right after it was learned to the end of the run; negative means forgetting. For comparison, gradient updates of the er-0 brain in Stage 1's pilot gave -0.84 (`notes/pilot_vocab2_rules.md`).
- **P8.** With the Kenyon backoff, a stored pair is recalled from its cue's look-alike twin (transfer), while twins stored with different answers stay distinct, because the exact entry wins.

## 2. The memory write

This section fixes what "write" means.

A plain count increment cannot work. A cue word such as "dinner" already has an exact one-token row with thousands of continuations, so one more count changes its ratios by about 1e-4.

So a write is one extra entry in the COUNT read. A table maps each written cue token to its answer token. Wherever the text token is a written cue, the answer's row of the model's untied head, times a write weight w, is added to the embedding. It goes in exactly where COUNT's own count-weighted head rows are added (`src/clm/countread.py`, `B0GPT.forward`). No weights change and no gradient flows, so writing a new pair cannot overwrite an old one.

The cost is that the cue word injects its answer wherever it appears, including ordinary text. We measure this as the language loss with the memory on.

The write weight is chosen once, on a separate calibration list (task seed 100, 384 pairs). Try w in {1, 4, 16, 64, 256, 1024} and take the smallest w whose recall is within 5% (relative) of the best recall. That w is then fixed for every test list. Choosing the smallest good w keeps the damage to ordinary text as low as the recall allows.

The grid reaches far above COUNT's own scores (at most 1/4 per order) because the written entry must outweigh the cue token's own embedding. In the trained model the head-row norm is about 2.3 and the token-embedding norm about 19.

A preliminary smoke measurement on a 50M-token pilot COUNT checkpoint is not a registered result. For 128 pairs it found recall 0.0 at w = 1, 0.06 at 8, 0.20 at 32 and 0.21 at 128 to 2048. The one-slot write recalls roughly a fifth of pairs and then saturates. That is registered here as a known limit, so a low recall is read as a property of this write rule and not as a surprise.

## 3. Tasks

Code: `src/clm/vocab_task.py`, as in the Stage 1 vocabulary pilot. Words are single-token, lowercase, 4 to 10 letters. A sequence is end-of-text, then cue, answer, cue, answer, for 64 pairs. Within a sequence each cue appears once, so attention cannot copy an answer from the same sequence. **Recall** is the share of cue positions where the model's top guess is the answer.

- **Sequential lists.** 4 lists of 384 random pairs. Cues and answers are disjoint, and no word is in two lists. The lists are learned or written one after another, with a quiz of every list after each phase. Three task seeds: 0, 1, 2. Metrics:
  - the accuracy matrix (row = after phase, column = list);
  - `final_mean`, the mean of the last row;
  - `backward_transfer`, the mean over all lists but the last of final accuracy minus accuracy right after its own phase (Lopez-Paz & Ranzato 2017).
- **Twins.** 512 cues made of the 256 most alike word pairs, each cue with its own random answer. Likeness is the cosine of positive-PMI context vectors within ±2 tokens over `data/fineweb/train.bin` (`src/clm/word_sim.py`). The top pairs are "said/says", "million/billion" and "north/south". Three task seeds.

## 4. Arms on the lists

All arms run on copies of the finished models: one fresh copy per arm and task seed.

- **gradient_b0 and gradient_count.** Fine-tune every parameter on each list in turn, with no memory. AdamW, lr 3e-4, betas 0.9 and 0.95, no weight decay, warmup 20 steps, 300 steps per list, batches of 16 sequences of 64 pairs, gradient clip 1.0. The loss is taken at the cue positions only. `gradient_count` keeps its COUNT read; `gradient_b0` has none. These show how much plain training forgets.
- **write.** No training. After each list is written, quiz every list with the memory on. Run on COUNT-M.
- **consolidate.** After each list is written, train the backbone with the memory off so that it matches what backbone plus memory says on every pair written so far. This is the replay step. The loss is the KL divergence from a frozen teacher to the student, at cue positions. **KL divergence** measures how far apart two probability distributions are, and it is zero when they agree.
  - The teacher is the backbone as it was at the start of the phase, with the memory on. The student is the same backbone with the memory off.
  - An anchor of weight 1.0 keeps the student's output distribution on ordinary validation text close to the original model's. Its batches are 4 windows of 512 from the second half of the fixed validation windows. Without it, the student could learn the pairs by wrecking everything else.
  - Same optimizer and 300 steps per list. After each phase, quiz every list with the memory removed. Run on COUNT-M.
- **Language loss**, measured after each phase. It uses 64 validation windows: every 32nd of the first 2,048 of the 4,096 fixed windows of 512 in `data/fineweb/val.bin`. It is taken with the memory on and with it off.

## 5. Twins protocol

`scripts/stage2c.py twins`, with the write weight fixed in section 2.

1. **Discrimination.** Write all 512 cues with their different answers and measure recall of each cue. The control writes 512 unrelated random pairs and measures recall, so it has the same count but no twins.
2. **Transfer.** Write one twin of each pair (256 entries). Then query the other twin, which was not written, for the stored twin's answer (this is "transfer") and for its own answer. Also query the stored twin for its own answer.

Overlays tried:

- **exact:** the cue token itself.
- **Kenyon:** an unwritten cue falls back to the entry whose cue falls in the same Kenyon bucket. This uses Stage 2B-0's selected hash applied to a one-token history, with its PPMI codes and calibration.
- **Kenyon with random codes:** the same, with the similarity removed.
- **token hash:** a plain hash of the token id into 4,194,304 buckets, with no similarity.

For each overlay we report the share of twin pairs whose two cues share a bucket.

The Kenyon overlays run only if Stage 2B-0's final stage produced its codes. Otherwise they are reported "not run" and P8 is reported untested.

## 6. Decision rules

All are fixed here. Every figure is a mean over the three task seeds.

- **P7 is supported** iff the write arm's `backward_transfer` is within ±0.05 of zero and both gradient arms' `backward_transfer` is at most -0.3. By construction a written cue never changes the entry of another cue, so the first condition is nearly certain. It tests that nothing else interferes. The second condition tests that fine-tuning does forget.
- **Usefulness** is reported beside P7 and is not part of it. It compares the write arm's `final_mean` with `gradient_count`'s `final_mean`, and the language loss with the memory on against off.
- **Consolidation works** iff all three hold, with the memory removed:
  - the consolidate arm's `final_mean` is at least the write arm's `final_mean` minus 0.05 (it keeps what the memory knows);
  - its `backward_transfer` is at least `gradient_count`'s plus 0.3 (it forgets less);
  - its language loss after the last phase is within 0.05 nats of the starting loss (the anchor holds).

  Otherwise it is reported as not working, with the numbers.
- **P8 is supported** iff both hold:
  - (a) with all 512 written, the exact overlay's recall of the twin cues is at least 0.9 times the unrelated-pairs control's recall;
  - (b) the Kenyon overlay's transfer rate exceeds both the exact overlay's and the token-hash overlay's by at least 0.02 (absolute), while its stored-twin recall stays at least 0.95 times the exact overlay's.

## 7. Conduct

Runs go under `systemd-run --user` with the GPU memory cap 0.75. Services are stopped and restarted as before (approved by Nate 2026-11-02). One model seed (0) and three task seeds. No model is retrained.

No change to code, config or data after this file is committed except crash fixes, each logged in `notes/deviations.md` with its commit hash. This stops the design drifting toward a result.

After any change to `src/`, the suite and `scripts/mutation_audit.py` must pass. This stage carries planted-bug mutants M75 to M88 in `scripts/mutation_audit.py` (the bucket fallback of the COUNT read, the memory and its arms). They include a write that clears earlier entries, an overlay slot read at the wrong positions, a consolidation teacher that updates with the student, and a language-loss helper that skips the memory, among others. A test that cannot catch those bugs does not count.

## 8. Limits stated in advance

- There is one write rule (single slot, fixed weight). Its recall is expected to saturate near a fifth of pairs (section 2). A different rule might recall more.
- The replay teacher is the backbone plus the memory, so consolidation can at best match the memory's own recall. It tests retention, not extra learning.
- The write weight is tuned on recall alone, so its cost on ordinary text may be large. That cost is measured and reported.
- The one-token Kenyon hash was never calibrated or tested at one-token histories in Stage 2B-0, which used 2, 3 and 4 tokens. P8 is therefore a weaker test of the backoff than 2B-0.
- Twin similarity from text statistics is a weak proxy. Stage 1's pilot found look-alike cues only slightly harder to learn than random ones.
- There is one model seed, and the model has 13.8M parameters.

TL:DR. We test whether a gradient-free write into the COUNT read stores new word pairs without forgetting old ones (P7), and whether the Kenyon backoff lets a stored pair be recalled from its look-alike twin without blurring twins that have different answers (P8). We also test whether replay can move the memory's pairs into the backbone with the memory off (consolidation). The write rule, the weight-selection rule, the arms, the twins protocol and every pass threshold are fixed before any run. The write is expected to recall only about a fifth of pairs, and that is registered now as a known limit. The work uses finished models, three task seeds and about 30 GPU minutes.

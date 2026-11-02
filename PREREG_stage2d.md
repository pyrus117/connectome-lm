# Stage 2D pre-registration: effort-priced depth in a looped model

Dated 2026-11-03. Committed before the Stage 2D run. Its sha256 is recorded in the run's `results.json`. Plan: `docs/superpowers/plans/2026-11-02-stage2-kenyon-memory.md` (§7.5, §9 P9, §10, §11). Stage 2D does not depend on gate 1 of Stage 2A.

## 1. Questions

**(a) P9.** DeepSeek-V4.1-Flash (App. C) shows that when the price of thinking falls exponentially with an effort level, the best amount of thinking rises in a straight line with effort. Here the "thinking" is hidden compute: running a looped model's shared blocks one more time (one extra pass). So: when the price of one extra pass falls exponentially with an effort level b, is the exact optimal depth per token, averaged over tokens, an affine function of b over the middle of the range? Affine means a straight line, depth = slope x b + offset. DeepSeek derives the line from a marginal benefit that decays exponentially and calls it a local reward-level approximation, so it is only claimed to hold in the middle. No test on hidden compute was found in the literature search.

**(b) Frontier.** At equal mean depth, does a per-token optimal depth beat fixed depth and a free-exit rule by at least the SESOI of 0.01 nats? The free-exit rule stops a token when its hidden state stops changing. Ouro found this within 1-2% of a trained gate.

**(c) The RL gate (plan §10).** Reinforcement-learning halting is run only if the exact optimum beats the free exit by at least the SESOI at equal mean depth.

No reinforcement learning is run in this stage. If the gate opens, the work stops and goes to Nate. RL needs a design of its own and about 6 GPU-hours, and the plan rates RL as a weak lever at this scale.

## 2. Fixed design

- **Model.** `src/clm/s2loop.py`, class `LoopGPT`. It reuses B0's six blocks under B0's own parameter names, so a seed starts from exactly B0-M's weights (tested) and has exactly B0-M's parameters (13.8M non-embedding). Layout: 2 prelude blocks, a 2-block loop shared across passes, 2 coda blocks, then the untied head. Depth r runs the loop r times, so 4 + 2r blocks execute. r = 1 is B0's own six-block computation. Maximum depth is 6 (16 blocks). Size M: d 384, 6 heads, context 512, the Stage 1 tokenizer (8,192 tokens). Config `configs/stage2d.json`.
- **Training.** `src/clm/s2train.py` with arch `loop`. One run, seed 0, the Stage 2.0 recipe unchanged: Muon lr 8e-3 (chosen in Stage 2.0), weight decay 0.1, warmup 200 steps then linear decay to zero, clip 1.0, bf16, GPU memory cap 0.75, 200M tokens = 12,207 steps of 32 x 512, same data order as B0-M seed 0.
- **Depth sampling.** Each micro-batch (8 windows) trains at one depth drawn uniformly from 1 to 6. The draw is a function of (seed, step, micro index) only, so a resumed run draws the same depths and ends identical. This is a cheaper stand-in for training every depth at every step (about 4.3 times B0's compute). It costs on average 11 blocks per pass against B0's 6.
- **No entropy term.** The plan's wording includes an entropy term (from Ouro). Its job is to stop a learned depth distribution from collapsing onto the last pass. Uniform sampling has no learned distribution, so there is nothing to collapse and no entropy term is used. This is a refinement of the plan's wording, made here before any run.
- **Evaluation.** Validation loss at each depth 1 to 6 on the first 2,097,152 tokens of `data/fineweb/val.bin` (4,096 windows of 512), at every eval and at the end. The headline `final_val_loss` is the depth-6 loss.

## 3. Analysis

Code: `scripts/stage2d_analyze.py`, with the arithmetic in `src/clm/depth_law.py` and the model glue in `src/clm/depth_eval.py`. Every constant below is fixed here.

- **Per-token loss at every depth.** One forward pass with all passes read out gives each validation token's loss at every depth. So the best depth for any price is exact, not estimated.
- **Price of one extra pass at effort b.** k(b) = 0.2 x exp(-b / tau) nats, with tau = 10 / ln(100) = 2.1715 and efforts b = 0, 1, ..., 10. Prices run from 0.2 down to 0.002 nats per pass. Why this range: the mean gain from one more pass is expected to run from about 0.1 nats at the first pass to about 0.01 at the last. So these prices run from "nobody goes deeper" to "everybody goes to 6".
- **Best depth of a token at effort b.** The depth r that minimises loss_r + k(b) x r. A tie goes to the shallower depth, so the rule has one answer.
- **P9 test.** Take the efforts whose mean best depth lies in [1.5, 5.5]. The ends bend because depth is capped at 1 and 6, so a straight line cannot hold there. With at least four such efforts, fit mean depth against b by least squares. P9 is **supported** if R^2 >= 0.9 and the slope is positive. Otherwise it is **not supported**. Fewer than four efforts means the range has no middle, and that is reported as not supported.
- **Diagnostic for why.** A fit of log(marginal gain between depths r and r+1) against r. The law holds exactly when gains shrink by a constant factor per pass (geometric decay), and then the slope is sigma / tau. This tells Nate whether a failed P9 failed because the gains are not geometric or because of something else.
- **Executed policies.** When a token exits, it keeps its state, and later tokens read that frozen state in the passes the exited token skips. The "predicted" loss of an oracle policy assumes no token's exit changes what the others read. The "executed" loss actually runs the policy. Their difference is the cross-token effect. It is the only thing RL could add over supervised halting, so it is reported per effort and on average.
- **Free exit.** For each token at each pass, the relative change of the loop state: norm(x_r - x_(r-1)) / norm(x_r). A token exits at the first pass whose change is below a threshold theta, else at pass 6. theta is set at each of the quantiles 0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.97 of all change values, which traces out the whole frontier.
- **Frontiers compared at equal mean depth, by linear interpolation.** Oracle (executed), free exit (executed), and fixed depth (the points (r, mean loss at r), r = 1 to 6).
- **RL gate.** OPEN if the mean, over oracle efforts whose mean depth lies inside the free-exit frontier's range, of (free-exit executed loss minus oracle executed loss at the same mean depth) is at least 0.01 nats. Otherwise CLOSED. The oracle sees each token's true loss, so it is an upper bound and not a policy anyone could deploy. The gate is the plan's literal rule and may well open. That is why an open gate is a stop-and-ask point and not a go signal.
- **Also reported.** Loss by depth against B0-M's three Stage 2.0 final losses (3.405540, 3.405689, 3.400939; mean 3.4040), with the blocks executed at each depth. This shows what each depth costs and whether any depth beats B0-M.

## 4. Procedure

One training run (`runs2/loop-M-0`) by `scripts/stage2d_driver.py`, then the analysis. The run goes under `systemd-run --user` with the GPU memory cap 0.75, because the harness kills long background jobs and the laptop GPU needs headroom. Services are stopped and restarted as in Stage 2.0 (approved by Nate 2026-11-02).

## 5. What each outcome does

P9 supported or not, the frontier numbers and the gate state are all written up whatever they are. Gate OPEN: stop and report to Nate before any RL work. No other result changes the plan.

## 6. Conduct

A diverged run is rerun once from scratch. A second divergence is reported. No change to code, config or data after this file is committed except crash fixes, each logged in `notes/deviations.md` with its commit hash. This stops the design drifting toward a result. Interrupted runs resume exactly (tested: `tests/test_s2loop.py::test_interrupted_and_resumed_loop_run_ends_bit_identical`). After any change to `src/`, the suite and `scripts/mutation_audit.py` must pass. The loop code carries planted-bug mutants: a loop that ignores depth, an exited token that keeps updating, a depth drawn from the wrong step, among others. A test that cannot catch those bugs does not count.

## 7. Limits stated in advance

- One seed. B0-M's seed standard deviation at 200M tokens is 0.0054 nats, so differences under 0.01 between frontiers are inside the noise.
- The price schedule is our choice. The law is DeepSeek's "local reward-level approximation", and the depth cap of 6 bends both ends.
- The oracle uses the true loss and is not deployable.
- Training at one random depth per micro-batch gives each depth about a sixth of the batches. That is a weaker objective than training all depths at once, so the depth losses may be worse than a model trained on all of them.
- 13.8M parameters, one corpus.
- Looping is not expected to help training efficiency at this token-to-parameter ratio (plan §8). The test here is about the shape of the compute-loss frontier, not about beating B0-M.

TL:DR. We train one looped version of B0-M (seed 0, 200M tokens, each micro-batch at a random depth from 1 to 6) and read every token's loss at every depth. Then we ask three things with all constants fixed in advance. Does the best depth rise in a straight line with effort when each extra pass gets exponentially cheaper (P9, R^2 >= 0.9 with a positive slope over the middle efforts)? Does a per-token optimal depth beat fixed depth and a free-exit rule by 0.01 nats at equal mean depth? Does it beat the free exit by that margin, which opens the RL gate? Everything is written up whatever it shows. If the gate opens, no RL is run and the work stops for Nate.

# Pre-registration — per-head Muon side test (2026-11-03)

Asked for by Nate on 2026-11-03 to inform the next baseline decision. Written and committed before any run.

## 1. Question

Does Kimi K3's per-head Muon (arXiv 2607.24653 §2.5: Newton–Schulz orthogonalisation on each head's slice of Q, K and V separately) give B0-M a lower validation loss than our fused Muon, which orthogonalises the whole `qkv` matrix at once?

## 2. Arms

- `muon` — fused, as in Stage 2.0 (unchanged code path).
- `muon_head` — `HeadMuon` in `src/clm/s2model.py`: each `qkv` weight is 3 x n_head row blocks of 64 x 384, each orthogonalised alone, lr scaled by the block shape (`match_rms_adamw`, so update RMS stays 0.2 x lr). Every other matrix is treated exactly as in `muon` (tested: with no split it matches `torch.optim.Muon` bit for bit).

Everything else is the Stage 2.0 pilot setting: B0-M, 50M tokens, `configs/stage2_0.json`, warmup 200 then linear decay to zero, weight decay 0.1, same data order per seed.

## 3. Procedure

1. Learning rate for `muon_head`: 4e-3, 8e-3, 16e-3 on seed 0. Lowest final validation loss wins. If the winner is on the grid edge, one step further (halve or double), once.
2. Seeds 1 and 2 for `muon_head` at the chosen rate and for `muon` at 8e-3 (the Stage 2.0 choice). Seed 0 of `muon` is the existing Stage 2.0 pilot run `runs2/pilot/muon-0.008` (3.6649), which used the same code path.

Runs go to `runs2/headmuon/`. Driver: `scripts/headmuon_driver.py`.

## 4. Decision rule

d_s = loss(`muon_head`) − loss(`muon`) for seeds 0, 1, 2. SESOI = 0.01 nats (Stage 2.0).

- **BETTER**: mean d ≤ −0.01 and all three d < 0.
- **WORSE**: mean d ≥ +0.01 and all three d > 0.
- **NO DIFFERENCE**: anything else.

Also reported: seconds per step for both arms, and the 50M-token seed SD of each arm.

## 5. What a verdict does

None changes Stage 2.0's B0, which stays frozen. BETTER makes per-head Muon a candidate for a revised baseline. That choice (rerun the ladder first, or go on to Stage 2A with B0 as is) goes to Nate with the numbers. NO DIFFERENCE or WORSE keeps fused Muon.

## 6. Limits stated in advance

50M tokens is a short-horizon proxy, as in the Stage 2.0 pilot. The lr choice for `muon_head` uses seed 0, which is also one of the three compared seeds; the fused rate was chosen the same way, so the bias is symmetric. Kimi K3 is a 2.8T MoE; a result at 13.8M parameters says nothing about that scale. The laptop runs with heretic-harness and EmbeddingGemma stopped, as for Stage 2.0.

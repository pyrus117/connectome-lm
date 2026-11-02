# Lacuna literature check and effective-rank measurement, 2026-11-02

Searched with the Lacuna research MCP (lacuna.tiptreesystems.com): about 14 paper, hypothesis and detail queries. Summaries were read through Lacuna's paper pages; no full PDF was opened. Stage 1 is finished, so reading primary-arm losses is allowed. Everything here is exploratory and post hoc.

## Coverage warning

Lacuna does not index arXiv 2604.04033 (Topological Sensitivity), FlyGPT, DrosophiLLM or Hering 2609.17251, all found by the 2026-10-21 checks. Its coverage of this niche is partial. Its hypothesis store holds no connectome-LM proposal.

## New prior work (not in the two earlier novelty checks)

| Paper | Lacuna | What it adds | Closeness 0–3 |
|---|---|---|---|
| Yu, ..., Vogelstein, "Biological Processing Units", arXiv 2507.10951 (AGI 2025) | [link](https://lacuna.tiptreesystems.com/work/biological-processing-units-leveraging-an-insect-connectome/version/art_27be006dbeb7430b9d6b996062093caa) | Fixed larval fly connectome as a recurrent core, trained I/O only; beats size-matched MLPs on MNIST/CIFAR/chess. Grows the graph with a degree-corrected stochastic block model (DCSBM). Comparison is against MLPs, not degree-preserving nulls. | 1–2 |
| Zou, Zang, Ji, "Structural features of the fly olfactory circuit mitigate the stability-plasticity dilemma", arXiv 2502.01427 | [link](https://lacuna.tiptreesystems.com/work/structural-features-of-the-fly-olfactory-circuit-mitigate/version/art_dbd2089e4c344d10ae572a98c1ad1d91) | PN→KC expansion, ~6 random inputs per KC, top-k (APL-like) sparse coding: +22% over SGD on class-incremental CIFAR-100. Random wiring, not the connectome. Sparse coding is the active ingredient. Our SharedBrain has no top-k step. | 1 (to the sequential-list task) |
| Clark, Marschall, van Meegen, Litwin-Kumar, "Connectivity structure and dynamics of nonlinear RNNs", arXiv 2409.01969 | [link](https://lacuna.tiptreesystems.com/work/connectivity-structure-and-dynamics-of-nonlinear-recurrent/version/art_b7fbd42a94bc4546ba2fe9bc4ff305f7) | The fly hemibrain's singular-value spectrum decays much faster than i.i.d.: low effective rank. Theory: activity dimension near chaos scales with connectivity effective rank. | 1 (mechanism) |
| Liu et al., "How connectivity structure shapes rich and lazy learning", ICLR 2024 | [link](https://lacuna.tiptreesystems.com/work/how-connectivity-structure-shapes-rich-and-lazy-learning/version/art_5a76f989b1b24481b7b7bde1ee242027) | Biological (EM, Dale) connectivity has lower effective rank than nulls and forces "rich" learning, unless aligned with the task. Final performance is named as future work. | 1 (mechanism) |
| Zhang, "Network structure governs Drosophila brain functionality", Fundamental Research 2025 | artifact `art_691651eaee4d9287055fe654375e223e` | Whole-brain simplified-neuron model; 20% random rewiring destroys sensory segregation. No training, no LM. | 0–1 |

Not found in Lacuna: any paper that links a connectome's effective rank to trained task loss against a degree-preserving null.

## Measurement: effective rank of the brain matrix (scripts/effective_rank.py)

Laptop CPU, float64 `torch.linalg.svdvals` on the dense 6,293 x 6,293 matrix, torch 2.14.0. Effective rank = participation ratio of singular values. "Topology" = edges weighted 1/sqrt(in-degree) (the fan-in init scale); "trained" = checkpoint synapse values at the final step.

| Run | PR topology | PR trained | Val loss (runs/analysis.json) |
|---|---|---|---|
| real-0 | 333.4 | 1580.5 | 3.9476 |
| shuf-0 | 703.7 | 1856.5 | 3.9350 |
| shuf-1 | 698.7 | 1871.3 | 3.9359 |
| shuf-2 | 700.1 | 1845.2 | 3.9309 |
| er-0 | 1919.2 | 3000.3 | 3.9224 |
| er-2 | 1920.3 | 3012.1 | 3.9190 |

The real wiring has half the effective rank of its degree-preserving shuffles and a sixth of ER's, and training does not close the gap. Across these three arms, higher rank goes with lower loss. real-1 and real-2 share real-0's wiring; er-1 is left out because of its gradient spikes.

Limits: three arms only, so this is an ordering, not a fit. `real_randio` has the real matrix (same PR) yet a lower loss than `real` (3.931 vs 3.948 on seeds 0 and 2), so matrix rank alone does not explain the result; input/readout placement also matters. `dense` (760 neurons) cannot be compared on raw PR. A readout-activity measure (PR of the 256 readout neurons over validation tokens) could cover all five arms; not yet measured.

## Link to critters (added the same day)

critters S2 (registered, 2026-10-28, `/home/nate/Projects/critters/PLAN.md:1059`): stratified lobe wiring decoded the one drive x colour conjunction it was built for at 0.998 (random R10 0.713), but its activity participation ratio fell to 1.2 (R10 17.0). That is the same trade-off: structured wiring, few dimensions, good only on the aligned task. Critters measured activity PR, this note measures weight-matrix PR; use activity PR for the next check so the two are comparable. Vault: `findings/connectome-lm - Real mushroom-body wiring has half the effective rank of its shuffles.md`.

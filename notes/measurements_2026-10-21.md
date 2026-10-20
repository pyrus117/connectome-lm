# Measurements, 2026-10-21

## Sparse matmul on the RTX 3050 Laptop GPU (4 GB), torch 2.14 (system python3)

Region: `data/mb_alpn_783.npz` (6,293 neurons, 570,682 edges). Input: dense 6,293 x 2,048 (2,048 tokens), float32. One call = build COO from a values Parameter, `.coalesce()`, `torch.sparse.mm`, `.sum().backward()`.

| Check | Result |
|---|---|
| Gradient reaches the values Parameter | yes, 100% of the 570,682 values got a non-zero gradient |
| Coalesced nnz | 570,682 (no duplicates) |
| Forward + backward, mean of 10 calls | 164.6 ms |
| Peak allocated memory | 497 MB |
| bf16 under autocast | fails: `NotImplementedError "addmm_sparse_cuda" not implemented for 'BFloat16'` |
| CSR `@` backward | out of memory: tried to allocate 606 MiB (it densifies) |

Conditions: llama-server held 384 MB of the GPU. A game (The Detective Servant, 2.6 GB of the GPU) started within about a minute after this run. A second benchmark (pre-sorted COO, masked-dense) failed with out-of-memory because of that game and produced no numbers. Treat 164.6 ms as one uncontended measurement, not a settled figure.

## Speed test on the RTX 3050 Laptop (results/speed.json), 10:36

Settings: steps 4, batch 64 x 256 tokens, micro 4, PyTorch capped at 75% of GPU memory, EmbeddingGemma 2 stopped for the test, 2 warm-up + 2 timed steps. No panel-link errors in the kernel log during the run.

| Layout | Arm | s/step | Peak GB | Hours per run |
|---|---|---|---|---|
| all 6 layers | real | 24.0 | 2.17 | 81.5 |
| all 6 layers | dense | 0.24 | 0.38 | 0.81 |
| layers 2, 3 | real | 8.2 | 0.97 | 27.7 |
| layers 2, 3 | dense | 0.22 | 0.36 | 0.74 |

Projected totals (12 connectome runs + 3 dense): 980.6 h (all 6 layers), 334.1 h (layers 2, 3). Neither fits the 150 h local budget, so the rule chose no layout (exit 3).

These are cool-GPU numbers and probably optimistic. The 20-step smoke run (layers 2, 3, micro 4, same code) slowed from 11.6 s to 16.3 s per step over its first steps and had a median of 23.6 s, about 3x the 8.2 s here. Likely thermal or power throttling under sustained load, not measured directly.

## CORRECTION: first-step active edges, three block designs

The `active-at-init` column in the next section is wrong. It combined "pre lit after S−2 hops" with "post reaches a readout in S−2 hops", conditions from different passes. The correct test for pass j (1 … S−1) is: pre lit after j−1 hops AND post's effect reaches the output. Recomputed (same draws as below):

- **A:** no per-step norm. Edge active at pass j iff pre ∈ fwd(j−1) and post ∈ rev(S−1−j).
- **B:** RMS norm over all neurons after every step, no readout norm. Every lit neuron moves the global scale, so edge active iff pre ∈ fwd(S−2).
- **C:** as B plus an RMS norm over the readout neurons. The readout norm cancels the last step's global scale, so the last pass needs post ∈ readout.

| Arm | A, S=3 | A, S=4 | B, S=3 | **B, S=4** | C, S=4 |
|---|---|---|---|---|---|
| real | 0.121 | 0.610 | 0.472 | **1.000** | 0.615 |
| shuf | 0.170 | 0.725 | 0.632 | **1.000** | 0.726 |
| er | 0.087 | 0.969 | 0.976 | **1.000** | 0.977 |
| real_randio | 0.080 | 0.927 | 0.940 | **1.000** | 0.943 |

Chosen 2026-10-21 (Nate): design B, S = 4.

## Edge reachability per arm (graph only, k_io = 256) — superseded, see correction above

`learn` = fraction of edges whose post neuron can reach a readout neuron in time (can ever get a gradient). `reach` = fraction of readout neurons that input activity reaches. `active-at-init` = fraction of edges that get a gradient on the first step with zero biases (pre reachable from input AND post reachable to readout). S = steps (S−1 sparse passes). One shuffle, ER and random-IO draw each (seeds 0/1); the shuffle is a plain permutation without self-loop/duplicate repair.

| Arm | S=2 learn / reach / active | S=3 | S=4 |
|---|---|---|---|
| real | 0.25 / 0.20 / 0.00 | 0.99 / 1.00 / **0.47** | 1.00 / 1.00 / 1.00 |
| shuf | 0.25 / 0.96 / 0.00 | 1.00 / 1.00 / **0.63** | 1.00 / 1.00 / 1.00 |
| er | 0.05 / 1.00 / 0.00 | 0.99 / 1.00 / 0.97 | 1.00 / 1.00 / 1.00 |
| real_randio | 0.04 / 0.91 / 0.00 | 0.98 / 1.00 / 0.92 | 1.00 / 1.00 / 1.00 |

S = 5 is 1.00 everywhere.

The degree rule recovers the anatomy. The 256 lowest in-degree neurons are 249 ALPN, 4 DAN and 3 KC. The 256 highest are 172 DAN, 73 MBON, 6 KC, 4 MBIN and 1 ALPN. Input neurons have mean out-degree 26.2; readout neurons have mean in-degree 567.

## Projection from that one number (not measured)

Per training step of 64 x 256 = 16,384 tokens, with steps = 3 (two sparse applications per block), each block needs 8 chunks x 2 applications x about 0.16 s, so about 2.6 s per connectome layer per step. 6 layers is about 16 s per step. 12,207 steps is about 54 hours per run on this laptop. 2 connectome layers is about 18 hours per run.

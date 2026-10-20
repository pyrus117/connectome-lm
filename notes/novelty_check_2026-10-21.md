# Novelty check, 2026-10-21

Raw report from a researcher subagent (about 12 searches and 9 page fetches). Items marked "snippet only" were not opened. Not yet independently verified.

Closeness scale: 0 = unrelated, 3 = nearly the same experiment.

**Verdict:** the core comparison (a real connectome against a degree-preserving shuffle, as a trainable transformer/LM layer, matched on parameters and tokens) was not found in print. Confidence: medium-high. A negative search does not prove absence.

1. **Topological Sensitivity in Connectome-Constrained Neural Networks** (arXiv 2604.04033, 2026). Abstract opened. Uses the flyvis Drosophila visual connectome with a self-loop-matched random graph and a degree-preserving rewired null. It says earlier topology advantages "can arise from initialization and null-model confounds, and largely disappear under fair from-scratch initialization and degree-preserving controls." Closeness 2: the same method, not the same task. Design consequence: train every arm from the same random initialisation, use several degree-preserving seeds, and check activation scale before training.
2. **DrosophiLLM** (GitHub Rob-bio4/DrosophiLLM, no paper). README opened. Constrains the residual stream of Qwen2.5-0.5B-Instruct with a 2,048-neuron, 59,963-synapse FlyWire sub-circuit. Includes a rewired ablation and a Kenyon-cell lesion. Closeness 2. It is a single-author repo with self-reported results, no peer review, and claims not verified. It routes a pretrained model; it does not train a replacement feed-forward layer.
3. **conn2res** (Suárez et al., Nat Commun 2024). Opened. Real connectomes beat 500 degree-preserving rewired nulls as reservoirs, "except for the fruit fly." Closeness 2. Fixed weights, readout-only training, no language.
4. **Cross-species reservoir comparisons** (Reimers/Nichele; arXiv 2211.00161; CCN 2024; bioRxiv 2025.10.29.685101). Snippet only. One snippet says biological weight values, not topology alone, drove much of the gain. Closeness 1.
5. **Biological connectomes as a representation for the architecture of ANNs** (Schmidgall et al., arXiv 2209.14406). Abstract opened. C. elegans; says the circuit's statistics carry a useful prior even when the exact wiring is not kept. Closeness 1.
6. **Elegans-AI / ElegansNet** (arXiv 2304.13538). Snippet only. Vision tasks, no degree-preserving null seen. Closeness 1.
7. **PHCSSM** (arXiv 2604.01295, 2026). Abstract opened. A spiking state-space model with connectome-style layers, but no real wiring diagram. Relevant to Stage 2. Closeness 1.
8. **Cannistraci-Hebb brain-network sparse transformers** (arXiv 2501.19107). Snippet only. Generated small-world and scale-free topologies in LLaMA 60M–1B, not a real connectome. Closeness 1–2.
9. **Randomly Wired Neural Networks** (Xie et al. 2019). Background only.
10. **Fly connectome as network:** Lappalainen et al. 2024 (Nature); FlyGM (arXiv 2602.17997; its rewired-null claim is unverified); a MIRE-org FlyWire olfactory SNN on Hugging Face (fixed AL+MB topology with shuffled, sparse-MLP and dense controls, used for classification). Closeness 1–2.
11. **Spiking LMs:** SpikeGPT (arXiv 2302.13939, snippet only). Uses no biological graph. Closeness 0–1.
12. **Not found:** connectome-constrained transformer FFN layers; mouse, macaque or human connectomes used for language; a connectome as a KV-cache alternative.

**Prior evidence leans against a large real-against-shuffled effect once degree is preserved** (item 1, and the fruit-fly exception in item 3). The expected answer is "the degree sequence explains most of it." A null result would itself be publishable.

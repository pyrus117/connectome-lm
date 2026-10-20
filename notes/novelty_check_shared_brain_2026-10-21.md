# Novelty check: one persistent, layer-shared connectome brain, 2026-10-21

Raw report from a researcher subagent (8 searches, 4 extended; 4 pages fetched). Items marked "snippet only" were not opened. Not independently verified.

Closeness scale: 0 = unrelated, 3 = nearly the same design.

**Verdict:** not found in print. No source combines (a) sparse, fixed, real-connectome topology, (b) one leaky recurrent step per transformer layer on a state that persists across depth within a token, (c) weights shared by all layers, and (d) a degree-preserving-shuffle comparison. The components exist separately. Absence of a hit is not proof of absence.

- **Closest architecture:** Hering, "Persistent Recurrent Memory Between Transformer Layers – Improves Language Model Generalization", arXiv 2609.17251 (Sep 2026). Abstract read. A dense GRU state bridges the lower and upper halves of a 6-layer decoder once, with no biological wiring and no topology controls. Closeness 2.
- **Closest real-versus-shuffle language-model test:** QuixiAI/FlyGPT (Hugging Face model card, not peer reviewed). A character-level recurrent LM, not a transformer, on a 5,000-neuron MaleCNS fly subgraph. It compares real wiring against degree-preserving scrambles with paired seeds. Snippet only: validation loss 1.6093 real vs 1.6125 scrambled, i.e. no detectable difference. Closeness 2. **A prior pointing toward a null.**
- flybook-git/flm, nftechie/flm, ngxson/fly-llm-hf: frozen-reservoir fly LMs. Snippet only. Closeness 1.
- Kuklev, arXiv 2609.24565 (Sep 2026): a frozen fly visual connectome as a vision feature extractor; degree-preserving rewiring costs 0.271–0.356 accuracy. Not an LM. Evidence that wiring can matter in some tasks. Closeness 0.
- Goyal et al., "Coordination Among Neural Modules Through a Shared Global Workspace", arXiv 2103.01197 (ICLR 2022): a dense attention workspace shared by all layers. Snippet only. Closeness 2 for the shared read/write structure only.
- Looped / depth-recurrent / shared-weight transformers (arXiv 2608.15062, 2604.14442, 2609.01924; MeSH): share weights across depth, no biological wiring. Snippet only. Closeness 1.
- **Check before freezing the null definition:** arXiv 2609.39248 reportedly finds that the choice of null model (degree-preserving swaps vs column shuffles) changes connectome conclusions. Snippet only.
- Prior warnings still apply: arXiv 2604.04033 (topology advantage vanishes under fair init and degree-preserving nulls).

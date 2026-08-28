# SAE_statind — Consolidated Project Dossier

*Assembled from conversation history, May–August 2026. Intended as a Project knowledge file — update in place as things change.*

---

## 1. Identity

| | |
|---|---|
| **Project** | `sae_statind` |
| **Repo** | `rhyderi1/SAE_statind` |
| **Cluster path** | `/project/aip-bahtol/rhyderi1/sae_statind` |
| **Compute** | Vulcan (Alliance Canada), account `aip-bahtol`, user `rhyderi1` |
| **Funding** | NSERC USRA |
| **Lab** | NeuBahar Lab, University of Alberta |
| **PI** | Bahareh Tolooshams |
| **Collaborator** | Valérie Costa |
| **Adjacent** | Thomas Fel, Ekdeep Singh Lubana, Demba Ba |

**Goal.** Build a *probe-free* detector for feature absorption in SAEs — a metric computable from joint latent statistics alone, with no ground-truth probe and minimal data assumptions. Chanin et al.'s pipeline is language-specific and probe-dependent; the payoff of removing both is generalizing absorption to other modalities (images, audio).

**Target setup.** Gemma-2-2B, layer 12 residual stream (some case-study work at layer 3). Four architectures: ReLU, TopK, JumpReLU, Matryoshka.

---

## 2. The metric

$$\text{Absorb}(A \to B) = \cos(d_A, d_B) \times (1 - P(A \mid B))$$

- Run in **both orderings**; parent identified as the higher-frequency latent ($f_A > f_B$).
- **Cosine factor** = geometric direction-sharing; identifies which pairs *could* share a feature.
- **Conditional factor** = the directional asymmetry. $P(A|B) \approx 1$ means intact hierarchy; absorption is $P(A|B) \ll 1$ *combined with* high decoder cosine.
- Headline validation is **ROC-AUC against SAEBench's labeled `S_main`/`S_abs` pairs** — threshold-free. The only threshold in play is the activation binarization threshold, inherited from SAEBench's own convention rather than chosen independently.

### Design decisions locked in
- **Multi-absorber directed edges**, not single-absorber attribution. Each $A \to B_i$ edge keeps its own weight; a decoder-projection threshold prevents attributing noise to hundreds of spurious latents. The single-absorber (argmax) view is derived *post hoc* for SAEBench comparability. Collapsing upfront would discard information irreversibly and hide the diffuse high-frequency signal.
- **Two-rung structure**, mapping onto the Bahareh/Valérie discussion of sparse non-Gaussian activation statistics:
  - *Rung 1* — indicator statistics (binary co-firing), immune to sparsity pathologies.
  - *Rung 2* — on-support code statistics (magnitude), requiring joint-support normalization and division by co-activation count.
- **Escalation ladder** (covariance → entropy → conditional entropy) is a ladder to climb only if binary AUC proves insufficient, *not* a set of metrics to compute in parallel.
- **Parent-axis identification without labels**: the x-axis range asymmetry in triangle plots is both the visual signature of the parent's higher base rate *and* the recovery mechanism for axis assignment on unlabeled pairs.

### Technical conventions
- `from_pretrained_no_processing` with `center_writing_weights=False` for Gemma-2-2B (RMSNorm requirement).
- `sae.encode(X)` rather than manual encoder-column projection.
- Joint-support standardization for STD panels.

---

## 3. Timeline

### May 2026 — Scoping and the H1–H4 framing

USRA proposal began as "inductive biases in SAEs," was critiqued as too broad and under-defined, then rewritten with: a formal definition of inductive bias in the SAE optimization context, narrowed scope (primary **L1 vs TopK**; secondary **Gated vs JumpReLU**), and four falsifiable hypotheses derived from theory *before* experiments:

- **H1** — lasso shrinkage bias
- **H2** — absorption as a downstream consequence of L1 shrinkage
- **H3** — TopK splitting when true sparsity $s < k$
- **H4** — Gated SAEs and the shrinkage/selection decoupling

Evaluation frame: SynthSAEBench + Hungarian matching. Confound controls: hyperparameter sensitivity, matched sparsity, penalty-strength sweeps, initialization variance, normalization, synthetic prior choice.

Deliverable: 12-slide plain-white deck (Cambria titles / Calibri body, thin gray rules), presented at lab meeting May 5–6 with Bahareh and Valérie.

Theory touchpoints from this period: the irrepresentable condition (Zhao & Yu, 2006), cross-polytope geometry of the L1 ball, completeness vs. precision as evaluation metrics, superposition geometry, connections to compressed sensing / RIP.

### Late May — First training run

Baseline ReLU SAE trained from scratch: `StandardTrainingSAEConfig`, `l1_coefficient=5`, `d_in=2304`, `lr=2e-4`, dataset `Skylion007/openwebtext`. 30,000 steps, 61.44M tokens, ~1h44m on an L40S-48GB.

SAEBench absorption eval → `absorption_rate: 0.0420`. Flagged as **artificially low due to undertraining** (MSE ≈ 503).

Environment blockers cleared: pyarrow on Vulcan (load `arrow/24.0.0` *before* venv activation, plus `PYTHONPATH` to the module's site-packages), `HF_TOKEN` for the gated Gemma-2-2B.

### June — Code comprehension

Full walkthroughs of both libraries:

- **SAELens** — `saes/sae.py` (base SAE, `TrainingSAE`, `fold_W_dec_norm`), architecture variants (JumpReLU / TopK / BatchTopK / Matryoshka), `activations_store.py`, `sae_trainer.py`.
- **SAEBench absorption pipeline** — three stages: GT probe training (`probing.py`, BCEWithLogitsLoss with `pos_weight` rebalancing, one-vs-rest 26-class, memmap activation storage) → k-sparse probing (`k_sparse_probing.py`, L1-regularized probe for main latents, F1-jump threshold defining `split_feats`) → absorption detection (`feature_absorption.py`, `feature_absorption_calculator.py`: decoder cosine similarities, probe projections, absorption fraction with top-3 absorber pooling).

**Metric distinctions that matter for claims:**
- `is_full_absorption` requires main latents to be *completely silent* (activation < EPS) as a hard prerequisite.
- `absorption_fraction` allows main latents to fire, measuring absorbers' share of jointly-explained probe signal: `absorption_probe_proj / (absorption_probe_proj + main_feats_probe_proj)`.
- Both use the 40% threshold, but in different roles (classification condition vs. computation gate).
- `mean_full_absorption_score` matches Chanin et al.; `mean_absorption_fraction_score` is Demian Till's extension. **Fraction score is more sensitive for discriminating architectures; full score is required for paper-comparable claims.**
- **Matryoshka dictionary-size confound** is baked in at the split-feat detection and top-3 absorber search stages — width-mismatched comparisons are invalid on *both* scores.

Multi-SAE sweeps within one model are natively supported via regex; multi-model sweeps require an outer loop over `run_eval` with separate output folders and explicit model deletion between iterations.

### June — Activation pipeline

`infer_z.py` (originally `june7test.py`): collects residual-stream activations X and SAE latents Z from Gemma-2-2B, running as SLURM array jobs.

**Bahareh's June 8 directives:** disable shuffling; stay under 1 TB; save X for only one SAE; compute $Z^\top Z$ heatmaps; produce sparsity-aware statistics; reproduce Valérie's `dict_vis` figure; write everything mathematically.

Also built: streaming activation histograms, Z-histogram scripts, extraction of `S_main` / `S_abs` latent sets from the absorption parquet (per-letter groupby, with the `.iloc[0]` assumption that `split_feats` is constant within a letter, and explicit `int()` casts for JSON serialization).

### June — DTD × ZTZ geometry

Scripts: `DTDxZTZ_scatter.py`, then `src/dtdxztz_scatter_simple.py`. Each point pairs the *same* $(i,j)$ entry from both Gram matrices — $D^\top D$ (decoder direction overlap) on x, $Z^\top Z$ (co-activation) on y. Off-diagonal only (`tril_indices`); `|DTD|` folding to use the full sample; log y-axis; `MAX_POINTS = 300_000` subsampling against ~134M pairs.

**Architecture signatures observed (layer 12):**

| Architecture | Signature |
|---|---|
| ReLU | Positive-correlation "fin"; near-duplicate features at DTD ≈ 1.0 |
| TopK | Tighter; geometry largely decoupled from co-activation |
| JumpReLU | Bimodal vertical structure driven by threshold-gating artifacts |
| Matryoshka | Near-orthogonal decoders (DTD ≈ 0) but structurally high co-activation |

**Caveats flagged:** raw ZTZ is *not* comparable across architectures (y-scales span 10⁶–10⁹) — normalize to correlation form first (Bahareh's directive). Mask the diagonal before any correlation analysis. Some plots were clipping at 10⁸.

**Recommended summarization strategy:** collapse scatters to scalars — Spearman ρ between DTD and log ZTZ, plus a quadrant-fraction metric — then present as an architecture×layer heatmap grid at canonical sparsity/width, a separate sparsity-vs-width plot for one focal architecture, one anchor hexbin for interpretability, and an interactive HTML viewer for browsing. A CSV with one row per configuration is the reproducibility backbone.

**Separate result:** OrtSAE vs. pretrained SAEs — OrtSAE flattens the absorption-vs-sparsity curve, outperforming at low sparsity and underperforming at high L0, with a crossover between k=40 and k=70.

### Late June — The directional turn

Started from symmetric $Z^\top Z$ / $D^\top D$; identified that **absorption is inherently directional and symmetric statistics cannot capture it**. Worked through the conditional-probability framing (parent A, child/absorber B) using asymmetric co-firing ratios. A sign error was caught and corrected — $P(A|B) \approx 1$ describes an *intact* hierarchy, not absorption — yielding the final metric above.

Triangle plot read geometrically: the y-axis cluster of absorption events is the binary co-firing deficit (rung 1); the negatively-sloped joint-activation blob is partial/magnitude absorption that the binary metric misses and on-support Pearson captures (rung 2).

### July — Case study 6510 / 1085

Pair: **6510** (parent, "starts with S") and **1085** (child, "short"), JumpReLU layer 3, k=59. Built $z_i$ vs $z_j$ scatters with a conditional-probability / conditional-expectation panel, under two conditions: all tokens vs. S-initial tokens only.

**Result:** S-token conditioning left the *continuous scatter geometry* unchanged, but shifted the firing conditional $P(\text{parent}|\text{child})$ from **0.13 → 0.31**.

**Interpretation:** this is positive evidence, not a null result. The absorption signature lives in **binary firing structure**, not activation magnitudes — which points toward the ZTZ indicator variant over continuous correlations. It also predicts *where* a firing-based estimator should beat an energy-based one: the low-magnitude regime.

Bar chart (replicating Chanin et al. Fig. 6a, with sample counts, standard deviations, and random control latents) cleanly shows the absorption hole: 6510 fires on other S-tokens but ≈0 on "short"; 1085 fires only on "short".

Caveat carried forward: a single pair is too noisy to conclude much either way — the right unit of analysis is the *aggregate* over many pairs.

Side deliverable: an 18-slide PPTX walkthrough of "A is for Absorption" with full speaker notes.

### July — The derivation program

**Stated preference, load-bearing:** a *mechanistic, derived* understanding of the formula from a generative model — not a metric selected empirically via AUROC search over candidates. An AUROC search can't tell you *why* the metric works, which means you can't predict its behavior on a vision SAE where there are no labels to check against.

**C2R (Jin et al.) adopted as the scaffold.** Their §3 (Eqs. 5–6, Table 2) is a two-latent generative story with a single $\alpha \in [0,1]$: the learned latent $L'_2 = [(1-\alpha)L_1 + \alpha L_2]/\|\cdot\|$ absorbs a fraction $\alpha$ of the parent's direction. Validated on ~4,555 labeled pairs on this exact model.

The fit is unusually good, and the gap is exactly the contribution:
- C2R's $\alpha$ is a purely **geometric/energy** quantity, recovered from an energy ratio A/B. It corresponds almost exactly to the **cosine factor**, giving that term a rigorous generative meaning.
- Their framework has **no analogue of the conditional-firing factor**. But their own Table 2 contains the signature: at $\alpha = 1$, the parent coefficient $(1-\alpha)z_1$ goes to *exactly zero* on absorbed samples. The parent is silent precisely when the child fires — which is what $P(A|B)$ measures directly and their energy-based $\hat\alpha$ captures only indirectly.
- Framing: the metric is a **thresholded-firing estimator of $\alpha$**, strictly *more expressive* than $\alpha$ (splitting is roughly symmetric, absorption is not — the two orderings separate what $\alpha$ conflates), and robust exactly where their estimator fails (low-magnitude pairs, their App. K).
- Inherited assumptions to state explicitly: their guarantee (Eq. 13) is **conditional**, holding for 88.1% of pairs, and rests on $\sum z_1^2 \gg \sum z_2^2$ plus the frequency gap behind Eq. 14. Worth checking whether the indicator formulation relaxes the magnitude-dominance assumption.

**Population assumption from EWG-SAE §2.3.** Hypernyms are necessarily *more frequent* than their hyponyms — every child instance implies the parent, not vice versa. So $P(\text{parent}) \gg P(\text{child})$ with the child's support nested inside the parent's, and **absorption is the gap between "child implies parent by frequency/logic" and "parent actually fires given child."** The null becomes: no absorption ⟺ $P(A|B)$ ≈ its value under independence-given-nesting.

> ⚠️ EWG-SAE's *empirical* claims are unreliable and should not be cited as load-bearing: their own table shows vanilla JumpReLU (0.0114) beating their method (0.0125) on absorption; the dimension-group list has nine entries while the text says K=5; and their reconstruction/L0 numbers (explained variance 0.73, L0 ≈ 2666) are far off from C2R/OrtSAE on the same model and layer.

**Open decision blocking the derivation:** the distributional assumption on $z_1$ (parent activation magnitude on shared samples). Half-normal or exponential keeps the threshold-crossing probability closed-form; the stated preference is to condition on the **empirical $z_1$ distribution from `infer_z`** instead.

**Agreed validation ladder:**
1. Derive $(1 - P(A|B))$ as a closed-form function of $\alpha$ under C2R's Table 2 model with a JumpReLU threshold. *(≈ a page of algebra, not a compute job)*
2. Synthetic simulation with known $\alpha$ — check monotone recovery and match to the closed form. Stress-test corners: independent latents → 0; splitting (symmetric, both fire) → directional term stays low while cosine is high; true absorption → both channels agree.
3. SAEBench retrieval AUROC as a **single confirmation**, not a search. Underperformance now becomes informative — it says the generative model is missing something, sending you back to the model rather than to a new formula.
4. Architecture-ranking probe-free check: does the metric reproduce the known TopK > BatchTopK > Ort > Matryoshka absorption ordering?
5. Cross-modal deployment on a vision SAE — the payoff.

### Late July – Aug 1 — Rank plots

Streaming batch-wise per-latent statistics over active tokens: `max(zi|zi>0)`, `min(zi|zi>0)`, `E(zi|zi>0)`, `std(zi|zi>0)`, plus off-diagonal `zi·zj` Gram magnitudes. Sorted into rank plots (interactive Plotly, HTML output) to check whether any correlate with absorption patterns. JumpReLU layer 3 k=59; ReLU layer 12 variant; `NeelNanda/pile-10k`.

**Bugs found and fixed** (worth not re-introducing):
- `min_z` initialized to zeros instead of `+inf` → never updates, silently returns 0 for every latent.
- `Z[Z>0].min(dim=0)` flattens via boolean indexing → one global scalar, not a per-latent vector. Use `Z.masked_fill(Z <= 0, inf).min(dim=0)`.
- `torch.load()` called without a path argument.
- **Most consequential:** independent `torch.sort` on each statistic discards the latent-index correspondence (`_` throwing away sort indices), destroying exactly the cross-stat and absorption-pattern correspondence the plots exist to reveal. Use `argsort` and carry original latent IDs through as `customdata`.
- Accumulators allocated on the wrong device; division by zero for dead latents; negative variance from float error before `sqrt` (clamp with `.clamp_min(0)`).

**Plot-quality fixes:** marker `circle` size 3 (not `diamond` size 14 with black outlines — total overplotting); log axes on both dimensions for heavy-tailed data; filter to finite positive values *before* sorting; one full-screen HTML per statistic using `autosize=True` with `config={"responsive": True, "scrollZoom": True}`.

---

## 4. Related-work map

**Core / load-bearing**
- **Chanin et al., "A is for Absorption"** (arXiv:2409.14507, NeurIPS 2025) — the canonical definition, the probe-based pipeline this project is trying to replace, and the source of the 6510/1085 example and Fig. 6a.
- **C2R** (Jin et al.) — generative model with the $\alpha$ coordinate; labeled pair set (N ≈ 4,555) on this exact model. *Mitigation* method, not a detector.
- **OrtSAE** (Korznikov et al.) — orthogonality-based mitigation; also introduces "composition" (independent co-occurring features merging, e.g. "red square") as a distinct-but-adjacent failure.
- **Hindupur, Lubana, Fel, Ba** (arXiv:2503.01822, NeurIPS 2025) — duality paper, theoretical backbone.

**Mechanism / cause**
- **Feature Hedging** (Chanin, Dulka & Garriga-Alonso, arXiv:2505.11756) — the generalization: narrow SAE + correlated features → merged components, attributed to *reconstruction loss* rather than the sparsity penalty. Absorption becomes the special case (perfect hierarchical dependence); hedging is the general second-order-correlation case. Also notes the **decoder bias behaves as an always-on feature** in a degenerate hierarchy with every other feature — one reason absorption-like effects are hard to eliminate.
- **Sparse but Wrong: Incorrect L0 Leads to Incorrect Features** (arXiv:2508.16560) — same phenomenon through the L0 lens, with a decoder-projection-magnitude diagnostic for correct L0.
- **LLNL masking-regularization work** — locates the cause squarely in co-occurrence rather than hierarchy per se.
- **Tree SAE** — co-occurrence framing.

*Bottom line from the literature search: no paper reports canonical first-letter-style absorption arising from genuinely independent features. Some form of asymmetric co-occurrence appears necessary — but "hierarchy" in this literature is really a statement about co-occurrence structure, and several lines push the cause outward from strict parent⟹child.*

**Theory / identifiability**
- **Taming Polysemanticity: Provable Feature Recovery via SAEs** (arXiv:2506.14002) — quantitative co-occurrence threshold ($\rho_2 \ll 1/\sqrt{n}$) below which recovery sharply declines; modality-neutral by construction.
- **A Stable Neural Statistical Dependence Estimator** (arXiv:2603.11428) — the methods piece if the claim needs to move from *covariance* to genuine *statistical dependence* (higher-order).
- **A Unified Theory of Sparse Dictionary Learning** (arXiv:2512.05534) — optimization landscape, piecewise biconvexity, spurious minima.

**Testbeds / skepticism**
- **SynthSAEBench** (arXiv:2602.14687) — synthetic data with configurable correlation, hierarchy, and ground-truth features.
- **SAEs Can Interpret Randomly Initialized Transformers** (arXiv:2501.17727) — many SAE features may reflect data/architecture statistics rather than learned computation. Directly relevant to whether ZTZ structure says anything about Gemma's computation vs. corpus statistics.
- **Sanity Checks for SAEs** (Korznikov et al., arXiv:2602.14111) — 9% true-feature recovery at 71% explained variance.
- **Open Problems in Mechanistic Interpretability** (Sharkey et al., arXiv:2501.16496).

**Lab-adjacent**
- **MP-SAE / matching pursuit** (Costa, Fel, Lubana, Tolooshams, Ba) — unrolls matching pursuit for residual-guided extraction of correlated features; the quasi-orthogonality assumption it identifies is exactly what the ZTZ heatmaps probe.
- **SAE Neural Operators** (Tolooshams, Shen, Anandkumar, arXiv:2509.03738) — parameterization as a driver of interpretability; the theoretical framing behind H1–H4.

**Cross-modal targets**
- Vision SAEs (Stevens et al.), hierarchical CLIP SAEs.

---

## 5. Where things stand

The infrastructure exists — pipeline, metric definition, visualization suite, a worked case study, a mapped literature. What's missing is the step that turns a metric *definition* into a validated *detector*.

**The blocking next step is the derivation** — item 1 on the validation ladder. It is a page of algebra, not a compute job, and it resolves the stated discomfort about understanding the formula rather than selecting it. Everything else in the ladder is queued behind it.

The one open decision inside it: fix the distribution on $z_1$, or condition on the empirical `infer_z` distribution.

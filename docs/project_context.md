# SAE_statind — Consolidated Project Dossier

> **Verified against code as of 2026-08-27.** Sections 1–5 carry forward the narrative history assembled from conversation logs (May–August 2026); §6 records what a full pass over the repository at commit `8d6cb68` confirmed, corrected, or found missing. Claims marked ✅ were checked against source; ⚠️ marks a discrepancy between this document and the code; ❔ marks a claim no artifact in the repo can confirm or deny.

*Intended as a Project knowledge file — update in place as things change.*

---

## 0. Provenance of this revision

| | |
|---|---|
| **Verified at** | commit `8d6cb68` ("aug 27"), working tree clean |
| **Source document** | `project_context.md` (repo root) — 221 lines, committed in `8d6cb68` |
| **Last compute run** | 2026-08-14 (job 482855, rank plots) |
| **Last data produced** | 2026-08-01 (job 277588, `ztz_layer3_jumprelu_k59.pt`) |

**Two caveats on this revision's sourcing.**

1. The document was referred to as `docs/project_context.md`. No such path existed. The matching document was `sae_statind_project_dossier.md` at the repo root; it was renamed to `project_context.md` and committed as part of `8d6cb68` on Aug 27. This revision is written to `docs/project_context.md`; **the root copy is left untouched**, so the two must be reconciled by hand (see §7).
2. **No past Claude Code session transcripts exist on the cluster.** `~/.claude/projects/-project-aip-bahtol-rhyderi1-sae-statind/` contains exactly one `.jsonl`, which is the session that produced this revision, and `memory/` is empty. Per `CLAUDE.md`, this project is normally driven from a laptop with the directory mounted over SSHFS, so prior transcripts live under *that* machine's `~/.claude/projects/`, not here. **Every implementation-level detail in §6 was therefore recovered from the code, the git history, the uncommitted diffs, and the 759 SLURM logs in `logs/` — not from transcripts.** If the laptop's transcripts are ever available, they are still worth a pass; they may contain reasoning that left no trace in the repo.

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

> ⚠️ **Target setup has drifted.** The SAE registry (`SAE_DATA` in `src/infer_z.py:41`) covers **layers 3, 12, and 19** — layer 19 is fully populated for ReLU/TopK/JumpReLU/Matryoshka and has 12 figure directories under `figures/dtdxztz_scatterplots/dtdxztz_scatter/`. Layer 19 is not mentioned anywhere in the original document. Separately, the *centre of gravity of recent work is layer 3 JumpReLU k=59*, not layer 12: it is the only config in `config/params.csv`, the only one with a current Gram matrix, and the subject of every job since July. See §6.2.

---

## 2. The metric

$$\text{Absorb}(A \to B) = \cos(d_A, d_B) \times (1 - P(A \mid B))$$

- Run in **both orderings**; parent identified as the higher-frequency latent ($f_A > f_B$).
- **Cosine factor** = geometric direction-sharing; identifies which pairs *could* share a feature.
- **Conditional factor** = the directional asymmetry. $P(A|B) \approx 1$ means intact hierarchy; absorption is $P(A|B) \ll 1$ *combined with* high decoder cosine.
- Headline validation is **ROC-AUC against SAEBench's labeled `S_main`/`S_abs` pairs** — threshold-free. The only threshold in play is the activation binarization threshold, inherited from SAEBench's own convention rather than chosen independently.

> ⚠️ **The headline validation plan has a coverage problem the document does not record, and the repo already contains both the diagnosis and the fix.** See §6.3 — SAEBench's labels cover only ~19.8% of the relevant vocabulary, and `src/absorption_full_vocab.py` exists specifically to widen that. This materially affects what a ROC-AUC against `S_main`/`S_abs` would mean.

### Design decisions locked in
- **Multi-absorber directed edges**, not single-absorber attribution. Each $A \to B_i$ edge keeps its own weight; a decoder-projection threshold prevents attributing noise to hundreds of spurious latents. The single-absorber (argmax) view is derived *post hoc* for SAEBench comparability. Collapsing upfront would discard information irreversibly and hide the diffuse high-frequency signal.
  - ✅ Consistent with `src/extract_absorption_sets.py`, whose docstring explicitly flags that using `is_full_absorption` forces exactly one absorber per event by construction (`top_projection_feat`) — i.e. the JSON is the *collapsed* view. The multi-absorber information survives in `results/absorption_full_*.csv`, which keeps `top_projection_feat` alongside `absorption_fraction`.
- **Two-rung structure**, mapping onto the Bahareh/Valérie discussion of sparse non-Gaussian activation statistics:
  - *Rung 1* — indicator statistics (binary co-firing), immune to sparsity pathologies.
  - *Rung 2* — on-support code statistics (magnitude), requiring joint-support normalization and division by co-activation count.
  - ⚠️ **Rung 1 is not currently being computed.** The indicator-Gram accumulation in `src/compute_gram_from_tokens.py:131-139` is commented out, and so is the line that saves it. See §6.1 — this is the single most consequential drift in the document.
- **Escalation ladder** (covariance → entropy → conditional entropy) is a ladder to climb only if binary AUC proves insufficient, *not* a set of metrics to compute in parallel.
- **Parent-axis identification without labels**: the x-axis range asymmetry in triangle plots is both the visual signature of the parent's higher base rate *and* the recovery mechanism for axis assignment on unlabeled pairs.

### Technical conventions
- `from_pretrained_no_processing` with `center_writing_weights=False` for Gemma-2-2B (RMSNorm requirement).
  - ⚠️ **Honoured in 14 of 16 scripts — but violated in exactly the two that generate the data.** See §6.4.
- `sae.encode(X)` rather than manual encoder-column projection. ✅ Confirmed throughout.
- Joint-support standardization for STD panels. ❔ No current script implements this; see §6.5.

---

## 3. Timeline

*Dates in this section come from conversation history and are ❔ unverifiable from the repo before mid-June, which is where the git history begins in earnest. Dated commits and SLURM logs corroborate June onward.*

### May 2026 — Scoping and the H1–H4 framing

USRA proposal began as "inductive biases in SAEs," was critiqued as too broad and under-defined, then rewritten with: a formal definition of inductive bias in the SAE optimization context, narrowed scope (primary **L1 vs TopK**; secondary **Gated vs JumpReLU**), and four falsifiable hypotheses derived from theory *before* experiments:

- **H1** — lasso shrinkage bias
- **H2** — absorption as a downstream consequence of L1 shrinkage
- **H3** — TopK splitting when true sparsity $s < k$
- **H4** — Gated SAEs and the shrinkage/selection decoupling

Evaluation frame: SynthSAEBench + Hungarian matching. Confound controls: hyperparameter sensitivity, matched sparsity, penalty-strength sweeps, initialization variance, normalization, synthetic prior choice.

Deliverable: 12-slide plain-white deck (Cambria titles / Calibri body, thin gray rules), presented at lab meeting May 5–6 with Bahareh and Valérie.

Theory touchpoints from this period: the irrepresentable condition (Zhao & Yu, 2006), cross-polytope geometry of the L1 ball, completeness vs. precision as evaluation metrics, superposition geometry, connections to compressed sensing / RIP.

> ❔ **Note on H1–H4.** No script in the repo tests any of H1–H4, and no Gated SAE appears in `SAE_DATA` or anywhere else. The `batchtopk` architecture is listed in the `--arch` argparse choices of both `infer_z.py:112` and `compute_gram_from_tokens.py:52` but has **no entry in `SAE_DATA`**, so passing it raises `KeyError`. Either the H1–H4 program was superseded by the absorption-detector program, or it is dormant. The document should say which; a reader would currently assume the hypotheses are live.

### Late May — First training run

Baseline ReLU SAE trained from scratch: `StandardTrainingSAEConfig`, `l1_coefficient=5`, `d_in=2304`, `lr=2e-4`, dataset `Skylion007/openwebtext`. 30,000 steps, 61.44M tokens, ~1h44m on an L40S-48GB.

SAEBench absorption eval → `absorption_rate: 0.0420`. Flagged as **artificially low due to undertraining** (MSE ≈ 503).

Environment blockers cleared: pyarrow on Vulcan (load `arrow/24.0.0` *before* venv activation, plus `PYTHONPATH` to the module's site-packages), `HF_TOKEN` for the gated Gemma-2-2B.

- ✅ `src/sae_train.py` survives and matches (`d_in=2304`, `d_sae=16384`, 30k steps, openwebtext). Its docstring now correctly marks it as **not part of the analysis pipeline** — all reported results use pretrained SAEs from `SAE_DATA`.
- ✅ The `arrow/24.0.0`-before-venv ordering is preserved in `scripts/run_inferz.sh` and `scripts/run_job7.sh` (`module load gcc arrow/24.0.0`, then `source .venv/bin/activate`). Note `scripts/run_job6.sh` **omits** the arrow load — fine for `sorted_lineplots.py`, which reads a `.pt` off disk, but it will break if that script ever touches a parquet or a streaming dataset.
- 🔴 **`HF_TOKEN` is hardcoded in plaintext in at least three committed shell scripts** (`run_inferz.sh`, `run_job6.sh`, `run_job7.sh`). It is in git history and therefore compromised. See §7.

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

✅ All four distinctions are corroborated by the eval outputs actually on disk, and the numbers are worth recording — the document never states them:

| SAE | `mean_full_absorption_score` | `mean_absorption_fraction_score` |
|---|---|---|
| `gemma-scope-2b-pt-res` layer 3, 16k, L0 59 (JumpReLU) | 0.0574 | 0.1165 |
| `sae_bench…vanilla_width-2pow14` blocks.12 trainer_0 (ReLU k20) | 0.0197 | 0.0911 |

*(from `eval_results/absorption/*.json`; std devs 0.0596 / 0.1153 and 0.0398 / 0.1310 respectively)*

The fraction score being ~2–5× the full score, and the gap being much wider for the layer-12 ReLU SAE, is exactly the "fraction score is more sensitive" claim in numbers.

> 🔴 **The width confound is live in the code right now.** `src/compute_feature_absorption.py:53-70` runs the SAEBench eval over a `selected_saes` list built entirely from **width-2pow16 (65k)** SAEs at **layers 12 and 20**. The geometry/statistics half of the project (`SAE_DATA`) uses **width-2pow14 (16k)** at layers 3/12/19. The two halves of the project are operating on **different SAE populations**, which is precisely the comparison the document warns is invalid. There is also a probable mislabel in that list: the block commented `# --- Layer 20 | TopK ---` points at `blocks.19.hook_resid_post__trainer_2`.

Multi-SAE sweeps within one model are natively supported via regex; multi-model sweeps require an outer loop over `run_eval` with separate output folders and explicit model deletion between iterations.

### June — Activation pipeline

`infer_z.py` (originally `june7test.py`): collects residual-stream activations X and SAE latents Z from Gemma-2-2B, running as SLURM array jobs. ✅ Confirmed, including the `--task_id` → `config/params.csv` array-index mechanism.

**Bahareh's June 8 directives:** disable shuffling; stay under 1 TB; save X for only one SAE; compute $Z^\top Z$ heatmaps; produce sparsity-aware statistics; reproduce Valérie's `dict_vis` figure; write everything mathematically.

- ✅ *Disable shuffling* is honoured and, better, **relied upon**: `src/z_hist_bin_stats.py`'s docstring notes that because no shuffling is configured, a fresh `ActivationsStore` replays the identical batch sequence, which is what makes its two-pass rewrite exactly equivalent to the original one-pass computation.
- ✅ *Stay under 1 TB* is what `compute_gram_from_tokens.py` exists to achieve (§6.1).

Also built: streaming activation histograms, Z-histogram scripts, extraction of `S_main` / `S_abs` latent sets from the absorption parquet (per-letter groupby, with the `.iloc[0]` assumption that `split_feats` is constant within a letter, and explicit `int()` casts for JSON serialization).

- ✅ Verified line-for-line at `src/extract_absorption_sets.py:55-66`: `df.groupby("letter")`, `grp["split_feats"].iloc[0]`, `[int(x) for x in ...]`, `feat = int(feat)`.
- ✅ Output confirmed: `results/absorption_sets_jumprelu_layer3_k59_20260716_113640.json` has all 26 letters, `s_main['s'] = [6510]`, and **147 absorbers** for `s`. Keys are lowercase (`s_main`, `s_abs`, `s_abs_tokens`), not `S_main`/`S_abs` — worth knowing before writing a loader.
- ✅ `s_main` for the other case-study letters matches the hardcoded pairs in `absorption_metrics.py`: `u→16033`, `e→5407`, `o→1006`.

### June — DTD × ZTZ geometry

Scripts: `DTDxZTZ_scatter.py`, then `src/dtdxztz_scatter_simple.py`. Each point pairs the *same* $(i,j)$ entry from both Gram matrices — $D^\top D$ (decoder direction overlap) on x, $Z^\top Z$ (co-activation) on y. Off-diagonal only (`tril_indices`); `|DTD|` folding to use the full sample; log y-axis; `MAX_POINTS = 300_000` subsampling against ~134M pairs.

> ⚠️ **The script family has grown well past what the document lists.** There are now **six** DTD×ZTZ scripts, in two generations:
>
> | Script | Status |
> |---|---|
> | `dtdxztz_heatmap.py` | LEGACY — hardcoded dead shard path, writes to cwd |
> | `dtdxztz_binary_heatmap.py` | LEGACY — same, but binarised ZTZ |
> | `dtdxztz_scatter_simple.py` | current, single hardcoded config (layer 12 ReLU k20) |
> | `dtdxztz_scatter_full.py` | **current workhorse** — sweeps `params.csv`, 3×2 grid |
> | `dtdxztz_scatter_full_overlay.py` | current — same grid, sparsities overlaid per (layer, arch) |
> | `fix_axes.py` | shared axis-limit convention record |
>
> `MAX_POINTS = 300_000` ✅ confirmed at `dtdxztz_scatter_full.py:32`.

**Architecture signatures observed (layer 12):**

| Architecture | Signature |
|---|---|
| ReLU | Positive-correlation "fin"; near-duplicate features at DTD ≈ 1.0 |
| TopK | Tighter; geometry largely decoupled from co-activation |
| JumpReLU | Bimodal vertical structure driven by threshold-gating artifacts |
| Matryoshka | Near-orthogonal decoders (DTD ≈ 0) but structurally high co-activation |

**Caveats flagged:** raw ZTZ is *not* comparable across architectures (y-scales span 10⁶–10⁹) — normalize to correlation form first (Bahareh's directive). Mask the diagonal before any correlation analysis. Some plots were clipping at 10⁸.

**Recommended summarization strategy:** collapse scatters to scalars — Spearman ρ between DTD and log ZTZ, plus a quadrant-fraction metric — then present as an architecture×layer heatmap grid at canonical sparsity/width, a separate sparsity-vs-width plot for one focal architecture, one anchor hexbin for interpretability, and an interactive HTML viewer for browsing. A CSV with one row per configuration is the reproducibility backbone.

> ⚠️ **This is listed as "recommended" but is substantially already built.** `dtdxztz_scatter_full.py` writes `spearman_summary.csv`, one row per config, carrying **four** rank correlations — `rho_cos_ztz`, `rho_cos_ratio`, `rho_cos_l0`, `rho_raw_ztz` — plus `d_sae`, `q_alive`, `n_dead`, `n_pairs`, `seed`. It also drops dead latents (never active in the corpus) before sampling, which the document's caveat list does not mention but which matters as much as diagonal masking. What is genuinely still missing: the **quadrant-fraction metric**, the **architecture×layer heatmap grid**, the **anchor hexbin**, and the **interactive HTML viewer**. Rewriting this paragraph as "CSV backbone ✅ done, presentation layer ⬜ pending" would keep the next reader from rebuilding the CSV.
>
> The script also implements the normalization directive more thoroughly than described: it plots three y-quantities (`zᵢᵀzⱼ`, `zᵢᵀzⱼ / co-activation count`, and the co-activation count itself) against two x-quantities (`dᵢᵀdⱼ` and `cos(dᵢ,dⱼ)`) — the middle y-quantity being the joint-support normalization the metric's rung 2 calls for.

**Separate result:** OrtSAE vs. pretrained SAEs — OrtSAE flattens the absorption-vs-sparsity curve, outperforming at low sparsity and underperforming at high L0, with a crossover between k=40 and k=70. ❔ No OrtSAE code or checkpoint in the repo; this result is not reproducible from here.

### Late June — The directional turn

Started from symmetric $Z^\top Z$ / $D^\top D$; identified that **absorption is inherently directional and symmetric statistics cannot capture it**. Worked through the conditional-probability framing (parent A, child/absorber B) using asymmetric co-firing ratios. A sign error was caught and corrected — $P(A|B) \approx 1$ describes an *intact* hierarchy, not absorption — yielding the final metric above.

Triangle plot read geometrically: the y-axis cluster of absorption events is the binary co-firing deficit (rung 1); the negatively-sloped joint-activation blob is partial/magnitude absorption that the binary metric misses and on-support Pearson captures (rung 2).

> ⚠️ **One directional metric was implemented and the document omits it entirely.** `src/absorption_metrics.py` computes what it calls **"Absorption Metric 3"**:
>
> $$\text{M3}(i,j) = \frac{E[z_i \mid i] - E[z_i \mid i \wedge j]}{E[z_i \mid i] + \varepsilon} \times P(j \mid i)$$
>
> This is *not* the headline metric — it is magnitude-based (rung 2) rather than indicator-based, uses $P(j|i)$ rather than $1 - P(A|B)$, and carries no cosine factor. Metrics 1 and 2 are referenced in commented-out print statements but their definitions are gone. The script's design is otherwise good and worth preserving: it evaluates the three top absorption pairs (`16033→12304` for *u*, `5407→10622` for *e*, `1006→731` for *o*) plus `6510→1085`, **against three matched control pairs that reuse the same absorber $j$ with a different letter's main latent $i$** — a real control for "does this absorber just fire a lot?". Whether M3 was superseded or abandoned is not recorded anywhere.

### July — Case study 6510 / 1085

Pair: **6510** (parent, "starts with S") and **1085** (child, "short"), JumpReLU layer 3, k=59. Built $z_i$ vs $z_j$ scatters with a conditional-probability / conditional-expectation panel, under two conditions: all tokens vs. S-initial tokens only.

**Result:** S-token conditioning left the *continuous scatter geometry* unchanged, but shifted the firing conditional $P(\text{parent}|\text{child})$ from **0.13 → 0.31**.

**Interpretation:** this is positive evidence, not a null result. The absorption signature lives in **binary firing structure**, not activation magnitudes — which points toward the ZTZ indicator variant over continuous correlations. It also predicts *where* a firing-based estimator should beat an energy-based one: the low-magnitude regime.

Bar chart (replicating Chanin et al. Fig. 6a, with sample counts, standard deviations, and random control latents) cleanly shows the absorption hole: 6510 fires on other S-tokens but ≈0 on "short"; 1085 fires only on "short".

- ✅ `src/chanin_fig6a.py:41-42` hardcodes `Latent1 = 6510`, `Latent2 = 1085`, plus two seeded random control latents (`Latent3`, `Latent4`) — matching the description exactly. Five output figures survive in `figures/chanin_fig6/jumprelu_layer3_k59/`, the last from 2026-07-19.
- ⚠️ **`zizj_scatter.py` no longer points at 6510/1085.** Its `PAIRS` list (line 40) is now `(477, 10069)`, `(7985, 3645)`, `(4014, 622)`. Anyone re-running the case study from that script will silently get three different pairs. Likewise `coact_match.py` defaults to `--latent 6840`, not 6510.

Caveat carried forward: a single pair is too noisy to conclude much either way — the right unit of analysis is the *aggregate* over many pairs.

> ✅ **This caveat was acted on and the document does not say so.** `src/coact_match.py` builds **co-activation-matched control pairs**: for an anchor latent $i$ it counts how many tokens every other latent co-fires with $i$, then for each true absorber $j$ reports non-absorbing controls whose co-activation with $i$ is closest to $j$'s. That is the correct control for the obvious confound — that absorbers just co-fire more — and it is the first step toward the aggregate analysis the caveat calls for.

Side deliverable: an 18-slide PPTX walkthrough of "A is for Absorption" with full speaker notes. ❔ Not in the repo.

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

> ✅ **The empirical option is now cheap, and the document predates the machinery that makes it so.** The per-latent on-support statistics needed to characterize the empirical $z_1$ distribution — `max(zᵢ|zᵢ>0)`, `min(zᵢ|zᵢ>0)`, `E(zᵢ|zᵢ>0)`, `std(zᵢ|zᵢ>0)` — are exactly what `src/sorted_lineplots.py` already streams. For latent 6510 specifically, those four numbers are one indexing operation away from a run that has already been done. This decision is more tractable than the document implies. (Two caveats before trusting those numbers: §6.6.)

**Agreed validation ladder:**
1. Derive $(1 - P(A|B))$ as a closed-form function of $\alpha$ under C2R's Table 2 model with a JumpReLU threshold. *(≈ a page of algebra, not a compute job)*
2. Synthetic simulation with known $\alpha$ — check monotone recovery and match to the closed form. Stress-test corners: independent latents → 0; splitting (symmetric, both fire) → directional term stays low while cosine is high; true absorption → both channels agree.
3. SAEBench retrieval AUROC as a **single confirmation**, not a search. Underperformance now becomes informative — it says the generative model is missing something, sending you back to the model rather than to a new formula.
4. Architecture-ranking probe-free check: does the metric reproduce the known TopK > BatchTopK > Ort > Matryoshka absorption ordering?
5. Cross-modal deployment on a vision SAE — the payoff.

> ⚠️ **Rung 3 and rung 4 both have unrecorded blockers.** Rung 3's label set covers ~19.8% of the vocabulary (§6.3). Rung 4 needs BatchTopK, which is in the argparse choices but absent from `SAE_DATA`, and OrtSAE, which is absent from the repo entirely. Neither is a large job, but neither is zero, and the ladder as written reads as though only rung 1 stands between the project and rung 5.

### Late July – Aug 1 — Rank plots

Streaming batch-wise per-latent statistics over active tokens: `max(zi|zi>0)`, `min(zi|zi>0)`, `E(zi|zi>0)`, `std(zi|zi>0)`, plus off-diagonal `zi·zj` Gram magnitudes. Sorted into rank plots (interactive Plotly, HTML output) to check whether any correlate with absorption patterns. JumpReLU layer 3 k=59; ReLU layer 12 variant; `NeelNanda/pile-10k`.

**Bugs found and fixed** (worth not re-introducing):
- `min_z` initialized to zeros instead of `+inf` → never updates, silently returns 0 for every latent. ✅ **fixed** — now `torch.full((p,), float('inf'), …)`.
- `Z[Z>0].min(dim=0)` flattens via boolean indexing → one global scalar, not a per-latent vector. ✅ **fixed** — though the implementation is `torch.where(Z > 0, Z, inf).min(dim=0).values`, not the `Z.masked_fill(Z <= 0, inf)` idiom the document records. Equivalent; the document's version is not what is in the file.
- `torch.load()` called without a path argument. ⚠️ **fixed, but badly** — replaced with a hardcoded absolute path (`/home/rhyderi1/projects/aip-bahtol/…/ztz_layer3_jumprelu_k59.pt`) that also routes through the `~/projects` symlink rather than `/project/…`. Not portable, and silently wrong the moment the config changes.
- **Most consequential:** independent `torch.sort` on each statistic discards the latent-index correspondence (`_` throwing away sort indices), destroying exactly the cross-stat and absorption-pattern correspondence the plots exist to reveal. Use `argsort` and carry original latent IDs through as `customdata`. ✅ **fixed** — all four statistics now keep their index vectors (`max_idxs`, `min_indxs`, `mean_indxs`, `std_indxs`) and pass them as Plotly `customdata`; the `zizj` panel carries both `i` and `j`.
- Accumulators allocated on the wrong device. ✅ **fixed** — all five now take `device=Z.device, dtype=Z.dtype`.
- Division by zero for dead latents. 🔴 **NOT fixed** — see §6.6.
- Negative variance from float error before `sqrt` (clamp with `.clamp_min(0)`). 🔴 **NOT fixed** — see §6.6.

> **Four more bug fixes are visible in the diff and absent from the document** — see §6.6 for the full list, including one silent correctness bug in the `zizj` extraction.

**Plot-quality fixes:** marker `circle` size 3 (not `diamond` size 14 with black outlines — total overplotting); log axes on both dimensions for heavy-tailed data; filter to finite positive values *before* sorting; one full-screen HTML per statistic using `autosize=True` with `config={"responsive": True, "scrollZoom": True}`.

> ⚠️ **This paragraph describes an output format the code does not produce.** What `sorted_lineplots.py` actually writes is:
> - **one static PNG** (`matplotlib`, 2×3 grid, `figsize=(15,10)`, dpi 150, marker `.` size 1, log–log) containing **all** points for all five statistics — because the full `zizj` set is too large for an interactive page; and
> - **one interactive HTML** (Plotly, 2×3 subplot grid, fixed `height=1400, width=1400`, marker `circle` **size 6**, log–log) in which the `zizj` panel is truncated to `TOP_K = 1000` pairs.
>
> There is no per-statistic HTML, no `autosize=True`, and no `config={"responsive": …, "scrollZoom": …}`. Marker size is 6, not 3. The PNG/HTML split is a sensible response to page weight (the surviving HTML is 5.8 MB even at TOP_K=1000) and is a better design than what the document describes — but a reader following the document will look for files that do not exist.

---

## 4. Related-work map

*❔ Nothing in this section is verifiable from the repository — there is no `references/`, no BibTeX, no PDF cache. Carried forward unchanged. Two entries are flagged below where the code bears on them.*

**Core / load-bearing**
- **Chanin et al., "A is for Absorption"** (arXiv:2409.14507, NeurIPS 2025) — the canonical definition, the probe-based pipeline this project is trying to replace, and the source of the 6510/1085 example and Fig. 6a. ✅ Vendored as `sae_bench.evals.absorption` and imported directly by `absorption_full_vocab.py` and `archived_scripts/replicate_absorption_figs67.py`.
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
- **SynthSAEBench** (arXiv:2602.14687) — synthetic data with configurable correlation, hierarchy, and ground-truth features. *(This is the natural home for validation-ladder rung 2.)*
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

> **This assessment still holds, with three amendments from the code review.**
>
> 1. The derivation is still the right next step and is still not blocked by compute. Nothing found contradicts that.
> 2. But **"the infrastructure exists" overstates the case for rung 1 of the metric.** The indicator statistics the binary metric is defined on are not currently being produced for the config all recent work uses (§6.1). That is a ~13-hour job, not a design problem — but it is not "done," and the document reads as though it is.
> 3. **The project has been idle since 2026-08-14** (last successful job, 482855). Whatever context was in working memory then is now 13 days cold; §6 is partly an attempt to reconstruct it.

---

## 6. Code verification — findings (2026-08-27)

### 6.1 🔴 The indicator Gram is not being computed

The document's rung 1 rests on binary co-firing statistics. The script that would produce them, `src/compute_gram_from_tokens.py`, has the accumulation **commented out**:

- lines 131–139: the `Z_ind_flat` / `ZindTZind` / `G_ind` block is entirely commented
- line 159: the `torch.save(G_ind.cpu(), …zindtzind…)` line is commented
- the uncommitted-then-committed diff in `8d6cb68` also commented out the NaN check, which had been dereferencing the now-`None` `G_ind` and would have raised `AttributeError`

Consequence, confirmed on disk:

| config | `ztz_*.pt` | `zindtzind_*.pt` |
|---|---|---|
| `layer12_relu_k20` | ✅ 2026-06-26 | ✅ 2026-06-26 |
| `gram_layer3_jumprelu_l059` | ✅ 2026-08-01 (1.07 GB) | ❌ **absent** |

So the indicator Gram exists only for the layer-12 ReLU config, and the current focal config (layer 3 JumpReLU k59) — the one behind the case study, the rank plots, and `params.csv` — has only the magnitude Gram. **Uncommenting three blocks and re-running `scripts/run_job7.sh` produces it**; the job took ~13 h wall for ZTZ alone (`--time=13:00:00`), and computing both roughly doubles the per-batch matmul cost, so budget accordingly or split into two jobs.

The script's docstring is also stale on two points: it advertises saving both matrices, and it gives the output path as `data/pile-10k-saes/layer{L}_{arch}_k{sp}/` while the code writes `data/pile-10k-saes/gram_layer{L}_{arch}_l0{sp}/`.

### 6.2 The current data path is a script the document never mentions

`src/compute_gram_from_tokens.py` (first appears in the tree ~2026-07-27) is now the primary data-generating script, and it is absent from the document. It is the answer to Bahareh's "stay under 1 TB" directive: same streaming setup as `infer_z.py`, but it accumulates the $p \times p$ Gram on the fly and **never writes Z shards**, avoiding the hundreds of GB `infer_z.py` produces.

It also fixes an epoch-boundary bug that `infer_z.py` still has. It calls `get_batch_tokens(batch_size, raise_at_epoch_end=True)` inside a `try/except StopIteration` and stops cleanly when the dataset wraps; `infer_z.py` calls `get_batch_tokens(batch_size)` with no such guard and will silently re-read the start of the corpus, double-counting it. This is what `src/check_batch_ceiling.py` exists to settle — it computes the exact number of full batches before wrap-around, in two modes (`fast`, replaying SAELens's `concat_and_batch_sequences` packing with the tokenizer only; `store`, driving the real `ActivationsStore` to `StopIteration`). The `n_batches` value moved **3823 → 3820** as a result; the old 3823 survives in `scripts/run_inferz.sh` and as a comment in `sorted_lineplots.py:28`.

Script provenance is worth recording, since git makes it confusing: `scripts/run_compute_gram_from_tokens.sh` was **renamed to `scripts/run_job7.sh`** in `8d6cb68` (git records it as a delete + add; `--find-renames` shows 6 lines changed — the job name, and `--n_batches 3823` → `3820`).

### 6.3 🔴 SAEBench's labels cover ~19.8% of the vocabulary — and the repo already documents this

`src/absorption_full_vocab.py` is a substantial, well-documented script that the document does not mention, and its docstring contains a finding that bears directly on the headline validation plan. SAEBench's absorption eval scores only a slice of the vocabulary, for two compounding reasons:

1. `probing.create_dataset_probe_training` splits the alphabetic vocab 80/20 and evaluates on the **20% test half only**;
2. `feature_absorption.get_stats_and_likely_false_negative_tokens` then keeps only "likely false negatives" — tokens where the LR probe fires but the k-sparse SAE probe does not.

Verified numerically on disk:

| file | rows | note |
|---|---|---|
| `letter_s.csv` (SAEBench output) | 2,883 tokens | **19.8%** of the vocabulary |
| `results/absorption_full_s_jumprelu_layer3_k59_*.csv` | 14,563 tokens | full single-token `s` vocabulary |

The concrete damage, per the docstring: `' short'`, `' shorter'`, `' shortly'` and `'short'` **all landed in the train half**, so latent 1085 — the canonical "short"-family absorber of the entire case study — shows exactly **one** absorbed token (`'Short'`) in SAEBench's output. Colouring the $z_i$-vs-$z_j$ scatter from that list marks ~20 of the ~950 y-axis positions it should.

`absorption_full_vocab.py` re-runs `FeatureAbsorptionCalculator` over every single-token vocab entry for a letter, reusing the trained probe and the chosen `S_main` latents, with no split and no false-negative filter. Its output is a strict superset of `letter_s.csv` and additionally keeps `top_projection_feat` (the absorbing latent). Results already exist for both configs:

| config | tokens | `is_full_absorption` | `absorption_fraction > 0` |
|---|---|---|---|
| layer 3 JumpReLU k59 | 14,563 | 1,758 (12.1%) | 4,054 (27.8%) |
| layer 12 ReLU k20 | 14,563 | 2,727 (18.7%) | 10,557 (72.5%) |

**Implications the document should absorb.** (a) Rung 3 of the validation ladder — "SAEBench retrieval AUROC" — should run against the full-vocab CSVs, not the eval's labels, or the AUROC is computed on a filtered, non-random ~20% subsample whose filter is *itself* correlated with the thing being detected (it keeps tokens where the SAE probe fails). (b) `src/zizj_scatter.py` has already been migrated: `absorbed_token_ids_from_csv()` reads the full-vocab CSV and is documented as superseding the older `absorbed_token_ids()` JSON path for exactly this reason. (c) The document's "ROC-AUC against SAEBench's labeled `S_main`/`S_abs` pairs, threshold-free" needs rewording — the label set is a design choice with a known bias, not a neutral ground truth.

One usage constraint from the docstring, worth not rediscovering: the calculator requires every prompt to tokenize to the same length. This holds because `get_alpha_tokens` returns single-token entries only — **do not pass hand-written multi-token words.**

### 6.4 ⚠️ The model-loading convention is violated in the two scripts that generate data

The document records `from_pretrained_no_processing` (with `center_writing_weights=False`) as a locked convention for Gemma-2-2B's RMSNorm. Across the repo, 14 of 16 call sites honour it. The two that do not are:

- `src/infer_z.py:266` — `HookedTransformer.from_pretrained(...)`
- `src/compute_gram_from_tokens.py:175` — `HookedTransformer.from_pretrained(...)`

These are precisely the scripts that produce X, Z, and the Gram matrices that every other script consumes. Every *analysis* script (`sorted_lineplots`, `chanin_fig6a`, `zizj_scatter`, `absorption_metrics`, `coact_match`, `absorption_full_vocab`, `z_histogram*`, `check_batch_ceiling`, …) uses `from_pretrained_no_processing`. So activations are being **generated** under one convention and **interpreted** under another.

The SLURM log for the run that produced the current Gram (`logs/cmput_gram-277588.err`) shows TransformerLens reacting to this:

```
WARNING:root:You tried to specify center_unembed=True for a model using logit softcap,
  but this can't be done! ... Setting center_unembed=False instead.
WARNING:root:You are not using LayerNorm, so the writing weights can't be centered! Skipping
```

So TransformerLens defended against the two most dangerous defaults on its own — `center_writing_weights` was skipped, `center_unembed` was overridden. `fold_ln=True` still applied, and it is mathematically a no-op on `hook_resid_post` (the RMSNorm scale is folded into the *following* layer's weights). **The practical impact is therefore probably small — but "probably" is doing real work in that sentence, and the SAEs were trained on the unprocessed stream.** Before the Gram matrices are used for any published number, this deserves an explicit check: run a few hundred tokens through both loaders and compare `blocks.3.hook_resid_post` directly. That is a 5-minute job and it converts an assumption into a fact.

### 6.5 Other drift, briefly

| Finding | Location |
|---|---|
| `batchtopk` accepted by `--arch` but absent from `SAE_DATA` → `KeyError` at `get_sae`. Blocks validation-ladder rung 4. | `infer_z.py:112`, `compute_gram_from_tokens.py:52` |
| `sae_data.py` is a **duplicate, unused** copy of the registry that **omits the layer-3 entries** the absorption work depends on. Its own docstring says to prefer `from infer_z import SAE_DATA`. A live footgun. | `src/sae_data.py` |
| `z_histogram.py` is **broken as written** — `max_val` is used in the `np.histogram` call but its definition is commented out → `NameError`. | `src/z_histogram.py` |
| Absorption eval driver runs **65k-width SAEs at layers 12/20**; the rest of the project uses **16k-width at layers 3/12/19**. Also `# --- Layer 20 | TopK ---` labels a `blocks.19` sae_id. | `compute_feature_absorption.py:53-70` |
| `force_rerun=True` in the eval driver — cached results are always discarded, so every run costs 30–60+ min. | `compute_feature_absorption.py` |
| Nine scripts are marked LEGACY in their own docstrings (hardcoded shard paths from a directory layout that no longer exists, outputs to cwd): `dtdxztz_heatmap`, `dtdxztz_binary_heatmap`, `ztz_heatmaps_all_shards`, `ztz_heatmaps_with_cov_and_corr`, `visualizations`, `save_expectations`, `expectation_histogram`, `scratch`, `archived_scripts/replicate_absorption_figs67`. | `src/` |
| "Joint-support standardization for STD panels" (§2 conventions) has no implementation. The nearest thing is the `zᵢᵀzⱼ / co-activation count` y-quantity in `dtdxztz_scatter_full.py`. | — |
| `src/helloooooo.txt` — six lines of keyboard mash, committed in `8d6cb68`. Harmless; presumably unintended. | `src/helloooooo.txt` |

### 6.6 `sorted_lineplots.py` — full bug ledger

The document lists seven fixes. Comparing `0df38c1` to the committed state, **four more were made that the document omits**, and **two it claims were made were not**.

**Additional fixes, undocumented:**

1. `Z.max(dim=0)` / `Z.min(dim=0)` return a `(values, indices)` namedtuple, not a tensor. Passing that straight into `torch.maximum` fails. Now `.values` on both.
2. Variance used the **global** active count: `variance = (sum_sq/active_n) - mean**2`, where `active_n` is a scalar summed over all latents, while `mean` used the per-latent count. Dimensionally inconsistent and wrong by orders of magnitude. Now both use `active_n_per_latent`. *(The now-unused `active_n = 0` accumulator still sits at line 62.)*
3. `x = list(range(1, len(max) + 1))` referenced the **builtin `max`**, not `max_z` → `TypeError`. Now `len(max_z)`.
4. **Silent correctness bug in the `zizj` extraction.** The old code did `coords = torch.nonzero(Gram_tri)` and *separately* `zizj_values = Gram_tri[Gram_tri != 0]`. These are two different traversals producing two independently-ordered lists, then concatenated column-wise — so **every value was paired with the wrong `(i,j)` coordinate.** This is the same class of error as the discarded sort indices, and arguably worse because nothing about the output looks wrong. Now `rows, cols = torch.nonzero(Gram_tri, as_tuple=True)` with `zizj_values = Gram_tri[rows, cols]`, guaranteeing a single consistent traversal.
5. Also `torch.argsort(zizj_and_coords[:, 0], …)` → `torch.argsort(zizj_values, …)`, and `del X_sae` added after `sae.encode` to cut peak memory.

**Claimed fixed, but still present in the committed code:**

6. 🔴 **No negative-variance clamp.** Lines 113–114 are `variance = (sum_sq/active_n_per_latent) - mean**2` then `std_dev = variance**(1/2)`. The document says this was fixed with `.clamp_min(0)`. It was not. $E[z^2] - E[z]^2$ in float32 over ~15M tokens will go slightly negative for low-variance latents, and `(negative)**0.5` yields **NaN**, which then propagates into the sort and the plot. Fix: `variance.clamp_min(0)`.
7. 🔴 **No dead-latent guard.** `mean = sum_z/active_n_per_latent` divides by zero for any latent that never fires, giving NaN/Inf. The document says this was fixed. It was not. Note `min_z` *does* get a guard one line later (`torch.where(torch.isinf(min_z), 0.0, min_z)`) — so the pattern was clearly in mind, just not applied to `mean`/`variance`. `dtdxztz_scatter_full.py` handles this properly by dropping dead latents up front (`n_dead` is reported in the summary CSV); the same approach would work here.

**Two further issues, neither in the document:**

8. `REPO_ROOT = Path(__file__).resolve().parent` resolves to **`src/`**, not the repo root — so both outputs land in `src/figures/` instead of the top-level `figures/`. This is why `src/figures/` exists and was committed in `8d6cb68`. `absorption_metrics.py:31` gets this right with `.parent.parent`.
9. The Gram is loaded from a hardcoded absolute path routed through the `~/projects` symlink. It does not track `arch`/`layer`/`sparsity`, which *are* parameterised at the top of the file — so changing the config silently loads the wrong matrix.

Given 6 and 7, **the statistics in the current rank-plot HTML/PNG (`src/figures/…20260813_185617.*`) should be treated as provisional** until re-run. Any latent with near-zero on-support variance, and every dead latent, is likely NaN.

### 6.7 Timeline of recent compute (from `logs/`, 759 files)

| date | job | outcome |
|---|---|---|
| 2026-08-01 01:46 | `cmput_gram` 277585, 277587 | both **killed** (SIGTERM, ~10 s in) |
| 2026-08-01 01:56 | `cmput_gram` 277588 | ✅ produced `ztz_layer3_jumprelu_k59.pt` (1.07 GB) |
| 2026-08-01 00:06 – 01:42 | `run_job6` ×6 | short runs, iterating on `sorted_lineplots.py` |
| 2026-08-14 00:45 | `run_job6` 482806 | ❌ **CANCELLED — TIME LIMIT** (`--time=0:30:00`) |
| 2026-08-14 00:57 | `run_job6` 482855 | ✅ wrote the current rank-plot HTML + PNG |
| 2026-08-14 → 2026-08-27 | — | **idle** |

The 482806 timeout is worth noting: `run_job6.sh` was changed from `--time=8:00:00` to `--time=0:30:00` (and `--mem` 32G→64G) in the same edit session, and 30 min is marginal — the successful rerun needed most of it just for model load plus a full corpus pass. If `sorted_lineplots.py` is re-run after fixing the NaN bugs, restore a longer wall time.

---

## 7. Recommended next actions

Ordered by ratio of value to effort. Items 1–3 are cheap and unblock things; 4–5 are the research path.

1. **Reconcile the two copies of this document.** `project_context.md` (root, committed) and `docs/project_context.md` (this revision) now both exist. Keep one — the root path is what `8d6cb68` committed, so moving this file there and deleting the duplicate is the least disruptive.
2. 🔴 **Rotate the HF token.** It is hardcoded in `run_inferz.sh`, `run_job6.sh`, and `run_job7.sh`, all committed, so it is in the git history and must be considered compromised. Revoke it at huggingface.co/settings/tokens, issue a new one, and load it from `~/.config/…` or a non-committed `.env` sourced by the job scripts. Purging git history is optional; rotating is not.
3. **Fix the two NaN bugs in `sorted_lineplots.py`** (§6.6, items 6–7) and re-run with a restored wall time. Two one-line changes. Until then the per-latent statistics — including the ones that would settle the $z_1$ distribution question — are not trustworthy.
4. **Uncomment the indicator-Gram block in `compute_gram_from_tokens.py` and run it for layer 3 JumpReLU k59** (§6.1). This is the rung-1 data the whole binary metric is defined on, and it does not exist for the focal config. ~13 h, one job, no new code.
5. **Do the derivation** (validation-ladder item 1). Still the right blocking step, still a page of algebra. The one open decision inside it — the distribution on $z_1$ — is now cheap to settle empirically once (3) lands, since `sorted_lineplots.py` already computes the on-support statistics for latent 6510.

**Two things to reconsider while doing (5):**

- **Rung 3's label set is biased, not neutral** (§6.3). Decide now whether the AUROC runs against SAEBench's ~20% filtered labels or the full-vocab CSVs already sitting in `results/`. The choice changes what a number means, and it is much easier to decide before seeing the number.
- **Verify the loader inconsistency doesn't matter** (§6.4) rather than assuming it. Five minutes, and it either removes a caveat from every future claim or catches something serious.

**One structural note.** `src/` holds 40 Python files, nine of which are self-declared LEGACY with hardcoded paths into a directory layout that no longer exists, plus a duplicate registry that silently omits layer 3. The AI-generated docstrings (commit `82ed753`) are genuinely good and honest — several say "BROKEN" or "LEGACY" or "UNUSED" outright, which is how §6.5 was assembled quickly. Moving the nine LEGACY scripts into `src/archived_scripts/` (which exists, and holds exactly one file) would make the live surface of the project about a dozen files. Worth an hour, not urgent.

---

*Revision written 2026-08-27 against commit `8d6cb68`. No files in the repository were modified in producing it; `project_context.md` at the root is unchanged. Sections 1–5 preserve the original text with inline verification marks; §6–7 are new.*

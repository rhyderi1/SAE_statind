c# Implementation Spec — Absorption Membership Extraction + DTD×ZTZ Offline Analysis

**Audience:** Claude Code, implementing inside the existing `sae_statind` project.
**Project root:** `/project/aip-bahtol/rhyderi1/sae_statind` (Vulcan, account `aip-bahtol`).
**Goal:** For one chosen atom `k` tied to one concept `c`, extract absorption *membership sets* (`S_main`, `S_abs`) two ways — atom-mode and probe-mode — save them to disk, then run offline DTD×ZTZ analysis on the resulting pairs. Do **not** record absorption *scores*; record *indices*.

This spec assumes the existing pipeline (`infer_z.py`, `SAE_DATA` registry, `DTDxZTZ_grams.py`, and an adapted copy of SAEBench's `feature_absorption_calculator.py`, `probing.py`, `k_sparse_probing.py`). Reuse those; do not reimplement geometry or SAE loading.

---

## 0. Locate-first (do this before writing anything)

Run these and record the answers in a scratch note; the rest of the spec depends on them:

1. `grep -rn "def .*pair" DTDxZTZ_grams.py` and any utils it imports — find the **existing pair function** that returns `(dtd_value, ztz_value)` for an atom pair `(i, j)`. Reuse it verbatim in Step 4. Do not write a second one.
2. In `DTDxZTZ_grams.py`, confirm the **axis conventions actually plotted**:
   - Is the x-axis (DTD) `d_i·d_j` on **unit-normalized** rows (→ cosine, range `[-1,1]`) or raw rows? Step 1 requires cosine for `x ∈ [−1,1]`.
   - Is DTD computed as `D @ D.T` (shape `[F,F]`, entry `d_i·d_j`)? The memory/notation in this project calls `D D^T` "DTD" — **honor whatever the existing script does** and make every new file consistent with it. State the convention you found at the top of each new file.
   - Is ZTZ `Z.T @ Z` (co-activation Gram, `[F,F]`), and is it normalized (e.g. by counts/√diag) or raw? Match it.
3. In the adapted `feature_absorption_calculator.py`, find the **two thresholds** that already govern membership and read their exact values — do not invent them:
   - `theta_fire` — the latent firing threshold (a code `z_j` counts as "on" when above this).
   - `tau` — the absorber decoder-alignment / cosine threshold.
   - Also find any **projection-share** cutoff (the fraction of the probe-direction projection an absorber must supply). If none exists, introduce `share_threshold` (default `0.10`) and mark it configurable.
4. Confirm where `X` (residual activations, `[N, d_model]`) and `Z` (SAE latents, `[N, F]`) for a given `(layer, architecture, sparsity)` are written by `infer_z.py`, and the token→row alignment. Steps 2–3 read these.
5. Find SAEBench's **concept dataset builder** (the first-letter / spelling task used by the absorption eval). Step 3's probe and both steps' positive/negative token sets come from it — do not hand-roll labels.

If any of (1)–(5) is missing or ambiguous, stop and surface it rather than guessing — the membership sets are sensitive to thresholds and to the DTD/ZTZ normalization.

---

## 1. Notation & data contracts

| Symbol | Shape | Meaning |
|---|---|---|
| `D` = `W_dec` | `[F, d_model]` | decoder; row `d_j` is atom `j`'s direction |
| `D_unit` | `[F, d_model]` | row-normalized `D` (cosine geometry) |
| `X` | `[N, d_model]` | residual-stream activations at the eval layer/position |
| `Z` | `[N, F]` | SAE latents (post-activation codes); column `z_j` |
| `c` | — | the chosen concept (e.g. "first letter = S") |
| `k` | int | the chosen atom = k-sparse-probing main latent for `c` |
| `p` | `[d_model]` | reference direction, **unit-normalized** (differs per step) |
| `S_main` | set[int] | main latent(s) for `c` |
| `S_abs` | set[int] | absorbers found for `c` |

Keep **both** `D` (raw) and `D_unit`. Geometry/threshold tests use `D_unit`; the actual contribution of latent `j` to the reconstruction is `z_j · d_j` (raw), so projection-share tests use raw magnitudes. The contribution of latent `j` along `p` on token `n` is `z[n,j] * (d_j · p)` with `p` unit-normalized.

Fix a **sign convention** everywhere: alignment is `d_j_unit · p` and absorbers must have it `> tau > 0` (same direction, not anti-aligned).

---

## 2. File plan (create these)

```
src/absorption_membership/
  config.py            # thresholds, paths, chosen (arch, layer, concept_id, k)
  membership.py        # core detector: emit indices, not scores (Steps 2 & 3)
  run_atom_mode.py     # Step 2 entrypoint
  run_probe_mode.py    # Step 3 entrypoint
  io_schema.py         # save/load of the membership artifact (Step 4 contract)
analysis/
  step4_pairs.py       # 4a: binned DTD/ZTZ comparison, absorber pairs vs control
  step4_overlay.py     # 4b: highlight S_main/S_abs on the existing scatter
  step4_summary_metric.py  # open question: compare candidate summary statistics
viz/
  fix_axes.py          # Step 1 carry-over: shared axis limits + replot
```

Everything is **offline** except activation/probe collection (which reuses `infer_z.py` output already on disk). No new GPU SLURM job is required for Steps 2–4 beyond what produced `X`/`Z`; probe training (Step 3) is CPU logistic regression on `X`.

---

## 3. Step 1 — axis fix (carry-over)

In `viz/fix_axes.py`, parametrize axis limits and apply **identical** limits to all four architecture panels (relu, topk, jumprelu, matryoshka):

- x-axis (DTD): hard `[-1, 1]` — requires cosine DTD (`D_unit`). If the existing scatter used raw `d_i·d_j`, switch to cosine or document the rescale.
- y-axis (ZTZ): one shared `[y_min, y_max]` computed as the global min/max across all four panels (or a shared robust quantile, e.g. 1st–99th percentile, to avoid one outlier squashing the rest). Record which.
- Provide a `replot(arch_results, xlim=(-1,1), ylim=SHARED)` that regenerates each figure with the same limits and identical tick spacing, so panels are visually comparable.

Acceptance: the four regenerated figures share x and y limits exactly; reading `ax.get_xlim()/get_ylim()` returns the same tuple for all four.

---

## 4. Step 2 — atom-mode membership (`p = d_k`)

**Reference direction:** `p = D_unit[k]`. No training.

**`S_main`:** declared `{k}`, but **verify** (one-step greedy that k-sparse probing would run):
over concept-positive tokens, score each latent `j` by `mean_n[ z[n,j] * (d_j · p) ]`; assert `argmax_j == k`. With `p = d_k`, alignment `d_k·p = 1` is maximal, so `k` explains the direction alone and the greedy stops after one pick. Log a warning (don't crash) if the argmax is a near-duplicate atom `k'` — that's feature splitting and is worth knowing.

**`S_abs` detection** — iterate concept-positive tokens `n`:

```
for n in positive_tokens:
    if z[n, k] > theta_fire:        # main fired → not an absorption event
        continue
    proj = X[n] · p                 # residual projection onto p
    if proj <= 0:                   # concept direction not present; skip
        continue
    for j != k:
        if z[n, j] <= theta_fire:               # j must fire
            continue
        align = D_unit[j] · p                    # decoder alignment
        if align <= tau:                         # must be positively aligned
            continue
        contrib = z[n, j] * (D[j] · p_raw_or_unit_consistent)   # use the same p as proj
        share = contrib / proj
        if share >= share_threshold:
            absorb_hits[j] += 1                  # support count
S_abs = { j : absorb_hits[j] >= min_support }
```

Keep `absorb_hits` (per-`j` support counts) — Step 4 may want to weight or threshold by support, and they let you drop one-off noise via `min_support` (default `2`; expose it).

**Why this version:** using the atom itself as `p` removes the probe-training degree of freedom and asks a purely geometric question — *which other atoms point along `k` and pick up `c` on tokens where `k` stays silent?* It needs no labels in principle (the trigger set could be defined label-free as "`X·d_k` large but `z_k` didn't fire"), which is what lets it scale to all `F` atoms × 4 architectures without training `F` probes. Record **membership, not a score**, because Step 4 indexes DTD/ZTZ by atom; the scalar absorption number discards *which* atoms form the pair, which is the only thing the Gram analysis uses.

---

## 5. Step 3 — probe-mode membership (`p` = linear probe)

There are **two** probes; do not conflate them.

**(a) Direction probe (the paper's ground truth) → gives `p`:**
train logistic regression on the **raw residual stream `X`** (`d_model` features), concept-positive vs negative, on a clean train split. `p = unit_normalize(probe.coef_)`. This is `probing.py`'s job. Keep train/eval disjoint so nothing downstream leaks.

**(b) Main set → gives `S_main`:**
run **k-sparse probing on the SAE latents `Z`** (`k_sparse_probing.py`): greedily select the smallest latent set whose firing predicts `c` at high F1. Expect it to recover `k`, possibly with split siblings `{k, k', …}`.

**`S_abs`:** identical loop to Step 2, with two changes:
- `p` is the trained probe direction (a);
- "main off" means **all** of `S_main` is below `theta_fire` on token `n` (replace the single `z[n,k]` test with `all(z[n, m] <= theta_fire for m in S_main)`).

**Diagnostic to record:** `cos(d_k, p_probe) = D_unit[k] · p`. If atom-mode and probe-mode diverge, this number explains most of it — the concept direction and the atom direction have drifted apart.

**Why this is the validation anchor:** the probe direction is data-defined (not tied to one atom) → the honest "what is the concept's direction"; the k-sparse set is the honest "what fires for the concept." Comparing to Step 2 makes two things diagnostic: the **size** of `S_main` (atom-mode is always `{k}`; probe-mode `{k,k',…}` exposes feature splitting directly) and the **overlap** of the two `S_abs` sets. High overlap ⇒ the cheap geometric shortcut faithfully stands in for the supervised method ⇒ licensed to run at scale. Divergence is itself the finding.

---

## 6. Output artifact schema (`io_schema.py`)

Save one artifact per `(architecture, layer, concept_id, mode)`. Use `.npz` for arrays + a sidecar `.json` for metadata, or a single JSON if sets are small. Path:

```
results/membership/{arch}_L{layer}_{concept_id}_{mode}.json
```

Contents:

```json
{
  "arch": "topk",
  "layer": 12,
  "concept_id": "first_letter_s",
  "mode": "atom",                 // "atom" | "probe"
  "atom_k": 1234,
  "S_main": [1234],
  "S_abs": [88, 902, 4471],
  "support_counts": {"88": 7, "902": 3, "4471": 2},
  "thresholds": {"theta_fire": 0.0, "tau": 0.5, "share_threshold": 0.10, "min_support": 2},
  "probe_cos": 0.83,             // probe mode only; null for atom mode
  "n_pos_tokens": 1500,
  "dtd_convention": "cosine, D @ D.T on unit rows",
  "ztz_convention": "Z.T @ Z, count-normalized",
  "git_commit": "abcd123"
}
```

Record `thresholds`, conventions, and `git_commit` in **every** artifact — the sets move with the thresholds, so they must travel with the data.

---

## 7. Step 4 — offline analysis

Load both artifacts (atom + probe) for the chosen `(arch, layer, concept)`. Use the **existing pair function** from Step 0 for all DTD/ZTZ values so Step 4 is consistent with the scatter.

### 4a — binned comparison (`step4_pairs.py`)

Build two bins (Bahareh's discrete framing):

- **Absorber/absorbed bin:** all pairs in `S_abs × S_main`, i.e. `{(j, i) : j ∈ S_abs, i ∈ S_main}`.
- **Control bin:** "the rest." Define explicitly and record the choice:
  - default = a random sample of off-diagonal atom pairs **excluding** the absorber bin, sample size matched to the absorber bin (e.g. 50× for stable distributions), seeded;
  - also support `control="all"` (every off-diagonal pair) for a global reference.

For each pair compute `(dtd, ztz)` via the existing pair function. Then:

1. Report per-bin summary stats for **each axis separately**: median, IQR, mean of `dtd` and of `ztz`.
2. Two-sample tests per axis (light, non-parametric): Mann–Whitney U on `dtd` and on `ztz`; report effect size (rank-biserial) not just p — the bins are tiny and p is not the point.
3. Save a 2D scatter coloring the two bins in the DTD×ZTZ plane (same axes as Step 1, `x∈[-1,1]`, shared y), plus the marginal histograms.

Output `results/step4/{arch}_L{layer}_{concept}_bins.{json,png}` with the stats and figure.

### 4b — scatter overlay (`step4_overlay.py`)

On the existing per-architecture DTD×ZTZ scatter (which plots one point per off-diagonal atom pair), **highlight** the points whose `(i, j)` index pair belongs to `S_abs × S_main`. Use the saved indices to mask. Distinct marker/color + a legend; annotate `k`. This is the visual form of the same question: do the absorption-identified pairs fall in the high-DTD / high-ZTZ region? Keep to the **one** chosen atom/concept for now. Honor Step 1 axis limits so this overlay is comparable across architectures.

### Open question — is `DTD × ZTZ` the right summary? (`step4_summary_metric.py`)

Do **not** hardcode `DTD × ZTZ`. Treat the combined metric as provisional. Compute a panel of candidate one-number summaries per pair and report how well each **separates the absorber bin from control**:

- candidates: `dtd` alone, `ztz` alone, `dtd * ztz`, `dtd + ztz` (z-scored), `|dtd| * ztz`, `log` variants, ratio, and a 2-feature logistic boundary on `(dtd, ztz)`.
- separation metric: ROC-AUC of each summary at distinguishing absorber pairs (label 1) from control (label 0). Also report the AUC of the logistic combination as the "best linear" reference.
- emit a small ranked table (summary → AUC) and a one-line recommendation. This directly answers the push-back: show whether `DTD × ZTZ` actually separates the bins or whether a different summary does better, **before** committing to it.

---

## 8. Acceptance checks (must pass before reporting results)

1. **Atom-mode `S_main` = `{k}`** and the one-step greedy argmax equals `k` (or logs the near-duplicate). 
2. **Probe split clean:** k-sparse `S_main` recovery is measured on the eval split, not the probe's train split.
3. **Sign convention:** every absorber satisfies `D_unit[j]·p > tau > 0`. Add an assert.
4. **Threshold provenance:** `theta_fire` and `tau` in the artifact match the values read from `feature_absorption_calculator.py` (Step 0), not new constants.
5. **DTD/ZTZ consistency:** Step 4 uses the same pair function and the same normalization conventions as `DTDxZTZ_grams.py`; the conventions string in the artifact matches the scatter.
6. **Overlap sanity:** print `|S_abs(atom) ∩ S_abs(probe)|`, `|S_main(atom) ∩ S_main(probe)|`, and `cos(d_k, p_probe)`. These are the validation numbers; surface them, don't bury them.
7. **Axis limits identical** across the four architecture figures (Step 1).
8. **Determinism:** control-bin sampling is seeded; rerunning reproduces the same bins and stats.

---

## 9. Run order

```
# Step 1 (carry-over)
python -m viz.fix_axes --replot

# Steps 2 & 3 (one chosen arch/layer/concept; set in config.py)
python -m src.absorption_membership.run_atom_mode
python -m src.absorption_membership.run_probe_mode

# Step 4 (offline)
python -m analysis.step4_pairs
python -m analysis.step4_overlay
python -m analysis.step4_summary_metric
```

Keep everything pinned to **one atom / one concept** for this pass. Scaling to all atoms and all four architectures is a later step and is exactly why atom-mode (label-free, no per-atom probe) exists — but validate the two-way agreement on this single anchor first.

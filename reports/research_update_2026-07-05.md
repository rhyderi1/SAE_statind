# Research Update — Week of Jun 29 – Jul 5, 2026

**Project:** SAE statistical independence & feature absorption
**Setup:** Gemma-2-2b, layer 12, ReLU SAE (width 2¹⁴ ≈ 16k, L0/sparsity label "20"), Pile-10k
**Author:** Raza · **Date:** 2026-07-05

> Draft for review — please sanity-check the numbers and claims marked *(preliminary)* before circulating.

---

## TL;DR

This week I moved from the absorption machinery into activation-distribution diagnostics, and built a co-activation scatter tool (`zizj_scatter`) to compare an **absorbed** feature pair against a **non-absorbed** pair. The tool works and produces clean figures, but the comparison surfaced a methodological question I'd like to discuss: **whether the co-activation scatter is actually the right lens for absorption, and how to make the two pairs comparable** (they currently differ ~10× in co-activation count).

---

## What I did

**1. Absorption set extraction.**
Ran `extract_absorption_sets.py` → produced `absorption_sets_relu_layer12_k20_*.json`. This gives, per first letter, the "main" letter feature(s) and the list of absorber features with counts (how often each acts as absorber). E.g. for "d", the main feature is latent **477** and its strongest absorber is latent **10069** (73 absorption events).

**2. Activation-distribution diagnostics.**
- Per-feature activation histograms and summary stats (`z_histogram.py`, `z_histogram_latent_stats.py`) → `feature_max_dist`, `feature_mean_dist`, `feature_max_vs_mean` figures.
- Per-bin stats and a BOS-token sanity check (`z_hist_bin_stats.py`, `z_histogram_bos_check.py`) to confirm the BOS position isn't distorting the distributions.

**3. Co-activation scatter (`zizj_scatter.py`).**
Built a tool that plots latent *zᵢ* vs *zⱼ* for chosen pairs, in raw and joint-support-standardized panels, reporting the on-support Pearson *r* (correlation computed only over tokens where both latents fire). This week I also rewrote it to be fully vectorized (removed Python per-token loops; earlier version took ~2 hrs), fixed the origin-dropping logic, and annotated the two study pairs.

## Preliminary observations *(from the current figure)*

| Pair | Type | Co-active tokens | On-support *r* |
|------|------|------------------|----------------|
| 477 ↔ 10069 | Absorbed (d-feature ↔ absorber) | ~52,200 | −0.07 |
| 1628 ↔ 2969 | Non-absorbed | ~4,500 | +0.15 |

- Both pairs show the expected sparse structure: two axis "strips" (one latent fires, the other silent) plus an off-axis blob (both fire).
- For the absorbed pair, absorption events (73) are a **tiny fraction** of the ~52k co-activations, and the on-support correlation is essentially zero. So absorption does **not** obviously show up as a gross change in blob shape — if there's a signature, it may live in a thin subset of tokens.

## Things to discuss

1. **Is the co-activation scatter the right instrument for absorption?** Given absorption events are rare relative to co-activation and *r* ≈ 0 for the absorbed pair, the scatter may not reveal the phenomenon. Worth deciding whether to push this or pivot to a more targeted view (e.g. conditioning on the absorbed tokens specifically).
2. **Making the two pairs comparable.** They differ ~10× in co-activation count, so the panels aren't directly comparable. My plan is to control this at analysis time (equal-N subsampling / density-normalized panels, bootstrap CI on *r*) rather than by cherry-picking pairs on point count — since firing frequency is entangled with absorption itself. Would like your read on this.
3. **Anecdote vs. distribution.** Two hand-picked pairs illustrate but don't establish anything. I think the real claim needs the *distribution* of on-support *r* across all absorbed vs. non-absorbed pairs. Agree?
4. **Standardization choice.** I standardize each latent on the *joint support* so the standardized blob's tilt equals the on-support *r*; happy to walk through the rationale if useful.

## Next steps (proposed)

- [ ] Compute per-latent **marginal firing rates** to select absorbed/non-absorbed pairs matched on frequency and "type."
- [ ] Add **equal-N subsampling** and a **bootstrap CI** on *r* to the scatter.
- [ ] Compute the **on-support *r* distribution** across all absorbed vs. non-absorbed pairs (the population result behind the illustrative panels).
- [ ] **Performance/reproducibility:** cache extracted activations to disk so plotting is decoupled from the ~2-hr model pass.
- [ ] Consider extending to layer 19 and other SAE architectures (topk / jumprelu / matryoshka) already configured.

## Blockers / notes

- Hit a Vulcan disconnect mid-session last week (recovered via file save) — flagging in case it recurs.
- Long runtimes (~2 hrs for the full-corpus pass) motivate the caching step above.

---

*Reconstructed from this week's repo activity; numbers from the latest `zizj_scatter` run — please verify against your own notes.*

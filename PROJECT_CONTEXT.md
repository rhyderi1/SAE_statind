# PROJECT_CONTEXT.md — `sae_statind`

A machine-readable summary of the codebase for anyone (human or agent) picking it up
cold. For the *research* narrative — the metric derivation, the literature map, the
timeline, and a line-by-line verification of code vs. intent — see
[`docs/project_context.md`](docs/project_context.md) (the consolidated dossier) and
its predecessor [`docs/project_info_archived.md`](docs/project_info_archived.md).
This file covers only what the code *is* and how to run it.

---

## 0. One-paragraph orientation

The project is building a **probe-free detector for feature absorption in sparse
autoencoders (SAEs)** — a metric computed from joint latent activation statistics
alone, without a ground-truth linear probe. The target model is **Gemma-2-2B**
(residual stream, layers 3 / 12 / 19), across four SAE families
(ReLU, TopK, JumpReLU, Matryoshka). The centre of gravity of recent work is
**layer 3, JumpReLU, average L0 ≈ 59**. The repo is a research scratch space: a
pipeline that streams a corpus through the model + an SAE to collect activations,
a set of Gram-matrix / co-activation accumulators, a suite of plotting scripts,
and a re-implementation of parts of SAEBench's absorption eval widened to the full
vocabulary. Everything runs as Slurm batch jobs on the Vulcan (Alliance Canada)
cluster.

---

## 1. Tech Stack & Dependencies

| Layer | Choice |
|---|---|
| **Language** | Python **3.11.5** (CVMFS module `python/3.11.5` on Vulcan) |
| **Numerics / ML** | `torch`, `numpy`, `scipy`, `scikit-learn` |
| **Interpretability libs** | `transformer_lens` (model loading + residual-stream hooks), `sae_lens` (SAE loading, `ActivationsStore` streaming data pipeline), `sae_bench` (absorption eval, k-sparse probing, probe training — vendored code imported directly) |
| **Data / IO** | `pandas`, `pyarrow` (parquet artifacts from SAEBench; on Vulcan needs `module load gcc arrow/24.0.0` *before* venv activation), HuggingFace `datasets` (streaming `NeelNanda/pile-10k`, `Skylion007/openwebtext`) |
| **Plotting** | `matplotlib` (`Agg` backend, always set before `pyplot` import), `seaborn`, `plotly` (interactive HTML line/scatter plots) |
| **Runtime env** | Vulcan HPC — Slurm, 4×L40S (48 GB) GPU nodes, CVMFS/Lmod software tree. Virtualenv at `.venv/` built with `--system-site-packages` against the CVMFS Python; wheels from the Alliance wheelhouse (`pip install --no-index`). A second stale env `.venv1/` exists — ignore it. |
| **Model access** | Gemma-2-2B is gated: jobs export `HF_TOKEN`. `HF_HOME` / `HF_DATASETS_CACHE` are redirected to `/scratch/rhyderi1/...`. |

`requirements.txt` is the authoritative list (unpinned):
`torch, numpy, matplotlib, seaborn, pandas, scipy, scikit-learn, transformer_lens, sae_lens, sae_bench`.

> ⚠️ **Known issue:** `HF_TOKEN` is hardcoded in plaintext in several committed
> scripts (`run_job.sh`, `run_inferz.sh`, `run_absorption_full.sh`, and others)
> and is therefore in git history. It should be rotated and moved to an
> un-committed source. Do not add new hardcoded tokens.

---

## 2. Project Structure

```
sae_statind/
├── src/                  All Python. ~40 files; ~12 are the live surface, the rest
│                         self-declare LEGACY/BROKEN/UNUSED in their docstrings.
├── scripts/              One Slurm sbatch wrapper per job. Thin: module loads,
│                         venv activate, cd, then one `python src/<x>.py ...` call.
├── config/              *Inputs, not outputs* — sweep definitions read by scripts.
│   ├── params.csv        layer,arch,sparsity → `3,jumprelu,59`  (the focal config)
│   └── params2.csv       `12,relu,20`
├── data/                 Generated (gitignored). X/ Z/ shards, pile-10k-saes/ Gram
│                         matrices, expectations/.
├── results/              Small non-figure outputs: absorption_full_*.csv (one per
│                         letter), absorption_sets_*.json, coact_*.pt.
├── figures/              PNG/HTML plots, foldered by figure type + SAE config.
├── eval_results/         SAEBench absorption eval JSON output.
├── artifacts/            Cached SAEBench probe/eval artifacts (k_sparse_probing JSON).
├── logs/                 Slurm %x-%j.out / .err (gitignored; ~780 files).
├── docs/                 project_context.md (research dossier), project_info_archived.md
├── letter_s.csv          A raw SAEBench absorption-eval output kept at root for ref.
├── requirements.txt · README.md · CLAUDE.md
```

### Core source files (the live surface)

| File | Role |
|---|---|
| **`src/infer_z.py`** | **Stage 1.** Streams the corpus through Gemma-2-2B + an SAE, writes `X` (residual, `d_model=2304`) and/or `Z` (SAE latents, `d_sae=16384`) as `Z_shard{NNN}.pt`. **Holds `SAE_DATA`**, the `(layer, arch, sparsity) → (release, sae_id)` registry that nearly every other script imports. |
| **`src/compute_gram_from_tokens.py`** | **Stage 1, low-footprint variant.** Same streaming setup, but accumulates the `p×p` Gram `Zᵀ Z` (and, when enabled, the indicator Gram `1{Z≠0}ᵀ 1{Z≠0}`) on the fly and writes *only* those — avoids the hundreds of GB of Z shards. Current primary data path. |
| `src/compute_gram_from_Z.py`, `compute_ind_gram.py`, `compute_ind_gram_2.py` | Accumulate (indicator) Gram matrices from Z shards already on disk. Scaled vs. raw-count variants. |
| `src/check_batch_ceiling.py` | Computes the exact number of full batches before `ActivationsStore` wraps to a new epoch (so `--n_batches` doesn't silently double-count the corpus start). Two modes: `fast` (tokenizer only) / `store` (drives the real store to `StopIteration`). |
| **`src/absorption_full_vocab.py`** | Re-runs SAEBench's `FeatureAbsorptionCalculator` over the **entire single-token vocabulary** for a letter (no 80/20 split, no false-negative filter), reusing the trained probe + chosen `S_main` latents. Output: `results/absorption_full_<letter>_*.csv`, a strict superset of SAEBench's `letter_s.csv` with the absorbing latent (`top_projection_feat`) kept. |
| `src/compute_feature_absorption.py` | Thin driver around `sae_bench.evals.absorption` over a hardcoded `selected_saes` list → `eval_results/absorption/*.json`. |
| `src/extract_absorption_sets.py` | Reads SAEBench's per-SAE parquet artifacts → per-letter `S_main` / `S_abs` latent sets → `results/absorption_sets_*.json` (lowercase keys `s_main`, `s_abs`, `s_abs_tokens`). |
| `src/absorption_metrics.py` | Streams the corpus once for hand-picked `(main, absorber)` latent pairs and computes "Absorption Metric 3", a magnitude-suppression × conditional-firing quantity, against matched control pairs. Config hardcoded at top. |
| `src/coact_match.py` | Builds co-activation-matched control latents for an anchor latent (controls for "the absorber just fires a lot"). |
| **`src/sorted_lineplots.py`** | Streams the corpus, accumulates per-latent on-support stats — `max/min/E/std (zᵢ | zᵢ>0)` — plus off-diagonal `zᵢ·zⱼ`. Emits one static PNG (all points) + one Plotly HTML (2×3 grid, `zizj` panel truncated to `TOP_K`). Used to check whether any stat tracks absorption. |
| `src/dtdxztz_scatter_full.py` | **DTD×ZTZ workhorse.** Per row of `params.csv`: streams Z, forms `ZᵀZ` + co-activation count, drops dead latents, samples ≤ `MAX_POINTS=300_000` off-diagonal pairs, plots 3 y-quantities (`zᵢᵀzⱼ`, `zᵢᵀzⱼ / co-act count`, co-act count) × 2 x-quantities (`dᵢᵀdⱼ`, `cos(dᵢ,dⱼ)`). Also writes `spearman_summary.csv` (one row per config: `rho_cos_ztz`, `rho_cos_ratio`, `rho_cos_l0`, `rho_raw_ztz`, `d_sae`, `q_alive`, `n_dead`, `n_pairs`, `seed`). |
| `src/dtdxztz_scatter_full_overlay.py` | Same grid, sparsities overlaid per `(layer, arch)`. |
| `src/dtdxztz_scatter_simple.py` | Single-config version, loads a precomputed ZTZ. Config hardcoded (layer 12 relu k20). |
| `src/chanin_fig6a.py` | Replicates Chanin et al. Fig. 6a for the case-study pair (latents `6510` "starts with S" / `1085` "short") + 2 seeded random controls. |
| `src/zizj_scatter.py` | `zᵢ` vs `zⱼ` scatter for pairs in a `PAIRS` list; red-colours tokens from full-absorption events. Migrated to read the full-vocab CSV (`absorbed_token_ids_from_csv`). |
| `src/z_histogram*.py`, `z_hist_bin_stats.py`, `z_histogram_latent_stats.py` | Activation-value / per-latent histograms with memory-bounded streaming. (`z_histogram.py` is currently BROKEN — `max_val` NameError.) |
| `src/check_*.py`, `src/scratch.py` | Small diagnostics (encoder-bias presence, Gram symmetry, token inspection, shapes). |
| `src/sae_train.py` | Standalone vanilla-SAE training run. **Not part of the analysis pipeline** — all reported results use pretrained SAEs. |
| `src/sae_data.py` | **Duplicate, UNUSED** copy of `SAE_DATA` that omits the layer-3 entries. Prefer `from infer_z import SAE_DATA`. Footgun. |
| `src/archived_scripts/` | Where LEGACY scripts belong (currently holds one). |

---

## 3. Architecture & Data Flow

The pipeline is linear and file-based; stages communicate through `.pt` / `.csv` /
`.parquet` / `.json` on disk, not through imported functions.

```
                 config/params.csv  (layer, arch, sparsity)
                          │  (--task_id = SLURM array index → one row)
                          ▼
        ┌─────────────────────────────────────────────────────────┐
        │ SAE_DATA[layer][arch][sparsity] → (release, sae_id)      │  src/infer_z.py
        └─────────────────────────────────────────────────────────┘
                          │
   HuggingFace dataset ───┤  sae_lens.ActivationsStore  (streaming, NO shuffle)
   (NeelNanda/pile-10k)   │  tokenises on the fly, packs to context_size=128,
                          │  batch_size=32
                          ▼
   transformer_lens.HookedTransformer(gemma-2-2b)
        hook: blocks.{layer}.hook_resid_post   ──►  X  (residual, d_model=2304)
                          │
                     sae.encode(X)             ──►  Z  (latents, d_sae=16384)
                          │
        ┌─────────────────┴───────────────────────────────┐
        ▼                                                 ▼
   infer_z.py:                                    compute_gram_from_tokens.py:
   write Z_shard{NNN}.pt (+ X shards)             accumulate p×p Gram on the fly
   → data/Z/... , data/X/...                      ZtZ  = Σ Zᵀ Z
        │                                         G_ind = Σ 1{Z≠0}ᵀ 1{Z≠0}  (rung 1)
        │                                         → data/pile-10k-saes/.../*.pt
        ▼
   compute_gram_from_Z.py / compute_ind_gram*.py
   (Gram from shards, if shards were kept)
        │
        ▼
   ┌────────────────────────── consumers ──────────────────────────┐
   │ dtdxztz_scatter_full.py   ZᵀZ  ×  DᵀD (= sae.W_dec Gram)       │→ figures/ + spearman_summary.csv
   │ sorted_lineplots.py       per-latent on-support stats + zizj   │→ figures/ (PNG + Plotly HTML)
   │ absorption_metrics.py     streams corpus again, per-pair stats │→ results/
   │ coact_match.py            co-activation-matched controls       │→ results/*.pt
   │ zizj_scatter.py           zᵢ vs zⱼ, absorption tokens in red   │→ figures/zizj_scatter/
   │ chanin_fig6a.py           bar chart, case-study latents        │→ figures/chanin_fig6/
   └───────────────────────────────────────────────────────────────┘

   ── parallel track: SAEBench ground-truth labels ──
   compute_feature_absorption.py ──► sae_bench absorption eval
        │  writes eval_results/absorption/*.json  +  per-SAE parquet artifacts
        ▼
   extract_absorption_sets.py  ──►  results/absorption_sets_*.json  (S_main / S_abs)
   absorption_full_vocab.py    ──►  results/absorption_full_<letter>_*.csv
        (widens SAEBench's ~20%-of-vocab label coverage to the full vocab;
         used as the retrieval target for validating the probe-free metric)
```

**Key objects & conventions in the data flow**

- **`SAE_DATA`** (`src/infer_z.py`) is the single source of truth for which
  pretrained SAE corresponds to a `(layer, arch, sparsity)`. `arch` choices in
  argparse include `batchtopk`, but it has **no `SAE_DATA` entry** → `KeyError`.
- **No shuffling** is configured in `ActivationsStore`. This is deliberate and
  *relied upon*: a fresh store replays the identical batch sequence, which makes
  two-pass streaming rewrites exactly equivalent to one-pass.
- **`X`** = residual stream at `blocks.{layer}.hook_resid_post`; **`Z`** =
  `sae.encode(X)`. Use `sae.encode`, never a manual encoder-column projection.
- **Gram matrices**: `ZᵀZ` (magnitude co-activation, "rung 2") and
  `1{Z≠0}ᵀ1{Z≠0}` (indicator co-firing, "rung 1"). `DᵀD` is the decoder Gram
  from `sae.W_dec`; scatter scripts pair the same `(i,j)` entry from a `Z`-Gram
  and a `D`-Gram.
- The absorption "metric" of interest is directional:
  `Absorb(A→B) = cos(d_A, d_B) · (1 − P(A|B))`, evaluated in both orderings, with
  the higher-frequency latent taken as the parent. Not all of this is implemented
  yet — see the dossier §2 and §6.

---

## 4. Setup & Run Commands

### Environment (Vulcan)

```bash
module load python/3.11.5
module load gcc arrow/24.0.0                 # needed BEFORE venv activate for pyarrow
source /project/aip-bahtol/rhyderi1/sae_statind/.venv/bin/activate
# first-time build:  virtualenv --no-download .venv && pip install --no-index -r requirements.txt
export HF_HOME=/scratch/rhyderi1/hf_home
export HF_DATASETS_CACHE=/scratch/rhyderi1/hf_datasets
export HF_TOKEN=<your gated-Gemma token>      # do not commit
```

### Submitting jobs

Never run the pipeline on the login node. Every entry point has a wrapper in
`scripts/`; edit the `#SBATCH` directives and the trailing `python ...` line, then:

```bash
ssh vulcan "cd /project/aip-bahtol/rhyderi1/sae_statind && sbatch scripts/<script>.sh"
ssh vulcan "squeue -u rhyderi1"
# outputs: logs/<job-name>-<job-id>.out  and  .err
```

| Script | Runs | Notable `#SBATCH` |
|---|---|---|
| `scripts/run_inferz.sh` | `infer_z.py` — collect X/Z shards (layer 3 jumprelu k59, `--store_x --store_z`) | `gpu:1`, 64G, 8 h |
| `scripts/run_job7.sh` | `compute_gram_from_tokens.py` — accumulate ZtZ (+ indicator Gram) | `gpu:1`, 64G, **13 h** |
| `scripts/run_ind_gram.sh` | `compute_ind_gram_2.py` — indicator Gram from shards | `gpu:1`, 32G, 5 h |
| `scripts/run_absorption.sh` | `compute_feature_absorption.py` — SAEBench eval | `gpu:1`, 64G, 12 h |
| `scripts/run_absorption_full.sh` | `absorption_full_vocab.py --letters all --arch jumprelu --layer 3 --sparsity 59 --seed 0` | `gpu:l40s:1`, 64G, 1 h |
| `scripts/run_saebench_main.sh` | `python -m sae_bench.evals.absorption.main` directly | `gpu:1`, 64G, 3 h |
| `scripts/run_job2.sh` | `zizj_scatter.py` | `gpu:1`, 64G, 10 min |
| `scripts/run_job4.sh` | `z_histogram_latent_stats.py` | `gpu:1`, 64G, 4 h |
| `scripts/run_job6.sh` | `sorted_lineplots.py` | `gpu:1`, 64G, 30 min |
| `scripts/run_z_hist.sh` | `z_histogram.py` | `gpu:1`, 64G, 8 h |
| `scripts/run_job.sh`, `run_job3.sh` | scratch slots (currently `check_batch_ceiling.py`, `check_tokens.py`) | small |

`run_job*.sh` are interchangeable scratch wrappers — the numbering only lets
several jobs run in parallel. Change the `python` line to repurpose one.

### Running a single config without Slurm array

```bash
python src/infer_z.py --modelchoice gemma-2-2b --layer 3 --arch jumprelu \
    --sparsity 59 --n_batches 3820 --batch_size 32 --context_size 128 \
    --shard_size 50000 --store_z
python src/absorption_full_vocab.py --letters s --arch jumprelu --layer 3 --sparsity 59
```

For a sweep, drop `--layer/--arch/--sparsity` and pass `--task_id $SLURM_ARRAY_TASK_ID`
(index into `config/params.csv`).

### Tests

There is **no test suite**. Verification is done by dedicated diagnostic scripts
(`src/check_*.py`) and by eyeballing figures. `check_batch_ceiling.py`,
`check_symmetric.py`, `check_b_enc.py` are the closest thing to assertions.

---

## 5. Coding Standards & Conventions

**Model loading**
- Analysis scripts load Gemma-2-2B with
  `HookedTransformer.from_pretrained_no_processing(..., center_writing_weights=False)`
  — required for RMSNorm models. **`infer_z.py` and `compute_gram_from_tokens.py`
  instead use plain `from_pretrained(...)`**, so data is *generated* under one
  convention and *analysed* under another. TransformerLens warns and self-corrects
  the two dangerous defaults; impact is believed small but unverified. Prefer
  `from_pretrained_no_processing` in new code.
- Encode with `sae.encode(X)`; never re-derive latents from encoder columns.

**Config & parameterisation**
- Sweep parameters live in `config/params.csv` (`layer,arch,sparsity`), read by
  array index via `--task_id`. These CSVs are **inputs** — `.gitignore` has an
  explicit `!config/*.csv` un-ignore.
- Scripts take `argparse` args for the intended-to-vary knobs; many older /
  single-purpose scripts hardcode `ARCH / LAYER / SPARSITY` constants at the top
  of the file instead. When hardcoded, they are near the top and clearly named.
- The registry `SAE_DATA` is imported from `infer_z.py`. Do not use
  `src/sae_data.py` (stale duplicate, missing layer 3).

**Filesystem / IO**
- All heavy IO goes under `data/` (shards, Gram matrices) or `/scratch`. `data/`,
  `logs/`, `*.pt`, `*.csv`, `*.out`, `*.err`, `*.npz`, `*.parquet` are gitignored.
- Output filenames encode the run: `<name>_<arch>_layer<L>_k<sp>_<YYYYMMDD_HHMMSS>`.
  Timestamp via `datetime.now()`. Figures are foldered by figure type then SAE
  config (`figures/<type>/<arch>_layer<L>_k<sp>/`).
- Consumers locate inputs by `glob` on those name patterns, so keep the naming
  scheme stable. A few scripts hardcode absolute paths through the `~/projects`
  symlink — non-portable; parameterise instead.
- `REPO_ROOT` should be `Path(__file__).resolve().parent.parent` (file is in
  `src/`). `absorption_metrics.py` gets this right; `sorted_lineplots.py` gets it
  wrong (`.parent` → writes to `src/figures/`).

**Streaming / memory**
- The dataset is streamed through `ActivationsStore`; never materialise the whole
  corpus. Per-latent statistics are kept as fixed-size `(p,)` accumulators
  (`device=Z.device, dtype=Z.dtype`) so memory is flat in the number of batches.
- When a computation needs two passes, exploit the no-shuffle guarantee to replay
  the identical batch sequence rather than caching.
- Guard against wrap-around: use
  `get_batch_tokens(batch_size, raise_at_epoch_end=True)` in a
  `try/except StopIteration` (as `compute_gram_from_tokens.py` does).
  `infer_z.py` still lacks this guard.
- Known numerical footguns (present in `sorted_lineplots.py` right now): compute
  `min_z` with `+inf` init (not zeros); carry `argsort` indices as Plotly
  `customdata` rather than sorting each stat independently; clamp variance with
  `.clamp_min(0)` before `sqrt`; guard divide-by-`active_n` for dead latents.

**Plotting**
- `import matplotlib; matplotlib.use('Agg')` **before** `import matplotlib.pyplot`
  (headless nodes).
- Gram / co-activation data is heavy-tailed: log axes, filter to finite positive
  values *before* sorting, small markers (overplotting), subsample large pair sets
  (`MAX_POINTS`, `TOP_K`). Mask the diagonal and drop dead latents before any
  correlation. Raw `ZᵀZ` is **not** comparable across architectures — normalise to
  correlation / co-activation-count form first.

**Docstrings & script lifecycle**
- Every `src/*.py` opens with a docstring: what it does, a runnable example, and
  an explicit status tag when relevant — `LEGACY` (hardcoded paths into a
  directory layout that no longer exists), `BROKEN` (named bug), `UNUSED`. Trust
  these tags; keep them accurate when you touch a file. New throwaway analysis
  should say so.
- Nine scripts are self-declared LEGACY; they belong in `src/archived_scripts/`.

**Secrets**
- No credentials in committed files. `HF_TOKEN` currently violates this in several
  `scripts/*.sh` — rotate and externalise; do not copy the pattern.

**Git**
- `main` is the working branch; commits are frequent and terse. Large/binary
  outputs are gitignored by design.

---

## 6. Pointers

- Research context, metric derivation, validation ladder, open decisions:
  [`docs/project_context.md`](docs/project_context.md).
- Cluster rules (Slurm, CVMFS, storage): `/etc/claude-code/CLAUDE.md` and the
  `alliance-*` skills.
- Per-config absorption ground truth: `results/absorption_full_*.csv` (full vocab,
  preferred) over `letter_s.csv` / SAEBench eval labels (~20% of vocab, filtered).

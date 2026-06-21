# Convention record (Step 0 findings — all new files must stay consistent with this):
#   DTD x-axis: cosine = d_i·d_j / (‖d_i‖‖d_j‖), range [-1, 1]   (gram_scatter.py:decoder_pair_values)
#   ZTZ y-axis: raw Z.T @ Z, log-scale, ylim=(1e-2, 1e5)            (gram_scatter.py:accumulate_z_grams)
#   Thresholds sourced from feature_absorption_calculator.py:
#     theta_fire = 1e-8 (EPS; latent fires when activation >= EPS)
#     tau        = 0.025 (full_absorption_probe_cos_sim_threshold)
#     share_threshold = 0.4 (probe_projection_proportion_threshold)

import os
from pathlib import Path

# ── Anchor: one arch / layer / concept for this validation pass ──────────────
ARCH      = "relu"
LAYER     = 12
SPARSITY  = "20"
CONCEPT_ID = "first_letter_s"
LETTER    = "S"

# atom_k: primary latent for the concept, determined by k-sparse probing.
# Set to -1 to force run_probe_mode to discover it; run_atom_mode will then
# read it from the probe artifact. Can also be hard-coded after a first probe run.
ATOM_K = 7983

# ── Thresholds — read verbatim from feature_absorption_calculator.py ─────────
THETA_FIRE       = 1e-8   # EPS; activation < this means "latent did not fire"
TAU              = 0.025  # full_absorption_probe_cos_sim_threshold
SHARE_THRESHOLD  = 0.4    # probe_projection_proportion_threshold
MIN_SUPPORT      = 2      # new: absorber must appear on >= this many tokens

# ── Data paths ────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[2]   # sae_statind/

# Directory written by infer_z.py for (layer=12, arch=relu, sparsity=20)
Z_DIR = PROJECT_ROOT / "Full_SAE_pt_files" / "1_acts_layer12_relu_20_20260610_132348"

RESULTS_DIR   = PROJECT_ROOT / "results" / "membership"
STEP4_DIR     = PROJECT_ROOT / "results" / "step4"
FIG_DIR       = PROJECT_ROOT / "results" / "step4"

# ── Scatter / axis conventions (must match gram_scatter.py) ──────────────────
XLIM = (-1.0, 1.0)
YLIM = (1e-2, 1e5)

# ── DTD/ZTZ convention strings recorded in every artifact ────────────────────
DTD_CONVENTION = "cosine, D @ D.T on unit rows, x in [-1,1]"
ZTZ_CONVENTION = "Z.T @ Z, raw (not count-normalized), log-scale"

# ── Control-bin sampling ──────────────────────────────────────────────────────
CONTROL_OVERSAMPLE = 50   # control pairs = oversample * len(absorber_pairs)
CONTROL_SEED       = 42

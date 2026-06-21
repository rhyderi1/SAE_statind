# Convention record (consistent with config.py / gram_scatter.py):
#   DTD x-axis: cosine d_i·d_j / (‖d_i‖‖d_j‖), range [-1,1]
#   ZTZ y-axis: raw Z.T @ Z, not normalized
#
# Thresholds sourced from feature_absorption_calculator.py (never reinvented here):
#   theta_fire = 1e-8   (EPS — latent fires when activation >= EPS)
#   tau        = 0.025  (full_absorption_probe_cos_sim_threshold)
#   share_threshold = 0.4 (probe_projection_proportion_threshold)

from __future__ import annotations

import logging
import warnings
from collections import Counter
from pathlib import Path

import numpy as np
import torch

log = logging.getLogger(__name__)


# ─── Data loading ────────────────────────────────────────────────────────────

def iter_xz_shards(z_dir: Path | str, *, max_shards: int | None = None):
    """Yield (X_shard, Z_shard) tensors one at a time. Never holds more than
    one shard of each in memory. Use for absorber detection over large datasets."""
    z_dir = Path(z_dir)
    x_paths = sorted(z_dir.glob("X_shard*.pt"))
    z_paths = sorted(z_dir.glob("Z_shard*.pt"))
    if not x_paths:
        raise FileNotFoundError(f"No X_shard*.pt in {z_dir}")
    if not z_paths:
        raise FileNotFoundError(f"No Z_shard*.pt in {z_dir}")
    if len(x_paths) != len(z_paths):
        raise ValueError(f"Shard count mismatch: {len(x_paths)} X vs {len(z_paths)} Z")
    if max_shards is not None:
        x_paths = x_paths[:max_shards]
        z_paths = z_paths[:max_shards]
    log.info(f"Streaming {len(x_paths)} shard(s) from {z_dir}")
    for xp, zp in zip(x_paths, z_paths):
        X = torch.load(xp, map_location="cpu").float()
        Z = torch.load(zp, map_location="cpu").float()
        yield X, Z
        del X, Z


def load_xz_shards(z_dir: Path | str, *, max_shards: int):
    """Load up to max_shards shards concatenated. max_shards is required to
    prevent accidental OOM on large datasets (93 shards = ~350 GB)."""
    Xs, Zs = [], []
    for X, Z in iter_xz_shards(z_dir, max_shards=max_shards):
        Xs.append(X); Zs.append(Z)
    X_cat = torch.cat(Xs, dim=0)
    Z_cat = torch.cat(Zs, dim=0)
    log.info(f"Loaded X={tuple(X_cat.shape)}, Z={tuple(Z_cat.shape)} ({len(Xs)} shards)")
    return X_cat, Z_cat


# ─── Atom-mode membership (Step 2) — streaming ───────────────────────────────

def detect_absorbers_atom_mode_streaming(
    z_dir: Path | str,
    D: torch.Tensor,       # [F, d_model] raw decoder rows
    D_unit: torch.Tensor,  # [F, d_model] row-normalized
    atom_k: int,
    *,
    theta_fire: float,
    tau: float,
    share_threshold: float,
    min_support: int,
    max_shards: int | None = None,
) -> tuple[list[int], dict[int, int], list[int], int]:
    """
    Streaming atom-mode detection. Processes one shard at a time.

    Returns (s_main, support_counts, s_abs, n_pos_tokens).
    s_main is always [atom_k]. Logs a warning if the greedy argmax disagrees.

    Acceptance checks:
      - Every absorber j satisfies D_unit[j]·p > tau > 0 (sign convention assert).
    """
    p = D_unit[atom_k]                    # [d_model]
    p_np = p.numpy()
    align_all = (D_unit @ p).numpy()      # [F]  — precomputed, same across shards

    # Candidate absorbers: aligned with p, not atom_k itself
    candidate_mask = (align_all > tau)
    candidate_mask[atom_k] = False
    j_candidates = np.where(candidate_mask)[0]
    Dp_candidates = (D[j_candidates] @ p).numpy()  # [n_cand]

    absorb_hits: Counter[int] = Counter()
    # Accumulators for greedy argmax check
    score_sum = np.zeros(D_unit.shape[0], dtype=np.float64)
    n_pos_total = 0

    for X, Z in iter_xz_shards(z_dir, max_shards=max_shards):
        X_np = X.numpy()
        Z_np = Z.numpy()
        N = Z_np.shape[0]

        # Eligible tokens: main off AND concept direction present
        main_off       = Z_np[:, atom_k] < theta_fire          # [N]
        proj           = X_np @ p_np                           # [N]
        concept_present = proj > 0
        eligible        = main_off & concept_present            # [N]
        n_pos_total    += int(eligible.sum())

        # Greedy check accumulator: score_j += sum_n[ z[n,j] * align_j ]  (eligible tokens)
        if eligible.sum() > 0:
            score_sum += (Z_np[eligible] * align_all[None, :]).sum(0)

        if eligible.sum() == 0:
            continue

        Z_elig   = Z_np[eligible]                              # [n_elig, F]
        proj_elig = proj[eligible]                             # [n_elig]

        # Vectorized absorber check over candidate j's
        Z_cand   = Z_elig[:, j_candidates]                    # [n_elig, n_cand]
        firing   = Z_cand >= theta_fire                        # [n_elig, n_cand]
        contrib  = Z_cand * Dp_candidates[None, :]             # [n_elig, n_cand]
        share    = contrib / proj_elig[:, None]                # [n_elig, n_cand]
        hits     = (firing & (share >= share_threshold)).sum(0)  # [n_cand]

        for idx, j in enumerate(j_candidates):
            absorb_hits[int(j)] += int(hits[idx])

    # Greedy argmax check
    if n_pos_total > 0:
        greedy_argmax = int(score_sum.argmax())
        if greedy_argmax != atom_k:
            warnings.warn(
                f"[atom-mode] Greedy argmax = {greedy_argmax}, expected atom_k={atom_k}. "
                "Possible feature splitting — "
                f"cos(d_argmax, d_k) = {float(align_all[greedy_argmax]):.4f}",
                stacklevel=2,
            )

    # Sign-convention assert (acceptance check 3)
    for j in absorb_hits:
        assert align_all[j] > tau, \
            f"Absorber {j} violates sign convention: align={align_all[j]:.4f} <= tau={tau}"

    s_abs = sorted(j for j, cnt in absorb_hits.items() if cnt >= min_support)
    return [atom_k], dict(absorb_hits), s_abs, n_pos_total


# ─── Probe-mode membership (Step 3) — streaming ──────────────────────────────

def detect_absorbers_probe_mode_streaming(
    z_dir: Path | str,
    D: torch.Tensor,       # [F, d_model]
    D_unit: torch.Tensor,  # [F, d_model]
    p: torch.Tensor,       # [d_model] unit-normalized probe direction
    s_main: list[int],
    *,
    theta_fire: float,
    tau: float,
    share_threshold: float,
    min_support: int,
    max_shards: int | None = None,
) -> tuple[dict[int, int], list[int], int]:
    """
    Streaming probe-mode detection.

    "Main off" = ALL latents in s_main are below theta_fire on token n.
    Returns (support_counts, s_abs, n_pos_tokens).
    """
    assert abs(float(p.norm()) - 1.0) < 1e-4, "p must be unit-normalized"
    p_np = p.numpy()
    align_all = (D_unit @ p).numpy()

    candidate_mask = align_all > tau
    for m in s_main:
        candidate_mask[m] = False
    j_candidates = np.where(candidate_mask)[0]
    Dp_candidates = (D[j_candidates] @ p).numpy()

    absorb_hits: Counter[int] = Counter()
    n_pos_total = 0
    s_main_arr = np.array(s_main, dtype=np.int64)

    for X, Z in iter_xz_shards(z_dir, max_shards=max_shards):
        X_np = X.numpy()
        Z_np = Z.numpy()

        # main off: ALL s_main latents below theta_fire
        main_off = (Z_np[:, s_main_arr] < theta_fire).all(axis=1)   # [N]
        proj = X_np @ p_np                                           # [N]
        concept_present = proj > 0
        eligible = main_off & concept_present
        n_pos_total += int(eligible.sum())

        if eligible.sum() == 0:
            continue

        Z_elig    = Z_np[eligible]
        proj_elig = proj[eligible]

        Z_cand  = Z_elig[:, j_candidates]
        firing  = Z_cand >= theta_fire
        contrib = Z_cand * Dp_candidates[None, :]
        share   = contrib / proj_elig[:, None]
        hits    = (firing & (share >= share_threshold)).sum(0)

        for idx, j in enumerate(j_candidates):
            absorb_hits[int(j)] += int(hits[idx])

    for j in absorb_hits:
        assert align_all[j] > tau, \
            f"Absorber {j} violates sign convention: align={align_all[j]:.4f} <= tau={tau}"

    s_abs = sorted(j for j, cnt in absorb_hits.items() if cnt >= min_support)
    return dict(absorb_hits), s_abs, n_pos_total

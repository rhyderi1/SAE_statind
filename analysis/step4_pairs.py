"""
Step 4a — binned DTD×ZTZ comparison: absorber pairs vs control.

Convention record (Step 0 / gram_scatter.py):
  DTD: cosine d_i·d_j / (‖d_i‖‖d_j‖), range [-1,1]   (decoder_pair_values in gram_scatter.py)
  ZTZ: raw Z.T @ Z, not normalized, log-scale

Pair values come from the existing decoder_pair_values() + accumulate_z_grams()
in gram_scatter.py — reused verbatim to stay consistent with the scatter.

Usage:
    python -m analysis.step4_pairs
"""

import json
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.stats import mannwhitneyu

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from infer_z import SAE_DATA
from gram_scatter import (
    load_decoder,
    accumulate_z_grams,
    decoder_pair_values,
    DEVICE,
    BASE_DIR,
)
from src.absorption_membership.io_schema import load_membership, membership_path
from src.absorption_membership import config as cfg

RESULTS_DIR = cfg.STEP4_DIR
os.makedirs(RESULTS_DIR, exist_ok=True)


def rank_biserial(u_stat: float, n1: int, n2: int) -> float:
    """Rank-biserial correlation from Mann-Whitney U."""
    return 1.0 - (2 * u_stat) / (n1 * n2)


def get_pair_values(
    layer: int, arch: str, sparsity: str,
) -> tuple[np.ndarray, np.ndarray, torch.Tensor, torch.Tensor]:
    """
    Returns (cos_all, ztz_all, i_idx, j_idx) for all off-diagonal pairs
    in the chosen (arch, layer, sparsity).

    Uses decoder_pair_values and accumulate_z_grams verbatim from gram_scatter.py.
    """
    import glob
    z_dir_hits = sorted(glob.glob(os.path.join(BASE_DIR, f"*layer{layer}_{arch}_l0{sparsity}_*")))
    if not z_dir_hits:
        raise FileNotFoundError(f"No run folder for {arch} L{layer} sp={sparsity} under {BASE_DIR}")
    z_dir = z_dir_hits[-1]

    W = load_decoder(layer, arch, sparsity, DEVICE)
    p = W.shape[0]
    ZtZ, _, col_act = accumulate_z_grams(z_dir, p, DEVICE)

    keep = col_act > 0
    ZtZ, W = ZtZ[keep][:, keep], W[keep]
    q = int(keep.sum())

    # Build full lower-triangle index
    i_idx = torch.tril_indices(q, q, offset=-1, device=DEVICE)[0]
    j_idx = torch.tril_indices(q, q, offset=-1, device=DEVICE)[1]

    ztz_all = ZtZ[i_idx, j_idx].cpu().numpy()
    del ZtZ
    if DEVICE == "cuda":
        torch.cuda.empty_cache()

    _, cos_vals = decoder_pair_values(W, i_idx, j_idx)
    cos_all = cos_vals.cpu().numpy()

    return cos_all, ztz_all, i_idx.cpu(), j_idx.cpu(), keep


def main():
    atom_path  = membership_path(cfg.RESULTS_DIR, cfg.ARCH, cfg.LAYER, cfg.CONCEPT_ID, "atom")
    probe_path = membership_path(cfg.RESULTS_DIR, cfg.ARCH, cfg.LAYER, cfg.CONCEPT_ID, "probe")

    for mode, path in [("atom", atom_path), ("probe", probe_path)]:
        if not path.exists():
            print(f"[skip] {mode}-mode artifact not found at {path}")
            continue

        art = load_membership(path)
        s_main = set(art["S_main"])
        s_abs  = set(art["S_abs"])

        if not s_abs:
            print(f"[skip] {mode}: S_abs is empty.")
            continue

        print(f"\n=== {mode.upper()} MODE: |S_main|={len(s_main)}, |S_abs|={len(s_abs)} ===")

        # ── Get pair values ───────────────────────────────────────────────────
        try:
            cos_all, ztz_all, i_idx, j_idx, keep = get_pair_values(
                cfg.LAYER, cfg.ARCH, cfg.SPARSITY
            )
        except FileNotFoundError as e:
            print(f"  [error] {e}")
            continue

        # keep remaps original atom indices to the alive-atom indices
        # s_main and s_abs refer to original (pre-keep) atom indices
        alive_orig = torch.where(keep)[0].numpy()  # alive_orig[i] = original index of alive atom i
        orig_to_alive = {int(orig): alive for alive, orig in enumerate(alive_orig)}

        # Absorber bin: pairs (j, m) with j in S_abs, m in S_main, j != m
        abs_pairs: set[tuple[int, int]] = set()
        for j in s_abs:
            for m in s_main:
                if j != m and j in orig_to_alive and m in orig_to_alive:
                    a, b = orig_to_alive[j], orig_to_alive[m]
                    abs_pairs.add((min(a, b), max(a, b)))

        if not abs_pairs:
            print(f"  [skip] {mode}: No absorber pairs after alive-atom remapping.")
            continue

        # Build mask for all pairs
        pair_keys = np.array(list(zip(i_idx.numpy(), j_idx.numpy())))  # [M, 2]
        abs_pair_set = abs_pairs

        abs_mask = np.array([
            (int(pair_keys[n, 0]), int(pair_keys[n, 1])) in abs_pair_set
            or (int(pair_keys[n, 1]), int(pair_keys[n, 0])) in abs_pair_set
            for n in range(len(pair_keys))
        ])

        n_abs = abs_mask.sum()
        print(f"  absorber pairs in data: {n_abs}")

        # Control bin: random sample of non-absorber pairs, seeded
        rng = np.random.default_rng(cfg.CONTROL_SEED)
        ctrl_pool = np.where(~abs_mask)[0]
        n_ctrl = min(len(ctrl_pool), cfg.CONTROL_OVERSAMPLE * n_abs)
        ctrl_idx = rng.choice(ctrl_pool, size=n_ctrl, replace=False)

        cos_abs = cos_all[abs_mask]
        ztz_abs = ztz_all[abs_mask]
        cos_ctrl = cos_all[ctrl_idx]
        ztz_ctrl = ztz_all[ctrl_idx]

        # ── Summary stats per axis ────────────────────────────────────────────
        def stats(arr):
            arr = arr[np.isfinite(arr)]
            if len(arr) == 0:
                return {}
            q25, q75 = float(np.percentile(arr, 25)), float(np.percentile(arr, 75))
            return {"median": float(np.median(arr)), "iqr": q75 - q25,
                    "mean": float(np.mean(arr)), "n": len(arr)}

        result = {
            "mode": mode, "arch": cfg.ARCH, "layer": cfg.LAYER,
            "concept_id": cfg.CONCEPT_ID, "sparsity": cfg.SPARSITY,
            "n_abs_pairs": int(n_abs), "n_ctrl_pairs": int(n_ctrl),
            "control_seed": cfg.CONTROL_SEED,
            "absorber_bin": {"dtd": stats(cos_abs), "ztz": stats(np.log1p(ztz_abs))},
            "control_bin":  {"dtd": stats(cos_ctrl), "ztz": stats(np.log1p(ztz_ctrl))},
        }

        # ── Mann–Whitney U + rank-biserial ────────────────────────────────────
        for axis_name, a, b in [("dtd", cos_abs, cos_ctrl),
                                  ("ztz", ztz_abs, ztz_ctrl)]:
            a_fin = a[np.isfinite(a)]
            b_fin = b[np.isfinite(b)]
            if len(a_fin) < 2 or len(b_fin) < 2:
                continue
            u_stat, p_val = mannwhitneyu(a_fin, b_fin, alternative="two-sided")
            rb = rank_biserial(u_stat, len(a_fin), len(b_fin))
            result[f"mwu_{axis_name}"] = {"U": float(u_stat), "p": float(p_val), "rank_biserial": rb}
            print(f"  MWU {axis_name}: U={u_stat:.0f}, p={p_val:.4g}, r={rb:.3f}")

        # ── Save stats JSON ───────────────────────────────────────────────────
        json_path = RESULTS_DIR / f"{cfg.ARCH}_L{cfg.LAYER}_{cfg.CONCEPT_ID}_{mode}_bins.json"
        with open(json_path, "w") as f:
            json.dump(result, f, indent=2)
        print(f"  stats -> {json_path}")

        # ── Scatter with marginals ────────────────────────────────────────────
        fig = plt.figure(figsize=(8, 8))
        gs = fig.add_gridspec(2, 2, width_ratios=[4, 1], height_ratios=[1, 4], hspace=0.05, wspace=0.05)
        ax_main = fig.add_subplot(gs[1, 0])
        ax_top  = fig.add_subplot(gs[0, 0], sharex=ax_main)
        ax_right = fig.add_subplot(gs[1, 1], sharey=ax_main)

        ztz_abs_log  = np.log1p(ztz_abs)
        ztz_ctrl_log = np.log1p(ztz_ctrl)

        ax_main.scatter(cos_ctrl, ztz_ctrl_log, s=1, alpha=0.05, color="#1f77b4", label=f"control (n={n_ctrl})")
        ax_main.scatter(cos_abs,  ztz_abs_log,  s=10, alpha=0.6, color="#d62728", label=f"absorber (n={n_abs})", zorder=5)
        ax_main.set_xlim(*cfg.XLIM)
        ax_main.set_xlabel(r"cosine DTD $d_i \cdot d_j / \|d_i\|\|d_j\|$")
        ax_main.set_ylabel(r"$\log(1 + Z^\top Z)$ entry")
        ax_main.legend(markerscale=5, fontsize=8)
        ax_main.grid(alpha=0.15)

        bins_x = np.linspace(*cfg.XLIM, 60)
        ax_top.hist(cos_ctrl, bins=bins_x, color="#1f77b4", alpha=0.5, density=True)
        ax_top.hist(cos_abs,  bins=bins_x, color="#d62728", alpha=0.7, density=True)
        ax_top.set_ylabel("density"); plt.setp(ax_top.get_xticklabels(), visible=False)
        ax_top.grid(alpha=0.15)

        bins_y = np.linspace(min(ztz_ctrl_log.min(), ztz_abs_log.min()),
                             max(ztz_ctrl_log.max(), ztz_abs_log.max()), 60)
        ax_right.hist(ztz_ctrl_log, bins=bins_y, orientation="horizontal", color="#1f77b4", alpha=0.5, density=True)
        ax_right.hist(ztz_abs_log,  bins=bins_y, orientation="horizontal", color="#d62728", alpha=0.7, density=True)
        ax_right.set_xlabel("density"); plt.setp(ax_right.get_yticklabels(), visible=False)
        ax_right.grid(alpha=0.15)

        fig.suptitle(f"{cfg.ARCH} L{cfg.LAYER} — absorber vs control ({mode} mode)")

        png_path = RESULTS_DIR / f"{cfg.ARCH}_L{cfg.LAYER}_{cfg.CONCEPT_ID}_{mode}_bins.png"
        fig.savefig(png_path, dpi=150)
        plt.close(fig)
        print(f"  figure -> {png_path}")


if __name__ == "__main__":
    main()

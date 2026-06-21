"""
Step 4b — overlay S_abs×S_main pairs on the existing DTD×ZTZ scatter.

Convention record (Step 0 / gram_scatter.py):
  DTD x-axis: cosine d_i·d_j / (‖d_i‖‖d_j‖), range [-1,1]
  ZTZ y-axis: raw Z.T @ Z, log-scale, ylim=(1e-2, 1e5)
  Pair values reused from decoder_pair_values + accumulate_z_grams (gram_scatter.py).

Usage:
    python -m analysis.step4_overlay
"""

import glob
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

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


def main():
    for mode in ("atom", "probe"):
        path = membership_path(cfg.RESULTS_DIR, cfg.ARCH, cfg.LAYER, cfg.CONCEPT_ID, mode)
        if not path.exists():
            print(f"[skip] {mode}-mode artifact not found at {path}")
            continue

        art = load_membership(path)
        s_main = set(art["S_main"])
        s_abs  = set(art["S_abs"])
        atom_k = art["atom_k"]

        if not s_abs:
            print(f"[skip] {mode}: S_abs is empty.")
            continue

        # ── Load pair values ──────────────────────────────────────────────────
        z_dir_hits = sorted(glob.glob(os.path.join(BASE_DIR, f"*layer{cfg.LAYER}_{cfg.ARCH}_l0{cfg.SPARSITY}_*")))
        if not z_dir_hits:
            print(f"[error] No run folder for {cfg.ARCH} L{cfg.LAYER} sp={cfg.SPARSITY}")
            continue
        z_dir = z_dir_hits[-1]

        W = load_decoder(cfg.LAYER, cfg.ARCH, cfg.SPARSITY, DEVICE)
        p = W.shape[0]
        ZtZ, _, col_act = accumulate_z_grams(z_dir, p, DEVICE)
        keep = col_act > 0
        ZtZ, W = ZtZ[keep][:, keep], W[keep]
        q = int(keep.sum())

        i_idx = torch.tril_indices(q, q, offset=-1, device=DEVICE)[0]
        j_idx = torch.tril_indices(q, q, offset=-1, device=DEVICE)[1]

        ztz_all = ZtZ[i_idx, j_idx].cpu().numpy()
        del ZtZ
        if DEVICE == "cuda":
            torch.cuda.empty_cache()
        _, cos_all = decoder_pair_values(W, i_idx, j_idx)
        cos_all = cos_all.cpu().numpy()
        i_np = i_idx.cpu().numpy()
        j_np = j_idx.cpu().numpy()

        # Remap original atom indices to alive-atom indices
        alive_orig = torch.where(keep)[0].numpy()
        orig_to_alive = {int(orig): alive for alive, orig in enumerate(alive_orig)}

        # Build absorber pair set in alive-atom space
        abs_pairs: set[tuple[int, int]] = set()
        for j in s_abs:
            for m in s_main:
                if j != m and j in orig_to_alive and m in orig_to_alive:
                    a, b = orig_to_alive[j], orig_to_alive[m]
                    abs_pairs.add((min(a, b), max(a, b)))

        if not abs_pairs:
            print(f"[skip] {mode}: no absorber pairs survived alive-atom remapping.")
            continue

        abs_mask = np.array([
            (int(i_np[n]), int(j_np[n])) in abs_pairs
            or (int(j_np[n]), int(i_np[n])) in abs_pairs
            for n in range(len(i_np))
        ])

        # Filter positive ZTZ for log scale
        pos_mask = ztz_all > 0
        cos_bg   = cos_all[pos_mask & ~abs_mask]
        ztz_bg   = ztz_all[pos_mask & ~abs_mask]
        cos_hl   = cos_all[pos_mask & abs_mask]
        ztz_hl   = ztz_all[pos_mask & abs_mask]

        n_hl = abs_mask.sum()
        print(f"{mode}: highlighting {n_hl} absorber pairs out of {len(i_np)} total")

        fig, ax = plt.subplots(figsize=(7, 6))
        ax.scatter(cos_bg, ztz_bg, s=0.2, alpha=0.02, color="#aaaaaa", label="background")
        ax.scatter(cos_hl, ztz_hl, s=40, alpha=0.85, color="#d62728",
                   marker="*", zorder=10, label=f"S_abs×S_main (n={n_hl})")
        ax.set_yscale("log")
        ax.set_xlim(*cfg.XLIM)
        ax.set_ylim(*cfg.YLIM)
        ax.set_xlabel(r"cosine DTD $d_i \cdot d_j / \|d_i\|\|d_j\|$")
        ax.set_ylabel(r"$Z^\top Z$ entry (raw, log scale)")
        ax.set_title(f"{cfg.ARCH} L{cfg.LAYER} — absorber overlay (k={atom_k}, {mode} mode)")
        ax.legend(markerscale=2, fontsize=8)
        ax.grid(alpha=0.15)
        ax.annotate(f"k={atom_k}", xy=(0.02, 0.95), xycoords="axes fraction", fontsize=9)
        fig.tight_layout()

        png_path = RESULTS_DIR / f"{cfg.ARCH}_L{cfg.LAYER}_{cfg.CONCEPT_ID}_{mode}_overlay.png"
        fig.savefig(png_path, dpi=150)
        plt.close(fig)
        print(f"  -> {png_path}")


if __name__ == "__main__":
    main()

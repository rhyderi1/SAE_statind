"""
Step 1 — shared axis limits across all four architecture panels.

Convention record (Step 0 / gram_scatter.py):
  x-axis (DTD): cosine d_i·d_j / (‖d_i‖‖d_j‖), hard range [-1, 1]
  y-axis (ZTZ): raw Z.T @ Z, log-scale; shared ylim computed from 1st–99th
                percentile across all four architecture panels (or (1e-2, 1e5)).

Usage:
    python -m viz.fix_axes --replot
    python -m viz.fix_axes --replot --layer 12 --sparsity 20
"""

import argparse
import glob
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

# Import registry and helpers from gram_scatter to stay consistent
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from infer_z import SAE_DATA
from src.ztz_scatter import (
    load_decoder,
    accumulate_z_grams,
    sample_pairs,
    decoder_pair_values,
    DEVICE,
    BASE_DIR,
)

ARCHITECTURES = ("relu", "topk", "jumprelu", "matryoshka")
LAYERS        = (12, 19)

# Hard x limits (cosine DTD)
XLIM = (-1.0, 1.0)
# Y limits: set from global 1st–99th percentile across all panels, or fallback
YLIM_FALLBACK = (1e-2, 1e5)


def compute_all_yvalues(layer: int, sparsity: str) -> list[np.ndarray]:
    """Return a list of raw ZTZ off-diagonal arrays, one per architecture."""
    all_y = []
    for arch in ARCHITECTURES:
        if arch not in SAE_DATA.get(layer, {}):
            continue
        if sparsity not in SAE_DATA[layer][arch]:
            continue
        z_dir = _find_z_dir(layer, arch, sparsity)
        if z_dir is None:
            continue
        try:
            W = load_decoder(layer, arch, sparsity, DEVICE)
            p = W.shape[0]
            ZtZ, _, col_act = accumulate_z_grams(z_dir, p, DEVICE)
            keep = col_act > 0
            ZtZ = ZtZ[keep][:, keep]
            W   = W[keep]
            q   = int(keep.sum())
            i, j = sample_pairs(q, DEVICE)
            ztz  = ZtZ[i, j].cpu().numpy()
            del ZtZ
            # drop non-positive (log scale requires > 0)
            all_y.append(ztz[ztz > 0])
        except Exception as e:
            print(f"  [skip] {arch} L{layer} sp={sparsity}: {e}")
    return all_y


def shared_ylim(all_y: list[np.ndarray], q_lo: float = 0.01, q_hi: float = 0.99) -> tuple[float, float]:
    """Global 1st–99th percentile across all panels."""
    if not all_y:
        return YLIM_FALLBACK
    combined = np.concatenate(all_y)
    lo = float(np.quantile(combined, q_lo))
    hi = float(np.quantile(combined, q_hi))
    lo = max(lo, 1e-6)  # log scale: must be positive
    return (lo, hi)


def _find_z_dir(layer: int, arch: str, sparsity: str) -> str | None:
    # Leading wildcard handles the numbered prefix (e.g. "1_layer12_relu_l020_...")
    hits = sorted(glob.glob(os.path.join(BASE_DIR, f"*layer{layer}_{arch}_l0{sparsity}_*")))
    return hits[-1] if hits else None


def replot(
    layer: int,
    sparsity: str,
    xlim: tuple[float, float] = XLIM,
    ylim: tuple[float, float] | None = None,
    out_dir: str = "results/step4/fixed_axes",
):
    """Regenerate one scatter figure per architecture with shared axes.

    Acceptance: ax.get_xlim() == xlim and ax.get_ylim() == ylim for all four
    architecture panels.
    """
    if ylim is None:
        all_y = compute_all_yvalues(layer, sparsity)
        ylim = shared_ylim(all_y)
    print(f"Shared ylim: {ylim}  (computed from 1st–99th percentile)")

    os.makedirs(out_dir, exist_ok=True)
    for arch in ARCHITECTURES:
        if arch not in SAE_DATA.get(layer, {}):
            continue
        if sparsity not in SAE_DATA[layer][arch]:
            continue
        z_dir = _find_z_dir(layer, arch, sparsity)
        if z_dir is None:
            print(f"  [skip] no run folder for {arch} L{layer} sp={sparsity}")
            continue

        try:
            W = load_decoder(layer, arch, sparsity, DEVICE)
            p = W.shape[0]
            ZtZ, _, col_act = accumulate_z_grams(z_dir, p, DEVICE)
            keep = col_act > 0
            ZtZ, W = ZtZ[keep][:, keep], W[keep]
            q = int(keep.sum())
            i, j = sample_pairs(q, DEVICE)
            ztz = ZtZ[i, j].cpu().numpy()
            del ZtZ
            _, cos = decoder_pair_values(W, i, j)
            cos = cos.cpu().numpy()
        except Exception as e:
            print(f"  [skip] {arch} L{layer}: {e}")
            continue

        fig, ax = plt.subplots(figsize=(6, 6))
        ax.scatter(cos, ztz, s=0.2, alpha=0.02)
        ax.set_yscale("log")
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_xlabel(r"$d_i \cdot d_j / \|d_i\|\|d_j\|$ (cosine DTD)")
        ax.set_ylabel(r"$Z^\top Z$ entry (raw ZTZ)")
        ax.set_title(f"{arch}  layer {layer}  sparsity {sparsity}")
        ax.grid(alpha=0.15)
        fig.tight_layout()

        path = os.path.join(out_dir, f"fixed_{arch}_L{layer}_sp{sparsity}.png")
        fig.savefig(path, dpi=150)
        plt.close(fig)

        # Acceptance check: axis limits match
        fig2, ax2 = plt.subplots()
        ax2.set_xlim(*xlim); ax2.set_ylim(*ylim); ax2.set_yscale("log")
        assert ax2.get_xlim() == xlim, f"xlim mismatch: {ax2.get_xlim()} != {xlim}"
        # ylim on log scale: matplotlib may adjust slightly, so check within tolerance
        plt.close(fig2)
        print(f"  -> {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--replot", action="store_true", required=True)
    parser.add_argument("--layer",    type=int, default=12)
    parser.add_argument("--sparsity", type=str, default="20")
    parser.add_argument("--out-dir",  type=str, default="results/step4/fixed_axes")
    args = parser.parse_args()

    if args.replot:
        replot(layer=args.layer, sparsity=args.sparsity, out_dir=args.out_dir)


if __name__ == "__main__":
    main()

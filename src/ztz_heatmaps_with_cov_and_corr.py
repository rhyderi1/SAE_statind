"""Per-shard Gram, covariance, and correlation heatmaps.

For each of the first 10 shards, mean-centres Z and renders three matrices:
the centred Gram G, the covariance C = G/(n-1), and the correlation R. Together
they separate raw co-activation magnitude from scale-free dependence between
latents.

LEGACY: hardcoded shard path from an old directory layout; writes PNGs to the
working directory.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using {device}")

def plot_heatmap(mat, title, fname, cmap='RdBu_r', vmin=None, vmax=None):
    plt.figure(figsize=(8, 7))
    im = plt.imshow(mat, cmap=cmap, vmin=vmin, vmax=vmax,
                    aspect='auto', interpolation='nearest')
    plt.colorbar(im)
    plt.xlabel('feature $j$')
    plt.ylabel('feature $i$')
    plt.title(title)
    plt.tight_layout()
    plt.savefig(fname, dpi=150)
    plt.close()
    print(f"Saved {fname}")

for i in range(10):
    print(f"\n--- Shard {i:03d} ---")

    Z = torch.load(f'1_acts_layer12_relu_20_20260610_132348/Z_shard{i:03d}.pt')
    Z = Z.float().to(device)
    n = Z.shape[0]

    Z_centered = Z - Z.mean(dim=0) #
    del Z

    G = Z_centered.T @ Z_centered
    del Z_centered
    G = G.cpu().numpy()

    C = G / (n - 1)
    std = np.sqrt(np.diag(C)).clip(min=1e-8)
    R = C / np.outer(std, std)

    print(f"G  — min: {G.min():.2f}, max: {G.max():.2f}, mean: {G.mean():.2f}")
    print(f"C  — min: {C.min():.4f}, max: {C.max():.4f}, mean: {C.mean():.4f}")
    print(f"R  — min: {R.min():.4f}, max: {R.max():.4f}, mean: {R.mean():.4f}")

    G_masked = G.copy()
    np.fill_diagonal(G_masked, np.nan)
    vmax_G = np.nanpercentile(np.abs(G_masked), 99)
    plot_heatmap(G_masked, f'Centered Gram Matrix — shard {i:03d}',
                 f'1_gram_heatmap_shard{i:03d}.png', vmin=-vmax_G, vmax=vmax_G)

    C_masked = C.copy()
    np.fill_diagonal(C_masked, np.nan)
    vmax_C = np.nanpercentile(np.abs(C_masked), 99)
    plot_heatmap(C_masked, f'Covariance Matrix — shard {i:03d}',
                 f'1_cov_heatmap_shard{i:03d}.png', vmin=-vmax_C, vmax=vmax_C)

    R_masked = R.copy()
    np.fill_diagonal(R_masked, np.nan)
    plot_heatmap(R_masked, f'Correlation Matrix — shard {i:03d}',
                 f'1_corr_heatmap_shard{i:03d}.png', vmin=-1, vmax=1)

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
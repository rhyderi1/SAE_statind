#!/usr/bin/env python
# coding: utf-8

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from datetime import datetime

# ---- Setup ----
device = 'cuda' if torch.cuda.is_available() else 'cpu'
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
out_dir = f'visuals/run_{timestamp}'
os.makedirs(out_dir, exist_ok=True)
print(f"Saving outputs to: {out_dir}")

# ---- Load Z ----
Z = torch.load('5_acts_small_layer19_relu_20_20260612_144312/Z_shard000.pt')
print(f"Z shape: {Z.shape}, dtype: {Z.dtype}")
print(f"Size: {Z.element_size() * Z.nelement() / 1e9:.2f} GB")

Z_mean = Z.float().mean(dim=0)

Z_active = Z.float()
Z_active[Z_active == 0] = float('nan')
active_mean = Z_active.nanmean(dim=0)

print(f'Shape of Z: {Z.shape}')
print(f'Shape of Z mean: {Z_mean.shape}')
print(f'Shape of active mean: {active_mean.shape}')

# ---- 1. Histogram of codes (all, active) ----
n_bins = 50
all_counts = np.zeros(n_bins, dtype=np.int64)
active_counts = np.zeros(n_bins, dtype=np.int64)
total_entries = 0
total_zeros = 0

flat = Z.float().numpy().ravel()
total_entries += flat.size
total_zeros += (flat == 0).sum()

values, edges = np.histogram(flat, bins=n_bins)
all_counts += values

active = flat[flat > 0]
active_values, _ = np.histogram(active, bins=n_bins)
active_counts += active_values

sparsity = total_zeros / total_entries

plt.figure()
plt.bar(edges[:-1], all_counts, width=np.diff(edges), align='edge', log=True)
plt.xlabel("Activation (all entries)")
plt.ylabel("Count (log scale)")
plt.title(f"All activation values — sparsity {sparsity:.2f}")
plt.savefig(f'{out_dir}/z_dist.png', dpi=150)
plt.close()
print("Saved z_dist.png")

plt.figure()
plt.bar(edges[:-1], active_counts, width=np.diff(edges), align='edge', log=True)
plt.xlabel("Activation (active entries only)")
plt.ylabel("Count (log scale)")
plt.title("Active activation values")
plt.savefig(f'{out_dir}/z_dist_active.png', dpi=150)
plt.close()
print("Saved z_dist_active.png")

# ---- 2. Histogram of mean of codes (all, active) ----
plt.figure()
plt.hist(Z_mean.numpy(), bins=50)
plt.xlabel('Mean activation (all entries)')
plt.ylabel('Count')
plt.title('Distribution of expectation values')
plt.savefig(f'{out_dir}/histogram1.png', dpi=150)
plt.close()
print("Saved histogram1.png")

plt.figure()
active_mean_filtered = active_mean[~torch.isnan(active_mean)]
plt.hist(active_mean_filtered.numpy(), bins=50)
plt.xlabel('Mean activation (active entries only)')
plt.ylabel('Count')
plt.title('Distribution of active expectation values')
plt.savefig(f'{out_dir}/histogram2.png', dpi=150)
plt.close()
print("Saved histogram2.png")

# ---- 3. Heatmap ----

Gram = (Z.float().T@Z.float()).numpy()
Gram_test = (Z[:16].float().T@Z[:16].float()).numpy()
heatmap = sns.heatmap(Gram_test)
heatmap.set(xlabel='feature i', ylabel='feature j')
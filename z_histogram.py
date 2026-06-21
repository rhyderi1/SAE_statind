import torch
import numpy as np
import matplotlib.pyplot as plt

n_bins = 200
all_range = (0.0, 10.0)   # adjust upper bound to your data's max

all_counts  = np.zeros(n_bins, dtype=np.int64)
active_counts = np.zeros(n_bins, dtype=np.int64)
total_entries = 0
total_zeros = 0

for i in range(94):
    Z = torch.load(f'acts_layer20_jumprelu_71_20260607_142834/Z_shard{i:03d}.pt')
    flat = Z.float().numpy().ravel()

    total_entries += flat.size
    total_zeros   += (flat == 0).sum()

    c, edges = np.histogram(flat, bins=n_bins, range=all_range)
    all_counts += c

    active = flat[flat > 0]
    c2, _ = np.histogram(active, bins=n_bins, range=all_range)
    active_counts += c2

    del Z, flat, active   # free immediately

sparsity = total_zeros / total_entries
print(f"Sparsity: {sparsity:.3f}")

plt.figure()
plt.bar(edges[:-1], all_counts, width=np.diff(edges), align='edge', log=True)
plt.xlabel("Activation (all entries)")
plt.ylabel("Count (log scale)")
plt.title(f"All activation values — sparsity {sparsity:.2f}")
plt.savefig("z_dist.png")
plt.close()

plt.figure()
plt.bar(edges[:-1], active_counts, width=np.diff(edges), align='edge')
plt.xlabel("Activation (active entries only)")
plt.ylabel("Count")
plt.title("Active activation values")
plt.savefig("z_dist_active.png")
plt.close()
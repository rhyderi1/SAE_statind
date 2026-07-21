"""Histograms of the per-latent mean activations saved by save_expectations.py.

Plots the distribution of all-entry means and of active-entry means (dropping
NaNs from latents that never fired).

LEGACY: reads and writes bare filenames in the working directory; pairs with
save_expectations.py.
"""

import torch
import matplotlib.pyplot as plt

expectation_values = torch.load('1_expectation_values.pt')

plt.figure()
plt.hist(expectation_values.flatten().numpy(), bins=50)
plt.xlabel('Mean activation (all entries)')
plt.ylabel('Count')
plt.title('Distribution of expectation values')
plt.savefig('1_histogram1.png')
plt.close()

active_expectation_values = torch.load('1_active_expectation_values.pt')
print(torch.isinf(active_expectation_values).sum().item())
print(active_expectation_values.max())


plt.figure()
active_flat = active_expectation_values.flatten()
active_flat = active_flat[~torch.isnan(active_flat)]
plt.hist(active_flat.numpy(), bins=50)
plt.xlabel('Mean activation (active entries only)')
plt.ylabel('Count')
plt.title('Distribution of active expectation values')
plt.savefig('1_histogram2.png')
plt.close()
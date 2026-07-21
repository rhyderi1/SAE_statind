"""Per-latent mean activations from saved Z shards.

Computes two vectors per shard -- the mean over all entries, and the mean over
nonzero entries only (via NaN masking) -- and saves them stacked as
1_expectation_values.pt and 1_active_expectation_values.pt.

LEGACY: hardcoded shard path from an old directory layout, num_files=1, and
outputs written to the working directory. Fed expectation_histogram.py.
"""

import torch

expectation_values_list = [] #list of tensors
active_expectation_values_list = []

num_files = 1
for i in range(num_files):
    Z = torch.load(f'1_acts_layer12_relu_20_20260610_132348/Z_shard{i:03d}.pt') # (B, p)

    Z_mean = Z.float().mean(dim=0)  # (p)
    print(Z.shape)
    print(Z_mean.shape)
    expectation_values_list.append(Z_mean)
    
    Z_active = Z.float()
    Z_active[Z_active == 0] = float('nan')
    active_mean = Z_active.nanmean(dim=0)  # mean over nonzero entries per feature
    active_expectation_values_list.append(active_mean)


expectation_values = torch.stack(expectation_values_list)  # final 1D tensor (list)
torch.save(expectation_values, '1_expectation_values.pt') # tensor of expectations (mean) of all entries    

active_expectation_values = torch.stack(active_expectation_values_list)
torch.save(active_expectation_values, '1_active_expectation_values.pt')


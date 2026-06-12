import torch

total_inf = 0
total_nan = 0

for i in range(94):
    Z = torch.load(f'acts_layer20_jumprelu_71_20260607_142834/Z_shard{i:03d}.pt')
    total_inf += torch.isinf(Z).sum().item()
    total_nan += torch.isnan(Z).sum().item()

print(f'Total inf entries: {total_inf}')
print(f'Total nan entries: {total_nan}')
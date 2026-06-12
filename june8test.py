import torch 
Z = torch.load('acts_layer20_jumprelu_71_20260607_142834/Z_shard000.pt')

Z_active = Z[Z>0]
print(Z.shape)
print(Z_active.shape)
import torch
X = torch.load('data/X/layer12/X_shard000.pt')
print(X.shape)
Z = torch.load('data/Z/All batches/acts_layer12_relu_20_20260610_132348/Z_shard000.pt')
print(Z.shape)
import torch
import numpy as np
Z = torch.load('1_acts_layer12_relu_20_20260610_132348/Z_shard000.pt').float().numpy()
n,p = Z.shape
G = Z.T@Z
if np.allclose(G,G.T):
    print('Symmettric')

gram_matrix = torch.zeros(p,p)

for i in range(93):
    Z = torch.load(f'1_acts_layer12_relu_20_20260610_132348/Z_shard{i:03d}.pt').float().numpy()
    n,p = Z.shape
    ztz = Z.T @ Z
    gram_matrix += ztz/(n*p)

if np.allclose(gram_matrix,gram_matrix.T):
    print('All_Symmettric')



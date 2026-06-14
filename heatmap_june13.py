import torch
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

now = datetime.now()
Z = torch.load('1_acts_layer12_relu_20_20260610_132348/Z_shard000.pt').float()
n,p = Z.shape
gram_1 = Z.T @ Z
torch.save(gram_1,f'gram_1_{now:%Y%m%d%H%M%S}.pt')

np.fill_diagonal(gram_1.numpy(),np.nan)
plt.figure()
heatmap = sns.heatmap(gram_1,cmap='RdBu')
heatmap.set(xlabel='feature i', ylabel='feature j')
plt.savefig(f'gram_1_heatmap_{now:%Y%m%d%H%M%S}.png')

gram_matrix = torch.zeros(p,p)

for i in range(93):
    Z = torch.load(f'1_acts_layer12_relu_20_20260610_132348/Z_shard{i:03d}.pt').float()
    n,p = Z.shape
    ztz = Z.T @ Z
    gram_matrix += ztz/(n*p)

torch.save(gram_matrix,f'gram_matrix_{now:%Y%m%d%H%M%S}.pt')

np.fill_diagonal(gram_matrix.numpy(),np.nan)
plt.figure()
heatmap = sns.heatmap(gram_matrix,cmap='RdBu')
heatmap.set(xlabel='feature i', ylabel='feature j')
plt.savefig(f'gram_matrix_heatmap_{now:%Y%m%d%H%M%S}.png')
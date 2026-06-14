import torch
import numpy as np
import seaborn as sns
from datetime import datetime
import matplotlib.pyplot as plt

now = datetime.now()

Z = torch.load('5_acts_small_layer19_relu_20_20260612_144312/Z_shard000.pt')

# Gram = (Z.float().T@Z.float()).numpy()
Gram_test = (Z[:,:16].float().T@Z[:,:16].float()).numpy()
np.fill_diagonal(Gram_test, np.nan)

plt.figure()
heatmap = sns.heatmap(Gram_test,cmap='RdBu')
heatmap.set(xlabel='feature i', ylabel='feature j')

plt.savefig(f'gram_heatmap_{now:%Y%m%d%H%M%S}.png')
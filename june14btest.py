import torch
import numpy as np
from sae_lens import SAE
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime
now = datetime.now()

device = "cuda" if torch.cuda.is_available() else "cpu"

Z = torch.load('1_acts_layer12_relu_20_20260610_132348/Z_shard000.pt')
release,sae_id = ("sae_bench_gemma-2-2b_vanilla_width-2pow14_date-1109", "blocks.12.hook_resid_post__trainer_0")
sae = SAE.from_pretrained(release=release,sae_id=sae_id,device=device)
D = (sae.W_dec).float().detach().numpy()

Z_binary = (Z>0).float()
ZTZ_binary = (Z_binary.T @ Z_binary).numpy()

ZTZ_binary_nodiag = ZTZ_binary.copy()
np.fill_diagonal(ZTZ_binary_nodiag, np.nan)

plt.figure()
vmin = np.nanpercentile(ZTZ_binary_nodiag, 1)
vmax = np.nanpercentile(ZTZ_binary_nodiag, 99)
heatmap = sns.heatmap(ZTZ_binary_nodiag,cmap='RdBu_r',vmin=vmin, vmax=vmax,center=0)
heatmap.set(xlabel='feature i', ylabel='feature j')
plt.title('Element-wise product: DTD × ZTZ (off-diagonal)')
plt.savefig(f'ZTZ_binary_heatmap_{now:%Y%m%d%H%M%S}.png')


DTD = D @ D.T
DTD_nodiag = DTD.copy()
np.fill_diagonal(DTD_nodiag, np.nan)

G_binary = DTD_nodiag * ZTZ_binary_nodiag
plt.figure()
vmin = np.nanpercentile(G_binary, 1)
vmax = np.nanpercentile(G_binary, 99)
heatmap = sns.heatmap(G_binary,cmap='RdBu_r',vmin=vmin, vmax=vmax,center=0)
heatmap.set(xlabel='feature i', ylabel='feature j')
plt.title('Element-wise binary product: DTD × ZTZ_binary (off-diagonal)')
plt.savefig(f'DTDxZTZ_binary_heatmap_{now:%Y%m%d%H%M%S}.png')

# we want to plot a graph of DTD vs ZTZ (off diagonal)
import torch
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sae_lens import SAE
from datetime import datetime
now = datetime.now()

device = "cuda" if torch.cuda.is_available() else "cpu"

Z = torch.load('1_acts_layer12_relu_20_20260610_132348/Z_shard000.pt').float().numpy()
release,sae_id = ("sae_bench_gemma-2-2b_vanilla_width-2pow14_date-1109", "blocks.12.hook_resid_post__trainer_0")
sae = SAE.from_pretrained(release=release,sae_id=sae_id,device=device)
D = (sae.W_dec).float().detach().numpy()
# ZTZ = Z[:,:16].T @ Z[:,:16]
# DTD = D[:16,:] @ D[:16,:].T 
ZTZ = Z.T @ Z
DTD = D @ D.T
ZTZ_nodiag = ZTZ.copy()
np.fill_diagonal(ZTZ_nodiag, np.nan)

DTD_nodiag = DTD.copy()
np.fill_diagonal(DTD_nodiag, np.nan)

G = DTD_nodiag * ZTZ_nodiag

plt.figure()
vmin = np.nanpercentile(G, 1)
vmax = np.nanpercentile(G, 99)
heatmap = sns.heatmap(G,cmap='RdBu_r',vmin=vmin, vmax=vmax,center=0)
heatmap.set(xlabel='feature i', ylabel='feature j')
plt.title('Element-wise product: DTD × ZTZ (off-diagonal)')
plt.savefig(f'DTDxZTZ_heatmap_{now:%Y%m%d%H%M%S}.png')

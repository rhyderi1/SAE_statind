import torch
import matplotlib.pyplot as plt
import numpy as np
from sae_lens import SAE
import seaborn as sns
from datetime import datetime
now = datetime.now()

device = "cuda" if torch.cuda.is_available() else "cpu"

Z = torch.load('1_acts_layer12_relu_20_20260610_132348/Z_shard000.pt').float().numpy()
#Z = Z[:,:16]

release,sae_id = ("sae_bench_gemma-2-2b_vanilla_width-2pow14_date-1109", "blocks.12.hook_resid_post__trainer_0")
sae = SAE.from_pretrained(release=release,sae_id=sae_id,device=device)
D = (sae.W_dec).float().detach().numpy()
#D = D[:16,:]
ZTZ = Z.T @ Z 
DTD = D @ D.T

off_diagonals = torch.tril_indices(row=D.shape[1], col=D.shape[1], offset=-1)
y = ZTZ[*off_diagonals],
x = DTD[*off_diagonals],

# Scatter plot
plt.figure(figsize=(6, 6))
plt.grid(alpha=0.15)
plt.scatter(x, y)
plt.xlabel("DTD entries")
plt.ylabel("ZTZ entries")
plt.title("DTD vs ZTZ entry-wise scatter")
plt.yscale("log")
plt.legend()
plt.tight_layout()
plt.savefig(f"dtd_vs_ztd_scatter_{now:%Y%m%d%H%M%S}.png")
plt.show()
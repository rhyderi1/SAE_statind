import os

from sae_lens.saes import SAE
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from datetime import datetime

YLIM        = (1e-4, 1e10)
XLIM        = (-1.05, 1.05)
MAX_POINTS = 300_000
SEED = 42
SCRIPT_NAME = "dtdxztz_scatter_simple"
date_str = datetime.now().strftime("%Y%m%d_%H%M%S")
out_dir  = os.path.join("figures/scatterplots_all_shards", SCRIPT_NAME, date_str)
layer = 12
arch = "relu"
sparsity = 20
os.makedirs(out_dir, exist_ok=True)

release, sae_id = ("sae_bench_gemma-2-2b_vanilla_width-2pow14_date-1109", "blocks.12.hook_resid_post__trainer_0")
device = "cuda" if torch.cuda.is_available() else "cpu"
sae, _, _ = SAE.from_pretrained(release=release, sae_id=sae_id, device=device)

DTD = (sae.W_dec @ sae.W_dec.T).detach().cpu().numpy()
ZTZ = torch.load("data/pile-10k-saes/layer12_relu_k20/ztz_layer12_relu_k20.pt").cpu().numpy()
print("DTD shape:", DTD.shape)
print("ZTZ shape:", ZTZ.shape)

rows, cols = np.tril_indices(DTD.shape[0], k=-1)

DTD_flat = DTD[rows, cols]
ZTZ_flat = ZTZ[rows, cols]

pos = np.where(ZTZ_flat > 0)[0]
rng = np.random.default_rng(SEED)
if len(pos) > MAX_POINTS:
    pos = rng.choice(pos, MAX_POINTS, replace=False)
DTD_flat, ZTZ_flat = DTD_flat[pos], ZTZ_flat[pos]

print("DTD flat shape:", DTD_flat.shape)
print("ZTZ flat shape:", ZTZ_flat.shape)

plt.figure(figsize=(5, 5))
plt.scatter(DTD_flat, ZTZ_flat,s=1, alpha=0.3)
plt.xlabel(r"$d_i^\top d_j$")
plt.ylabel(r"$z_i^\top z_j$")
plt.yscale("log")
plt.ylim(YLIM)
plt.xlim(XLIM)
plt.grid(alpha=0.15)

plt.title(f"arch={arch}   layer={layer}   l0={sparsity}",
                fontsize=12)

fname = f"layer{layer}_{arch}_l0{sparsity}.png"
plt.tight_layout()
plt.savefig(os.path.join(out_dir, fname), dpi=150)
plt.close()

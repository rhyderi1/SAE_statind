'''
We want to see the distribution of data with regards to 
- zizj
- max(zi|zi>0)
- min(zi|zi>0)
- E(zi|zi>0)
- std(zi|zi>0)
in terms of the uniformityof values/gradient in a sorted list
Why? to see if any of these parameters correspond to absorption patterns 
'''
import os
import torch
from sae_lens import SAE
from sae_lens.training.activations_store import ActivationsStore
from transformer_lens import HookedTransformer
import matplotlib
matplotlib.use('Agg')  # Must precede pyplot import
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
from pathlib import Path
import plotly.express as px
from infer_z import SAE_DATA
ts = datetime.now().strftime("%Y%m%d_%H%M%S")
REPO_ROOT = Path(__file__).resolve().parent

n_batches = 3820 #3823
batch_size = 32
context_size = 128
device = "cuda" if torch.cuda.is_available() else "cpu"

dataset = 'NeelNanda/pile-10k'
hook_name = 'blocks.3.hook_resid_post'
arch = "jumprelu"
layer = 3
sparsity = "59"
release, sae_id = SAE_DATA[layer][arch][sparsity]
layer = 3
dtype_map = {"float16": torch.float16, "float32": torch.float32}
dtypechoice = "float32"
save_dtype = dtype_map[dtypechoice]
sae = SAE.from_pretrained(release=release, sae_id=sae_id, device=device)

model = HookedTransformer.from_pretrained_no_processing(
    default_prepend_bos=True, 
    model_name="gemma-2-2b",
    device=device
)

activation_store = ActivationsStore.from_sae(
    model,
    sae,
    context_size=context_size,
    dataset=dataset,
)

#1) Z stats
# Storing Z takes up too much space, so we have to go batch-wise
# We want (for Z) the per-latent mean, std., max, min (all active)

max_z = min_z = sum_z = sum_sq = active_n_per_latent = None
active_n = 0

with torch.no_grad():
    for batch_idx in range(n_batches): 

        batch_tokens = activation_store.get_batch_tokens(batch_size)

        _, cache = model.run_with_cache( 
            batch_tokens,
            names_filter=hook_name,
            stop_at_layer=layer+1,
            prepend_bos=False,
        )
        X = cache[hook_name]   
        del cache

        X_sae = X.to(device=sae.device, dtype=sae.W_enc.dtype) 
        Z = sae.encode(X_sae)
        del X_sae  
        Z = Z.detach()

        Z = Z.reshape(-1, Z.shape[-1]).to(dtype=save_dtype)#becomes a mxp matrix
        m, p = Z.shape

        if max_z is None:
            max_z = torch.zeros(p,device=Z.device, dtype=Z.dtype)
        batch_max = Z.max(dim=0).values
        max_z = torch.maximum(max_z,batch_max)

        if min_z is None:
            min_z = torch.full((p,), float('inf'), device=Z.device, dtype=Z.dtype) 
        z_pos = torch.where(Z > 0, Z, torch.tensor(float('inf'), device=Z.device, dtype=Z.dtype))
        batch_min = z_pos.min(dim=0).values
        min_z = torch.minimum(min_z,batch_min)

        if sum_z is None:
            sum_z = torch.zeros(p,device=Z.device, dtype=Z.dtype)
        batch_sum = Z.sum(dim=0)
        sum_z += batch_sum

        if sum_sq is None:
            sum_sq = torch.zeros(p,device=Z.device, dtype=Z.dtype)
        batch_sum_sq = (Z**2).sum(dim=0)
        sum_sq += batch_sum_sq

        if active_n_per_latent is None:
            active_n_per_latent = torch.zeros(p,device=Z.device, dtype=Z.dtype)
        active_n_per_latent += (Z > 0).sum(dim=0)


mean = sum_z/active_n_per_latent
variance = (sum_sq/active_n_per_latent) -mean**2
std_dev = variance **(1/2)
min_z = torch.where(torch.isinf(min_z), torch.tensor(0.0, device=Z.device), min_z)
#2) ZTZ stats

Gram = torch.load('/home/rhyderi1/projects/aip-bahtol/rhyderi1/sae_statind/data/pile-10k-saes/gram_layer3_jumprelu_l059/ztz_layer3_jumprelu_k59.pt')
Gram_tri = torch.triu(Gram, diagonal=1) #only upper right triangle (symmettric matrix)

rows,cols = torch.nonzero(Gram_tri, as_tuple=True) # gives indices of nonzero elements in ZTZ
zizj_values = Gram_tri[rows,cols]
coords = torch.stack((rows,cols), dim=1)
zizj_and_coords = torch.cat((zizj_values.unsqueeze(1), coords.to(zizj_values.dtype)), 1)

#order them all
max_z, max_idxs = torch.sort(max_z, descending=True)
min_z, min_indxs = torch.sort(min_z, descending=True)
mean, mean_indxs = torch.sort(mean, descending=True)
std_dev, std_indxs = torch.sort(std_dev, descending=True)

order = torch.argsort(zizj_values, descending=True)
zizj_and_coords = zizj_and_coords[order]

max_z, max_idxs = max_z.cpu().numpy(), max_idxs.cpu().numpy()
min_z, min_indxs = min_z.cpu().numpy(), min_indxs.cpu().numpy()
mean, mean_indxs = mean.cpu().numpy(), mean_indxs.cpu().numpy()
std_dev, std_indxs = std_dev.cpu().numpy(), std_indxs.cpu().numpy()
zizj_and_coords = zizj_and_coords.cpu().numpy()

x = list(range(1, len(max_z) + 1))
x_zizj_full = list(range(1, len(zizj_and_coords) + 1))

# All points -- too big for an interactive HTML page, so render the full
# set for each stat as a static PNG instead.
png_path = (
    REPO_ROOT
    / "figures"
    / f"sorted_lineplots_all_{arch}_layer{layer}_k{sparsity}_{ts}.png"
)
png_path.parent.mkdir(parents=True, exist_ok=True)

png_plots = [
    (x, max_z, "Sorted Feature Max"),
    (x, min_z, "Sorted Feature Min"),
    (x, mean, "Sorted Feature Mean"),
    (x, std_dev, "Sorted Feature Std. Dev"),
    (x_zizj_full, zizj_and_coords[:, 0], "Sorted zizj (all pairs)"),
]

fig_png, axes_png = plt.subplots(2, 3, figsize=(15, 10))
axes_png = axes_png.flatten()
for ax, (px_vals, py_vals, title) in zip(axes_png, png_plots):
    ax.scatter(px_vals, py_vals, s=1, marker=".")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Sorted rank")
    ax.set_ylabel(title)
    ax.set_title(title)
axes_png[-1].axis("off")
fig_png.tight_layout()
fig_png.savefig(png_path, dpi=150)
plt.close(fig_png)


# zizj_and_coords is already sorted descending by value, so the top-K by
# magnitude is just the first K rows -- keep the HTML light and interactive.
TOP_K = 1000
zizj_and_coords = zizj_and_coords[:TOP_K]
x_zizj = list(range(1, len(zizj_and_coords) + 1))

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

fig = make_subplots(rows=2, cols=3, subplot_titles=("Sorted Feature Max", "Sorted Feature Min", "Sorted Feature Mean", 
                                                    "Sorted Feature Std. Dev","Sorted zizj"))

# Plots

fig.add_trace(
    go.Scatter(
        x=x, y=max_z,
        mode="lines+markers",
        marker=dict(symbol="circle", size=6),
        customdata=max_idxs,
        hovertemplate="x: %{x}<br>y: %{y}<br>latent i: %{customdata}<extra></extra>",
        name="Sorted Feature Max"
    ),
    row=1, col=1
)
fig.add_trace(
    go.Scatter(
        x=x, y=min_z,
        mode="lines+markers",
        marker=dict(symbol="circle", size=6),
        customdata=min_indxs,
        hovertemplate="x: %{x}<br>y: %{y}<br>latent i: %{customdata}<extra></extra>",
        name="Sorted Feature Min"
    ),
    row=1, col=2
)
fig.add_trace(
    go.Scatter(
        x=x, y=mean,
        mode="lines+markers",
        marker=dict(symbol="circle", size=6),
        customdata=mean_indxs,
        hovertemplate="x: %{x}<br>y: %{y}<br>latent i: %{customdata}<extra></extra>",
        name="Sorted Feature Mean"
    ),
    row=1, col=3
)
fig.add_trace(
    go.Scatter(
        x=x, y=std_dev,
        mode="lines+markers",
        marker=dict(symbol="circle", size=6),
        customdata=std_indxs,
        hovertemplate="x: %{x}<br>y: %{y}<br>latent i: %{customdata}<extra></extra>",
        name="Sorted Feature Standard Deviation"
    ),
    row=2, col=1
)
fig.add_trace(
    go.Scatter(
        x=x_zizj, y=zizj_and_coords[:, 0],
        mode="lines+markers",
        marker=dict(symbol="circle", size=6),
        customdata=zizj_and_coords[:, 1:3],
        hovertemplate="x: %{x}<br>y: %{y}<br>latent i: %{customdata[0]}<br>latent j: %{customdata[1]}<extra></extra>",
        name="Sorted zizj"
    ),
    row=2, col=2
)

out_path = (
    REPO_ROOT
    / "figures"
    / f"sorted_lineplots_{arch}_layer{layer}_k{sparsity}_{ts}.html"
)
out_path.parent.mkdir(parents=True, exist_ok=True)
fig.update_layout(height=1400, width=1400)
fig.update_xaxes(type="log")
fig.update_yaxes(type="log")
fig.write_html(out_path)  # interactive — keeps your hovers
print(f"Done. Saved files to {out_path}")
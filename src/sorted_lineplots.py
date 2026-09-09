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
REPO_ROOT = Path(__file__).resolve().parent.parent
FIG_DIR = REPO_ROOT / "figures" / "sorted_lineplots"
n_batches = 3823
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

# Normalized zizj = cosine similarity of the latent activation vectors:
#   (Z^T Z)_ij / ( sqrt((Z^T Z)_ii) * sqrt((Z^T Z)_jj) )
gram_norms = torch.diagonal(Gram).clamp(min=0).sqrt()          # sqrt(z_i^T z_i)
denom = gram_norms[rows] * gram_norms[cols]
zizj_norm_values = torch.where(denom > 0, zizj_values / denom,
                               torch.zeros_like(zizj_values))
zizj_norm_and_coords = torch.cat(
    (zizj_norm_values.unsqueeze(1), coords.to(zizj_norm_values.dtype)), 1
)

#order them all
max_z, max_idxs = torch.sort(max_z, descending=True)
min_z, min_indxs = torch.sort(min_z, descending=True)
mean, mean_indxs = torch.sort(mean, descending=True)
std_dev, std_indxs = torch.sort(std_dev, descending=True)

order = torch.argsort(zizj_values, descending=True)
zizj_and_coords = zizj_and_coords[order]

# sort normalized pairs
norm_order = torch.argsort(zizj_norm_values, descending=True)
zizj_norm_and_coords = zizj_norm_and_coords[norm_order]

max_z, max_idxs = max_z.cpu().numpy(), max_idxs.cpu().numpy()
min_z, min_indxs = min_z.cpu().numpy(), min_indxs.cpu().numpy()
mean, mean_indxs = mean.cpu().numpy(), mean_indxs.cpu().numpy()
std_dev, std_indxs = std_dev.cpu().numpy(), std_indxs.cpu().numpy()
zizj_and_coords = zizj_and_coords.cpu().numpy()
zizj_norm_and_coords = zizj_norm_and_coords.cpu().numpy()

x = list(range(1, len(max_z) + 1))
x_zizj_full = list(range(1, len(zizj_and_coords) + 1))
x_zizj_norm_full = list(range(1, len(zizj_norm_and_coords) + 1))

# Pairs to call out as red dots on the zizj panels. Add as many as you like;
# order within a pair does not matter (normalized to i < j to match Gram_tri).
HIGHLIGHT_PAIRS = [
    (6510, 1085),    # S / Short
    (13715, 310),    # A / Apply
    (15497, 355),    # C / Collection
    (5407, 1061),    # E / Explain
    (8987, 14214),   # F / Follow
]
HIGHLIGHT_PAIRS = [(min(a, b), max(a, b)) for a, b in HIGHLIGHT_PAIRS]

# Individual latents for the per-latent stat panels. Defaults to every latent
# that appears in HIGHLIGHT_PAIRS.
HIGHLIGHT_LATENTS = sorted({L for pair in HIGHLIGHT_PAIRS for L in pair})


def _latent_hits(idx_arr, y_vals, latents):
    """Return (xs, ys, labels) for the sorted-rank positions of `latents`."""
    idx_arr = np.asarray(idx_arr)
    xs, ys, labels = [], [], []
    for L in latents: # find latent position than append x and y rank (+1)
        pos = np.where(idx_arr == L)[0]
        if len(pos):
            r = int(pos[0])
            xs.append(r + 1)
            ys.append(y_vals[r])
            labels.append(f"latent {L}")
    return xs, ys, labels


def _pair_hits(coords_arr, y_vals, pairs):
    """Return (xs, ys, labels) for the sorted-rank positions of `pairs`."""
    xs, ys, labels = [], [], []
    for i, j in pairs:
        match = np.where((coords_arr[:, 1] == i) & (coords_arr[:, 2] == j))[0]
        if len(match):
            r = int(match[0])
            xs.append(r + 1)
            ys.append(y_vals[r])
            labels.append(f"({i}, {j})")
    return xs, ys, labels


def _logsample(n, k=4000):
    """Log-spaced row indices; on a log x-axis this looks like all n points."""
    if n <= k:
        return np.arange(n)
    idx = np.unique(np.round(np.logspace(0, np.log10(n), k)).astype(np.int64)) - 1
    return idx[(idx >= 0) & (idx < n)]

# ---- static PNG: every point, all six stats ----
png_path = FIG_DIR / f"sorted_lineplots_all_{arch}_layer{layer}_k{sparsity}_{ts}.png"
png_path.parent.mkdir(parents=True, exist_ok=True)

png_plots = [
    (x, max_z, "Sorted Feature Max"),
    (x, min_z, "Sorted Feature Min"),
    (x, mean, "Sorted Feature Mean"),
    (x, std_dev, "Sorted Feature Std. Dev"),
    (x_zizj_full, zizj_and_coords[:, 0], "Sorted zizj (all pairs)"),
    (x_zizj_norm_full, zizj_norm_and_coords[:, 0], "Sorted normalized zizj (all pairs)"),
]

# Parallel to png_plots: a latent-index array for the per-latent stats, or a
# (value, i, j) coords array for the pair panels.
png_highlight_src = [
    max_idxs, min_indxs, mean_indxs, std_indxs,
    zizj_and_coords, zizj_norm_and_coords,
]

fig_png, axes_png = plt.subplots(2, 3, figsize=(15, 10))
axes_png = axes_png.flatten()
for ax, (px_vals, py_vals, title), src in zip(axes_png, png_plots, png_highlight_src):
    ax.scatter(px_vals, py_vals, s=1, marker=".")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Sorted rank")
    ax.set_ylabel(title)
    ax.set_title(title)
    if src.ndim == 2:                       # zizj / normalized zizj -> per pair
        hx, hy, _ = _pair_hits(src, py_vals, HIGHLIGHT_PAIRS)
    else:                                   # per-latent stat -> per latent
        hx, hy, _ = _latent_hits(src, py_vals, HIGHLIGHT_LATENTS)
    if hx:
        ax.scatter(hx, hy, s=40, c="red", marker="o", zorder=5,
                   edgecolors="black", linewidths=0.5)
fig_png.tight_layout()
fig_png.savefig(png_path, dpi=150)
plt.close(fig_png)
print(f"Saved static PNG to {png_path}")

# ---- interactive HTML ----
import plotly.graph_objects as go
from plotly.subplots import make_subplots

x_zizj_full = np.asarray(x_zizj_full)
x_zizj_norm_full = np.asarray(x_zizj_norm_full)
zizj_keep = _logsample(len(zizj_and_coords))
zizj_norm_keep = _logsample(len(zizj_norm_and_coords))

fig = make_subplots(
    rows=2, cols=3,
    subplot_titles=("Sorted Feature Max", "Sorted Feature Min",
                    "Sorted Feature Mean", "Sorted Feature Std. Dev",
                    "Sorted zizj", "Sorted normalized zizj"),
)

fig.add_trace(
    go.Scatter(x=x, y=max_z, mode="lines+markers",
               marker=dict(symbol="circle", size=6), customdata=max_idxs,
               hovertemplate="x: %{x}<br>y: %{y}<br>latent i: %{customdata}<extra></extra>",
               name="Sorted Feature Max"),
    row=1, col=1,
)
fig.add_trace(
    go.Scatter(x=x, y=min_z, mode="lines+markers",
               marker=dict(symbol="circle", size=6), customdata=min_indxs,
               hovertemplate="x: %{x}<br>y: %{y}<br>latent i: %{customdata}<extra></extra>",
               name="Sorted Feature Min"),
    row=1, col=2,
)
fig.add_trace(
    go.Scatter(x=x, y=mean, mode="lines+markers",
               marker=dict(symbol="circle", size=6), customdata=mean_indxs,
               hovertemplate="x: %{x}<br>y: %{y}<br>latent i: %{customdata}<extra></extra>",
               name="Sorted Feature Mean"),
    row=1, col=3,
)
fig.add_trace(
    go.Scatter(x=x, y=std_dev, mode="lines+markers",
               marker=dict(symbol="circle", size=6), customdata=std_indxs,
               hovertemplate="x: %{x}<br>y: %{y}<br>latent i: %{customdata}<extra></extra>",
               name="Sorted Feature Standard Deviation"),
    row=2, col=1,
)
fig.add_trace(
    go.Scatter(x=x_zizj_full[zizj_keep], y=zizj_and_coords[zizj_keep, 0],
               mode="lines+markers", marker=dict(symbol="circle", size=6),
               customdata=zizj_and_coords[zizj_keep, 1:3],
               hovertemplate="x: %{x}<br>y: %{y}<br>latent i: %{customdata[0]}<br>latent j: %{customdata[1]}<extra></extra>",
               name="Sorted zizj"),
    row=2, col=2,
)
fig.add_trace(
    go.Scatter(x=x_zizj_norm_full[zizj_norm_keep], y=zizj_norm_and_coords[zizj_norm_keep, 0],
               mode="lines+markers", marker=dict(symbol="circle", size=6),
               customdata=zizj_norm_and_coords[zizj_norm_keep, 1:3],
               hovertemplate="x: %{x}<br>y: %{y}<br>latent i: %{customdata[0]}<br>latent j: %{customdata[1]}<extra></extra>",
               name="Sorted normalized zizj"),
    row=2, col=3,
)

# Red dots 
_html_highlights = [
    (1, 1, _latent_hits(max_idxs, max_z, HIGHLIGHT_LATENTS)),
    (1, 2, _latent_hits(min_indxs, min_z, HIGHLIGHT_LATENTS)),
    (1, 3, _latent_hits(mean_indxs, mean, HIGHLIGHT_LATENTS)),
    (2, 1, _latent_hits(std_indxs, std_dev, HIGHLIGHT_LATENTS)),
    (2, 2, _pair_hits(zizj_and_coords, zizj_and_coords[:, 0], HIGHLIGHT_PAIRS)),
    (2, 3, _pair_hits(zizj_norm_and_coords, zizj_norm_and_coords[:, 0], HIGHLIGHT_PAIRS)),
]
for r, c, (hx, hy, labels) in _html_highlights:
    if not hx:
        continue
    fig.add_trace(
        go.Scatter(
            x=hx, y=hy,
            mode="markers",                       # -> "markers+text" for on-plot labels
            marker=dict(symbol="circle", size=12, color="red",
                        line=dict(width=1, color="black")),
            text=labels,
            hovertemplate="x: %{x}<br>y: %{y}<br>%{text}<extra></extra>",
            name="highlight", legendgroup="highlight",
            showlegend=(r == 1 and c == 1),
        ),
        row=r, col=c,
    )

for label, coords_arr in (("zizj", zizj_and_coords),
                          ("normalized zizj", zizj_norm_and_coords)):
    missing = [pr for pr in HIGHLIGHT_PAIRS
               if not len(np.where((coords_arr[:, 1] == pr[0]) & (coords_arr[:, 2] == pr[1]))[0])]
    if missing:
        print(f"Note: pairs {missing} not present in {label} coords; no dot drawn.")

out_path = FIG_DIR / f"sorted_lineplots_{arch}_layer{layer}_k{sparsity}_{ts}.html"
out_path.parent.mkdir(parents=True, exist_ok=True)
fig.update_layout(height=1400, width=2000)
fig.update_xaxes(type="log", title_text="Sorted rank")
fig.update_yaxes(type="log")
fig.write_html(out_path)  # interactive — keeps your hovers
print(f"Done. Saved files to {out_path}")
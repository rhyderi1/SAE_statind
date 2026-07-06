'''
z_i vs z_j scatter, raw and standardized, for a handful of chosen SAE latent pairs.

We drop ONLY the (0,0) point at the origin (both latents inactive): it would be a
single dot with a huge number of overlapping tokens on it. 

Each pair gets two panels: RAW and STD (each latent centered/scaled using the mean
and std computed over the joint support -- tokens where BOTH latents are active).
'''

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
from pathlib import Path
from datetime import datetime
from infer_z import SAE_DATA
from sae_lens import SAE
from transformer_lens import HookedTransformer
from sae_lens import ActivationsStore

device = "cuda" if torch.cuda.is_available() else "cpu"
dataset = 'NeelNanda/pile-10k'
context_size = 128
batch_size = 32
n_batches = 1209

ARCH = "relu"
LAYER = 12
SPARSITY = "20"
PAIRS = [
    (477, 10069), #Absorbed
    (1628,2969) #Non-absorbed
]
MARKER_SIZE = 2.5
ALPHA = 0.3

hook_name = f'blocks.{LAYER}.hook_resid_post'
release, sae_id = SAE_DATA[LAYER][ARCH][SPARSITY]
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


# 1) Stream batches

wanted_latents = []
for (i, j) in PAIRS:
    if i not in wanted_latents:
        wanted_latents.append(i)
    if j not in wanted_latents:
        wanted_latents.append(j)

batches_per_latent = {latent: [] for latent in wanted_latents}

with torch.no_grad():
    for batch_num in range(n_batches):
        batch_tokens = activation_store.get_batch_tokens(batch_size)

        _, cache = model.run_with_cache(
            batch_tokens,
            names_filter=hook_name,
            stop_at_layer=LAYER + 1,
            prepend_bos=False,
        )
        X = cache[hook_name]
        del cache

        X_sae = X.to(device=sae.W_enc.device, dtype=sae.W_enc.dtype)
        Z = sae.encode(X_sae)
        Z = Z.reshape(-1, Z.shape[-1])  # (tokens, d_sae)

        for latent in wanted_latents:
            # .clone() so we keep a small (tokens,) column, not the entire Z matrix
            batches_per_latent[latent].append(Z[:, latent].detach().clone().cpu())
        del Z

columns = {}
for latent in wanted_latents:
    columns[latent] = torch.cat(batches_per_latent[latent]).float().to(device)


#2) Compute joint-support means/stds and on-support Pearson r, then standardize

def joint_support_stats(zi, zj):
    """
    Joint-support means/stds and on-support Pearson r, all computed over the
    joint support (tokens where BOTH latents are active). Fully vectorized.
    Returns (mean_i, std_i, mean_j, std_j) as tensors, plus (r, n).
    """
    m = (zi > 0) & (zj > 0)
    n = int(m.sum().item())

    zi_m, zj_m = zi[m], zj[m]
    mean_i, mean_j = zi_m.mean(), zj_m.mean()
    di, dj = zi_m - mean_i, zj_m - mean_j

    std_i = di.square().mean().sqrt().clamp_min(1e-8)   
    std_j = dj.square().mean().sqrt().clamp_min(1e-8)


    #compute R value
    denom = di.square().sum().sqrt() * dj.square().sum().sqrt()
    r = float((di * dj).sum() / denom) if denom > 0 else float("nan")

    return mean_i, std_i, mean_j, std_j, r, n


def standardize_all(zi, zj, mean_i, std_i, mean_j, std_j):

    return (zi - mean_i) / std_i, (zj - mean_j) / std_j


# 3) Plot: one row per pair, RAW panel and STD panel side by side

n_pairs = len(PAIRS)
fig, axes = plt.subplots(n_pairs, 2, figsize=(7.2, 3.4 * n_pairs), squeeze=False)

for row in range(n_pairs):
    i, j = PAIRS[row]
    zi = columns[i]
    zj = columns[j]

    mean_i, std_i, mean_j, std_j, r, coact_count = joint_support_stats(zi, zj)
    zi_std, zj_std = standardize_all(zi, zj, mean_i, std_i, mean_j, std_j)

    # Drop ONLY the exact (0,0) origin; keep every other point. 
    keep = (zi != 0) | (zj != 0)
    n_plotted = int(keep.sum().item())
    total_tokens = int(zi.numel())
    print(
        f"pair (z_{i}, z_{j}): plotted {n_plotted} points "
        f"(co-active={coact_count}, total tokens={total_tokens})"
    )

    raw_x = zi[keep].cpu().numpy()
    raw_y = zj[keep].cpu().numpy()
    ax_raw = axes[row][0]
    ax_raw.scatter(raw_x, raw_y, s=MARKER_SIZE, alpha=ALPHA, linewidths=0)
    ax_raw.axhline(0, lw=0.6, color="0.55")
    ax_raw.axvline(0, lw=0.6, color="0.55")
    ax_raw.set_xlabel(f"z_{i}")
    ax_raw.set_ylabel(f"z_{j}")
    ax_raw.set_title(f"z_{i} vs z_{j} -- RAW (co-act={coact_count}, on-supp r={r:+.2f})", fontsize=9)

    std_x = zi_std[keep].cpu().numpy()
    std_y = zj_std[keep].cpu().numpy()
    ax_std = axes[row][1]
    ax_std.scatter(std_x, std_y, s=MARKER_SIZE, alpha=ALPHA, linewidths=0)
    ax_std.axhline(0, lw=0.6, color="0.55")
    ax_std.axvline(0, lw=0.6, color="0.55")
    ax_std.set_xlabel(f"z_{i} (std)")
    ax_std.set_ylabel(f"z_{j} (std)")
    ax_std.set_title(f"z_{i} vs z_{j} -- STD (joint support)", fontsize=9)

fig.suptitle(f"z_i vs z_j, raw and standardized (Arch={ARCH}, layer={LAYER}, k={SPARSITY})", y=1.0, fontsize=11)
fig.tight_layout()

ts = datetime.now().strftime("%Y%m%d_%H%M%S")
out_path = (
    Path(__file__).parent.parent
    / "figures"
    / f"zizj_scatter_{ARCH}_layer{LAYER}_k{SPARSITY}_{ts}.png"
)
out_path.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out_path, dpi=160, bbox_inches="tight")
plt.close(fig)

print(f"Saved -> {out_path}")

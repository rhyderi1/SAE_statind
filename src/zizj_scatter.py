'''
z_i vs z_j scatter, raw and standardized, for a handful of chosen SAE latent pairs.

Because SAE codes are sparse, most tokens leave one or both latents at exactly 0,
so each panel splits into three visible parts:
    z_i > 0, z_j = 0   -> a strip sitting exactly on the x-axis
    z_i = 0, z_j > 0   -> a strip sitting exactly on the y-axis
    z_i > 0, z_j > 0   -> the off-axis "blob" -- the only region where a
                          correlation between the two latents is even defined
We drop the both-zero point at the origin: it would be a single dot with millions
of overlapping tokens on it, and would visually swamp everything else.

Each pair gets two panels: RAW (native activation units) and STD (each latent
centered/scaled using the mean and std computed ONLY over the jointly-active
tokens -- z_i>0 AND z_j>0). 

We use the joint support, not each latent's own support, because the joint-support standard 
deviation is the one that makes the STD panel's tilt equal the actual on-support correlation; standardizing each
latent on its own support would inflate the variance with tokens where the other
latent is off, distorting the blob's apparent shape.
'''

import matplotlib
matplotlib.use('Agg')  # Must precede pyplot import
import matplotlib.pyplot as plt

import torch
from pathlib import Path
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

# Each tuple is one (latent_i, latent_j) pair to scatter against each other.
PAIRS = [
    (477, 10069),
    (1628,2969)
]

MARKER_SIZE = 2.5
ALPHA = 0.3
MAX_POINTS_PER_PANEL = 60_000  # keep the plot fast/legible; excess points are thinned out, not chosen at random

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


# ----------------------------------------------------------------------------
# 1) Stream batches, keep only the columns for latents we actually plot
# ----------------------------------------------------------------------------

# Every latent index that appears in any pair, with duplicates removed. These
# are the only SAE columns we ever pull out of Z, so memory stays small (a
# handful of columns x every token) instead of the full (tokens x n_features)
# matrix.

wanted_latents = []
for (i, j) in PAIRS:
    if i not in wanted_latents:
        wanted_latents.append(i)
    if j not in wanted_latents:
        wanted_latents.append(j)

# One accumulator list per wanted latent, keyed by latent index.
batches_per_latent = {}
for latent in wanted_latents:
    batches_per_latent[latent] = []

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

        X_sae = X.to(device=sae.device, dtype=sae.W_enc.dtype)
        Z = sae.encode(X_sae)
        Z = Z.reshape(-1, Z.shape[-1])  # shape (tokens, features)

        for latent in wanted_latents:
            batches_per_latent[latent].append(Z[:, latent].cpu())

# Stitch each latent's per-batch pieces into one plain Python list covering
# every token, and keep working in plain lists (not torch/numpy) from here on.
columns = {}
for latent in wanted_latents:
    columns[latent] = torch.cat(batches_per_latent[latent]).float().tolist()


# ----------------------------------------------------------------------------
# 2) Basic-loop statistics: joint-support mean/std and on-support correlation
# ----------------------------------------------------------------------------

def joint_support_mean_and_std(zi, zj):
    """
    Mean and std of zi and zj, computed only over tokens where BOTH are active
    (> 0) -- the "joint support". Written as a plain loop over every token
    instead of boolean-mask array indexing, so it reads like a running total.
    """
    sum_i = 0.0
    sum_j = 0.0
    count = 0
    for k in range(len(zi)):
        if zi[k] > 0 and zj[k] > 0:
            sum_i += zi[k]
            sum_j += zj[k]
            count += 1

    if count < 2:
        return 0.0, 1.0, 0.0, 1.0

    mean_i = sum_i / count
    mean_j = sum_j / count

    sq_diff_i = 0.0
    sq_diff_j = 0.0
    for k in range(len(zi)):
        if zi[k] > 0 and zj[k] > 0:
            sq_diff_i += (zi[k] - mean_i) ** 2
            sq_diff_j += (zj[k] - mean_j) ** 2
    std_i = (sq_diff_i / count) ** 0.5
    std_j = (sq_diff_j / count) ** 0.5

    if std_i == 0:
        std_i = 1e-8
    if std_j == 0:
        std_j = 1e-8

    return mean_i, std_i, mean_j, std_j


def standardize(zi, zj, mean_i, std_i, mean_j, std_j):
    """
    Center/scale zi and zj using the joint-support mean/std passed in, but only
    for entries where that particular latent is itself active. Inactive (zero)
    entries are left at exactly 0 -- otherwise subtracting the mean would move
    the axis strips off their axis, and the strip/blob shape we care about
    would be lost.
    """
    zi_std = []
    for value in zi:
        if value > 0:
            zi_std.append((value - mean_i) / std_i)
        else:
            zi_std.append(0.0)

    zj_std = []
    for value in zj:
        if value > 0:
            zj_std.append((value - mean_j) / std_j)
        else:
            zj_std.append(0.0)

    return zi_std, zj_std


def on_support_correlation(zi, zj):
    """Pearson correlation of zi and zj, computed only over jointly-active tokens."""
    paired_i = []
    paired_j = []
    for k in range(len(zi)):
        if zi[k] > 0 and zj[k] > 0:
            paired_i.append(zi[k])
            paired_j.append(zj[k])

    n = len(paired_i)
    if n < 2:
        return float("nan"), n

    mean_i = sum(paired_i) / n
    mean_j = sum(paired_j) / n

    numerator = 0.0
    sum_sq_i = 0.0
    sum_sq_j = 0.0
    for k in range(n):
        di = paired_i[k] - mean_i
        dj = paired_j[k] - mean_j
        numerator += di * dj
        sum_sq_i += di * di
        sum_sq_j += dj * dj

    denominator = (sum_sq_i ** 0.5) * (sum_sq_j ** 0.5)
    if denominator == 0:
        return float("nan"), n

    return numerator / denominator, n


def drop_both_zero_and_thin(zi, zj, max_points):
    """
    Keep only tokens where at least one of zi, zj is active (drops the
    uninformative both-zero pile at the origin), then thin the remainder down
    to at most max_points by keeping every Nth point. Plain step-slicing
    instead of numpy's random-choice subsampling -- simpler, though it means
    the kept points are evenly spaced through the stream rather than a random
    sample of it.
    """
    kept_i = []
    kept_j = []
    for k in range(len(zi)):
        if zi[k] > 0 or zj[k] > 0:
            kept_i.append(zi[k])
            kept_j.append(zj[k])

    if len(kept_i) <= max_points:
        return kept_i, kept_j

    step = len(kept_i) // max_points
    return kept_i[::step], kept_j[::step]


# ----------------------------------------------------------------------------
# 3) Plot: one row per pair, RAW panel and STD panel side by side
# ----------------------------------------------------------------------------

n_pairs = len(PAIRS)
fig, axes = plt.subplots(n_pairs, 2, figsize=(7.2, 3.4 * n_pairs), squeeze=False)

for row in range(n_pairs):
    i, j = PAIRS[row]
    zi = columns[i]
    zj = columns[j]

    r, coact_count = on_support_correlation(zi, zj)
    mean_i, std_i, mean_j, std_j = joint_support_mean_and_std(zi, zj)
    zi_std, zj_std = standardize(zi, zj, mean_i, std_i, mean_j, std_j)

    raw_x, raw_y = drop_both_zero_and_thin(zi, zj, MAX_POINTS_PER_PANEL)
    ax_raw = axes[row][0]
    ax_raw.scatter(raw_x, raw_y, s=MARKER_SIZE, alpha=ALPHA, linewidths=0)
    ax_raw.axhline(0, lw=0.6, color="0.55")
    ax_raw.axvline(0, lw=0.6, color="0.55")
    ax_raw.set_xlabel(f"z_{i}")
    ax_raw.set_ylabel(f"z_{j}")
    ax_raw.set_title(f"z_{i} vs z_{j} -- RAW (co-act={coact_count}, on-supp r={r:+.2f})", fontsize=9)

    std_x, std_y = drop_both_zero_and_thin(zi_std, zj_std, MAX_POINTS_PER_PANEL)
    ax_std = axes[row][1]
    ax_std.scatter(std_x, std_y, s=MARKER_SIZE, alpha=ALPHA, linewidths=0)
    ax_std.axhline(0, lw=0.6, color="0.55")
    ax_std.axvline(0, lw=0.6, color="0.55")
    ax_std.set_xlabel(f"z_{i} (std)")
    ax_std.set_ylabel(f"z_{j} (std)")
    ax_std.set_title(f"z_{i} vs z_{j} -- STD (joint support)", fontsize=9)

fig.suptitle(f"z_i vs z_j, raw and standardized (Arch={ARCH}, layer={LAYER}, k={SPARSITY})", y=1.0, fontsize=11)
fig.tight_layout()

out_path = (
    Path(__file__).parent.parent
    / "figures"
    / f"zizj_scatter_{ARCH}_layer{LAYER}_k{SPARSITY}.png"
)
out_path.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out_path, dpi=160, bbox_inches="tight")
plt.close(fig)

print(f"Saved -> {out_path}")

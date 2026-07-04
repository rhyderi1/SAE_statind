import os
import resource
import torch
from infer_z import SAE_DATA
from sae_lens import SAE
from transformer_lens import HookedTransformer
from sae_lens import ActivationsStore

device = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", device)

dataset = 'NeelNanda/pile-10k'
context_size = 128
batch_size = 32
n_batches = 250  # enough to see growth rate; full run already confirmed flat at 8 CPUs

ARCH = "relu"
LAYER = 12
SPARSITY = "20"
PAIRS = [(477, 10069), (1628, 2969)]

hook_name = f'blocks.{LAYER}.hook_resid_post'
release, sae_id = SAE_DATA[LAYER][ARCH][SPARSITY]

sae = SAE.from_pretrained(release=release, sae_id=sae_id, device=device)
model = HookedTransformer.from_pretrained_no_processing(
    default_prepend_bos=True, model_name="gemma-2-2b", device=device
)
activation_store = ActivationsStore.from_sae(model, sae, context_size=context_size, dataset=dataset)

wanted_latents = []
for (i, j) in PAIRS:
    if i not in wanted_latents:
        wanted_latents.append(i)
    if j not in wanted_latents:
        wanted_latents.append(j)

batches_per_latent = {}
for latent in wanted_latents:
    batches_per_latent[latent] = []


def rss_mb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024  # KB->MB on Linux


print(f"before loop  maxrss_mb={rss_mb():.1f}")

with torch.no_grad():
    for batch_num in range(n_batches):
        batch_tokens = activation_store.get_batch_tokens(batch_size)

        _, cache = model.run_with_cache(
            batch_tokens, names_filter=hook_name, stop_at_layer=LAYER + 1, prepend_bos=False,
        )
        X = cache[hook_name]
        del cache

        X_sae = X.to(device=sae.device, dtype=sae.W_enc.dtype)
        Z = sae.encode(X_sae)
        Z = Z.reshape(-1, Z.shape[-1])

        for latent in wanted_latents:
            batches_per_latent[latent].append(Z[:, latent].cpu())

        if batch_num % 25 == 0:
            print(f"batch {batch_num:4d}  maxrss_mb={rss_mb():.1f}")

print(f"after loop, before cat  maxrss_mb={rss_mb():.1f}")

columns = {}
for latent in wanted_latents:
    columns[latent] = torch.cat(batches_per_latent[latent]).float().tolist()

print(f"after cat/tolist  maxrss_mb={rss_mb():.1f}  (tokens per column: {len(columns[wanted_latents[0]])})")


def joint_support_mean_and_std(zi, zj):
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


MAX_POINTS_PER_PANEL = 60_000

for row, (i, j) in enumerate(PAIRS):
    zi = columns[i]
    zj = columns[j]
    print(f"pair {row} ({i},{j})  before analysis  maxrss_mb={rss_mb():.1f}")

    r, coact_count = on_support_correlation(zi, zj)
    print(f"pair {row}  after on_support_correlation (coact={coact_count})  maxrss_mb={rss_mb():.1f}")

    mean_i, std_i, mean_j, std_j = joint_support_mean_and_std(zi, zj)
    print(f"pair {row}  after joint_support_mean_and_std  maxrss_mb={rss_mb():.1f}")

    zi_std, zj_std = standardize(zi, zj, mean_i, std_i, mean_j, std_j)
    print(f"pair {row}  after standardize  maxrss_mb={rss_mb():.1f}")

    raw_x, raw_y = drop_both_zero_and_thin(zi, zj, MAX_POINTS_PER_PANEL)
    print(f"pair {row}  after drop_both_zero_and_thin(raw) kept={len(raw_x)}  maxrss_mb={rss_mb():.1f}")

    std_x, std_y = drop_both_zero_and_thin(zi_std, zj_std, MAX_POINTS_PER_PANEL)
    print(f"pair {row}  after drop_both_zero_and_thin(std) kept={len(std_x)}  maxrss_mb={rss_mb():.1f}")

print(f"FINAL  maxrss_mb={rss_mb():.1f}")

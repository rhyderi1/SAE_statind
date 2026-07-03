'''
For each bin in the histogram in z_histogram.py, find the latents contributing most to it (ie: have the most entries in that range)
'''
import json
from pathlib import Path

import torch
import numpy as np
from infer_z import SAE_DATA
from sae_lens import SAE
from transformer_lens import HookedTransformer
from sae_lens import ActivationsStore

device = "cuda" if torch.cuda.is_available() else "cpu"

dataset = 'NeelNanda/pile-10k'
context_size = 128
batch_size = 32
hook_name = 'blocks.12.hook_resid_post'
release, sae_id = SAE_DATA[12]["relu"]["20"]
n_batches = 1209
layer = 12
num_bins = 100
top_k = 10

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

n_features = sae.W_dec.shape[0]

Z_act_all = []
feat_idx_all = []
num_entries = 0

with torch.no_grad():
    for i in range(n_batches):
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
        Z = Z.reshape(-1, Z.shape[-1])  # shape (tokens, features)

        nz_mask = Z != 0
        # z_histogram.py does Z[Z != 0], which flattens and discards which
        # feature column each surviving value came from. nonzero() instead
        # gives us (row, col) pairs so we can keep feat_idx aligned with Z_act.
        row_idx, feat_idx = nz_mask.nonzero(as_tuple=True)
        Z_act = Z[row_idx, feat_idx]

        Z_act_all.append(Z_act.cpu())
        feat_idx_all.append(feat_idx.cpu())
        num_entries += Z.numel()

Z_final = torch.cat(Z_act_all).float().numpy()
feat_idx_final = torch.cat(feat_idx_all).numpy()
num_zeros = num_entries - len(Z_final)

max_val = Z_final.max() if len(Z_final) > 0 else 1.0
# Force the range to start at 0.0 (matches the same fix in z_histogram.py) so
# bin 0 actually covers 0.0 instead of Z_final.min() (some small positive
# value), which is where the num_zeros / zero_count_per_feature entries below
# get added. Without this, bin 0's reported range wouldn't contain the zero
# values being counted in it.
counts, bin_edges = np.histogram(Z_final, bins=num_bins, range=(0.0, max_val))
# z_histogram.py's plotted histogram is over ALL entries, not just the
# active ones: it computes bins from the nonzero values, then stuffs the
# zero count into bin 0 with `counts[0] += num_zeros`. Match that here so
# bin_total_count reflects the same "all entries" histogram being analyzed.
counts = counts.copy()
counts[0] += num_zeros

# Reusing the exact bin_edges from the histogram above (rather than
# recomputing bins) guarantees these results line up 1:1 with the plot
# produced by z_histogram.py. feature_edges gives each latent its own
# width-1 bin so histogram2d effectively tallies a 2D (value, latent)
# co-occurrence table in one vectorized pass instead of looping per bin.
feature_edges = np.arange(n_features + 1)
joint_counts, _, _ = np.histogram2d(Z_final, feat_idx_final, bins=[bin_edges, feature_edges])

# Per-latent zero counts, so bin 0 can get the same "all entries" treatment
# as `counts` above, but broken out by latent instead of lumped together.
# Latents fire at very different rates, so this isn't a uniform split: a
# latent that's almost always dead will dominate bin 0's top contributors.
nonzero_count_per_feature = joint_counts.sum(axis=0)
total_tokens = num_entries // n_features
zero_count_per_feature = total_tokens - nonzero_count_per_feature
joint_counts[0] += zero_count_per_feature

results = []
for b in range(num_bins):
    row = joint_counts[b]  # counts[f] = how many of latent f's entries landed in bin b
    top_latents = np.argsort(row)[::-1][:top_k]  # latent indices, most entries first
    results.append({
        "bin_index": b,
        "bin_range": [float(bin_edges[b]), float(bin_edges[b + 1])],
        "bin_total_count": int(counts[b]),
        "top_latents": [int(f) for f in top_latents],
        "top_latent_counts": [int(row[f]) for f in top_latents],
    })

out_path = Path(__file__).parent.parent / "results" / "z_hist_latent_contributors.json"
out_path.parent.mkdir(parents=True, exist_ok=True)
with open(out_path, "w") as f:
    json.dump({
        "layer": layer,
        "hook_name": hook_name,
        "release": release,
        "sae_id": sae_id,
        "num_bins": num_bins,
        "top_k": top_k,
        "num_zero_entries_in_bin0": int(num_zeros),
        "bins": results,
    }, f, indent=2)

print(f"Saved -> {out_path}")
for r in results:
    lo, hi = r["bin_range"]
    print(f"bin {r['bin_index']:3d} [{lo:.4f}, {hi:.4f}) total={r['bin_total_count']}  "
          f"top_latents={r['top_latents']}  counts={r['top_latent_counts']}")

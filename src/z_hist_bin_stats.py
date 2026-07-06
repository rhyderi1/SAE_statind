'''
For each bin in the histogram in z_histogram.py, find the latents contributing most to it (ie: have the most entries in that range)

Memory-bounded two-pass rewrite: the original version accumulated every single nonzero
SAE activation (value + feature index) across the whole ~5M-token dataset in RAM before
binning. That grows without bound as batches are processed and OOM-killed the job even
at 128G. Histogram counts are additive over any partition of the data, so instead this
runs the model over the dataset twice and never keeps more than one batch's worth of
data resident:
  pass 1 - track the running max activation only (needed to fix bin edges at (0, max),
           matching z_histogram.py's range) and the total entry count.
  pass 2 - bin each batch immediately into small, fixed-size running totals (`counts`,
           sized num_bins; `joint_counts`, sized num_bins x n_features) and discard the
           batch's raw values right after.
The activation stream has no shuffling configured (only `shuffle_input_dataset` would
add that, and it's never called here), so a fresh ActivationsStore replays the exact
same sequence of batches in both passes -- the result is identical to the original
single-pass computation, just computed without ever holding the full dataset in memory.
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

n_features = sae.W_dec.shape[0]


def encode_batch(activation_store):
    """Run one batch through the model + SAE, returning (tokens, features) activations."""
    batch_tokens = activation_store.get_batch_tokens(batch_size)

    _, cache = model.run_with_cache(
        batch_tokens,
        names_filter=hook_name,
        stop_at_layer=layer + 1,
        prepend_bos=False,
    )
    X = cache[hook_name]
    del cache

    X_sae = X.to(device=sae.device, dtype=sae.W_enc.dtype)
    Z = sae.encode(X_sae)
    return Z.reshape(-1, Z.shape[-1]).float()  # shape (tokens, features)


# --- pass 1: find the true max activation and total entry count ---
# Only running scalars are kept here, so memory doesn't grow with n_batches.
max_val = 0.0
num_entries = 0

activation_store = ActivationsStore.from_sae(
    model, sae, context_size=context_size, dataset=dataset,
)
with torch.no_grad():
    for i in range(n_batches):
        Z = encode_batch(activation_store)
        num_entries += Z.numel()
        batch_max = Z.max().item()
        if batch_max > max_val:
            max_val = batch_max

# if max_val == 0.0:
#     max_val = 1.0
max_val = np.float32(max_val)

_, bin_edges = np.histogram(np.array([], dtype=np.float32), bins=num_bins, range=(0.0, max_val))
feature_edges = np.arange(n_features + 1)

# --- pass 2: bin each batch immediately, discard its raw values right after ---
counts = np.zeros(num_bins, dtype=np.int64)
joint_counts = np.zeros((num_bins, n_features))
num_nonzero = 0

activation_store = ActivationsStore.from_sae(
    model, sae, context_size=context_size, dataset=dataset,
)
with torch.no_grad():
    for i in range(n_batches):
        Z = encode_batch(activation_store)

        nz_mask = Z != 0
        row_idx, feat_idx = nz_mask.nonzero(as_tuple=True)
        Z_act = Z[row_idx, feat_idx].cpu().numpy()
        feat_idx = feat_idx.cpu().numpy()

        num_nonzero += len(Z_act)
        batch_counts, _ = np.histogram(Z_act, bins=bin_edges)
        counts += batch_counts
        batch_joint, _, _ = np.histogram2d(Z_act, feat_idx, bins=[bin_edges, feature_edges])
        joint_counts += batch_joint

num_zeros = num_entries - num_nonzero
counts[0] += num_zeros

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

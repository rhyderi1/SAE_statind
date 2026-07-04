'''
BOS-candidate check.

Copy of z_histogram_latent_stats.py with the per-feature firing count
(`num_nonzeros`) turned back on, so we can GROUND the "BOS feature" story with
measurements instead of inferring it from the aggregate histogram.

For every SAE latent we accumulate, in a single streaming pass and with no
per-token storage:
    num_nonzeros       -- how many tokens the latent fires on
    value_sum          -- sum of its activations (-> active mean = firing magnitude)
    max                -- its largest activation (-> which histogram bin it lands in)

A BOS feature should fire ~once per sequence:
    once_per_sequence = n_batches * batch_size = 1209 * 32 = 38,688
so its `num_nonzeros` should sit right at that value. This script prints the
latents whose count matches, plus the specific candidates we hypothesised from
the histogram spikes, with the exact numbers behind each claim.

NOTE: matching count == once-per-sequence confirms "fires once per sequence at
high magnitude" -- it does NOT by itself prove the firing token is position 0.
That still needs the position-index check. This closes the gap in section (2)
(are the counts really ~38,688?) and (3a) (which latent owns which bin), not the
BOS-position claim itself.
'''

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

# The latents we hypothesised are BOS features from the histogram spikes.
HYPOTHESISED_BOS = [1611, 3547, 4090, 6451, 7677, 7983, 8647, 12615, 15284]

sae = SAE.from_pretrained(release=release, sae_id=sae_id, device=device)

model = HookedTransformer.from_pretrained_no_processing(
    default_prepend_bos=True,
    model_name="gemma-2-2b",
    device=device,
)

activation_store = ActivationsStore.from_sae(
    model,
    sae,
    context_size=context_size,
    dataset=dataset,
)

d_sae = sae.W_dec.shape[0]
value_sum_per_feature = torch.zeros(d_sae, dtype=torch.float32, device=device)
max_per_feature       = torch.zeros(d_sae, dtype=torch.float32, device=device)
num_nonzeros          = torch.zeros(d_sae, dtype=torch.float32, device=device)  # (1) re-enabled
num_entries_total = 0
num_sequences_total = 0

with torch.no_grad():
    for i in range(n_batches):
        batch_tokens = activation_store.get_batch_tokens(batch_size)
        num_sequences_total += batch_tokens.shape[0]

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
        Z = Z.reshape(-1, Z.shape[-1])  # (tokens, d_sae)

        value_sum_per_feature += Z.sum(dim=0)
        max_per_feature = torch.maximum(max_per_feature, Z.max(dim=0).values)
        num_nonzeros += (Z != 0).sum(dim=0)  # (1) how many tokens each latent fires on
        num_entries_total += Z.shape[0]

# ---------------------------------------------------------------------------
# Move the small per-feature vectors to numpy for reporting.
# ---------------------------------------------------------------------------
num_nonzeros_np = num_nonzeros.cpu().numpy()
max_np          = max_per_feature.cpu().numpy()                                   # (2)
# Average firing magnitude = sum over active tokens / number of active tokens.
# For a near-constant-magnitude spike this equals the activation value that
# decides which histogram bin the latent's counts pile into.
active_mean_np  = (value_sum_per_feature / num_nonzeros.clamp(min=1)).cpu().numpy()

once_per_sequence = num_sequences_total  # = n_batches * batch_size for full batches
global_max = float(max_np.max())
n_bins = 100  # must match z_histogram.py so bin indices are comparable
bin_width = global_max / n_bins


def bin_index(value):
    """Which of z_histogram.py's 100 bins (range 0..global_max) a value lands in."""
    if bin_width == 0:
        return 0
    return min(int(value / bin_width), n_bins - 1)


def print_row(latent):
    count = num_nonzeros_np[latent]
    fires_per_seq = count / once_per_sequence
    amean = active_mean_np[latent]
    amax = max_np[latent]
    print(
        f"  {latent:>6d}  count={int(count):>7d}  "
        f"fires/seq={fires_per_seq:5.2f}  "
        f"active_mean={amean:8.2f} (bin {bin_index(amean):>2d})  "
        f"max={amax:8.2f} (bin {bin_index(amax):>2d})"
    )


print("=" * 78)
print(f"num_sequences_total (once-per-sequence target) = {once_per_sequence}")
print(f"num_entries_total (tokens)                      = {num_entries_total}")
print(f"context_size (tokens/seq)                       = {num_entries_total // once_per_sequence}")
print(f"global max activation                           = {global_max:.2f}")
print(f"histogram bin width (0..max / {n_bins})              = {bin_width:.2f}")
print("=" * 78)

# --- Hypothesised candidates: show the exact measured numbers behind the claim
print("\nHypothesised BOS latents (from histogram spikes) -- measured values:")
print(f"  {'latent':>6}  {'count':>13}  {'fires/seq':>10}  {'active_mean(bin)':>18}  {'max(bin)':>14}")
for latent in HYPOTHESISED_BOS:
    print_row(latent)

# --- Data-driven: every latent that actually fires ~once per sequence.
# Band chosen loosely so we also catch BOS latents whose magnitude was too
# small to spike visibly in the aggregate histogram (the point of section 3).
lo, hi = 0.97 * once_per_sequence, 1.03 * once_per_sequence
band = np.where((num_nonzeros_np >= lo) & (num_nonzeros_np <= hi))[0]
band = band[np.argsort(-num_nonzeros_np[band])]
print(f"\nAll latents firing within +-3% of once-per-sequence "
      f"([{int(lo)}, {int(hi)}]): {len(band)} found")
for latent in band:
    print_row(int(latent))

# --- Also the raw top-40 by firing count, to see where the once-per-seq band
# sits relative to the genuinely high-frequency (content) features.
print("\nTop 40 latents by firing count (for context):")
top = np.argsort(-num_nonzeros_np)[:40]
for latent in top:
    print_row(int(latent))

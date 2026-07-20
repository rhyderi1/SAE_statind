'''
Replicate figure 6a from 'A is for Absorption' by Chanin et al.

PSEUDOCODE:
- Process input (Pile 10K) into X into Z
- Before processing, filter/save indices of words you would like
- Find activations by indexing the mxp matrix
'''
import argparse
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
from pathlib import Path
from datetime import datetime
from sae_lens import SAE
from transformer_lens import HookedTransformer
from sae_lens import ActivationsStore
from infer_z import SAE_DATA
import numpy as np
from collections import defaultdict
import random



REPO_ROOT = Path(__file__).parent.parent

device = "cuda" if torch.cuda.is_available() else "cpu"
dataset = 'NeelNanda/pile-10k'
context_size = 128
batch_size = 32
Arch = "jumprelu"
Sparsity = "59"
Layer = 3
num_batches = 1209
hook_name = f"blocks.{Layer}.hook_resid_post"

release, sae_id = SAE_DATA[Layer][Arch][Sparsity]


Latent1 = 6510
Latent2 = 1085

sae = SAE.from_pretrained(
    release=release,
    sae_id=sae_id,
    device = device
)

# Two random control latents
# Seeded --> the same two controls are drawn on every run (reproducible figure).
SEED = 0
rng = random.Random(SEED)
d_sae = sae.W_enc.shape[1]
Latent3, Latent4 = rng.sample(
    [i for i in range(d_sae) if i not in (Latent1, Latent2)], 2
)
print(f"Random control latents (seed={SEED}, d_sae={d_sae}): {Latent3}, {Latent4}")

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
tokenizer = activation_store.model.tokenizer

target_words = ["snake",  "steam", "short",  "soccer", "sax"]
#note to self: soggy tokenizes to ['so', 'ggy'] so we need to implement a sliding window, later
target_token_ids = []
token_word_map = {}

for word in target_words:
    word_set = [word, " " + word]
    for w in word_set:
        target_token = tokenizer.encode(w, add_special_tokens=False) #gives tokenized version of the word, without bos, eos tokens
        if len(target_token) == 1:
            target_token_ids.append(target_token[0]) #take [0] bc tokenizer always returns a list
            token_word_map[target_token[0]] = word
            print(f"Word: {w}, Token: {target_token[0]}")

target_token_ids = torch.tensor(target_token_ids, device = device)

Z_act_list = []

with torch.no_grad():
    for batch in range (num_batches):
        batch_tokens = activation_store.get_batch_tokens(batch_size).to(device) # Shape: (batch_size, context_size

        _, cache = model.run_with_cache(
            batch_tokens,
            names_filter=hook_name,
            stop_at_layer=Layer + 1,
            prepend_bos=False,
        )

        X = cache[hook_name]
        del cache

        X_sae = X.to(device=sae.W_enc.device, dtype=sae.W_enc.dtype)
        Z = sae.encode(X_sae) # Shape: (batch_size, context_size, d_sae)

        # Check if tokens in the current batch contain any of the target tokens, then find positions
        check_batch_tokens = torch.isin(batch_tokens, target_token_ids)
        positions = torch.nonzero(check_batch_tokens, as_tuple=False)

        for pos in positions:
            batch_idx, seq_idx = pos[0].item(), pos[1].item()
            Z_act = {
                "word": token_word_map.get(batch_tokens[batch_idx, seq_idx].item(), "Unknown"),
                "coordinates": (batch_idx, seq_idx),
                "activations":{Latent1: Z[batch_idx, seq_idx, Latent1].item(),
                                Latent2: Z[batch_idx, seq_idx, Latent2].item(),
                                Latent3: Z[batch_idx, seq_idx, Latent3].item(),
                                Latent4: Z[batch_idx, seq_idx, Latent4].item()
                                }
            }
            Z_act_list.append(Z_act)


print(Z_act_list)

# aggregate  per-word activations

# Z_act_list is one dict per *occurrence*: collapse those occurrences down to a single number per latent.
#
# each word has two lists of raw activations (one per latent).
#   {"short": {1085: [12.4, 9.8, ...], 6510:[]...}
aggregation_map = defaultdict(lambda: {Latent1: [], Latent2: [], Latent3: [], Latent4: []})

for item in Z_act_list:
    w = item["word"]
    aggregation_map[w][Latent1].append(item["activations"][Latent1])
    aggregation_map[w][Latent2].append(item["activations"][Latent2])
    aggregation_map[w][Latent3].append(item["activations"][Latent3])
    aggregation_map[w][Latent4].append(item["activations"][Latent4])


# Compute averages matching the original target word order sequence
# x positions, the 1085 heights and the 6510 heights as separate equal-length sequences.
plot_words = []
mean_latent1 = []
mean_latent2 = []
mean_latent3 = []
mean_latent4 = []


for word in target_words:
    if word in aggregation_map:
        plot_words.append(word)
        mean_latent1.append(np.mean(aggregation_map[word][Latent1]))
        mean_latent2.append(np.mean(aggregation_map[word][Latent2]))
        mean_latent3.append(np.mean(aggregation_map[word][Latent3]))
        mean_latent4.append(np.mean(aggregation_map[word][Latent4]))



# plot

# One tick per word; 
x = np.arange(len(plot_words)) # returns evenly spaced values

fig, ax = plt.subplots(figsize=(9, 5))
width = 0.2
x1 = x - 1.5 * width
x2 = x - 0.5 * width
x3 = x + 0.5 * width
x4 = x + 1.5 * width


rng_j = np.random.default_rng(0)

for i, word in enumerate(plot_words):
    acts_latent1 = np.array(aggregation_map[word][Latent1])
    acts_latent2 = np.array(aggregation_map[word][Latent2])
    acts_latent3 = np.array(aggregation_map[word][Latent3])
    acts_latent4 = np.array(aggregation_map[word][Latent4])

    j1 = rng_j.uniform(-width*0.15, width*0.15, size=len(acts_latent1))
    j2 = rng_j.uniform(-width*0.15, width*0.15, size=len(acts_latent2))
    j3 = rng_j.uniform(-width*0.15, width*0.15, size=len(acts_latent3))
    j4 = rng_j.uniform(-width*0.15, width*0.15, size=len(acts_latent4))


    ax.scatter(np.full_like(acts_latent1, x1[i]+j1), acts_latent1,color = 'none',edgecolor='#858483', s=45, alpha=0.75, zorder =5, clip_on=False)
    ax.scatter(np.full_like(acts_latent2, x2[i]+j2), acts_latent2, color = 'none', edgecolor='#858483', s=45, alpha=0.75,zorder=5, clip_on=False)
    ax.scatter(np.full_like(acts_latent3, x3[i]+j3), acts_latent3, color = 'none', edgecolor='#858483', s=45, alpha=0.75,zorder=5, clip_on=False)
    ax.scatter(np.full_like(acts_latent4, x4[i]+j4), acts_latent4, color = 'none', edgecolor='#858483', s=45, alpha=0.75,zorder=5, clip_on=False)

    std_k = acts_latent1.std()
    mean_k = acts_latent1.mean()
    yerr_k = np.array([[min(std_k, mean_k)], [std_k]])   # [lower, upper]
    std_k2 = acts_latent2.std()
    mean_k2 = acts_latent2.mean()
    yerr_k2 = np.array([[min(std_k2, mean_k2)], [std_k2]])   # [lower, upper]
    std_k3 = acts_latent3.std()
    mean_k3 = acts_latent3.mean()
    yerr_k3 = np.array([[min(std_k3, mean_k3)], [std_k3]])   # [lower, upper]
    std_k4 = acts_latent4.std()
    mean_k4 = acts_latent4.mean()
    yerr_k4 = np.array([[min(std_k4, mean_k4)], [std_k4]])   # [lower, upper]



    ax.errorbar(x1[i], mean_k, yerr=yerr_k, fmt='_', markersize=18, mew=2.5, ecolor='black', color='black', capsize=6, capthick=1.5, elinewidth=1.5, zorder=6)
    ax.errorbar(x2[i], mean_k2, yerr=yerr_k2, fmt='_', markersize=18, mew=2.5, ecolor='black', color='black', capsize=6, capthick=1.5, elinewidth=1.5, zorder=6)
    ax.errorbar(x3[i], mean_k3, yerr=yerr_k3, fmt='_', markersize=18, mew=2.5, ecolor='black', color='black', capsize=6, capthick=1.5, elinewidth=1.5, zorder=6)
    ax.errorbar(x4[i], mean_k4, yerr=yerr_k4, fmt='_', markersize=18, mew=2.5, ecolor='black', color='black', capsize=6, capthick=1.5, elinewidth=1.5, zorder=6)


rects1 = ax.bar(x1, mean_latent1, width, label=str(Latent1), facecolor='#F2DCD6', linewidth=1.2)
rects2 = ax.bar(x2, mean_latent2, width, label=str(Latent2), facecolor='#DEF6FF', linewidth=1.2)
rects3 = ax.bar(x3, mean_latent3, width, label=f'{Latent3} (random control)', facecolor='#C1FFB7', linewidth=1.2)
rects4 = ax.bar(x4, mean_latent4, width, label=f'{Latent4} (random control)',  facecolor='#FFDE98', linewidth=1.2)



ax.set_ylabel('Activation', fontsize=12)
#ax.set_yscale('symlog', linthresh=0.1)
ax.set_xlabel('Token', fontsize=12)
ax.set_title("'S' activations by token, layer 3, 16k width, 59 L0", fontsize=13, pad=15)
ax.set_xticks(x)
ax.set_xticklabels([f'{w}, n={len(aggregation_map[w][Latent1])}' for w in plot_words], rotation=45, ha='right', fontsize=11)

ax.legend(title='Latent ID', loc='upper right', frameon=True)

# Add standard visual cleanups
ax.grid(axis='y', linestyle='-', alpha=0.3)
# Draw the gridlines behind the bars 
ax.set_axisbelow(True)
plt.tight_layout()

# Save image file
ts = datetime.now().strftime("%Y%m%d_%H%M%S")
out_path = (
    REPO_ROOT
    / "figures"
    / f"chanin_fig6a_{Arch}_layer{Layer}_k{Sparsity}_{ts}.png"
)
out_path.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out_path, dpi=300, bbox_inches="tight")
plt.close(fig)

print(f"Saved -> {out_path}")

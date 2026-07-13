'''
Replicate figure 6 from A is for Absorption

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

Main_Latent = 6510
Absorbing_Latent = 1085

sae = SAE.from_pretrained(
    release=release,
    sae_id=sae_id,
    device = device
)

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
#note to self: soggy tokenizes to ['so', 'ggy'] so we need to implement a sliding window
target_token_ids = []
token_word_map = {}

for word in target_words:
    word_set = [word, " " + word]
    for w in word_set:
        target_token = tokenizer.encode(w, add_special_tokens=False) #gives tokenized version of the word
        if len(target_token) == 1:
            target_token_ids.append(target_token[0])
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
                "activations":{6510: Z[batch_idx, seq_idx, Main_Latent].item(), 
                                1085: Z[batch_idx, seq_idx, Absorbing_Latent].item()}
            }
            Z_act_list.append(Z_act)


print(Z_act_list)

# aggregate  per-word activations

# Z_act_list is one dict per *occurrence*: collapse those occurrences down to a single number per latent.
#
# each word has two lists of raw activations (one per latent).
#   {"short": {1085: [12.4, 9.8, ...], 6510:[]...}
aggregation_map = defaultdict(lambda: {1085: [], 6510: []})

for item in Z_act_list:
    w = item["word"]
    aggregation_map[w][1085].append(item["activations"][1085])
    aggregation_map[w][6510].append(item["activations"][6510])

# Compute averages matching the original target word order sequence
# x positions, the 1085 heights and the 6510 heights as separate equal-length sequences.
plot_words = []
mean_1085 = []
mean_6510 = []

for word in target_words:
    if word in aggregation_map:
        plot_words.append(word)
        mean_1085.append(np.mean(aggregation_map[word][1085]))
        mean_6510.append(np.mean(aggregation_map[word][6510]))


# plot

# One tick per word; 
x = np.arange(len(plot_words))
width = 0.35  

fig, ax = plt.subplots(figsize=(8, 5))

rects1 = ax.bar(x - width/2, mean_1085, width, label='1085', color='#FF0000')
rects2 = ax.bar(x + width/2, mean_6510, width, label='6510', color='#0000FF')

ax.set_ylabel('Activation', fontsize=12)
ax.set_xlabel('Token', fontsize=12)
ax.set_title("'S' activations by token, layer 3, 16k width, 59 L0", fontsize=13, pad=15)
ax.set_xticks(x)
ax.set_xticklabels(plot_words, rotation=45, ha='right', fontsize=11)

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

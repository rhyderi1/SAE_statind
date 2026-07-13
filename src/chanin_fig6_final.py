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

sae = SAE.from_pretrained(
    release=release,
    sae_id=sae_id
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
# batch_tokens = activation_store.get_batch_tokens(batch_size)# Shape: (batch_size, context_size
#Claude: removed the pre-loop batch fetch. Searching this batch for target words is
#Claude: pointless because the loop below fetches a NEW batch each iteration; the search
#Claude: must run inside the loop against the same tokens that produce Z.

target_words = ["snake",  "soggy",  "steam", "short",  "soccer", "sax"]

# token_ids = []

#Claude: Build each target word's token-id sequence(s) ONCE, before the loop.
#Claude: - tokenizer.encode returns a LIST, so multi-token words (e.g. "soggy" -> ["sog","gy"])
#Claude:   are kept as full sequences instead of being force-fit into a single-id compare.
#Claude: - We keep both "word" and " word": mid-sentence Gemma emits the leading-space token
#Claude:   (the SentencePiece marker), which has a different id than the start-of-text form.
word_seqs = {}
for word in target_words:
    word_seqs[word] = [tokenizer.encode(v, add_special_tokens=False) for v in (word, " " + word)]
    print(f"{word}: {word_seqs[word]}")

#Claude: d_sae = number of SAE latents (encoder is d_model x d_sae); used to size the
#Claude: empty fallback tensor for any word that never appears in the corpus.
d_sae = sae.W_enc.shape[1]

# Z_act_list = []
# final_pos_list = torch.empty(0,0)
#Claude: replaced the single unlabelled activation list + shared position list with a
#Claude: per-word dict so every saved activation stays tagged with the word it came from.
saved = {w: [] for w in target_words}   #Claude: word -> list of (n_hits, d_sae) tensors on CPU

with torch.no_grad():
    for batch in range (num_batches):
        batch_tokens = activation_store.get_batch_tokens(batch_size)# Shape: (batch_size, context_size
        B, S = batch_tokens.shape  #Claude: batch and sequence lengths, needed for masking + flattening

        _, cache = model.run_with_cache(
            batch_tokens,
            names_filter=hook_name,
            stop_at_layer=Layer + 1,
            prepend_bos=False,
        )

        X = cache[hook_name]
        del cache

        X_sae = X.to(device=sae.W_enc.device, dtype=sae.W_enc.dtype)
        Z = sae.encode(X_sae)
        Z = Z.reshape(-1, Z.shape[-1])  # (tokens, d_sae)
        # for pos in final_pos_list:
        #     Z_act_list.append(Z[pos])
        #Claude: old indexing was doubly broken: final_pos_list was still empty here, and
        #Claude: its 2-D [batch, seq] coords index a FLATTENED Z (rows = batch*S + seq) wrongly.

        #Claude: Search THIS batch for each target word and collect the matching Z rows.
        for word, seqs in word_seqs.items():
            #Claude: boolean mask over (B, S); marked True at the LAST token of each occurrence
            hits = torch.zeros(B, S, dtype=torch.bool, device=batch_tokens.device)
            for seq in seqs:
                L = len(seq)
                #Claude: shifted-AND sequence match: m[b,i] is True iff seq starts at (b, i).
                #Claude: This is what lets multi-token words match, unlike a single == compare.
                m = torch.ones(B, S - L + 1, dtype=torch.bool, device=batch_tokens.device)
                for j, tid in enumerate(seq):
                    m &= (batch_tokens[:, j:S - L + 1 + j] == tid)
                #Claude: shift the mark to the word's final subword, whose residual activation
                #Claude: represents the completed word (the token Chanin et al. attribute to).
                hits[:, L - 1:] |= m
            if hits.any():
                #Claude: hits.reshape(-1) is flattened the same way as Z, so it selects the
                #Claude: correct rows; .cpu() keeps GPU memory flat across 1209 batches.
                saved[word].append(Z[hits.reshape(-1)].cpu())

# for word in target_words:
#     word_set = [word, " " + word]
#     for w in word_set:
#         token = tokenizer.encode(w, add_special_tokens=False) #gives tokenized version of the word
#         token_ids.append(token)
#         print(f"Word: {w}, Token: {token}")
#
#     for id in token_ids:
#         pos = (batch_tokens == id).nonzero(as_tuple=False)
#         final_pos_list = torch.cat((final_pos_list, pos), dim=0)
#Claude: this whole block ran AFTER the loop (so Z was already gone), had broken indentation,
#Claude: and searched the stale last batch. Its job now happens inside the loop above.

#Claude: Collapse each word's chunks into one (n_occurrences, d_sae) tensor and persist them.
saved = {w: (torch.cat(v) if v else torch.empty(0, d_sae)) for w, v in saved.items()}
for w, t in saved.items():
    print(f"{w}: {t.shape[0]} occurrences")
torch.save(saved, "target_activations.pt")

#Claude: ---------- Figure 6a-style bar chart ----------
#Claude: Fig 6a shows one main "first-letter" latent's activation across tokens: it fires
#Claude: strongly for most, but drops to ~0 on tokens where the feature is absorbed.
#Claude: All target words start with "s", so we take a proxy "starts-with-S" latent as the
#Claude: one with the highest MEAN activation over every collected S-word token, then plot
#Claude: each word's mean activation of that single latent. Replace main_latent with the
#Claude: paper's probed latent index if you have it.
nonempty = [t for t in saved.values() if t.shape[0] > 0]
if not nonempty:
    raise SystemExit("No target words found in the corpus - nothing to plot.")

all_acts = torch.cat(nonempty, dim=0)                 #Claude: (N_tokens, d_sae)
main_latent = all_acts.mean(dim=0).argmax().item()    #Claude: proxy first-letter latent
print(f"main (starts-with-S) latent = {main_latent}")

#Claude: per-word mean activation of the main latent (0.0 if the word never appeared)
word_names = list(saved.keys())
word_means = [
    saved[w][:, main_latent].mean().item() if saved[w].shape[0] > 0 else 0.0
    for w in word_names
]

#Claude: single-series bar chart -> one recessive hue (height carries magnitude, not color),
#Claude: no legend (title names the series), light y-grid behind the bars, direct value labels.
fig, ax = plt.subplots(figsize=(8, 5))
bars = ax.bar(word_names, word_means, color="#4C72B0", width=0.65, zorder=3)
ax.bar_label(bars, fmt="%.2f", padding=3, fontsize=9)
ax.set_ylabel(f"Latent {main_latent} activation")
ax.set_xlabel("Target token")
ax.set_title(f'Fig 6a replication - main "S" latent (layer {Layer}, {Arch} L0={Sparsity})')
ax.grid(axis="y", color="0.85", linewidth=0.8, zorder=0)
ax.set_axisbelow(True)
for spine in ("top", "right"):
    ax.spines[spine].set_visible(False)
plt.xticks(rotation=45, ha="right")
plt.tight_layout()

out_dir = Path("figures")
out_dir.mkdir(exist_ok=True)
stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
out_path = out_dir / f"chanin_fig6a_layer{Layer}_{stamp}.png"
plt.savefig(out_path, dpi=150)
print(f"saved figure to {out_path}")

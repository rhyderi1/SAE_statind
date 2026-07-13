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
batch_tokens = activation_store.get_batch_tokens(batch_size)# Shape: (batch_size, context_size

target_words = ["snake",  "soggy",  "steam", "short",  "soccer", "sax"]

token_ids = []

Z_act_list = []
final_pos_list = torch.empty(0,0)

with torch.no_grad():
    for batch in range (num_batches):
        batch_tokens = activation_store.get_batch_tokens(batch_size)# Shape: (batch_size, context_size

        _, cache = model.run_with_cache(
            batch_tokens,
            names_filter=hook_name,
            stop_at_layer=Layer + 1,
            prepend_bos=False,
        )
    for word in target_words:
        word_set = [word, " " + word]
        for w in word_set:
            token = tokenizer.encode(w, add_special_tokens=False) #gives tokenized version of the word
            token_ids.append(token)
            print(f"Word: {w}, Token: {token}")

    for id in token_ids:
        pos = (batch_tokens == id).nonzero(as_tuple=False)
        final_pos_list = torch.cat((final_pos_list, pos), dim=0)


        X = cache[hook_name]
        del cache

        X_sae = X.to(device=sae.W_enc.device, dtype=sae.W_enc.dtype)
        Z = sae.encode(X_sae)
        Z = Z.reshape(-1, Z.shape[-1])  # (tokens, d_sae)
        for pos in final_pos_list:
            Z_act_list.append(Z[pos])


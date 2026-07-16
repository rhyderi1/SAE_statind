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
num_batches = 3820
hook_name = f"blocks.{Layer}.hook_resid_post"
release, sae_id = SAE_DATA[Layer][Arch][Sparsity]

Eps = 1E-8
total_tokens = num_batches * batch_size * context_size # ~15M

latent_pairs = [
    # top 3 absorption pairs (main, absorber) — one per letter
    (16033, 12304),   # u
    (5407,  10622),   # e
    (1006,  731),     # o
    (6510, 1085),
    # 3 control pairs: same absorber j, but a different letter's main i
    (9795,  12304),   # b-main vs u-absorber
    (11993, 10622),   # k-main vs e-absorber
    (1024,  731),     # j-main vs o-absorber
]
pair_stats = {
    pair: {
        "i_count": 0,
        "j_count": 0,
        "i_and_j_count": 0,
        "i_sum": 0.0,
        "i_and_j_sum": 0.0,
    } for pair in latent_pairs
}

sae = SAE.from_pretrained(
    release=release,
    sae_id=sae_id,
    device=device
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
    streaming=False
)

print(f"Starting processing for {num_batches} batches...")
print(f"Layer {Layer}, {Arch}, sparsity {Sparsity}")

with torch.no_grad():
    for batch in range(num_batches):
        batch_tokens = activation_store.get_batch_tokens(batch_size).to(device)

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
        Z = Z.reshape(-1, Z.shape[-1])

        for latent_i, latent_j in latent_pairs:
            Z_i = Z[:, latent_i]
            Z_j = Z[:, latent_j]

            i_active = Z_i > 0
            j_active = Z_j > 0

            pair_stats[(latent_i, latent_j)]["i_count"] += i_active.sum().item()
            pair_stats[(latent_i, latent_j)]["j_count"] += j_active.sum().item()
            pair_stats[(latent_i, latent_j)]["i_and_j_count"] += (i_active & j_active).sum().item()

            pair_stats[(latent_i, latent_j)]["i_sum"] += Z_i[i_active].sum().item()
            pair_stats[(latent_i, latent_j)]["i_and_j_sum"] += Z_i[i_active & j_active].sum().item()


for latent_i, latent_j in latent_pairs:
    stats = pair_stats[(latent_i, latent_j)]
    
    i_count = stats["i_count"]
    j_count = stats["j_count"]
    i_and_j_count = stats["i_and_j_count"]
    i_sum = stats["i_sum"]
    i_and_j_sum = stats["i_and_j_sum"]

    if i_count == 0 or i_and_j_count ==0:
        print(f"\nLatent i: {latent_i}\nLatent j: {latent_j}")
        print("ERROR: j_count is 0. Computations undefined.\n")
        continue

    # Probabilities
    mean_zi_zi_active = i_sum/i_count
    mean_zi_zizj_active = i_and_j_sum/i_and_j_count
    prob_zj_zi_active = i_and_j_count/i_count


    Absorption_Metric3 = ((mean_zi_zi_active - mean_zi_zizj_active)/(mean_zi_zi_active + Eps)) * prob_zj_zi_active

    # Print formatting identical to your request
    print(f'\nLatent i: {latent_i}\nLatent j: {latent_j}')
    #print(f'Metric 1: {Absorption_Metric1}')
    #print(f'Metric 2: {Absorption_Metric2}')
    print(f'Metric 3: {Absorption_Metric3:.6f}\n')
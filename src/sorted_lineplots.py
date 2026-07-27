'''
We want to see the distributin of data with regards to 
- zizj
- max(zi|zi>0)
- min(zi|zi>0)
- E(zi|zi>0)
- std(zi|zi>0)
in terms of the uniformityof values/gradient in a sorted list
Why? to see if any of these parameters correspond to absorption patterns 
'''
import os
import torch
from sae_lens import SAE
from sae_lens.training.activations_store import ActivationsStore
from transformer_lens import HookedTransformer
import matplotlib
matplotlib.use('Agg')  # Must precede pyplot import
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import csv
import time
import plotly.express as px

n_batches = 3823
batch_size = 32
context_size = 128
device = "cuda" if torch.cuda.is_available() else "cpu"

dataset = 'NeelNanda/pile-10k'
hook_name = 'blocks.12.hook_resid_post'
release, sae_id = SAE_DATA[12]["relu"]["20"]
n_batches = 1209
layer = 3
dtype_map = {"float16": torch.float16, "float32": torch.float32}
dtypechoice = "float32"
save_dtype = dtype_map[dtypechoice]
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

#1) Z stats
# Storing Z takes up too much space, so we have to go batch-wise
# We want (for Z) the per-latent mean, std., max, min (all active)

max_z = min_z = sum_z = sum_sq = active_n_per_latent = None
active_n = 0

with torch.no_grad():
    for batch_idx in range(n_batches): 

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
        Z = Z.detach()

        Z = Z.reshape(-1, Z.shape[-1]).to(dtype=save_dtype)#becomes a mxp matrix
        m, p = Z.shape
        active_n += (Z>0).sum()

        if max_z is None:
            max_z = torch.zeros(p,)
        batch_max = Z.max(dim=0)
        max_z = torch.maximum(max_z,batch_max)

        if min_z is None:
            min_z = torch.zeros(p,)
        batch_min = Z.min(dim=0)
        min_z = torch.minimum(min_z,batch_min)

        if sum_z is None:
            sum_z = torch.zeros(p,)
        batch_sum = Z.sum(dim=0)
        sum_z += batch_sum

        if sum_sq is None:
            sum_sq = torch.zeros(p,)
        batch_sum_sq = (Z**2).sum(dim=0)
        sum_sq += batch_sum_sq

        if active_n_per_latent is None:
            active_n_per_latent = torch.zeros(p,)
        active_n_per_latent += (Z > 0).sum(dim=0)


mean = sum_z/active_n_per_latent
variance = (sum_sq/active_n) -mean**2
std_dev = variance **(1/2)

#2) ZTZ stats

Gram = torch.load()
Gram_tri = torch.triu(Gram, diagonal=1) #only upper right triangle (symmettric matrix)

coords = torch.nonzero(Gram_tri)
zizj_values = Gram_tri[Gram_tri != 0]

zizj_and_coords = torch.cat([zizj_values.unsqueeze(1), coords], dim = 1)

#order them all
max_z, _ = torch.sort(max_z, descending=True)
min_z,_ = torch.sort(min_z, descending=True)
mean,_ = torch.sort(mean, descending=True)
std_dev, _ = torch.sort(std_dev, descending=True)
order = torch.argsort(zizj_and_coords[:, 0], descending=True)
zizj_and_coords = zizj_and_coords[order]

x = list(range(1, len(max) + 1))






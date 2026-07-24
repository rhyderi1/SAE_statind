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
import argparse
import matplotlib
matplotlib.use('Agg')  # Must precede pyplot import
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import csv
import time
import plotly.express as px


#1) Z stats
# Storing Z takes up too much space, so we have to go batch-wise
# We want (for Z) the per-latent mean, std., max, min (all active)

max = min = sum = None
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
        m, p = Z.shape()
        active_n += (Z>0).sum()

        if max is None:
            max = torch.zeros(p,)
        batch_max = Z.max(dim=1)
        max = torch.maximum(max,batch_max)

        if min is None:
            min = torch.zeros(p,)
        batch_min = Z.min(dim=1)
        max = torch.minimum(min,batch_min)

        if sum is None:
            sum = torch.zeros(p,)
        batch_sum = Z.sum(dim=1)
        sum += batch_sum

        if sum_sq is None:
            sum_sq = torch.zeros(p,)
        batch_sum_sq = batch_sum **2
        sum_sq += batch_sum_sq
        


mean = sum/active_n
variance = (sum_sq/active_n) -mean**2
std_dev = variance **(1/2)

#2) ZTZ stats

Gram = torch.load()
Gram_tri = torch.triu(Gram, diagonal=1) #only upper right triangle (symmettric matrix)

coords = torch.nonzero(Gram_tri)
zizj_values = Gram_tri[Gram_tri>0]

zizj_and_coords = torch.cat(zizj.unsqueeze(), coords)

#order them all
torch.sort(max, descending=True)
torch.sort(min, descending=True)
torch.sort(mean, descending=True)
torch.sort(std_dev, descending=True)
torch.sort(zizj_and_coords,dim=0, descending=True)

x = list(range(1, len(max) + 1))






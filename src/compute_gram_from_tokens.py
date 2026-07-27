"""Single-pass ZTZ and indicator-Gram accumulation, straight from the corpus.

Same streaming setup as infer_z.py, but instead of writing Z shards to disk it
accumulates two p x p matrices on the fly and saves only those:

    ztz_layer{L}_{arch}_k{sp}.pt        G      = sum over batches of Z^T Z
    zindtzind_layer{L}_{arch}_k{sp}.pt  G_ind  = sum of 1{Z!=0}^T 1{Z!=0}

Use this when the Gram matrices are all that's needed -- it avoids the hundreds
of GB of Z shards that infer_z.py produces. Outputs land in
data/pile-10k-saes/layer{L}_{arch}_k{sp}/.

Reads config/params.csv by default.
"""

import os
import torch
from sae_lens import SAE
from sae_lens.training.activations_store import ActivationsStore
# ActivationsStore handles the entire data loading pipeline. 
# It takes the dataset (tokenizes on the fly(streaming)), intercepts and gets activations at some layer (using Hooked Transformer), 
# and returns the tokens in (batch size) batches

from transformer_lens import HookedTransformer
import argparse
import matplotlib
matplotlib.use('Agg')  # Must precede pyplot import
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import csv
import time

from infer_z import SAE_DATA

def read_config(task_id, file_path):
    """Reads the config row for the given task ID ."""
    with open(file_path, 'r') as file:
        reader = csv.DictReader(file)
        for i, row in enumerate(reader):
            if i == task_id:
                return row
    return None


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task_id",    type=int, default=None)
    parser.add_argument("--config_csv", type=str, default="config/params.csv")
    parser.add_argument("--layer",    type=int, choices=[3, 12, 19])
    parser.add_argument("--arch",     type=str, choices=["relu","topk","batchtopk","jumprelu","matryoshka"])
    parser.add_argument("--sparsity", type=str)
    parser.add_argument("--modelchoice",  type=str, required=True, choices=["gemma-2-2b"])
    parser.add_argument("--dtypechoice",  type=str, default="float32", choices=["float16", "float32"])
    parser.add_argument("--dataset",      type=str, default="NeelNanda/pile-10k")
    parser.add_argument("--n_batches",    type=int, default=200)
    parser.add_argument("--batch_size",   type=int, default=32)
    parser.add_argument("--context_size", type=int, default=128)
    parser.add_argument("--out_dir",      type=str)
    parser.add_argument("--shard_size",   type=int, default=50_000)
    parser.add_argument("--store_x",      action="store_true") #if the argument is present, set to True (required for Boolean)
    parser.add_argument("--store_z",      action="store_true")
    parser.add_argument("--plot", action="store_true")
    return parser.parse_args()



def get_sae(layer, arch, sparsity, device: str) -> SAE: # returns an object of class SAE (SAELens base architecture for all SAEs)
    release, sae_id = SAE_DATA[layer][arch][sparsity] # extracts release and sae_id from tuple
    print(f"Selected: layer={layer}, arch={arch}, sparsity={sparsity}")
    print(f"  release = {release}")
    print(f"  sae_id  = {sae_id}")
    sae = SAE.from_pretrained(release=release, sae_id=sae_id, device=device) # The from_pretrained function from the class SAE in sae.py in SAELens
    # from_pretrained only returns the first value from 'from_pretrained_with_cfg_and sparsity
    return sae.to(device).eval() # move to GPU, turns to evaluation mode (no dropout, batchnorm)

def collect_and_save_gram(layer,
    model, sae, activation_store,
    out_dir, n_batches, batch_size, shard_size,
    save_dtype,arch,sparsity
):
    G = None
    G_ind = None
    os.makedirs(out_dir, exist_ok=True) #creates output folder if it doesn't exist

    hook_name     = f"blocks.{layer}.hook_resid_post" # where hooked transformer should intercept: layer, and location (res, mlp, att)


    shard_idx     = 0

    print(f"\nCollecting {n_batches} batches of {batch_size} x {activation_store.context_size} tokens")
    print(f"  hook: {hook_name}  |  stop_at_layer: {layer + 1}") # only go to up to the layer we need
    print(f"  shard_size: {shard_size} token positions\n")

    sae.eval() # testing mode, not training (eg. no dropout, batch norm) (tells layers how to behave)
    with torch.no_grad():
        # can think of as a subset of sae.eval() --> saves memory by telling computer not to build computation 
        # graph from intermediate activations, since backward() function will not be called
        for batch_idx in range(n_batches): # looping, per batch

            # (from activation_store) get fixed-length batch of tokens
            try:
                batch_tokens = activation_store.get_batch_tokens(
                    batch_size, raise_at_epoch_end=True
                )
            except StopIteration:
                print(f"  Dataset exhausted after {batch_idx} batches "
                      f"({batch_idx * batch_size} windows); stopping.")
                break
            # shape: (batch_size, context_size)

            _, cache = model.run_with_cache( # the core: actually obtaining the model's intenal activations
                batch_tokens,
                names_filter=hook_name,
                stop_at_layer=layer+1,
                prepend_bos=False,
            )
            X = cache[hook_name]   # extracts residual stream activations at layer.(dictionary lookup) (batch_size, context_size, d_model)
            del cache # delete the rest of the cache to save memory

            X_sae = X.to(device=sae.device, dtype=sae.W_enc.dtype) # so that when you multiply the two in the encode step, you don't get errors
            Z = sae.encode(X_sae)  # (batch_size, context_size, d_sae)
            Z = Z.detach()

            ####### get ZTZ
            Z_flat = Z.reshape(-1, Z.shape[-1]).to(dtype=save_dtype)
            _, p = Z_flat.shape
            if G is None:
                G = torch.zeros(p, p, device=Z_flat.device, dtype=save_dtype)
            ZTZ = Z_flat.T @ Z_flat
            G += ZTZ

            # ####### get ZindTZind
            # Z_ind_flat=(Z_flat != 0).float()
            # ZindTZind = Z_ind_flat.T @ Z_ind_flat
            # if G_ind is None:
            #     G_ind = torch.zeros(p, p, device=Z_flat.device, dtype=save_dtype)
            # G_ind += ZindTZind
            # if (batch_idx + 1) % 50 == 0:
            #     print(f"  batch {batch_idx + 1}/{n_batches}")
            # #break  # for debugging, only run one batch

    # flush any remaining tokens

    gram_nancount = torch.isnan(G).sum().item()
    ind_gram_nancount = torch.isnan(G_ind).sum().item()
    print(f"\nGram matrix has {gram_nancount} NaNs, ZindTZind has {ind_gram_nancount} NaNs")
    print(f"\nDone. Shards written to {out_dir}/")
    torch.save(G.cpu(), os.path.join(out_dir, f"ztz_layer{layer}_{arch}_k{sparsity}.pt"))
    # torch.save(G_ind.cpu(), os.path.join(out_dir, f"zindtzind_layer{layer}_{arch}_k{sparsity}.pt"))


def main():
    args = parse_args()

    if args.task_id is not None:
        cfg = read_config(args.task_id, args.config_csv)
        if cfg is None:
            raise SystemExit(f"No config found for task_id {args.task_id}")
        args.layer    = int(cfg["layer"])
        args.arch     = cfg["arch"]
        args.sparsity = cfg["sparsity"]

    if args.layer is None or args.arch is None or args.sparsity is None:
        raise SystemExit("Provide --task_id, or all of --layer/--arch/--sparsity")

    if args.out_dir is None:
        args.out_dir = f"data/pile-10k-saes/gram_layer{args.layer}_{args.arch}_l0{args.sparsity}"

    dtype_map = {"float16": torch.float16, "float32": torch.float32}
    save_dtype = dtype_map[args.dtypechoice]

    device     = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"\nLoading model + SAE on {device} ...")
    model = HookedTransformer.from_pretrained(
        args.modelchoice, device=device, dtype=save_dtype
    )
    sae = get_sae(args.layer, args.arch, args.sparsity, device)

    print(f"\nBuilding ActivationsStore (dataset='{args.dataset}', "
          f"context_size={args.context_size}) ...")
    activation_store = ActivationsStore.from_sae(
        model=model,
        sae=sae,
        context_size=args.context_size,
        dataset=args.dataset,
    )

    collect_and_save_gram(
        layer = args.layer,
        model=model,
        sae = sae, 
        activation_store = activation_store,
        out_dir     = args.out_dir,
        n_batches   = args.n_batches,
        batch_size  = args.batch_size,
        shard_size  = args.shard_size,
        save_dtype  = save_dtype,
        arch = args.arch,
        sparsity = args.sparsity

    )


if __name__ == "__main__":
    main()
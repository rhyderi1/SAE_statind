"""Stage 1 of the pipeline: stream the corpus and save SAE activations to disk.

Loads gemma-2-2b under TransformerLens, hooks the residual stream at
blocks.{layer}.hook_resid_post, encodes each batch with the selected SAE, and
writes X (residual, d_model) and/or Z (SAE latents, d_sae) as Z_shard{NNN}.pt
files of ~shard_size token positions each.

SAE_DATA is the registry mapping (layer, arch, sparsity) -> (release, sae_id);
most other scripts in this repo import it from here.

Run one config directly, or a whole sweep via --task_id as a SLURM array index
into config/params.csv:

    python src/infer_z.py --modelchoice gemma-2-2b --layer 12 --arch relu \
        --sparsity 20 --store_z
    sbatch scripts/run_inferz.sh        # array 0-19, one row of params.csv each

Output layout (matches the downstream scatter scripts, which glob
data/Z/*/acts_layer{L}_{arch}_{sp}_*):
  X shards -> data/X/layer{L}/X_shard{NNN}.pt
  Z shards -> data/Z/All batches/acts_layer{L}_{arch}_{sparsity}_{ts}/Z_shard{NNN}.pt
--out_dir, if given, overrides the Z directory only.
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
from datetime import datetime
import csv
import time
now = datetime.now()
YLIM = (1e-4, 1e10)

SAE_DATA = {
    3: {
        "jumprelu": {
            "59": ("gemma-scope-2b-pt-res", "layer_3/width_16k/average_l0_59")
        }
    },
    12: {
        "relu": {
            "20": ("sae_bench_gemma-2-2b_vanilla_width-2pow14_date-1109", "blocks.12.hook_resid_post__trainer_0"),
            "40": ("sae_bench_gemma-2-2b_vanilla_width-2pow14_date-1109", "blocks.12.hook_resid_post__trainer_1"),
            "80": ("sae_bench_gemma-2-2b_vanilla_width-2pow14_date-1109", "blocks.12.hook_resid_post__trainer_2"),
        },
        "topk": {
            "20": ("sae_bench_gemma-2-2b_topk_width-2pow14_date-1109", "blocks.12.hook_resid_post__trainer_0"),
            "40": ("sae_bench_gemma-2-2b_topk_width-2pow14_date-1109", "blocks.12.hook_resid_post__trainer_1"),
            "80": ("sae_bench_gemma-2-2b_topk_width-2pow14_date-1109", "blocks.12.hook_resid_post__trainer_2"),
        },
        "jumprelu": {
            "22": ("gemma-scope-2b-pt-res", "layer_12/width_16k/average_l0_22"),
            "41": ("gemma-scope-2b-pt-res", "layer_12/width_16k/average_l0_41"),
            "82": ("gemma-scope-2b-pt-res-canonical", "layer_12/width_16k/average_l0_82"),
        },
        "matryoshka": {
            "40": ("gemma-2-2b-res-matryoshka-dc", "blocks.12.hook_resid_post"),
        },
    },
    19: {
        "relu": {
            "20": ("sae_bench_gemma-2-2b_vanilla_width-2pow14_date-1109", "blocks.19.hook_resid_post__trainer_0"),
            "40": ("sae_bench_gemma-2-2b_vanilla_width-2pow14_date-1109", "blocks.19.hook_resid_post__trainer_1"),
            "80": ("sae_bench_gemma-2-2b_vanilla_width-2pow14_date-1109", "blocks.19.hook_resid_post__trainer_2"),
        },
        "topk": {
            "20": ("sae_bench_gemma-2-2b_topk_width-2pow14_date-1109", "blocks.19.hook_resid_post__trainer_0"),
            "40": ("sae_bench_gemma-2-2b_topk_width-2pow14_date-1109", "blocks.19.hook_resid_post__trainer_1"),
            "80": ("sae_bench_gemma-2-2b_topk_width-2pow14_date-1109", "blocks.19.hook_resid_post__trainer_2"),
        },
    
        "jumprelu": {
            "23":        ("gemma-scope-2b-pt-res", "layer_19/width_16k/average_l0_23"),
            "40":       ("gemma-scope-2b-pt-res", "layer_19/width_16k/average_l0_40"),
            "73": ("gemma-scope-2b-pt-res", "layer_19/width_16k/average_l0_73"),
        },
        "matryoshka": {
            "40": ("gemma-2-2b-res-matryoshka-dc", "blocks.19.hook_resid_post"),
        },
    },
}

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


def _save_shard(all_X, all_Z, x_dir, z_dir, shard_idx, store_x, store_z):
    if store_x and all_X:
        X_cat = torch.cat(all_X, dim=0) # concatenates the list of tensors (13 tensors for 13 batches, where each batch is (4096,2304)) into 1 tensor with dimension (53k, 2304)
        x_path = os.path.join(x_dir, f"X_shard{shard_idx:03d}.pt")
        torch.save(X_cat, x_path)
        print(f"  Saved {x_path}  {tuple(X_cat.shape)}")
    if store_z and all_Z:
        Z_cat = torch.cat(all_Z, dim=0) # concatenates the list of tensors (13 tensors for 13 batches, where each batch is (4096,16k)) into 1 tensor with dimension (53k, 16k)
        z_path = os.path.join(z_dir, f"Z_shard{shard_idx:03d}.pt")
        torch.save(Z_cat, z_path)
        print(f"  Saved {z_path}  {tuple(Z_cat.shape)}")
    if store_z:
        return z_path

def collect_and_save_sharded(layer,
    model, sae, activation_store,
    x_dir, z_dir, n_batches, batch_size, shard_size,
    save_dtype, store_x, store_z,plot,arch,sparsity
):

    if store_x:
        os.makedirs(x_dir, exist_ok=True) # X shards -> data/X/layer{L}/
    if store_z:
        os.makedirs(z_dir, exist_ok=True) # Z shards -> data/Z/All batches/{name}/

    hook_name     = f"blocks.{layer}.hook_resid_post" # where hooked transformer should intercept: layer, and location (res, mlp, att)

    all_X, all_Z  = [], []
    tokens_in_buf = 0
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
            batch_tokens = activation_store.get_batch_tokens(batch_size)
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

            # flatten token dimension and save dtype 
            if store_x:
                X_flat = X.reshape(-1, X.shape[-1]).to(dtype=save_dtype).cpu() # -1 in reshaping means you fix the other parameters and let it fill it in (the number of objects must be constant)
                all_X.append(X_flat)

            if store_z:
                Z_flat = Z.reshape(-1, Z.shape[-1]).to(dtype=save_dtype).cpu()
                all_Z.append(Z_flat)

            # shapes: (batch_size * context_size, d_model / d_sae)

            tokens_in_buf += batch_size*activation_store.context_size

            # flush shard when full 
            if tokens_in_buf >= shard_size:
                if store_z:
                    z_path = os.path.join(z_dir, f"Z_shard{shard_idx:03d}.pt")
                _save_shard(all_X, all_Z, x_dir, z_dir, shard_idx, store_x, store_z)
                all_X, all_Z  = [], []
                tokens_in_buf = 0
                shard_idx    += 1
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                #if store_z and plot:
                 #   plot_dtd_ztz_scatter(z_path, sae, out_dir,arch,layer,sparsity)
                #break


            if (batch_idx + 1) % 50 == 0:
                print(f"  batch {batch_idx + 1}/{n_batches} "
                      f"({tokens_in_buf} tokens buffered so far this shard) ...")

    # flush any remaining tokens
    if all_X or all_Z:
        _save_shard(all_X, all_Z, x_dir, z_dir, shard_idx, store_x, store_z)

    total = (shard_idx + 1) * shard_size  # approximate
    if store_x:
        print(f"\nDone. X shards written to {x_dir}/")
    if store_z:
        print(f"Done. Z shards written to {z_dir}/")


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

    # X shards -> data/X/layer{L}/ ; Z shards -> data/Z/All batches/{name}/
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    name = f"acts_layer{args.layer}_{args.arch}_{args.sparsity}_{ts}"
    args.x_dir = f"data/X/layer{args.layer}"
    args.z_dir = args.out_dir if args.out_dir is not None else f"data/Z/All batches/{name}"

    if not args.store_x and not args.store_z: # if we don't choose to save X, Z --> warning
        raise SystemExit("Nothing to save — pass --store_x and/or --store_z")

    dtype_map = {"float16": torch.float16, "float32": torch.float32}
    save_dtype = dtype_map[args.dtypechoice]

    device     = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"\nLoading model + SAE on {device} ...")
    model = HookedTransformer.from_pretrained(
        # HookedTransfomer is a transformer architecture (with the model choice you select)
        #that has HookedRootModule (allows you to intercept model forward pass at some layer)
        args.modelchoice, device=device, dtype=save_dtype
    )
    sae = get_sae(args.layer, args.arch, args.sparsity, device)

    print(f"\nBuilding ActivationsStore (dataset='{args.dataset}', "
          f"context_size={args.context_size}) ...")
    activation_store = ActivationsStore.from_sae(# from_sae initializes an ActivationsStore class (with the necessary parameters)(here, called activation_store)  
        model,
        sae,
        context_size=args.context_size,
        dataset=args.dataset,
    )

    collect_and_save_sharded(
        layer = args.layer,
        model=model,
        sae = sae, 
        activation_store = activation_store,
        x_dir       = args.x_dir,
        z_dir       = args.z_dir,
        n_batches   = args.n_batches,
        batch_size  = args.batch_size,
        shard_size  = args.shard_size,
        save_dtype  = save_dtype,
        store_x     = args.store_x,
        store_z     = args.store_z,
        plot = args.plot,
        arch = args.arch,
        sparsity = args.sparsity
    )


if __name__ == "__main__":
    main()
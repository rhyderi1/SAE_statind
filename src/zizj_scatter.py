'''
z_i vs z_j scatter for the latent pairs in PAIRS.

  --save   stream the corpus and write one .pt per latent (plus the token ids)
  --plot   scatter each pair; reuses saved .pt files when one --save run covers
           every latent in PAIRS, otherwise streams the corpus itself
Passing both streams once, saves, and plots.

Red points: every occurrence of a token that appeared in a full-absorption
event of j under i's letter (from the newest absorption_sets JSON in results/).
Pairs where j never absorbs from i's letter get no red points.
'''

import argparse
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from pathlib import Path
from datetime import datetime
from sae_lens import SAE, ActivationsStore
from transformer_lens import HookedTransformer
from transformers import AutoTokenizer
import plotly.express as px

device = "cuda" if torch.cuda.is_available() else "cpu"
dataset = 'NeelNanda/pile-10k'
context_size = 128
batch_size = 32
n_batches = 1209

ARCH = "jumprelu"
LAYER = 3
SPARSITY = "59"
SAE_RELEASE = "gemma-scope-2b-pt-res"
SAE_ID = "layer_3/width_16k/average_l0_59"   # the paper's exact checkpoint (L0=59)

PAIRS = [
    (6510, 1085)
]

REPO_ROOT = Path(__file__).parent.parent
Z_DIR = REPO_ROOT / "data" / "Z" / "zizj_pairs"
FIG_DIR = REPO_ROOT / "figures"

LATENTS = set() # like a list, but prevents duplicates
for pair in PAIRS:
    for latent in pair:
        LATENTS.add(latent)
LATENTS = sorted(LATENTS) # deduplicated 'list' of all latents


def latent_file(latent, ts):
    return f"zi_latent{latent}_{ARCH}_layer{LAYER}_k{SPARSITY}_{ts}.pt"


def tokens_file(ts):
    return f"tokens_{ARCH}_layer{LAYER}_k{SPARSITY}_{ts}.pt"


# 1) Stream the corpus and pull out the z columns

def collect_columns():
    """Returns ({latent: (tokens,) tensor}, (tokens,) tensor of token ids)."""
    hook_name = f'blocks.{LAYER}.hook_resid_post'
    sae = SAE.from_pretrained(release=SAE_RELEASE, sae_id=SAE_ID, device=device)
    model = HookedTransformer.from_pretrained_no_processing(
        default_prepend_bos=True,
        model_name="gemma-2-2b",
        device=device,
    )
    store = ActivationsStore.from_sae(
        model, sae, context_size=context_size, dataset=dataset)

    cols = {}
    for latent in LATENTS:
        cols[latent] = []
    toks = []
    with torch.no_grad():
        for _ in range(n_batches):
            batch_tokens = store.get_batch_tokens(batch_size)
            toks.append(batch_tokens.reshape(-1).cpu())
            # shape: batch_size, context_size --> batch_size * context_size 
            # each row corresponds cleanly with each row of Z (individual tokens)

            _, cache = model.run_with_cache(
                batch_tokens,
                names_filter=hook_name,
                stop_at_layer=LAYER + 1,
                prepend_bos=False,
            )
            X = cache[hook_name].to(device=sae.W_enc.device, dtype=sae.W_enc.dtype)
            Z = sae.encode(X).reshape(-1, sae.cfg.d_sae)   # (tokens, p)
            for latent in LATENTS:
                # .clone() saves memory instead of simple slicing (view)
                cols[latent].append(Z[:, latent].clone().cpu())
                # appends the column for that feature (all tokens)
                # .append() makes sure we don't overwrite columsn for previous batches
                # meaning each latent has (n_batches) cols (activations for those tokens)

    columns = {}
    for latent, c in cols.items():
        columns[latent] = torch.cat(c).float()
        # concatenates to 1 long vertical column per latent (all tokens)
    token_ids = torch.cat(toks).long()
    # 1 dimensional tensor of token ids (ie: tokenized words)
    # concatenates to 1 long vertical column (all tokens)
    #.long() saves to int64
    return columns, token_ids # saves the dictionary with individual latent: activation columns


def save_columns(columns, token_ids):
    # saves the columns by index (separate plotting from computing)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    Z_DIR.mkdir(parents=True, exist_ok=True)
    for latent, z in columns.items():
        torch.save({"latent": latent, "z": z, "timestamp": ts},
                   Z_DIR / latent_file(latent, ts))
    torch.save({"token_ids": token_ids, "timestamp": ts},
               Z_DIR / tokens_file(ts))
    print(f"Saved run {ts} -> {Z_DIR}")


def load_columns():
    # Boolean: either run collect_columns() and save_columns() (if no timestamp intersection) or run load_columns()
    """Newest --save run that has every latent in LATENTS"""
    if not Z_DIR.is_dir():
        return None, None
    # every file of one run shares its trailing YYYYmmdd_HHMMSS timestamp;
    # keep the newest timestamp present for all latents
    runs_per_latent = []
    for latent in LATENTS:
        runs = set() # 1 set per latent, no duplicates
        for p in Z_DIR.glob(latent_file(latent, "*")):
            # .glob() is pattern matching. It looks for a file as described in latent_file function
            # passing * into the timestamp means it doesn't matter what's there (take teh file regardless of timestamp)(wildcard)
            runs.add("_".join(p.stem.split("_")[-2:]))
            # split the file into underscore separated parts, and only save the last 2 (datestamp, and timestamp)
        runs_per_latent.append(runs)
    common = set.intersection(*runs_per_latent) 
    # find a timestamp common to all saved runs for latents in question
    # important: not sufficient that we have all latents saved but for differing (or even some (not all) matching) timestamps
    # because tokens may not be exactly the same across timestamps (changed settings, shuffle, seed, etc...)
    if not common:
        print("No saved run covers every latent in PAIRS; streaming instead.")
        return None, None
    ts = max(common) # if more than 1 common, take the most recent

    columns = {}
    for latent in LATENTS:
        columns[latent] = torch.load(
            Z_DIR / latent_file(latent, ts), map_location="cpu")["z"]
            # latent_file saves a dict (see line 125 or in save_columns UDF). We only want the activations for a latent
    tok_path = Z_DIR / tokens_file(ts)
    token_ids = (torch.load(tok_path, map_location="cpu")["token_ids"]
                 if tok_path.exists() else None)
    print(f"Loaded run {ts} <- {Z_DIR}")
    return columns, token_ids


# 2) Which token ids get painted red for each pair

def absorbed_token_ids():
    '''
    We want to find token ids for absorption events, so we can plot it on our scatter plots (in red)
    '''
    pattern = f"absorption_sets_{ARCH}_layer{LAYER}_k{SPARSITY}_*.json"
    # find absorption_sets dict with s_main, s_abs, # absorption events, and each token (may have mutiple absorptio occurrences)
    candidates = sorted((REPO_ROOT / "results").glob(pattern)) # alpahbetical order is the same as chronological in this case!

    if not candidates:
        print("No absorption_sets JSON in results/; skipping red coloring.")
        return {}
    with open(candidates[-1]) as f: #open the newest absorption sets dict
        blob = json.load(f)
    print(f"Coloring absorption events from {candidates[-1]}")

    tok_map = blob["s_abs_tokens"]          
    # finds the s_abs_tokens part of the dict
    # these tokens are stored as words (need to be tokenized)
    letter_of = {} # get first letter for latents
    for L, mains in blob["s_main"].items():
        # create a reverse mapping of s_main. Instead of letter --> latent, latent --> letter
        for m in mains:
            letter_of[m] = L

    tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-2b")

    out = {}
    for i, j in PAIRS:
        tokens = sorted(set(tok_map.get(letter_of.get(i), {}).get(str(j), [])))
        # get the starting letter of the s_main latent, the in s_abs, find that starting letters dict for latent j that absorbed it, 
        # then for the absorber latent j in their, take the list of tokens where the absorption happened
        if not tokens:
            print(f"pair ({i}, {j}): j never absorbs from i's letter -- no red")
            continue
        ids = []
        for s in tokens:
            enc = tokenizer.encode(s, add_special_tokens=False)
            if len(enc) == 1:
                ids.append(enc[0]) # take only the first tokenized index if the tokenizer splits the words into multiple words (or should I ignore it altogether)
        out[(i, j)] = {"ids": torch.tensor(sorted(ids)), "tokens": tokens}
        # maps absorber tokens with their ids (tokenized)
        print(f"pair ({i}, {j}): {len(tokens)} absorbed tokens -> {len(ids)} ids")
    return out

def descriptive_stats(columns):
    '''Per-pair descriptive metrics for the scatter panels.'''
    stats = {}
    for i, j in PAIRS:
        zi, zj = columns[i], columns[j]
        n_total = zi.numel()

        i_on = zi > 0
        j_on = zj > 0
        both = i_on & j_on

        n_i = int(i_on.sum())
        n_j = int(j_on.sum())
        n_coact = int(both.sum())
        n_x_only = int((i_on & ~j_on).sum())
        n_y_only = int((~i_on & j_on).sum())
        n_support = int((i_on | j_on).sum())
        assert n_x_only + n_y_only + n_coact == n_support

        stats[(i, j)] = {
            "n_total":     n_total,
            "n_x_only":    n_x_only,
            "n_y_only":    n_y_only,
            "n_coact":     n_coact,
            "n_support":   n_support,
            "p_i":         n_i / n_total,
            "p_j":         n_j / n_total,
            "p_i_given_j": n_coact / n_j,
            "p_j_given_i": n_coact / n_i,
            "e_i":         float(zi.mean()),
            "e_j":         float(zj.mean()),
            "e_i_on_i":    float(zi[i_on].mean()),
            "e_j_on_j":    float(zj[j_on].mean()),
            "e_i_on_j":    float(zi[j_on].mean()),
            "e_j_on_i":    float(zj[i_on].mean()),
            "e_i_joint":   float(zi[both].mean()),
            "e_j_joint":   float(zj[both].mean()),
        }
    return stats

def stats_text(s): #turns stats into summary able to be plotted
    return "\n".join([
        "--- Summary Stats ---",
        f"Total points:   {s['n_total']:,}",
        f"X-axis points:  {s['n_x_only']:,}",
        f"Y-axis points:  {s['n_y_only']:,}",
        f"Co-active:      {s['n_coact']:,}",
        f"Support:        {s['n_support']:,}",
        "",
        f"P(zi>0):        {s['p_i']:.4f}",
        f"P(zj>0):        {s['p_j']:.4f}",
        f"P(zi>0|zj>0):   {s['p_i_given_j']:.4f}",
        f"P(zj>0|zi>0):   {s['p_j_given_i']:.4f}",
        "",
        f"E[zi]:          {s['e_i']:.3f}",
        f"E[zj]:          {s['e_j']:.3f}",
        f"E[zi|zi>0]:     {s['e_i_on_i']:.3f}",
        f"E[zj|zj>0]:     {s['e_j_on_j']:.3f}",
        f"E[zi|zj>0]:     {s['e_i_on_j']:.3f}",
        f"E[zj|zi>0]:     {s['e_j_on_i']:.3f}",
        f"E[zi|both]:     {s['e_i_joint']:.3f}",
        f"E[zj|both]:     {s['e_j_joint']:.3f}",
    ])


# 3) One scatter per pair

def plot_pairs(columns, token_ids, abs_ids,stats): #abs_ids just takes out dict above
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    for i, j in PAIRS:
        zi, zj = columns[i], columns[j]
        coact = int(((zi > 0) & (zj > 0)).sum())
        keep = (zi != 0) | (zj != 0)        # drop only the exact (0,0) points, to increase plotting efficiency

        abs_info = abs_ids.get((i, j))
        red = None
        if abs_info is not None and token_ids is not None:
            red = torch.isin(token_ids, abs_info["ids"]) & keep
            # important, we can't just use abs_info["ids"] because those are unique deduplicaetd token ids
            # a absorber token may appear in multiple places, so we create a boolean mask of size (context_size * batch_size) ~ 4.95 M, 
            # and see that at every position is the absorbed token there


        fig, (ax, ax_stats) = plt.subplots(
            1, 2,
            figsize=(7.0, 3.4),
            gridspec_kw={'width_ratios': [3.8, 2.8]}
        )

        base = keep if red is None else keep & ~red
        ax.plot(zi[base].numpy(), zj[base].numpy(), "o", ms=0.1, alpha=0.3)
        if red is not None:
            n_red = int(red.sum())
            name = (f"'{abs_info['tokens'][0]}'" if len(abs_info["tokens"]) == 1
                    else f"{len(abs_info['tokens'])} tokens")
            ax.plot(zi[red].numpy(), zj[red].numpy(), "o", ms=1.5, alpha=0.8,
                    color="red", zorder=3,
                    label=f"{name}, total {n_red} occurrences")
            ax.legend(loc="upper right", fontsize=6, markerscale=4)

        ax.axhline(0, lw=0.6, color="0.55")
        ax.axvline(0, lw=0.6, color="0.55")
        ax.set_xlabel(f"z_{i}")
        ax.set_ylabel(f"z_{j}")
        ax.set_title(f"z_{i} vs z_{j} ({ARCH} layer {LAYER} k={SPARSITY}, "
                     f"co-act={coact})", fontsize=9)

        ax_stats.axis('off')
        ax_stats.text(
            0.15, 0.5, stats_text(stats[i,j]),        
            transform=ax_stats.transAxes,
            fontsize=8,                   
            fontfamily='monospace',
            va='center',
            ha='left',
            bbox=dict(
                boxstyle='round,pad=1',
                facecolor='white',
                edgecolor='gray',
                alpha=0.9
            )
        )
        fig.tight_layout()

        out = FIG_DIR / f"zizj_scatter_{ARCH}_layer{LAYER}_k{SPARSITY}_pair{i}_{j}_{ts}.png"
        fig.savefig(out, dpi=160, bbox_inches="tight")
        plt.close(fig)
        print(f"pair ({i}, {j}): co-active={coact} -> {out.name}")

    

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--save", action="store_true")
    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args()
    if not (args.save or args.plot):
        raise SystemExit("Nothing to do -- pass --save and/or --plot")

    columns = token_ids = None
    if args.plot and not args.save:
        columns, token_ids = load_columns()
    if columns is None:
        columns, token_ids = collect_columns()
        if args.save:
            save_columns(columns, token_ids)
    if args.plot:
        stats = descriptive_stats(columns)
        plot_pairs(columns, token_ids, absorbed_token_ids(),stats)


if __name__ == "__main__":
    main()

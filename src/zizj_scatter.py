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
import torch
from pathlib import Path
from datetime import datetime
from sae_lens import SAE, ActivationsStore
from transformer_lens import HookedTransformer
from transformers import AutoTokenizer

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
    # fixed main i = 6840 ('g'), varied child j
    (6840, 2141),    # absorber x57
    (6840, 2809),
    (6840, 5904),
    (6840, 7852),
    # controls: absorbers of other letters
    (6840, 10622),
    (6840, 12353),
    (6840, 8480),
    (6840, 2240),
]

REPO_ROOT = Path(__file__).parent.parent
Z_DIR = REPO_ROOT / "data" / "Z" / "zizj_pairs"
FIG_DIR = REPO_ROOT / "figures"

LATENTS = sorted({latent for pair in PAIRS for latent in pair})


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

    cols = {latent: [] for latent in LATENTS}
    toks = []
    with torch.no_grad():
        for _ in range(n_batches):
            batch_tokens = store.get_batch_tokens(batch_size)
            # flattened the same way as Z below, so row t of every z column
            # belongs to token toks[t]
            toks.append(batch_tokens.reshape(-1).cpu())

            _, cache = model.run_with_cache(
                batch_tokens,
                names_filter=hook_name,
                stop_at_layer=LAYER + 1,
                prepend_bos=False,
            )
            X = cache[hook_name].to(device=sae.W_enc.device, dtype=sae.W_enc.dtype)
            Z = sae.encode(X).reshape(-1, sae.cfg.d_sae)   # (tokens, d_sae)
            for latent in LATENTS:
                # .clone() keeps just this small column; a plain slice would
                # hold the whole (tokens, 16k) Z matrix alive in memory
                cols[latent].append(Z[:, latent].clone().cpu())

    columns = {latent: torch.cat(c).float() for latent, c in cols.items()}
    token_ids = torch.cat(toks).long()
    return columns, token_ids


def save_columns(columns, token_ids):
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    Z_DIR.mkdir(parents=True, exist_ok=True)
    for latent, z in columns.items():
        torch.save({"latent": latent, "z": z, "timestamp": ts},
                   Z_DIR / latent_file(latent, ts))
    torch.save({"token_ids": token_ids, "timestamp": ts},
               Z_DIR / tokens_file(ts))
    print(f"Saved run {ts} -> {Z_DIR}")


def load_columns():
    """Newest --save run that has every latent in LATENTS"""
    if not Z_DIR.is_dir():
        return None, None
    # every file of one run shares its trailing YYYYmmdd_HHMMSS timestamp;
    # keep the newest timestamp present for all latents
    runs_per_latent = [
        {"_".join(p.stem.split("_")[-2:])
         for p in Z_DIR.glob(latent_file(latent, "*"))}
        for latent in LATENTS
    ]
    common = set.intersection(*runs_per_latent)
    if not common:
        print("No saved run covers every latent in PAIRS; streaming instead.")
        return None, None
    ts = max(common)

    columns = {
        latent: torch.load(Z_DIR / latent_file(latent, ts), map_location="cpu")["z"]
        for latent in LATENTS
    }
    tok_path = Z_DIR / tokens_file(ts)
    token_ids = (torch.load(tok_path, map_location="cpu")["token_ids"]
                 if tok_path.exists() else None)
    print(f"Loaded run {ts} <- {Z_DIR}")
    return columns, token_ids


# 2) Which token ids get painted red for each pair

def absorbed_token_ids():
    pattern = f"absorption_sets_{ARCH}_layer{LAYER}_k{SPARSITY}_*.json"
    candidates = sorted((REPO_ROOT / "results").glob(pattern))
    if not candidates:
        print("No absorption_sets JSON in results/; skipping red coloring.")
        return {}
    with open(candidates[-1]) as f:
        blob = json.load(f)
    print(f"Coloring absorption events from {candidates[-1]}")

    tok_map = blob["s_abs_tokens"]          # letter -> {absorber j -> [tokens]}
    letter_of = {m: L for L, mains in blob["s_main"].items() for m in mains}
    tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-2b")

    out = {}
    for i, j in PAIRS:
        tokens = sorted(set(tok_map.get(letter_of.get(i), {}).get(str(j), [])))
        if not tokens:
            print(f"pair ({i}, {j}): j never absorbs from i's letter -- no red")
            continue
        # map each token string back to its token id so its occurrences can be
        # found in the stream; strings that encode to >1 id can't be matched
        ids = [enc[0] for s in tokens
               if len(enc := tokenizer.encode(s, add_special_tokens=False)) == 1]
        out[(i, j)] = {"ids": torch.tensor(sorted(ids)), "tokens": tokens}
        print(f"pair ({i}, {j}): {len(tokens)} absorbed tokens -> {len(ids)} ids")
    return out


# 3) One scatter per pair

def plot_pairs(columns, token_ids, abs_ids):
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    for i, j in PAIRS:
        zi, zj = columns[i], columns[j]
        coact = int(((zi > 0) & (zj > 0)).sum())
        keep = (zi != 0) | (zj != 0)        # drop only the exact (0,0) points

        abs_info = abs_ids.get((i, j))
        red = None
        if abs_info is not None and token_ids is not None:
            red = torch.isin(token_ids, abs_info["ids"]) & keep

        fig, ax = plt.subplots(figsize=(3.8, 3.4))
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
        plot_pairs(columns, token_ids, absorbed_token_ids())


if __name__ == "__main__":
    main()

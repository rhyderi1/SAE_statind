'''
z_i vs z_j scatter, from a saved Z matrix.

Two modes, selected by flags (at least one required):
  --save   stream the corpus, extract the z columns for the latents in PAIRS,
           and write one .pt per latent (labeled with its latent index)
  --plot   scatter the pairs. Reads the saved per-latent .pt files when a run
           covering every latent is found, otherwise streams the corpus itself.

Passing both collects once, saves, and plots from what it just collected.
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

device = "cuda" if torch.cuda.is_available() else "cpu"
dataset = 'NeelNanda/pile-10k'
context_size = 128
batch_size = 32
n_batches = 1209 # not 1209

ARCH = "jumprelu"
LAYER = 3
SPARSITY = "59"
PAIRS = [
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
MARKER_SIZE = 2.5
ALPHA = 0.3

REPO_ROOT = Path(__file__).parent.parent
Z_DIR = REPO_ROOT / "data" / "Z" / "zizj_pairs"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--save", action="store_true",
                        help="collect the z columns for PAIRS and write them to a .pt")
    parser.add_argument("--plot", action="store_true",
                        help="scatter the pairs")
    parser.add_argument("--z_dir", type=str, default=None,
                        help="with --plot alone: read the per-latent .pt files from "
                             "this directory instead of streaming. Defaults to Z_DIR.")
    parser.add_argument("--n_batches", type=int, default=n_batches)
    parser.add_argument("--batch_size", type=int, default=batch_size)
    parser.add_argument("--context_size", type=int, default=context_size)
    return parser.parse_args()


def wanted_latents(pairs):
    latents = []
    for (i, j) in pairs:
        if i not in latents:
            latents.append(i)
        if j not in latents:
            latents.append(j)
    return latents


# 1) Stream batches

def collect_columns(pairs, n_batches, batch_size, context_size):
    """Stream the corpus and return {latent: (tokens,) float tensor on CPU}."""
    latents = wanted_latents(pairs)

    hook_name = f'blocks.{LAYER}.hook_resid_post'
    SAE_RELEASE = "gemma-scope-2b-pt-res"
    SAE_ID = "layer_3/width_16k/average_l0_59"   # the paper's exact checkpoint (L0=59)
    #SAE_RELEASE = "sae_bench_gemma-2-2b_vanilla_width-2pow14_date-1109"
    #SAE_ID = "blocks.12.hook_resid_post__trainer_0"   # the paper's exact checkpoint (L0=59)
    sae = SAE.from_pretrained(release=SAE_RELEASE, sae_id=SAE_ID, device=device)

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

    batches_per_latent = {latent: [] for latent in latents}

    with torch.no_grad():
        for batch_num in range(n_batches):
            batch_tokens = activation_store.get_batch_tokens(batch_size)

            _, cache = model.run_with_cache(
                batch_tokens,
                names_filter=hook_name,
                stop_at_layer=LAYER + 1,
                prepend_bos=False,
            )
            X = cache[hook_name]
            del cache

            X_sae = X.to(device=sae.W_enc.device, dtype=sae.W_enc.dtype)
            Z = sae.encode(X_sae)
            Z = Z.reshape(-1, Z.shape[-1])  # (tokens, d_sae)

            for latent in latents:
                # .clone() so we keep a small (tokens,) column, not the entire Z matrix
                batches_per_latent[latent].append(Z[:, latent].detach().clone().cpu())
            del Z

    columns = {}
    for latent in latents:
        columns[latent] = torch.cat(batches_per_latent[latent]).float()
    return columns


def latent_filename(latent, ts):
    return f"zi_latent{latent}_{ARCH}_layer{LAYER}_k{SPARSITY}_{ts}.pt"


def latent_glob(latent):
    return f"zi_latent{latent}_{ARCH}_layer{LAYER}_k{SPARSITY}_*.pt"


def save_columns(columns, pairs, n_batches, batch_size, context_size):
    """Write one .pt per latent, each labeled with its latent index.

    Every file from a single run shares a timestamp, so load_columns can tell
    which columns were streamed together. Returns the list of paths.
    """
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    Z_DIR.mkdir(parents=True, exist_ok=True)

    paths = []
    for latent, z in columns.items():
        out_path = Z_DIR / latent_filename(latent, ts)
        torch.save(
            {
                "latent": latent,
                "z": z,
                "pairs": pairs,
                "arch": ARCH,
                "layer": LAYER,
                "sparsity": SPARSITY,
                "dataset": dataset,
                "n_batches": n_batches,
                "batch_size": batch_size,
                "context_size": context_size,
                "timestamp": ts,
            },
            out_path,
        )
        print(f"Saved -> {out_path}  (latent {latent}, {z.numel()} tokens)")
        paths.append(out_path)
    return paths


def _run_timestamp(name, latent):
    """Pull the trailing YYYYmmdd_HHMMSS out of a per-latent filename."""
    stem = Path(name).stem
    prefix = f"zi_latent{latent}_{ARCH}_layer{LAYER}_k{SPARSITY}_"
    return stem[len(prefix):]


def load_columns(latents, z_dir=None):
    """Load one column per latent from the newest run that has all of them.

    Returns {latent: (tokens,) tensor}, or None when no such run exists.
    """
    z_dir = Z_DIR if z_dir is None else Path(z_dir)
    if not z_dir.is_dir():
        raise SystemExit(f"No such directory: {z_dir}")

    # timestamps present for each latent, then the newest one common to all
    per_latent = {
        latent: {_run_timestamp(p.name, latent) for p in z_dir.glob(latent_glob(latent))}
        for latent in latents
    }
    missing = [latent for latent, ts_set in per_latent.items() if not ts_set]
    if missing:
        print(f"No saved columns in {z_dir} for latent(s): {missing}")
        return None

    common = set.intersection(*per_latent.values())
    if not common:
        print(f"No single run in {z_dir} covers all of {latents}; "
              f"re-run with --save to collect them together.")
        return None
    ts = max(common)

    columns = {}
    for latent in latents:
        path = z_dir / latent_filename(latent, ts)
        blob = torch.load(path, map_location="cpu")
        columns[latent] = blob["z"]
        print(f"Loaded <- {path}  (latent {latent}, n_batches={blob['n_batches']})")
    return columns


#2) Compute joint-support means/stds and on-support Pearson r, then standardize

def joint_support_stats(zi, zj):
    """
    Joint-support means/stds and on-support Pearson r, all computed over the
    joint support (tokens where BOTH latents are active). Fully vectorized.
    Returns (mean_i, std_i, mean_j, std_j) as tensors, plus (r, n).
    """
    m = (zi > 0) & (zj > 0)
    n = int(m.sum().item())

    zi_m, zj_m = zi[m], zj[m]
    mean_i, mean_j = zi_m.mean(), zj_m.mean()
    di, dj = zi_m - mean_i, zj_m - mean_j

    std_i = di.square().mean().sqrt().clamp_min(1e-8)
    std_j = dj.square().mean().sqrt().clamp_min(1e-8)


    #compute R value
    denom = di.square().sum().sqrt() * dj.square().sum().sqrt()
    r = float((di * dj).sum() / denom) if denom > 0 else float("nan")

    return mean_i, std_i, mean_j, std_j, r, n


# 3) Plot: one row per pair, RAW panel only

def plot_columns(columns, pairs):
    n_pairs = len(pairs)
    fig, axes = plt.subplots(n_pairs, 1, figsize=(3.8, 3.4 * n_pairs), squeeze=False)

    for row in range(n_pairs):
        i, j = pairs[row]
        zi = columns[i].to(device)
        zj = columns[j].to(device)

        mean_i, std_i, mean_j, std_j, r, coact_count = joint_support_stats(zi, zj)

        # Drop ONLY the exact (0,0) origin; keep every other point.
        keep = (zi != 0) | (zj != 0)
        n_plotted = int(keep.sum().item())
        total_tokens = int(zi.numel())
        print(
            f"pair (z_{i}, z_{j}): plotted {n_plotted} points "
            f"(co-active={coact_count}, total tokens={total_tokens})"
        )

        raw_x = zi[keep].cpu().numpy()
        raw_y = zj[keep].cpu().numpy()
        ax_raw = axes[row][0]
        ax_raw.plot(raw_x, raw_y, "o", ms=0.1, alpha=0.3)

        ax_raw.axhline(0, lw=0.6, color="0.55")
        ax_raw.axvline(0, lw=0.6, color="0.55")
        ax_raw.set_xlabel(f"z_{i}")
        ax_raw.set_ylabel(f"z_{j}")
        ax_raw.set_title(f"z_{i} vs z_{j} -- RAW (co-act={coact_count}, on-supp r={r:+.2f})", fontsize=9)

    fig.suptitle(f"z_i vs z_j, raw (Arch={ARCH}, layer={LAYER}, k={SPARSITY})", y=1.0, fontsize=11)
    fig.tight_layout()

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = (
        REPO_ROOT
        / "figures"
        / f"zizj_scatter_{ARCH}_layer{LAYER}_k{SPARSITY}_{ts}.png"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved -> {out_path}")
    return out_path


def main():
    args = parse_args()
    if not args.save and not args.plot:
        raise SystemExit("Nothing to do -- pass --save and/or --plot")

    columns = None

    # plot-only against saved columns: never load the model or the SAE
    if args.plot and not args.save:
        columns = load_columns(wanted_latents(PAIRS), args.z_dir)

    if columns is None:
        columns = collect_columns(
            PAIRS, args.n_batches, args.batch_size, args.context_size
        )

    if args.save:
        save_columns(columns, PAIRS, args.n_batches, args.batch_size, args.context_size)

    if args.plot:
        plot_columns(columns, PAIRS)


if __name__ == "__main__":
    main()

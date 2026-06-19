import os, glob, csv
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sae_lens import SAE

# Single source of truth for the registry — import, don't duplicate.
try:
    from infer_z import SAE_DATA
except ImportError:
    raise SystemExit("Point this import at your infer_z module, or paste SAE_DATA here.")

DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
MAX_POINTS  = 200_000          # sampled off-diagonal pairs, per sparsity, per panel
SEED        = 0
BASE_DIR    = "june17outputs"  # where infer_z wrote the timestamped run folders
FIG_DIR     = "figures/grams"
SUMMARY_CSV = os.path.join(FIG_DIR, "spearman_summary.csv")

CLIP_XLIM   = True             # match reference: both x-columns pinned to [-1, 1]
YLIM        = (1e-1, 1e4)      # match reference y-range on every panel
SCATTER     = dict(s=0.3, alpha=0.03)   # tune alpha down if 3 overlaid clouds saturate
COLORS      = ["#1f77b4", "#ff7f0e", "#2ca02c", "#9467bd"]  # one per sparsity

# Which (layer, arch) figures to produce. Each makes ONE figure, sparsities overlaid.
FIGURES = [(layer, arch)
           for layer in (12, 19)
           for arch in ("relu", "topk", "jumprelu", "matryoshka")]

# Valérie's three y-quantities and two x-quantities, verbatim labels.
Y_LABELS = [r"$z_i^\top z_j$",
            r"$\frac{z_i^\top z_j}{\mathbf{1}_{z_i}^\top\mathbf{1}_{z_j}}$",
            r"$\mathbf{1}_{z_i}^\top \mathbf{1}_{z_j}$"]
X_LABELS = [r"$d_i^\top d_j$",
            r"$\frac{d_i^\top d_j}{\|d_i\|\|d_j\|}$"]
#   To match the 2-row render exactly, drop the middle entry of Y_LABELS
#   and the matching block in compute_quantities / the plot loop.


def find_z_dir(base, layer, arch, sparsity):
    """Newest run folder matching infer_z's naming: layer{L}_{arch}_l0{sp}_{ts}."""
    hits = sorted(glob.glob(os.path.join(base, f"layer{layer}_{arch}_l0{sparsity}_*")))
    return hits[-1] if hits else None


def load_decoder(layer, arch, sparsity, device):
    release, sae_id = SAE_DATA[layer][arch][sparsity]
    sae = SAE.from_pretrained(release=release, sae_id=sae_id, device=device)
    return sae.W_dec.float().detach().to(device)        # [d_sae, d_model], rows = atoms


def accumulate_z_grams(z_dir, p, device):
    """Stream shards; sum ZtZ and the L0 co-activation count. Never hold all of Z."""
    shards = sorted(glob.glob(os.path.join(z_dir, "Z_shard*.pt")))
    if not shards:
        raise FileNotFoundError(f"no Z_shard*.pt under {z_dir}")
    ZtZ, ZtZ_l0 = torch.zeros(p, p, device=device), torch.zeros(p, p, device=device)
    col_act = torch.zeros(p, device=device)
    for s in shards:
        z = torch.load(s, map_location="cpu").float().to(device)   # [n, p], bare tensor
        ind = (z != 0).float()
        ZtZ    += z.T @ z
        ZtZ_l0 += ind.T @ ind
        col_act += ind.sum(0)
        del z, ind
        if device == "cuda":
            torch.cuda.empty_cache()
    return ZtZ, ZtZ_l0, col_act


def sample_pairs(q, n, device, seed=SEED):
    """Uniform off-diagonal (i>j) pairs without building the full q^2/2 index set."""
    if q * (q - 1) // 2 <= n:
        off = torch.tril_indices(q, q, offset=-1, device=device)
        return off[0], off[1]
    g = torch.Generator(device=device).manual_seed(seed)
    i = torch.randint(0, q, (2 * n,), generator=g, device=device)
    j = torch.randint(0, q, (2 * n,), generator=g, device=device)
    m = i > j
    return i[m][:n], j[m][:n]


def decoder_pair_values(W, i, j, batch=100_000):
    """Raw d_i.d_j and cosine on the sampled pairs only (batched to cap memory)."""
    wn = W.norm(dim=1)
    raw = torch.empty(i.numel(), device=W.device)
    for s in range(0, i.numel(), batch):
        e = s + batch
        raw[s:e] = (W[i[s:e]] * W[j[s:e]]).sum(-1)
    return raw, raw / (wn[i] * wn[j])


def spearman(x, y):
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    if x.size < 2:
        return float("nan")
    rx, ry = np.argsort(np.argsort(x)), np.argsort(np.argsort(y))
    return float(np.corrcoef(rx, ry)[0, 1])


def compute_quantities(layer, arch, sparsity):
    """Returns (xs, ys, meta) for one SAE, or None if its shards are missing."""
    z_dir = find_z_dir(BASE_DIR, layer, arch, sparsity)
    if z_dir is None:
        print(f"  [skip] no run folder for layer{layer}_{arch}_l0{sparsity}")
        return None

    W = load_decoder(layer, arch, sparsity, DEVICE)          # [d_sae, d_model]
    p = W.shape[0]
    ZtZ, ZtZ_l0, col_act = accumulate_z_grams(z_dir, p, DEVICE)

    keep = col_act > 0                                       # drop dead latents both sides
    ZtZ, ZtZ_l0, W = ZtZ[keep][:, keep], ZtZ_l0[keep][:, keep], W[keep]
    q = int(keep.sum())

    i, j = sample_pairs(q, MAX_POINTS, DEVICE)               # one idx, shared across panels
    ztz, zl0 = ZtZ[i, j], ZtZ_l0[i, j]
    del ZtZ, ZtZ_l0
    if DEVICE == "cuda":
        torch.cuda.empty_cache()

    raw, cos = decoder_pair_values(W, i, j)
    ratio = ztz / zl0                                        # 0/0 -> nan, scatter skips it

    xs = [raw.cpu().numpy(), cos.cpu().numpy()]
    ys = [ztz.cpu().numpy(), ratio.cpu().numpy(), zl0.cpu().numpy()]
    n_clip = int(np.sum(np.abs(xs[0]) > 1.0))               # raw pairs outside [-1, 1]
    meta = dict(q=q, n_dead=int((~keep).sum()), n_pairs=int(i.numel()),
                n_clip=n_clip, z_dir=z_dir)
    return xs, ys, meta


def make_figure(layer, arch, writer):
    sparsities = sorted(SAE_DATA[layer][arch].keys(), key=int)
    fig, axes = plt.subplots(len(Y_LABELS), len(X_LABELS),
                             figsize=(8, 11), constrained_layout=True, squeeze=False)
    drawn = 0
    for k, sp in enumerate(sparsities):
        res = compute_quantities(layer, arch, sp)
        if res is None:
            continue
        xs, ys, meta = res
        color = COLORS[k % len(COLORS)]
        label = f"l0={sp} (q={meta['q']})"
        for r_, y in enumerate(ys):
            for c_, x in enumerate(xs):
                axes[r_, c_].scatter(x, y, color=color, label=label if (r_ == 0 and c_ == 0) else None,
                                     **SCATTER)
        drawn += 1
        if meta["n_clip"]:
            frac = meta["n_clip"] / meta["n_pairs"]
            print(f"  l0={sp}: clipping {frac:.1%} of raw pairs outside [-1,1]"
                  + ("  -> consider CLIP_XLIM=False" if frac > 0.02 else ""))
        writer.writerow({
            "layer": layer, "arch": arch, "sparsity": sp,
            "q_alive": meta["q"], "n_dead": meta["n_dead"], "n_pairs": meta["n_pairs"],
            "n_clip_raw": meta["n_clip"], "seed": SEED,
            "rho_cos_ztz":   spearman(xs[1], ys[0]),
            "rho_cos_ratio": spearman(xs[1], ys[1]),
            "rho_cos_l0":    spearman(xs[1], ys[2]),
            "rho_raw_ztz":   spearman(xs[0], ys[0]),
        })

    if drawn == 0:
        plt.close(fig)
        return

    for r_ in range(len(Y_LABELS)):
        for c_ in range(len(X_LABELS)):
            ax = axes[r_, c_]
            ax.set_yscale("log")
            ax.set_ylim(*YLIM)
            if CLIP_XLIM:
                ax.set_xlim(-1, 1)
            ax.set_xlabel(X_LABELS[c_])
            ax.set_ylabel(Y_LABELS[r_])
            ax.grid(alpha=0.15)
    leg = axes[0, 0].legend(markerscale=20, framealpha=0.9, loc="lower right", fontsize=8)
    for h in leg.legend_handles:      # legend dots inherit alpha=0.03 otherwise
        h.set_alpha(1.0)
    fig.suptitle(f"{arch}  —  layer {layer}", fontsize=14)
    os.makedirs(FIG_DIR, exist_ok=True)
    path = os.path.join(FIG_DIR, f"grams_{arch}_L{layer}.png")
    fig.savefig(path, dpi=300)
    plt.close(fig)
    print(f"  -> {path}")


def main():
    os.makedirs(FIG_DIR, exist_ok=True)
    fields = ["layer","arch","sparsity","q_alive","n_dead","n_pairs","n_clip_raw","seed",
              "rho_cos_ztz","rho_cos_ratio","rho_cos_l0","rho_raw_ztz"]
    with open(SUMMARY_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for layer, arch in FIGURES:
            print(f"\n{arch} L{layer}")
            make_figure(layer, arch, writer)
    print(f"\nsummary -> {SUMMARY_CSV}")


if __name__ == "__main__":
    main()
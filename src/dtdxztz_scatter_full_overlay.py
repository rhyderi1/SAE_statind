import os, glob, csv
from collections import defaultdict
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sae_lens import SAE
from datetime import datetime

try:
    from infer_z import SAE_DATA
except ImportError:
    raise SystemExit("Point this import at your infer_z module, or paste SAE_DATA here.")

SCRIPT_NAME = "dtdxztz_scatter_full_overlay"
Z_BASE      = "data/Z"
CSV_PATH    = "config/params.csv"
MAX_POINTS  = 200_000
SEED        = 0
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
YLIM        = (1e-2, 1e5)
CLIP_XLIM   = True
SCATTER     = dict(s=0.3, alpha=0.03)
COLORS      = ["#1f77b4", "#ff7f0e", "#2ca02c", "#9467bd"]

Y_LABELS = [r"$z_i^\top z_j$",
            r"$\frac{z_i^\top z_j}{\mathbf{1}_{z_i}^\top\mathbf{1}_{z_j}}$",
            r"$\mathbf{1}_{z_i}^\top \mathbf{1}_{z_j}$"]
X_LABELS = [r"$d_i^\top d_j$",
            r"$\frac{d_i^\top d_j}{\|d_i\|\|d_j\|}$"]


def load_configs(csv_path):
    configs = []
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            configs.append({"layer": int(row["layer"]), "arch": row["arch"], "sparsity": row["sparsity"]})
    return configs


def group_by_figure(configs):
    """Returns dict (layer, arch) -> [sparsity, ...] sorted numerically."""
    groups = defaultdict(list)
    for cfg in configs:
        groups[(cfg["layer"], cfg["arch"])].append(cfg["sparsity"])
    return {k: sorted(v, key=int) for k, v in groups.items()}


def find_z_dir(layer, arch, sparsity):
    hits = sorted(glob.glob(os.path.join(Z_BASE, "*", f"acts_layer{layer}_{arch}_{sparsity}_*")))
    return hits[-1] if hits else None


def load_decoder(layer, arch, sparsity):
    release, sae_id = SAE_DATA[layer][arch][sparsity]
    sae = SAE.from_pretrained(release=release, sae_id=sae_id, device=DEVICE)
    return sae.W_dec.float().detach().to(DEVICE)   # [d_sae, d_model], rows = atoms


def accumulate_z_grams(z_dir, p):
    shards = sorted(glob.glob(os.path.join(z_dir, "Z_shard*.pt")))
    if not shards:
        raise FileNotFoundError(f"no Z_shard*.pt under {z_dir}")
    ZtZ     = torch.zeros(p, p, device=DEVICE)
    ZtZ_l0  = torch.zeros(p, p, device=DEVICE)
    col_act = torch.zeros(p, device=DEVICE)
    for s in shards:
        z   = torch.load(s, map_location="cpu", weights_only=True).float().to(DEVICE)
        ind = (z != 0).float()
        ZtZ    += z.T @ z
        ZtZ_l0 += ind.T @ ind
        col_act += ind.sum(0)
        del z, ind
        if DEVICE == "cuda":
            torch.cuda.empty_cache()
    return ZtZ, ZtZ_l0, col_act


def sample_pairs(q, n, seed=SEED):
    if q * (q - 1) // 2 <= n:
        off = torch.tril_indices(q, q, offset=-1, device=DEVICE)
        return off[0], off[1]
    g = torch.Generator(device=DEVICE).manual_seed(seed)
    i = torch.randint(0, q, (2 * n,), generator=g, device=DEVICE)
    j = torch.randint(0, q, (2 * n,), generator=g, device=DEVICE)
    m = i > j
    return i[m][:n], j[m][:n]


def decoder_pair_values(W, i, j, batch=100_000):
    wn  = W.norm(dim=1)
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
    """Returns (xs, ys, meta) for one SAE, or None if data/SAE_DATA entry is missing."""
    z_dir = find_z_dir(layer, arch, sparsity)
    if z_dir is None:
        print(f"  [skip] no Z dir for layer{layer}_{arch}_{sparsity}")
        return None

    if (layer not in SAE_DATA
            or arch not in SAE_DATA[layer]
            or sparsity not in SAE_DATA[layer][arch]):
        print(f"  [skip] layer{layer}/{arch}/{sparsity} not in SAE_DATA")
        return None

    W = load_decoder(layer, arch, sparsity)   # [d_sae, d_model]
    p = W.shape[0]
    ZtZ, ZtZ_l0, col_act = accumulate_z_grams(z_dir, p)

    keep = col_act > 0
    ZtZ, ZtZ_l0, W = ZtZ[keep][:, keep], ZtZ_l0[keep][:, keep], W[keep]
    q = int(keep.sum())

    i, j = sample_pairs(q, MAX_POINTS)
    ztz, zl0 = ZtZ[i, j], ZtZ_l0[i, j]
    del ZtZ, ZtZ_l0
    if DEVICE == "cuda":
        torch.cuda.empty_cache()

    raw, cos = decoder_pair_values(W, i, j)
    ratio    = ztz / zl0

    xs = [raw.cpu().numpy(), cos.cpu().numpy()]
    ys = [ztz.cpu().numpy(), ratio.cpu().numpy(), zl0.cpu().numpy()]
    n_clip = int(np.sum(np.abs(xs[0]) > 1.0))
    meta   = dict(q=q, n_dead=int((~keep).sum()), n_pairs=int(i.numel()),
                  n_clip=n_clip, z_dir=z_dir)
    return xs, ys, meta


def make_figure(layer, arch, sparsities, out_dir, writer):
    fig, axes = plt.subplots(len(Y_LABELS), len(X_LABELS),
                             figsize=(8, 11), constrained_layout=True, squeeze=False)
    drawn = 0
    for k, sp in enumerate(sparsities):
        res = compute_quantities(layer, arch, sp)
        if res is None:
            continue
        xs, ys, meta = res
        color = COLORS[k % len(COLORS)]
        label = f"l0={sp}  (q={meta['q']})"
        for r_, y in enumerate(ys):
            for c_, x in enumerate(xs):
                axes[r_, c_].scatter(x, y, color=color,
                                     label=label if (r_ == 0 and c_ == 0) else None,
                                     **SCATTER)
        drawn += 1
        if meta["n_clip"]:
            frac = meta["n_clip"] / meta["n_pairs"]
            print(f"  l0={sp}: {frac:.1%} of raw pairs outside [-1,1]"
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
    for h in leg.legend_handles:
        h.set_alpha(1.0)

    sp_str = ", ".join(sparsities)
    fig.suptitle(f"arch={arch}   layer={layer}   l0 ∈ {{{sp_str}}}",
                 fontsize=13)

    fname = f"layer{layer}_{arch}.png"
    fig.savefig(os.path.join(out_dir, fname), dpi=300)
    plt.close(fig)
    print(f"  -> {fname}")


def main():
    date_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir  = os.path.join("figures/scatterplots_1shard", SCRIPT_NAME, date_str)
    os.makedirs(out_dir, exist_ok=True)

    figures  = group_by_figure(load_configs(CSV_PATH))  # (layer, arch) -> [sparsities]
    summary_csv = os.path.join(out_dir, "spearman_summary.csv")
    fields = ["layer", "arch", "sparsity", "q_alive", "n_dead", "n_pairs",
              "n_clip_raw", "seed",
              "rho_cos_ztz", "rho_cos_ratio", "rho_cos_l0", "rho_raw_ztz"]

    with open(summary_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for (layer, arch), sparsities in figures.items():
            print(f"\narch={arch}  layer={layer}  sparsities={sparsities}")
            make_figure(layer, arch, sparsities, out_dir, writer)

    print(f"\nFigures and summary saved to {out_dir}/")


if __name__ == "__main__":
    main()

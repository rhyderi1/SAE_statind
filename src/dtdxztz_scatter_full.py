"""DTD vs ZTZ scatter grid -- one 3x2 figure per SAE config.

For every row of config/params.csv, streams that config's Z shards into ZtZ and
the co-activation count matrix, samples up to MAX_POINTS off-diagonal latent
pairs, and plots three decoder-geometry y-quantities against two x-quantities:

    y: z_i^T z_j  |  z_i^T z_j / co-activation count  |  co-activation count
    x: d_i^T d_j  |  cosine(d_i, d_j)

Dead latents (never active in the corpus) are dropped first. Also writes
spearman_summary.csv with rank correlations per config. Output goes to
figures/dtdxztz_scatter_full/<timestamp>/.
"""

import os, glob, csv
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
    raise SystemExit("Set this import to your infer_z module name, or paste SAE_DATA in directly.")

SCRIPT_NAME = "dtdxztz_scatter_full"
Z_BASE      = "data/Z"
CSV_PATH    = "config/params.csv"
MAX_POINTS  = 300_000
SEED        = 0
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
YLIM        = (1e-2, 1e5)


def load_configs(csv_path):
    configs = []
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            configs.append({"layer": int(row["layer"]), "arch": row["arch"], "sparsity": row["sparsity"]})
    return configs


def find_z_dir(layer, arch, sparsity):
    hits = sorted(glob.glob(os.path.join(Z_BASE, "*", f"acts_layer{layer}_{arch}_{sparsity}_*")))
    return hits[-1] if hits else None


def load_decoder(layer, arch, sparsity):
    release, sae_id = SAE_DATA[layer][arch][sparsity]
    sae = SAE.from_pretrained(release=release, sae_id=sae_id, device=DEVICE)
    return sae.W_dec.float().detach().to(DEVICE)   # [d_sae, d_model], rows = atoms


def accumulate_z_grams(z_dir, p):
    """Stream all shards; accumulate ZtZ and L0 co-activation counts."""
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
    g = torch.Generator(device=DEVICE).manual_seed(seed)
    n_all = q * (q - 1) // 2
    if n_all <= n:
        off = torch.tril_indices(q, q, offset=-1, device=DEVICE)
        return off[0], off[1]
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
    cos = raw / (wn[i] * wn[j])
    return raw, cos


def spearman(x, y):
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    if x.size < 2:
        return float("nan")
    rx = np.argsort(np.argsort(x))
    ry = np.argsort(np.argsort(y))
    return float(np.corrcoef(rx, ry)[0, 1])


def run_config(layer, arch, sparsity, out_dir, writer):
    z_dir = find_z_dir(layer, arch, sparsity)
    if z_dir is None:
        print(f"  [skip] no Z dir for layer{layer}_{arch}_{sparsity}")
        return

    if (layer not in SAE_DATA
            or arch not in SAE_DATA[layer]
            or sparsity not in SAE_DATA[layer][arch]):
        print(f"  [skip] layer{layer}/{arch}/{sparsity} not in SAE_DATA")
        return

    W = load_decoder(layer, arch, sparsity)   # [d_sae, d_model]
    p = W.shape[0]

    ZtZ, ZtZ_l0, col_act = accumulate_z_grams(z_dir, p)

    keep = col_act > 0
    n_dead = int((~keep).sum())
    ZtZ, ZtZ_l0, W = ZtZ[keep][:, keep], ZtZ_l0[keep][:, keep], W[keep]
    q = int(keep.sum())

    i, j   = sample_pairs(q, MAX_POINTS)
    ztz    = ZtZ[i, j]
    zl0    = ZtZ_l0[i, j]
    del ZtZ, ZtZ_l0
    if DEVICE == "cuda":
        torch.cuda.empty_cache()

    raw, cos = decoder_pair_values(W, i, j)
    ratio    = ztz / zl0   # nan where pair never co-activated

    Zg, Zl0, R = ztz.cpu().numpy(), zl0.cpu().numpy(), ratio.cpu().numpy()
    Xraw, Xcos = raw.cpu().numpy(), cos.cpu().numpy()

    ys = [(Zg,  r"$z_i^\top z_j$"),
          (R,   r"$z_i^\top z_j\,/\,\mathbf{1}_{z_i}^\top\mathbf{1}_{z_j}$"),
          (Zl0, r"$\mathbf{1}_{z_i}^\top \mathbf{1}_{z_j}$")]
    xs = [(Xraw, r"$d_i^\top d_j$",                              None),
          (Xcos, r"$d_i^\top d_j\,/\,\|d_i\|\|d_j\|$",          (-1, 1))]

    fig, axes = plt.subplots(3, 2, figsize=(8, 11), constrained_layout=True)
    for r_, (y, ylab) in enumerate(ys):
        for c_, (x, xlab, xlim) in enumerate(xs):
            ax = axes[r_, c_]
            ax.scatter(x, y, s=0.2, alpha=0.02)
            ax.set_yscale("log")
            ax.set_ylim(*YLIM)
            if xlim:
                ax.set_xlim(*xlim)
            ax.set_xlabel(xlab)
            ax.set_ylabel(ylab)
            ax.grid(alpha=0.15)

    fig.suptitle(f"arch={arch}   layer={layer}   l0={sparsity}"
                 f"\n(q={q} alive, {n_dead} dead, {i.numel()} pairs)",
                 fontsize=11)

    fname = f"layer{layer}_{arch}_l0{sparsity}.png"
    fig.savefig(os.path.join(out_dir, fname), dpi=300)
    plt.close(fig)
    print(f"  -> {fname}  (q={q}, dead={n_dead})")

    writer.writerow({
        "layer": layer, "arch": arch, "sparsity": sparsity,
        "d_sae": p, "q_alive": q, "n_dead": n_dead,
        "n_pairs": int(i.numel()), "seed": SEED,
        "rho_cos_ztz":   spearman(Xcos, Zg),
        "rho_cos_ratio": spearman(Xcos, R),
        "rho_cos_l0":    spearman(Xcos, Zl0),
        "rho_raw_ztz":   spearman(Xraw, Zg),
    })


def main():
    date_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir  = os.path.join("figures", SCRIPT_NAME, date_str)
    os.makedirs(out_dir, exist_ok=True)

    summary_csv = os.path.join(out_dir, "spearman_summary.csv")
    fields = ["layer", "arch", "sparsity", "d_sae", "q_alive", "n_dead", "n_pairs", "seed",
              "rho_cos_ztz", "rho_cos_ratio", "rho_cos_l0", "rho_raw_ztz"]

    with open(summary_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for cfg in load_configs(CSV_PATH):
            layer, arch, sp = cfg["layer"], cfg["arch"], cfg["sparsity"]
            print(f"arch={arch}  layer={layer}  l0={sp}")
            run_config(layer, arch, sp, out_dir, writer)

    print(f"\nFigures and summary saved to {out_dir}/")


if __name__ == "__main__":
    main()

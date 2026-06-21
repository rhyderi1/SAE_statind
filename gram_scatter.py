import os, glob, csv
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sae_lens import SAE

# Pull the registry straight from infer_z so the two files never diverge.
# If your module isn't named infer_z.py, change this import (or paste SAE_DATA here).
try:
    from infer_z import SAE_DATA
except ImportError:
    raise SystemExit("Set this import to your infer_z module name, or paste SAE_DATA in directly.")

DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"
BASE_DIR   = "SAE_DATA_files"  # where infer_z wrote the timestamped run folders
#MAX_POINTS = 300_000           # sampled off-diagonal pairs per panel
SEED       = 0                 # reproducibility -> recorded in the summary CSV
FIG_DIR    = "figuresjune19/grams"
SUMMARY_CSV = "figuresjune19/grams/spearman_summary.csv"

# One entry per SAE. z_dir is the timestamped folder infer_z.py wrote, e.g.
#   june17outputs/layer12_relu_l020_20250617_143022
# (layer, arch, sparsity) must be a valid key path into SAE_DATA.
CONFIGS = [
    {"layer": 12, "arch": "relu",       "sparsity": "20", "z_dir": "/home/rhyderi1/projects/aip-bahtol/rhyderi1/sae_statind/june17outputs/layer12_relu_l020_20260617_161507/Z_shard000.pt"},
    {"layer": 12, "arch": "topk",       "sparsity": "20", "z_dir": "/home/rhyderi1/projects/aip-bahtol/rhyderi1/sae_statind/june17outputs/layer12_topk_l020_20260617_161513/Z_shard000.pt"},
    {"layer": 12, "arch": "jumprelu",   "sparsity": "22", "z_dir": "/home/rhyderi1/projects/aip-bahtol/rhyderi1/sae_statind/june17outputs/layer12_jumprelu_l022_20260617_161645/Z_shard000.pt"},
    {"layer": 12, "arch": "matryoshka", "sparsity": "40", "z_dir": "/home/rhyderi1/projects/aip-bahtol/rhyderi1/sae_statind/june17outputs/layer12_matryoshka_l040_20260617_161756/Z_shard000.pt"},
    {"layer": 19, "arch": "relu",       "sparsity": "20", "z_dir": "/home/rhyderi1/projects/aip-bahtol/rhyderi1/sae_statind/june17outputs/layer19_relu_l020_20260617_161800/Z_shard000.pt"},
    {"layer": 19, "arch": "topk",       "sparsity": "20", "z_dir": "/home/rhyderi1/projects/aip-bahtol/rhyderi1/sae_statind/june17outputs/layer19_topk_l020_20260617_161928/Z_shard000.pt"},
    {"layer": 19, "arch": "jumprelu",   "sparsity": "23", "z_dir": "/home/rhyderi1/projects/aip-bahtol/rhyderi1/sae_statind/june17outputs/layer19_jumprelu_l023_20260617_162032/Z_shard000.pt"},
    {"layer": 19, "arch": "matryoshka", "sparsity": "40", "z_dir": "/home/rhyderi1/projects/aip-bahtol/rhyderi1/sae_statind/june17outputs/layer19_matryoshka_l040_20260617_162128/Z_shard000.pt"},
    # ... layer 19 likewise
]


def load_decoder(layer, arch, sparsity, device):
    """Mirror get_sae() from infer_z, but keep only W_dec. [d_sae, d_model], rows = atoms."""
    release, sae_id = SAE_DATA[layer][arch][sparsity]
    sae = SAE.from_pretrained(release=release, sae_id=sae_id, device=device)  # single obj in your SAELens
    return sae.W_dec.float().detach().to(device)


def accumulate_z_grams(z_path, p, device):
    """Stream shards; sum ZtZ and the L0 co-activation count. Never hold all of Z at once.
    z_path may be a run directory (globs all Z_shard*.pt) or a single .pt file."""
    if os.path.isdir(z_path):
        shards = sorted(glob.glob(os.path.join(z_path, "Z_shard*.pt")))
    else:
        shards = [z_path]                       # a single shard file was given directly
    if not shards:
        raise FileNotFoundError(f"no Z_shard*.pt under {z_path}")

    ZtZ     = torch.zeros(p, p, device=device)
    ZtZ_l0  = torch.zeros(p, p, device=device)
    col_act = torch.zeros(p, device=device)
    for s in shards:
        z = torch.load(s, map_location="cpu").float().to(device)   # [n, p]
        ind = (z != 0).float()
        ZtZ    += z.T @ z
        ZtZ_l0 += ind.T @ ind
        col_act += ind.sum(0)
        del z, ind
        if device == "cuda":
            torch.cuda.empty_cache()
    return ZtZ, ZtZ_l0, col_act

# def sample_pairs(q, n, device, seed=SEED):
#     """Uniform off-diagonal (i>j) pairs. Equivalent to subsampling the full lower triangle,
#     but never builds the ~q^2/2 index tensor. Falls back to all pairs when q is small."""
#     g = torch.Generator(device=device).manual_seed(seed)
#     n_all = q * (q - 1) // 2
#     if n_all <= n:
#         off = torch.tril_indices(q, q, offset=-1, device=device)
#         return off[0], off[1]
#     i = torch.randint(0, q, (2 * n,), generator=g, device=device)
#     j = torch.randint(0, q, (2 * n,), generator=g, device=device)
#     m = i > j
#     i, j = i[m][:n], j[m][:n]
#     return i, j

def sample_pairs(q, device):
    """All off-diagonal (i>j) pairs."""
    off = torch.tril_indices(q, q, offset=-1, device=device)
    return off[0], off[1]


def decoder_pair_values(W, i, j, batch=100_000):
    """Raw d_i.d_j and cosine, computed only on the sampled pairs (batched to cap memory)."""
    wn = W.norm(dim=1)
    raw = torch.empty(i.numel(), device=W.device)
    for s in range(0, i.numel(), batch):
        e = s + batch
        raw[s:e] = (W[i[s:e]] * W[j[s:e]]).sum(-1)
    cos = raw / (wn[i] * wn[j])
    return raw, cos


def spearman(x, y):
    """Rank correlation, scipy-free. Drops non-finite pairs."""
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    if x.size < 2:
        return float("nan")
    rx = np.argsort(np.argsort(x))
    ry = np.argsort(np.argsort(y))
    return float(np.corrcoef(rx, ry)[0, 1])


def run_config(cfg, writer):
    layer, arch, sp = cfg["layer"], cfg["arch"], cfg["sparsity"]
    title = f"{arch} L{layer} l0={sp}"

    W = load_decoder(layer, arch, sp, DEVICE)        # [d_sae, d_model]
    p = W.shape[0]

    ZtZ, ZtZ_l0, col_act = accumulate_z_grams(cfg["z_dir"], p, DEVICE)

    keep = col_act > 0                               # drop dead latents (Z side and decoder side)
    n_dead = int((~keep).sum())
    ZtZ, ZtZ_l0, W = ZtZ[keep][:, keep], ZtZ_l0[keep][:, keep], W[keep]
    q = int(keep.sum())

    #i, j = sample_pairs(q, MAX_POINTS, DEVICE)
    i, j = sample_pairs(q, DEVICE)
    ztz  = ZtZ[i, j]
    zl0  = ZtZ_l0[i, j]
    del ZtZ, ZtZ_l0
    if DEVICE == "cuda":
        torch.cuda.empty_cache()

    raw, cos = decoder_pair_values(W, i, j)
    ratio = ztz / zl0                                # nan where the pair never co-activated

    # to numpy for plotting / stats
    Zg, Zl0, R = ztz.cpu().numpy(), zl0.cpu().numpy(), ratio.cpu().numpy()
    Xraw, Xcos = raw.cpu().numpy(), cos.cpu().numpy()

    ys = [(Zg,  r"$z_i^\top z_j$"),
          (R,   r"$z_i^\top z_j / \mathbf{1}_{z_i}^\top\mathbf{1}_{z_j}$"),
          (Zl0, r"$\mathbf{1}_{z_i}^\top \mathbf{1}_{z_j}$")]
    xs = [(Xraw, r"$d_i^\top d_j$",                         None),       # autoscale: not unit-norm
          (Xcos, r"$d_i^\top d_j / \|d_i\|\|d_j\|$",        (-1, 1))]    # cosine: bounded

    fig, axes = plt.subplots(3, 2, figsize=(8, 11), constrained_layout=True)
    for r_, (y, ylab) in enumerate(ys):
        for c_, (x, xlab, xlim) in enumerate(xs):
            ax = axes[r_, c_]
            ax.scatter(x, y, s=0.2, alpha=0.02)
            ax.set_yscale("log")
            if xlim:
                ax.set_xlim(*xlim)
            ax.set_xlabel(xlab); ax.set_ylabel(ylab); ax.grid(alpha=0.15)
            ax.set_ylim(1e-2, 1e5)  # shared y across architectures
    axes[0, 0].set_title(title)
    os.makedirs(FIG_DIR, exist_ok=True)
    fig.savefig(os.path.join(FIG_DIR, title.replace(" ", "_").replace("=", "") + ".png"), dpi=300)
    plt.close(fig)

    writer.writerow({
        "layer": layer, "arch": arch, "sparsity": sp,
        "d_sae": p, "q_alive": q, "n_dead": n_dead,
        "n_pairs": int(i.numel()), "seed": SEED,
        "rho_cos_ztz":   spearman(Xcos, Zg),
        "rho_cos_ratio": spearman(Xcos, R),
        "rho_cos_l0":    spearman(Xcos, Zl0),
        "rho_raw_ztz":   spearman(Xraw, Zg),
    })
    print(f"{title}: q={q}, dead={n_dead}, pairs={i.numel()}")


def main():
    os.makedirs(FIG_DIR, exist_ok=True)
    fields = ["layer","arch","sparsity","d_sae","q_alive","n_dead","n_pairs","seed",
              "rho_cos_ztz","rho_cos_ratio","rho_cos_l0","rho_raw_ztz"]
    with open(SUMMARY_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for cfg in CONFIGS:
            run_config(cfg, writer)
    print(f"summary -> {SUMMARY_CSV}")


if __name__ == "__main__":
    main()
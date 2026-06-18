import torch
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from pathlib import Path

# ── Config ────────────────────────────────────────────────────────────────────
# Each entry: (label_in_plot, path_to_Z_shard_dir)
RUNS = [
    ("TopK",      Path("/home/rhyderi1/projects/aip-bahtol/rhyderi1/sae_statind/june16outputs/layer19_topk_l040_20260616_185630")),
    ("JumpReLU",  Path("/home/rhyderi1/projects/aip-bahtol/rhyderi1/sae_statind/june16outputs/layer19_jumprelu_l040_20260616_185739")),
    ("ReLU",      Path("/home/rhyderi1/projects/aip-bahtol/rhyderi1/sae_statind/june16outputs/layer19_relu_l040_20260616_185508")),
    ("Matryoshka",Path("/home/rhyderi1/projects/aip-bahtol/rhyderi1/sae_statind/june16outputs/layer19_matryoshka_l040_20260616_185809")),
]

PALETTE = {
    "TopK":       "#0084D1",
    "JumpReLU":   "#FF8C00",
    "ReLU":       "#53B480",
    "Matryoshka": "#CC0000",
}

OUT_PATH = "feature_distributions.png"
# ──────────────────────────────────────────────────────────────────────────────

def load_zhats(shard_dir: Path) -> torch.Tensor:
    """Concatenate all Z_shard*.pt files in a directory."""
    shards = sorted(shard_dir.glob("Z_shard*.pt"))
    if not shards:
        raise FileNotFoundError(f"No Z_shard*.pt files found in {shard_dir}")
    return torch.cat([torch.load(s, weights_only=True) for s in shards], dim=0)


# ── Build dataframe ───────────────────────────────────────────────────────────
data = []
for model_name, shard_dir in RUNS:
    print(f"Loading {model_name} from {shard_dir} ...")
    zhat = load_zhats(shard_dir)           # (n_tokens, d_sae)

    l0 = torch.sum(zhat != 0, dim=0).float()          # activation frequency per feature
    l1 = torch.sum(zhat,      dim=0) / l0.clamp(min=1e-8)  # mean activation intensity

    l0_np = l0.cpu().numpy()
    l1_np = l1.cpu().numpy()

    for i in range(len(l0_np)):
        if l0_np[i] > 0:   # skip dead features
            data.append({
                "log10_L0":  np.log10(l0_np[i]),
                "L1_div_L0": l1_np[i],
                "Model":     model_name,
            })

df = pd.DataFrame(data)


# ── Plot ──────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 7))
ax.tick_params(axis="x", direction="out")
ax.tick_params(axis="y", direction="out")
ax.spines["right"].set_visible(False)
ax.spines["top"].set_visible(False)

levels = np.linspace(0.1, 0.7, 3)
levels_fill = [*levels, 1.0]

for model_name, color in PALETTE.items():
    model_df = df[df["Model"] == model_name]
    if len(model_df) < 20:
        print(f"  Skipping {model_name}: not enough data")
        continue

    x = model_df["log10_L0"].values
    y = model_df["L1_div_L0"].values

    kde = gaussian_kde(np.vstack([x, y]))
    xgrid = np.linspace(x.min() - 1, x.max() + 0.2, 100)
    ygrid = np.linspace(y.min() - 0.3, y.max() + 0.2, 100)
    X, Y = np.meshgrid(xgrid, ygrid)
    Z = kde(np.vstack([X.ravel(), Y.ravel()])).reshape(X.shape)
    Z = Z / Z.max()

    for i in range(len(levels_fill) - 1):
        ax.contourf(X, Y, Z,
                    levels=[levels_fill[i], levels_fill[i+1]],
                    colors=[color],
                    alpha=0.1 + 0.2 * (i / (len(levels_fill) - 2)))
    ax.contour(X, Y, Z, levels=levels, colors=[color], linewidths=2)

ax.set_xlabel("Activation Frequency (log10)")
ax.set_ylabel("Average Activation Intensity")
ax.set_title("Comparison of Feature Distributions")
ax.set_xticks(range(5))
ax.set_xticklabels([f"$10^{{{i}}}$" for i in range(5)])
ax.grid(alpha=0.3)

for name, color in PALETTE.items():
    ax.plot([], [], color=color, label=name, linewidth=2)
ax.legend()

plt.tight_layout()
plt.savefig(OUT_PATH, dpi=300, bbox_inches='tight')
print(f"Saved to {OUT_PATH}")
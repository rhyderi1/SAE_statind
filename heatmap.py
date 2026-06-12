import torch
import matplotlib.pyplot as plt


device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using {device}")

Z = torch.load('acts_layer20_jumprelu_71_20260607_142834/Z_shard000.pt')
Z = Z.float().to(device)

# ── prep ──────────────────────────────────────────────────────────────
n = Z.shape[0]
Z_centered = Z - Z.mean(dim=0)

# ── compute ───────────────────────────────────────────────────────────
G = Z_centered.T @ Z_centered
G=G.cpu()                        # centered Gram matrix
C = G / (n - 1)                                      # covariance matrix
std = torch.sqrt(torch.diag(C)).clamp(min=1e-8)
R = C / torch.outer(std, std)                        # correlation matrix

# ── plot ──────────────────────────────────────────────────────────────
matrices = [
    (G.numpy(), 'Centered Gram Matrix $(Z-\mu)^\\top(Z-\mu)$', 'gram_heatmap.png',     'RdBu_r', None, None),
    (C.numpy(), 'Covariance Matrix',                            'cov_heatmap.png',      'RdBu_r', None, None),
    (R.numpy(), 'Correlation Matrix',                           'corr_heatmap.png',     'RdBu_r', -1,   1  ),
]

for mat, title, fname, cmap, vmin, vmax in matrices:
    plt.figure(figsize=(8, 7))
    im = plt.imshow(mat, cmap=cmap, vmin=vmin, vmax=vmax,
                    aspect='auto', interpolation='nearest')
    plt.colorbar(im)
    plt.xlabel('feature $j$')
    plt.ylabel('feature $i$')
    plt.title(title)
    plt.tight_layout()
    plt.savefig(fname, dpi=150)
    plt.close()
    print(f"Saved {fname}")
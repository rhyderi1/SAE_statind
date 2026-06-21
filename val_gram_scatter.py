import os
import sys

import matplotlib.pyplot as plt
import torch
import torchvision
from torchvision.transforms import v2

sys.path.append("/home/vcosta/projects/aip-bahtol/vcosta/inductive_saes")

from src.utils.metrics import norms
import src.models.architecture
import torchvision
from torchvision.transforms import v2

FOLDER = "models/shallow_saes_/MNIST/"

SAE = "TopK"

DATA_ROOT = "data"
VAL_RANGE = (50000, 60000)
BATCH_SIZE = 1000
DEVICE = "cpu"


def find_checkpoints(folder, sae_name):
    matches = [
        f
        for f in os.listdir(folder)
        if f.startswith(sae_name) and os.path.isdir(os.path.join(folder, f))
    ]
    if not matches:
        raise FileNotFoundError(
            f"No checkpoint found for prefix '{sae_name}' in {folder}"
        )
    return sorted(matches)


def build_val_loader(data_root, val_range, batch_size):
    transform = v2.Compose(
        [
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Lambda(lambda x: x.flatten()),
        ]
    )
    dataset = torchvision.datasets.MNIST(data_root, train=True, transform=transform)
    val_dataset = torch.utils.data.Subset(dataset, range(*val_range))
    return torch.utils.data.DataLoader(
        val_dataset, shuffle=False, batch_size=batch_size
    )


def load_sae(folder, filename, device):
    parts = filename.split("_")
    sae_class = getattr(src.models.architecture, parts[0])
    m = 784
    p = int(parts[1][1:])
    k = int(parts[2][1:])
    tied = "_ti_" in filename
    sae = sae_class({"m": m, "p": p, "k": k, "tied": tied}, device)
    path = os.path.join(folder, filename, "model.pt")
    sae.load_state_dict(torch.load(path, map_location=device))
    sae.eval()
    return sae


def compute_grams_off_diagonals(sae, val_loader, device):
    Z_list = []
    for x, _ in val_loader:
        with torch.no_grad():
            zhat = sae.encode(x.to(device))
        Z_list.append(zhat.detach())

    Z = torch.vstack(Z_list)
    dead = Z.sum(dim=0) == 0.0
    Z = Z[:, ~dead]

    # Z gram
    Z_gram = Z.T @ Z
    Z_gram.fill_diagonal_(0)

    # L0 indicator gram
    Z_ind = (Z != 0).float()
    Z_gram_l0 = Z_ind.T @ Z_ind
    Z_gram_l0.fill_diagonal_(0)

    D = sae.D.detach()[:, ~dead]
    D_gram = D.T @ D
    D_gram.fill_diagonal_(0)

    D = sae.D.detach()[:, ~dead]
    D_norm = D / norms(D)
    D_normalized_gram = D_norm.T @ D_norm
    D_normalized_gram.fill_diagonal_(0)

    off_diagonals = torch.tril_indices(row=D.shape[1], col=D.shape[1], offset=-1)

    return (
        Z_gram[*off_diagonals],
        Z_gram_l0[*off_diagonals],
        D_gram[*off_diagonals],
        D_normalized_gram[*off_diagonals],
    )


def sublplots(folders, axes, val_loader):

    for name in folders:
        sae = load_sae(FOLDER, name, DEVICE)
        Z_gram, Z_gram_l0, D_gram, D_normalized_gram = compute_grams_off_diagonals(
            sae, val_loader, DEVICE
        )

        axes[0, 0].scatter(D_gram, Z_gram, s=0.1, alpha=0.01)
        axes[0, 0].set_yscale("log")
        axes[0, 0].set_xlim(-1, 1)
        axes[0, 0].set_xlabel(r"$d_i^\top d_j$")
        axes[0, 0].set_ylabel(r"$z_i^\top z_j$")
        axes[0, 0].set_ylim(1e-1, 1e4)
        axes[0, 0].grid(alpha=0.15)

        axes[1, 0].scatter(D_gram, Z_gram / Z_gram_l0, s=0.1, alpha=0.01)
        axes[1, 0].set_yscale("log")
        axes[1, 0].set_xlim(-1, 1)
        axes[1, 0].set_xlabel(r"$d_i^\top d_j$")
        axes[1, 0].set_ylabel(
            r"$\frac{z_i^\top z_j}{\mathbf{1}_{z_i}^\top \mathbf{1}_{z_j}}$"
        )
        axes[1, 0].set_ylim(1e-1, 1e4)
        axes[1, 0].grid(alpha=0.15)

        axes[2, 0].scatter(D_gram, Z_gram_l0, s=0.1, alpha=0.01)
        axes[2, 0].set_yscale("log")
        axes[2, 0].set_xlim(-1, 1)
        axes[2, 0].set_xlabel(r"$d_i^\top d_j$")
        axes[2, 0].set_ylabel(r"$\mathbf{1}_{z_i}^\top \mathbf{1}_{z_j}$")
        axes[2, 0].set_ylim(1e-1, 1e4)
        axes[2, 0].grid(alpha=0.15)

        axes[0, 1].scatter(D_normalized_gram, Z_gram, s=0.1, alpha=0.01)
        axes[0, 1].set_yscale("log")
        axes[0, 1].set_xlim(-1, 1)
        axes[0, 1].set_xlabel(r"$\frac{d_i^\top d_j}{\|d_i\| \|d_j\|}$")
        axes[0, 1].set_ylabel(r"$z_i^\top z_j$")
        axes[0, 1].set_ylim(1e-1, 1e4)
        axes[0, 1].grid(alpha=0.15)

        axes[1, 1].scatter(D_normalized_gram, Z_gram / Z_gram_l0, s=0.1, alpha=0.01)
        axes[1, 1].set_yscale("log")
        axes[1, 1].set_xlim(-1, 1)
        axes[1, 1].set_xlabel(r"$\frac{d_i^\top d_j}{\|d_i\| \|d_j\|}$")
        axes[1, 1].set_ylabel(
            r"$\frac{z_i^\top z_j}{\mathbf{1}_{z_i}^\top \mathbf{1}_{z_j}}$"
        )
        axes[1, 1].set_ylim(1e-1, 1e4)
        axes[1, 1].grid(alpha=0.15)

        axes[2, 1].scatter(D_normalized_gram, Z_gram_l0, s=0.1, alpha=0.01)
        axes[2, 1].set_yscale("log")
        axes[2, 1].set_xlim(-1, 1)
        axes[2, 1].set_xlabel(r"$\frac{d_i^\top d_j}{\|d_i\| \|d_j\|}$")
        axes[2, 1].set_ylabel(r"$\mathbf{1}_{z_i}^\top \mathbf{1}_{z_j}$")
        axes[2, 1].set_ylim(1e-1, 1e4)
        axes[2, 1].grid(alpha=0.15)


def main():
    val_loader = build_val_loader(DATA_ROOT, VAL_RANGE, BATCH_SIZE)
    folders = find_checkpoints(FOLDER, SAE)

    fig = plt.figure(constrained_layout=True, figsize=(15, 15))
    subfigs = fig.subfigures(2, 3, wspace=0.1, hspace=0.1)

    # tied norm
    axs = subfigs[0, 0].subplots(3, 2)
    sublplots(folders[21:24], axs, val_loader)
    subfigs[0, 0].suptitle("tied norm")

    # tied
    axs = subfigs[0, 1].subplots(3, 2)
    sublplots(folders[12:15], axs, val_loader)
    subfigs[0, 1].suptitle("tied")

    # tied norm benc
    axs = subfigs[1, 0].subplots(3, 2)
    sublplots(folders[18:21], axs, val_loader)
    subfigs[1, 0].suptitle("tied norm benc")

    # tied benc
    axs = subfigs[1, 1].subplots(3, 2)
    sublplots(folders[15:18], axs, val_loader)
    subfigs[1, 1].suptitle("tied benc")

    # untied norm
    axs = subfigs[0, 2].subplots(3, 2)
    sublplots(folders[9:12], axs, val_loader)
    subfigs[0, 2].suptitle("untied norm ")

    # untied norm benc
    axs = subfigs[1, 2].subplots(3, 2)
    sublplots(folders[6:9], axs, val_loader)
    subfigs[1, 2].suptitle("untied norm benc")

    folder = "figures/grams"
    os.makedirs(folder, exist_ok=True)

    plt.savefig(f"{folder}/{SAE}.png", dpi=300)


if __name__ == "__main__":
    main()

"""
Compute the accumulated indicator Gram matrix G = sum_r ind_r^T ind_r
where ind_r = (Z_r != 0).float()  — the binary co-activation count matrix.

Also saves ind^T ind from the first shard alone for comparison.
No scaling is applied.
"""

import os
import argparse
import torch
from datetime import datetime


def parse_args():
    parser = argparse.ArgumentParser(description="Accumulate indicator Gram matrix over Z shards.")
    parser.add_argument("--z_dir",     type=str, required=True,
                        help="Directory containing Z_shard*.pt files")
    parser.add_argument("--num_shards", type=int, default=93,
                        help="Number of shards to use (indices 0..num_shards-1)")
    parser.add_argument("--shard_ids", type=int, nargs="+", default=None,
                        help="Explicit shard indices to use (overrides --num_shards)")
    parser.add_argument("--out_dir",   type=str, default=None,
                        help="Directory to save outputs (default: results/ind_gram/<timestamp>)")
    return parser.parse_args()


def main():
    args = parse_args()

    shard_ids = args.shard_ids if args.shard_ids is not None else list(range(args.num_shards))
    R = len(shard_ids)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.out_dir is None:
        args.out_dir = os.path.join("results", "ind_gram", ts)
    os.makedirs(args.out_dir, exist_ok=True)

    G = None           # accumulated indicator gram, shape (p, p)
    ind_first = None   # ind^T ind from shard_ids[0] for comparison
    B_first = None

    print(f"z_dir    : {args.z_dir}")
    print(f"shards   : {shard_ids[0]}..{shard_ids[-1]}  (R={R})")
    print(f"out_dir  : {args.out_dir}")
    print(f"scaling  : none\n")

    for step, shard_idx in enumerate(shard_ids):
        path = os.path.join(args.z_dir, f"Z_shard{shard_idx:03d}.pt")
        Z = torch.load(path, weights_only=True).float()  # (B, p)
        B, p = Z.shape

        if G is None:
            G = torch.zeros(p, p, dtype=torch.float64)
            B_first = B
            print(f"  p={p}, B={B}")

        if B != B_first:
            print(f"  [warn] shard {shard_idx}: B={B} differs from first shard B={B_first}")

        ind = (Z != 0).double()                  # (B, p)
        ITI = ind.T @ ind                        # (p, p)

        if step == 0:
            ind_first = ITI.float()

        G += ITI

        if (step + 1) % 10 == 0 or step == 0:
            print(f"  shard {shard_idx:3d}  ({step+1}/{R})")

    G = G.float()

    gram_path  = os.path.join(args.out_dir, f"ind_gram_G_{ts}.pt")
    ind1_path  = os.path.join(args.out_dir, f"ind_gram_shard{shard_ids[0]:03d}_{ts}.pt")

    torch.save(G, gram_path)
    torch.save(ind_first, ind1_path)

    print(f"\nSaved ind gram G         -> {gram_path}  {tuple(G.shape)}")
    print(f"Saved ind gram (shard 0) -> {ind1_path}  {tuple(ind_first.shape)}")

    diag = G.diagonal()
    off  = G.tril(diagonal=-1).flatten()
    print(f"\nG diag    : min={diag.min():.4g}  max={diag.max():.4g}  mean={diag.mean():.4g}")
    print(f"G off-diag: min={off.min():.4g}  max={off.max():.4g}  mean={off.mean():.4g}")


if __name__ == "__main__":
    main()

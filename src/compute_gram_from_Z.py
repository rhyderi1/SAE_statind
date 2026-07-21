"""Accumulate the mean Gram matrix from Z shards already on disk.

G = sum_r (Z_r^T Z_r) / (B * R) over R shards of B tokens each, plus the
unscaled Z^T Z of shard 000 alone for comparison. Outputs to
results/gram/<timestamp>/.

The 1/(B*R) scaling is what distinguishes this from the raw-count accumulation
in compute_ind_gram.py.
"""

import os, argparse
import torch
from datetime import datetime


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--z_dir",      type=str, required=True)
    p.add_argument("--num_shards", type=int, default=93)
    p.add_argument("--out_dir",    type=str, default=None)
    return p.parse_args()


def main():
    args = parse_args()
    R  = args.num_shards
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = args.out_dir or os.path.join("results", "gram", ts)
    os.makedirs(out_dir, exist_ok=True)

    G         = None
    ztz_first = None

    for i in range(R):
        Z = torch.load(os.path.join(args.z_dir, f"Z_shard{i:03d}.pt"), weights_only=True).float()
        B, p = Z.shape
        if G is None:
            G = torch.zeros(p, p)
        ZTZ = Z.T @ Z
        if i == 0:
            ztz_first = ZTZ
        G += ZTZ / (B * R)

    torch.save(G,         os.path.join(out_dir, f"gram_{ts}.pt"))
    torch.save(ztz_first, os.path.join(out_dir, f"ztz_shard000_{ts}.pt"))
    print(f"Saved gram {tuple(G.shape)}      -> {out_dir}/gram_{ts}.pt")
    print(f"Saved ZTZ shard 0 {tuple(ztz_first.shape)} -> {out_dir}/ztz_shard000_{ts}.pt")


if __name__ == "__main__":
    main()

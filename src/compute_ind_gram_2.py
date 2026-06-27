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
    out_dir = args.out_dir or os.path.join("results", "ind_gram", ts)
    os.makedirs(out_dir, exist_ok=True)

    G         = None
    iti_first = None

    for i in range(R):
        Z = torch.load(os.path.join(args.z_dir, f"Z_shard{i:03d}.pt"), weights_only=True).float()
        B, p = Z.shape
        if G is None:
            G = torch.zeros(p, p)
        ind = (Z != 0).float()
        ITI = ind.T @ ind
        if i == 0:
            iti_first = ITI
        G += ITI / (B * R)

    torch.save(G,         os.path.join(out_dir, f"ind_gram_{ts}.pt"))
    torch.save(iti_first, os.path.join(out_dir, f"iti_shard000_{ts}.pt"))
    print(f"Saved ind_gram {tuple(G.shape)}      -> {out_dir}/ind_gram_{ts}.pt")
    print(f"Saved ITI shard 0 {tuple(iti_first.shape)} -> {out_dir}/iti_shard000_{ts}.pt")


if __name__ == "__main__":
    main()

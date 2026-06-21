"""
Step 2 — atom-mode membership (p = D_unit[k]).

Streams shards one at a time — never loads more than one shard of X+Z into
memory simultaneously (~3.7 GB per shard for layer-12 relu-16k SAE).

Run with:
    python -m src.absorption_membership.run_atom_mode --atom-k K

Convention record (Step 0):
  DTD: cosine d_i·d_j / (‖d_i‖‖d_j‖)   ZTZ: raw Z.T @ Z
  theta_fire=1e-8, tau=0.025, share_threshold=0.4  (from feature_absorption_calculator.py)
"""

import argparse
import logging

import torch
from sae_lens import SAE

from src.absorption_membership import config as cfg
from src.absorption_membership.membership import detect_absorbers_atom_mode_streaming
from src.absorption_membership.io_schema import save_membership, membership_path

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--atom-k",   type=int,   default=None)
    p.add_argument("--layer",    type=int,   default=cfg.LAYER)
    p.add_argument("--arch",     type=str,   default=cfg.ARCH)
    p.add_argument("--sparsity", type=str,   default=cfg.SPARSITY)
    p.add_argument("--z-dir",    type=str,   default=str(cfg.Z_DIR))
    p.add_argument("--max-shards", type=int, default=None,
                   help="Limit number of shards (default: all). Each shard ~3.7 GB.")
    p.add_argument("--theta-fire",      type=float, default=cfg.THETA_FIRE)
    p.add_argument("--tau",             type=float, default=cfg.TAU)
    p.add_argument("--share-threshold", type=float, default=cfg.SHARE_THRESHOLD)
    p.add_argument("--min-support",     type=int,   default=cfg.MIN_SUPPORT)
    return p.parse_args()


def main():
    args = parse_args()

    atom_k = args.atom_k if args.atom_k is not None else cfg.ATOM_K
    if atom_k < 0:
        raise SystemExit(
            "atom_k not set. Either run run_probe_mode first to discover S_main, "
            "or pass --atom-k explicitly."
        )

    # ── Load decoder ─────────────────────────────────────────────────────────
    from infer_z import SAE_DATA
    device = "cuda" if torch.cuda.is_available() else "cpu"
    release, sae_id = SAE_DATA[args.layer][args.arch][args.sparsity]
    log.info(f"Loading SAE: {release} / {sae_id}")
    sae = SAE.from_pretrained(release=release, sae_id=sae_id, device=device)
    D      = sae.W_dec.float().detach().cpu()
    D_unit = D / (D.norm(dim=1, keepdim=True) + 1e-12)
    del sae

    # ── Stream shards for absorber detection ──────────────────────────────────
    log.info(f"Streaming shards from {args.z_dir}  (max_shards={args.max_shards})")
    s_main, support_counts, s_abs, n_pos = detect_absorbers_atom_mode_streaming(
        args.z_dir, D, D_unit, atom_k,
        theta_fire=args.theta_fire,
        tau=args.tau,
        share_threshold=args.share_threshold,
        min_support=args.min_support,
        max_shards=args.max_shards,
    )
    log.info(f"Done. n_pos_tokens={n_pos}, |S_abs|={len(s_abs)}")

    # ── Save artifact ─────────────────────────────────────────────────────────
    out_path = membership_path(cfg.RESULTS_DIR, args.arch, args.layer, cfg.CONCEPT_ID, "atom")
    save_membership(
        out_path,
        arch=args.arch,
        layer=args.layer,
        concept_id=cfg.CONCEPT_ID,
        mode="atom",
        atom_k=atom_k,
        s_main=s_main,
        s_abs=s_abs,
        support_counts=support_counts,
        theta_fire=args.theta_fire,
        tau=args.tau,
        share_threshold=args.share_threshold,
        min_support=args.min_support,
        n_pos_tokens=n_pos,
        dtd_convention=cfg.DTD_CONVENTION,
        ztz_convention=cfg.ZTZ_CONVENTION,
        probe_cos=None,
    )

    print(f"\n=== ATOM-MODE RESULTS ===")
    print(f"S_main = {s_main}")
    print(f"S_abs  = {s_abs}  (|S_abs|={len(s_abs)})")
    print(f"n_pos_tokens = {n_pos}")
    print(f"support_counts top-10: {sorted(support_counts.items(), key=lambda x:-x[1])[:10]}")


if __name__ == "__main__":
    main()

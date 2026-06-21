"""
Step 3 — probe-mode membership.

Memory strategy:
  - Probe training: loads only `--train-shards` shards (default 3, ~11 GB).
    150k tokens is ample for logistic regression.
  - Absorber detection: streams all shards one at a time (~3.7 GB peak per shard).

Two probes:
  (a) Direction probe: logistic regression on raw X → unit-normalized p.
  (b) K-sparse probe on Z → S_main (via F1-jump greedy criterion).

Convention record (Step 0):
  DTD: cosine d_i·d_j / (‖d_i‖‖d_j‖)   ZTZ: raw Z.T @ Z
  theta_fire=1e-8, tau=0.025, share_threshold=0.4  (from feature_absorption_calculator.py)
  Train/eval split is disjoint (acceptance check 2).
"""

import argparse
import logging

import numpy as np
import torch
from sae_lens import SAE
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

from src.absorption_membership import config as cfg
from src.absorption_membership.membership import (
    load_xz_shards,
    detect_absorbers_probe_mode_streaming,
)
from src.absorption_membership.io_schema import (
    save_membership, load_membership, membership_path,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--layer",    type=int,   default=cfg.LAYER)
    p.add_argument("--arch",     type=str,   default=cfg.ARCH)
    p.add_argument("--sparsity", type=str,   default=cfg.SPARSITY)
    p.add_argument("--z-dir",    type=str,   default=str(cfg.Z_DIR))
    p.add_argument("--train-shards", type=int, default=3,
                   help="Shards to load for probe training. 3 shards ≈ 150k tokens ≈ 11 GB.")
    p.add_argument("--train-frac",   type=float, default=0.8)
    p.add_argument("--k-max",    type=int,   default=5)
    p.add_argument("--f1-jump-threshold", type=float, default=0.03)
    p.add_argument("--theta-fire",      type=float, default=cfg.THETA_FIRE)
    p.add_argument("--tau",             type=float, default=cfg.TAU)
    p.add_argument("--share-threshold", type=float, default=cfg.SHARE_THRESHOLD)
    p.add_argument("--min-support",     type=int,   default=cfg.MIN_SUPPORT)
    p.add_argument("--seed",    type=int,   default=42)
    return p.parse_args()


def train_direction_probe(X_train: np.ndarray, y_train: np.ndarray) -> np.ndarray:
    """Logistic regression on raw X; returns unit-normalized weight vector."""
    lr = LogisticRegression(max_iter=1000, C=1.0, solver="lbfgs").fit(X_train, y_train)
    w = lr.coef_[0]
    return w / (np.linalg.norm(w) + 1e-12)


def greedy_k_sparse(
    Z_train: np.ndarray,
    y_train: np.ndarray,
    Z_eval: np.ndarray,
    y_eval: np.ndarray,
    *,
    k_max: int,
    f1_jump_threshold: float,
) -> list[int]:
    """
    Greedy k-sparse probing on SAE latents Z.
    Selects latents greedily while each addition gives >= f1_jump_threshold F1 gain.
    Evaluated on eval split only (acceptance check 2).
    """
    mean_act_pos = Z_eval[y_eval == 1].mean(0)
    ranked = np.argsort(-mean_act_pos)

    selected: list[int] = []
    prev_f1 = 0.0

    for j in ranked[:k_max * 5]:
        candidate = selected + [int(j)]
        feat_train = Z_train[:, candidate].sum(axis=1)
        feat_eval  = Z_eval[:, candidate].sum(axis=1)
        probe = LogisticRegression(max_iter=500, class_weight="balanced").fit(
            feat_train.reshape(-1, 1), y_train
        )
        f1 = f1_score(y_eval, probe.predict(feat_eval.reshape(-1, 1)), zero_division=0)

        if f1 >= prev_f1 + f1_jump_threshold:
            selected.append(int(j))
            prev_f1 = f1
            log.info(f"  k={len(selected)}: added latent {j}, eval F1={f1:.4f}")
            if len(selected) >= k_max:
                break
        elif len(selected) >= 1:
            break

    if not selected:
        selected = [int(ranked[0])]
        log.warning("No F1-jump found; falling back to top-activation latent.")

    return selected


def main():
    args = parse_args()

    # ── Load decoder ─────────────────────────────────────────────────────────
    from infer_z import SAE_DATA
    device = "cuda" if torch.cuda.is_available() else "cpu"
    release, sae_id = SAE_DATA[args.layer][args.arch][args.sparsity]
    log.info(f"Loading SAE: {release} / {sae_id}")
    sae = SAE.from_pretrained(release=release, sae_id=sae_id, device=device)
    D      = sae.W_dec.float().detach().cpu()
    D_unit = D / (D.norm(dim=1, keepdim=True) + 1e-12)
    del sae

    # ── Load small set of shards for probe training ───────────────────────────
    log.info(f"Loading {args.train_shards} shard(s) for probe training...")
    X_all, Z_all = load_xz_shards(args.z_dir, max_shards=args.train_shards)
    # X: [N, d_model], Z: [N, F]

    # ── Build concept labels ──────────────────────────────────────────────────
    # Check if atom-mode artifact exists to borrow its atom_k as anchor label
    atom_path = membership_path(cfg.RESULTS_DIR, args.arch, args.layer, cfg.CONCEPT_ID, "atom")
    if atom_path.exists():
        atom_art = load_membership(atom_path)
        anchor_k = atom_art["atom_k"]
        log.info(f"Using atom_k={anchor_k} from atom-mode artifact for labels.")
    else:
        # Fall back to latent with highest mean activation as proxy
        anchor_k = int(Z_all.mean(0).argmax().item())
        log.warning(f"No atom-mode artifact; using highest-mean latent {anchor_k} as proxy labels.")

    X_np = X_all.numpy()
    Z_np = Z_all.numpy()
    y_all = (Z_np[:, anchor_k] >= args.theta_fire).astype(np.int64)
    log.info(f"Labels: {y_all.sum()} positive / {len(y_all)} total")
    del X_all, Z_all

    # ── Train/eval split — MUST be disjoint (acceptance check 2) ─────────────
    idx = np.arange(len(y_all))
    idx_train, idx_eval = train_test_split(idx, train_size=args.train_frac, random_state=args.seed)
    X_train = X_np[idx_train];  X_eval = X_np[idx_eval]
    Z_train = Z_np[idx_train];  Z_eval = Z_np[idx_eval]
    y_train = y_all[idx_train]; y_eval = y_all[idx_eval]
    del X_np, Z_np, y_all
    log.info(f"Train: {len(idx_train)} ({y_train.sum()} pos)  "
             f"Eval: {len(idx_eval)} ({y_eval.sum()} pos)")

    # ── (a) Direction probe → p ───────────────────────────────────────────────
    log.info("Training direction probe on X_train...")
    p_np = train_direction_probe(X_train, y_train)
    p    = torch.tensor(p_np, dtype=torch.float32)
    assert abs(float(p.norm()) - 1.0) < 1e-4, "p not unit-normalized"

    probe_cos = float(D_unit[anchor_k] @ p)
    log.info(f"cos(d_{anchor_k}, p_probe) = {probe_cos:.4f}")

    # ── (b) K-sparse probing → S_main ────────────────────────────────────────
    log.info("Running greedy k-sparse probing (eval split)...")
    s_main = greedy_k_sparse(
        Z_train, y_train, Z_eval, y_eval,
        k_max=args.k_max,
        f1_jump_threshold=args.f1_jump_threshold,
    )
    log.info(f"S_main = {s_main}")

    # ── Absorber detection — stream ALL shards (acceptance check 2: eval only) ─
    # We stream the full dataset; per-token "positive" here is defined by
    # concept present (X·p > 0) AND main off, so no train/eval label needed.
    log.info(f"Streaming all shards for absorber detection from {args.z_dir}...")
    support_counts, s_abs, n_pos = detect_absorbers_probe_mode_streaming(
        args.z_dir, D, D_unit, p, s_main,
        theta_fire=args.theta_fire,
        tau=args.tau,
        share_threshold=args.share_threshold,
        min_support=args.min_support,
        max_shards=None,   # stream all
    )
    log.info(f"Done. n_pos_tokens={n_pos}, |S_abs|={len(s_abs)}")

    # ── Save artifact ─────────────────────────────────────────────────────────
    out_path = membership_path(cfg.RESULTS_DIR, args.arch, args.layer, cfg.CONCEPT_ID, "probe")
    save_membership(
        out_path,
        arch=args.arch,
        layer=args.layer,
        concept_id=cfg.CONCEPT_ID,
        mode="probe",
        atom_k=anchor_k,
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
        probe_cos=probe_cos,
    )

    # ── Validation diagnostics (acceptance check 6) ───────────────────────────
    print(f"\n=== PROBE-MODE RESULTS ===")
    print(f"S_main (probe)   = {s_main}")
    print(f"S_abs  (probe)   = {s_abs}  (|S_abs|={len(s_abs)})")
    print(f"cos(d_{anchor_k}, p_probe) = {probe_cos:.4f}")
    if atom_path.exists():
        a = load_membership(atom_path)
        atom_smain = set(a["S_main"]); atom_sabs = set(a["S_abs"])
        probe_smain = set(s_main);     probe_sabs = set(s_abs)
        print(f"\n|S_main(atom) ∩ S_main(probe)| = {len(atom_smain & probe_smain)}")
        print(f"|S_abs(atom)  ∩ S_abs(probe)|  = {len(atom_sabs & probe_sabs)}")
        print(f"|S_main(atom)|={len(atom_smain)}, |S_main(probe)|={len(probe_smain)}")
        print(f"|S_abs(atom)|={len(atom_sabs)},  |S_abs(probe)|={len(probe_sabs)}")
    else:
        print("[info] Run run_atom_mode first to get overlap diagnostics.")


if __name__ == "__main__":
    main()

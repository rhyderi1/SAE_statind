"""
Step 4 open question — which one-number summary best separates absorber pairs from control?

Candidates evaluated:
  dtd           — cosine DTD alone
  ztz           — raw ZTZ alone (log-transformed for ROC)
  dtd_x_ztz     — DTD × ZTZ
  dtd_p_ztz     — DTD + ZTZ (z-scored)
  abs_dtd_x_ztz — |DTD| × ZTZ
  log_dtd       — log(|DTD| + ε)
  log_ztz       — log(ZTZ + ε)
  ratio         — DTD / (ZTZ + ε)
  logistic2d    — 2-feature logistic regression on (DTD, log(ZTZ+ε)) [best-linear reference]

Separation metric: ROC-AUC distinguishing absorber pairs (label 1) vs control (label 0).

Convention record (Step 0 / gram_scatter.py):
  DTD: cosine d_i·d_j / (‖d_i‖‖d_j‖)   ZTZ: raw Z.T @ Z

Usage:
    python -m analysis.step4_summary_metric
"""

import glob
import json
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from infer_z import SAE_DATA
from gram_scatter import (
    load_decoder,
    accumulate_z_grams,
    decoder_pair_values,
    DEVICE,
    BASE_DIR,
)
from src.absorption_membership.io_schema import load_membership, membership_path
from src.absorption_membership import config as cfg

RESULTS_DIR = cfg.STEP4_DIR
os.makedirs(RESULTS_DIR, exist_ok=True)

EPS = 1e-8


def compute_summaries(dtd: np.ndarray, ztz: np.ndarray) -> dict[str, np.ndarray]:
    log_ztz = np.log(np.abs(ztz) + EPS)
    log_dtd = np.log(np.abs(dtd) + EPS)
    z = StandardScaler()
    dtd_z = z.fit_transform(dtd.reshape(-1, 1)).ravel()
    ztz_z = z.fit_transform(ztz.reshape(-1, 1)).ravel()
    return {
        "dtd":           dtd,
        "ztz":           ztz,
        "dtd_x_ztz":     dtd * ztz,
        "dtd_p_ztz":     dtd_z + ztz_z,
        "abs_dtd_x_ztz": np.abs(dtd) * ztz,
        "log_dtd":       log_dtd,
        "log_ztz":       log_ztz,
        "ratio":         dtd / (ztz + EPS),
    }


def main():
    for mode in ("atom", "probe"):
        path = membership_path(cfg.RESULTS_DIR, cfg.ARCH, cfg.LAYER, cfg.CONCEPT_ID, mode)
        if not path.exists():
            print(f"[skip] {mode}-mode artifact not found at {path}")
            continue

        art = load_membership(path)
        s_main = set(art["S_main"])
        s_abs  = set(art["S_abs"])

        if not s_abs:
            print(f"[skip] {mode}: S_abs is empty.")
            continue

        # ── Load pair values ──────────────────────────────────────────────────
        z_dir_hits = sorted(glob.glob(os.path.join(BASE_DIR, f"*layer{cfg.LAYER}_{cfg.ARCH}_l0{cfg.SPARSITY}_*")))
        if not z_dir_hits:
            print(f"[error] No run folder for {cfg.ARCH} L{cfg.LAYER} sp={cfg.SPARSITY}")
            continue
        z_dir = z_dir_hits[-1]

        W = load_decoder(cfg.LAYER, cfg.ARCH, cfg.SPARSITY, DEVICE)
        p = W.shape[0]
        ZtZ, _, col_act = accumulate_z_grams(z_dir, p, DEVICE)
        keep = col_act > 0
        ZtZ, W = ZtZ[keep][:, keep], W[keep]
        q = int(keep.sum())

        i_idx = torch.tril_indices(q, q, offset=-1, device=DEVICE)[0]
        j_idx = torch.tril_indices(q, q, offset=-1, device=DEVICE)[1]
        ztz_all = ZtZ[i_idx, j_idx].cpu().numpy()
        del ZtZ
        if DEVICE == "cuda":
            torch.cuda.empty_cache()
        _, cos_all = decoder_pair_values(W, i_idx, j_idx)
        cos_all = cos_all.cpu().numpy()
        i_np = i_idx.cpu().numpy()
        j_np = j_idx.cpu().numpy()

        alive_orig = torch.where(keep)[0].numpy()
        orig_to_alive = {int(orig): alive for alive, orig in enumerate(alive_orig)}

        abs_pairs: set[tuple[int, int]] = set()
        for j in s_abs:
            for m in s_main:
                if j != m and j in orig_to_alive and m in orig_to_alive:
                    a, b = orig_to_alive[j], orig_to_alive[m]
                    abs_pairs.add((min(a, b), max(a, b)))

        if not abs_pairs:
            print(f"[skip] {mode}: no absorber pairs survived remapping.")
            continue

        abs_mask = np.array([
            (int(i_np[n]), int(j_np[n])) in abs_pairs
            or (int(j_np[n]), int(i_np[n])) in abs_pairs
            for n in range(len(i_np))
        ])

        # Control: seeded random sample of non-absorber pairs
        rng = np.random.default_rng(cfg.CONTROL_SEED)
        ctrl_pool = np.where(~abs_mask)[0]
        n_abs = abs_mask.sum()
        n_ctrl = min(len(ctrl_pool), cfg.CONTROL_OVERSAMPLE * n_abs)
        ctrl_idx = rng.choice(ctrl_pool, size=n_ctrl, replace=False)

        abs_idx = np.where(abs_mask)[0]
        sel_idx = np.concatenate([abs_idx, ctrl_idx])
        labels  = np.concatenate([np.ones(n_abs, dtype=int), np.zeros(n_ctrl, dtype=int)])

        dtd = cos_all[sel_idx]
        ztz = ztz_all[sel_idx]

        # ── Compute summaries ─────────────────────────────────────────────────
        summaries = compute_summaries(dtd, ztz)

        # ── ROC-AUC per summary ───────────────────────────────────────────────
        aucs: dict[str, float] = {}
        for name, scores in summaries.items():
            fin = np.isfinite(scores)
            if fin.sum() < 10:
                aucs[name] = float("nan")
                continue
            try:
                aucs[name] = roc_auc_score(labels[fin], scores[fin])
            except Exception:
                aucs[name] = float("nan")

        # 2-feature logistic boundary on (DTD, log(ZTZ+ε))
        feat2d = np.stack([dtd, np.log(np.abs(ztz) + EPS)], axis=1)
        fin2d  = np.isfinite(feat2d).all(axis=1)
        if fin2d.sum() >= 20:
            lr = LogisticRegression(max_iter=1000, C=1.0).fit(
                feat2d[fin2d], labels[fin2d]
            )
            aucs["logistic2d"] = roc_auc_score(labels[fin2d], lr.predict_proba(feat2d[fin2d])[:, 1])
        else:
            aucs["logistic2d"] = float("nan")

        # ── Ranked table ─────────────────────────────────────────────────────
        ranked = sorted(aucs.items(), key=lambda x: -x[1] if np.isfinite(x[1]) else -999)
        print(f"\n=== {mode.upper()} — Summary metric AUC ranking ===")
        for name, auc in ranked:
            print(f"  {name:<20} AUC={auc:.4f}")

        best_name, best_auc = ranked[0]
        if best_name == "dtd_x_ztz":
            rec = "DTD × ZTZ is the best linear summary — supports the existing choice."
        else:
            rec = (
                f"'{best_name}' outperforms DTD×ZTZ (AUC {best_auc:.4f} vs "
                f"{aucs.get('dtd_x_ztz', float('nan')):.4f}). "
                "Consider revising the summary metric."
            )
        print(f"\nRecommendation: {rec}")

        # ── Save JSON ─────────────────────────────────────────────────────────
        out = {
            "mode": mode, "arch": cfg.ARCH, "layer": cfg.LAYER,
            "concept_id": cfg.CONCEPT_ID, "sparsity": cfg.SPARSITY,
            "n_abs": int(n_abs), "n_ctrl": int(n_ctrl),
            "auc_ranking": ranked,
            "recommendation": rec,
        }
        json_path = RESULTS_DIR / f"{cfg.ARCH}_L{cfg.LAYER}_{cfg.CONCEPT_ID}_{mode}_summary_metric.json"
        with open(json_path, "w") as f:
            json.dump(out, f, indent=2)
        print(f"  -> {json_path}")

        # ── Bar chart ─────────────────────────────────────────────────────────
        names_plot = [n for n, _ in ranked]
        aucs_plot  = [a if np.isfinite(a) else 0.0 for _, a in ranked]
        colors = ["#d62728" if n == "dtd_x_ztz" else "#1f77b4" for n in names_plot]

        fig, ax = plt.subplots(figsize=(9, 4))
        bars = ax.barh(names_plot[::-1], aucs_plot[::-1], color=colors[::-1])
        ax.axvline(0.5, color="k", ls="--", lw=1, label="random")
        ax.set_xlabel("ROC-AUC")
        ax.set_title(f"{cfg.ARCH} L{cfg.LAYER} — summary metric comparison ({mode} mode)")
        ax.set_xlim(0, 1.05)
        ax.legend(fontsize=8)
        fig.tight_layout()
        png_path = RESULTS_DIR / f"{cfg.ARCH}_L{cfg.LAYER}_{cfg.CONCEPT_ID}_{mode}_summary_metric.png"
        fig.savefig(png_path, dpi=150)
        plt.close(fig)
        print(f"  figure -> {png_path}")


if __name__ == "__main__":
    main()

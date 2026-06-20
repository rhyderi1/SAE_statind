"""
Extract S_main and S_abs latent indices for the DTDxZTZ statistical analysis.
Uses the SAEBench geometric projection method (label-free).

Step 3 (PROBE mode): Loads cached Parquet files from run_eval().
Step 2 (ATOM mode) : Runs FeatureAbsorptionCalculator directly bypassing k-sparse probing.
"""

import os
import json
import argparse
from collections import Counter
from pathlib import Path

import pandas as pd
import torch

from sae_bench.evals.absorption.feature_absorption_calculator import (
    FeatureAbsorptionCalculator,
    AbsorptionResults,
)
from sae_bench.evals.absorption.vocab import get_alpha_tokens
from sae_bench.evals.absorption.prompting import first_letter_formatter

EPS = 1e-8


# --------------------------------------------------------------------------- #
# STEP 3: PROBE mode -- read the cached parquet, no recomputation.
# --------------------------------------------------------------------------- #
def extract_membership_from_cached_df(df: pd.DataFrame, letter: str) -> dict:
    """
    df is the parquet written by calculate_projection_and_cos_sims. 
    Filter to one letter and pull membership.
    """
    sub = df[df["letter"] == letter]
    if len(sub) == 0:
        raise ValueError(f"No rows for letter={letter!r} in cached results.")

    s_main = sub["split_feats"].iloc[0]  
    s_abs_counts: Counter = Counter()
    pairs: list[tuple[int, int, str]] = []

    confirmed = sub[sub["is_full_absorption"]]
    for _, row in confirmed.iterrows():
        absorber_i = int(row["top_projection_feat"])
        
        silent_mains = [
            fid
            for fid, act in zip(row["split_feats"], row["split_feat_acts"])
            if act < EPS
        ]
        for absorbed_j in silent_mains:
            pairs.append((absorber_i, absorbed_j, row["token"]))
            s_abs_counts[absorber_i] += 1

    return {
        "s_main": list(s_main),
        "s_abs_counts": dict(s_abs_counts),
        "s_abs": sorted(s_abs_counts.keys()),
        "pairs": pairs,
        "n_words_total": len(sub),
        "n_words_absorbed": len(confirmed),
    }


def load_cached_probe_results(sae_name: str, layer: int, results_dir: str) -> pd.DataFrame:
    path = Path(results_dir) / sae_name / f"layer_{layer}_{sae_name}.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Ensure run_eval() has finished and --results-dir is correct."
        )
    return pd.read_parquet(path)


# --------------------------------------------------------------------------- #
# STEP 2: ATOM mode -- call calculate_absorption directly with p = W_dec[k].
# --------------------------------------------------------------------------- #
def extract_membership_from_atom_run(results: AbsorptionResults) -> dict:
    """Extracts S_main and S_abs from a live AbsorptionResults object."""
    s_main = list(results.main_feature_ids)
    s_abs_counts: Counter = Counter()
    pairs: list[tuple[int, int, str]] = []

    for r in results.word_results:
        if not r.is_full_absorption:
            continue
        absorber_i = r.top_projection_feature_scores[0].feature_id
        silent_mains = [
            s.feature_id for s in r.main_feature_scores if s.activation < EPS
        ]
        for absorbed_j in silent_mains:
            pairs.append((absorber_i, absorbed_j, r.word))
            s_abs_counts[absorber_i] += 1

    return {
        "s_main": s_main,
        "s_abs_counts": dict(s_abs_counts),
        "s_abs": sorted(s_abs_counts.keys()),
        "pairs": pairs,
        "n_words_total": len(results.word_results),
        "n_words_absorbed": sum(1 for r in results.word_results if r.is_full_absorption),
    }


def run_atom_mode(
    calculator: FeatureAbsorptionCalculator,
    sae,
    *,
    atom_k: int,
    words: list[str],
    layer: int,
) -> AbsorptionResults:
    atom_dir = sae.W_dec[atom_k].detach().float()  # calculator unit-normalizes internally
    return calculator.calculate_absorption(
        sae,
        words=words,
        probe_direction=atom_dir,
        main_feature_ids=[atom_k],   # S_main = {k}, converges in 1 step
        layer=layer,
        show_progress=True,
    )


def save_membership(membership: dict, path: str, meta: dict) -> None:
    out = {"meta": meta, **membership}
    out["pairs"] = [list(p) for p in membership["pairs"]]
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[*] Wrote {path} "
          f"(|S_main|={len(membership['s_main'])}, |S_abs|={len(membership['s_abs'])}, "
          f"{membership['n_words_absorbed']}/{membership['n_words_total']} words absorbed)")


# --------------------------------------------------------------------------- #
def main():
    parser = argparse.ArgumentParser(description="Extract S_main and S_abs from SAEs")
    parser.add_argument("--letter", type=str, default="S", help="The concept letter to evaluate")
    parser.add_argument("--layer", type=int, default=3, help="Layer number")
    parser.add_argument("--sae-name", type=str, default="gemma-scope-2b-pt-res_layer_3_width_16k_average_l0_59", help="Name of the SAE")
    parser.add_argument("--sae-release", type=str, default="gemma-scope-2b-pt-res", help="SAE release group")
    parser.add_argument("--sae-id", type=str, default="layer_3/width_16k/average_l0_59", help="SAE ID in sae_lens")
    parser.add_argument("--results-dir", type=str, default="eval_results/absorption", help="Directory where run_eval Parquet files live")
    parser.add_argument("--hf-token", type=str, default=None, help="Your HuggingFace Hub Token")
    
    args = parser.parse_args()

    if args.hf_token:
        os.environ["HF_TOKEN"] = args.hf_token

    print(f"=== Starting extraction for Letter '{args.letter}' on Layer {args.layer} ===")

    # ---- STEP 3: PROBE mode, from cache ------------------------------------ #
    print("\n[PROBE mode] Loading cached results from parquet...")
    df = load_cached_probe_results(args.sae_name, args.layer, args.results_dir)
    probe_membership = extract_membership_from_cached_df(df, args.letter)
    
    probe_out = f"smain_sabs_probe_{args.letter}_L{args.layer}.json"
    save_membership(
        probe_membership, probe_out,
        meta={"mode": "probe", "letter": args.letter, "sae_name": args.sae_name, "layer": args.layer},
    )
    
    # We use the primary split feature as our shared atom k
    ATOM_K = probe_membership["s_main"][0]  

    # ---- STEP 2: ATOM mode, direct call ------------------------------------ #
    print(f"\n[ATOM mode] Running direct calculation for atom k={ATOM_K}...")
    from transformer_lens import HookedTransformer
    from sae_lens import SAE

    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print("Loading HookedTransformer...")
    model = HookedTransformer.from_pretrained("gemma-2-2b", device=device)
    
    print("Loading SAE...")
    sae = SAE.from_pretrained(
        release=args.sae_release,
        sae_id=args.sae_id,
        device=device,
    )

    vocab = get_alpha_tokens(model.tokenizer)
    calculator = FeatureAbsorptionCalculator(
        model=model,
        icl_word_list=vocab,
        max_icl_examples=10,
        answer_formatter=first_letter_formatter(),
    )

    # Reuse the exact SAME candidate words the probe run used to ensure fair comparison
    same_words = sorted({w for (_, _, w) in probe_membership["pairs"]}) or \
                 df[df["letter"] == args.letter]["token"].tolist()

    atom_results = run_atom_mode(calculator, sae, atom_k=ATOM_K, words=same_words, layer=args.layer)
    atom_membership = extract_membership_from_atom_run(atom_results)
    
    atom_out = f"smain_sabs_atom_{args.letter}_k{ATOM_K}_L{args.layer}.json"
    save_membership(
        atom_membership, atom_out,
        meta={"mode": "atom", "letter": args.letter, "atom_k": ATOM_K, "layer": args.layer},
    )

    # ---- VALIDATION -------------------------------------------------------- #
    print("\n=== VALIDATION ===")
    print(f"PROBE S_abs list: {probe_membership['s_abs']}")
    print(f"ATOM  S_abs list: {atom_membership['s_abs']}")
    
    shared = sorted(set(probe_membership['s_abs']) & set(atom_membership['s_abs']))
    print(f"Shared absorbers found by both methods: {shared}")


if __name__ == "__main__":
    main()
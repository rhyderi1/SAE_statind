
import argparse
import json
from collections import Counter
from pathlib import Path
import torch
from sae_lens import SAE
from transformer_lens import HookedTransformer
from sae_bench.evals.absorption.eval_config import AbsorptionEvalConfig
from sae_bench.evals.absorption.feature_absorption import run_feature_absortion_experiment
from sae_bench.evals.absorption.k_sparse_probing import run_k_sparse_probing_experiment
from sae_bench.sae_bench_utils import activation_collection, general_utils

EPS = 1e-8

HERE       = Path(__file__).parent
ARTIFACTS  = HERE / "artifacts"
RESULTS    = HERE / "results"
PROBES_DIR = ARTIFACTS / "probes"
K_SPARSE_DIR  = ARTIFACTS / "k_sparse_probing"
ABSORPTION_DIR = ARTIFACTS / "feature_absorption"


def parse_args():
    p = argparse.ArgumentParser(
        description="Extract absorber/absorbee index pairs (probe mode, one letter).")
    p.add_argument("--release", required=True)
    p.add_argument("--sae-id", required=True)
    p.add_argument("--letter", default="A")
    p.add_argument("--model", default="gemma-2-2b")
    p.add_argument("--force-rerun", action="store_true",default=False)
    return p.parse_args()


def get_sae_name(release: str, sae_id: str) -> str:
    """SAEBench names artifacts with this key; slashes must become underscores."""
    return f"{release}_{sae_id}".replace("/", "_")


def main() -> None:
    args = parse_args()
    letter = args.letter.upper()

    device   = "cuda" if torch.cuda.is_available() else "cpu"
    dtype_str = activation_collection.LLM_NAME_TO_DTYPE.get(args.model, "float32")
    dtype     = general_utils.str_to_dtype(dtype_str)

    # ── load model & SAE ──────────────────────────────────────────────────────
    print(f"Loading model {args.model!r} on {device} ...")
    model = HookedTransformer.from_pretrained_no_processing(
        args.model, device=device, dtype=dtype
    )

    print(f"Loading SAE {args.release!r} / {args.sae_id!r} ...")
    sae = SAE.from_pretrained(
        release=args.release,
        sae_id=args.sae_id,
        device=device,
    )
    sae = sae.to(device=device, dtype=dtype)
    # hook_name looks like "blocks.12.hook_resid_post"; parse the layer number from it
    hook_name = sae.cfg.metadata.hook_name
    layer = int(hook_name.split(".")[1])
    sae_name = get_sae_name(args.release, args.sae_id)

    config = AbsorptionEvalConfig(model_name=args.model)#take values and thresholds from them

    # ── Step 1: k-sparse probing → S_main (split features) ───────────────────
    print("\n[1/2] Running k-sparse probing ...")
    metrics_df = run_k_sparse_probing_experiment(
        # 1. Trains a logistic regression probe on raw residual-stream activations to classify first-letter (26-class).
        # 2. Trains k-sparse probes on the SAE latents Z for k = 1, 2, …, 10 (based on change in F1).
        # 3. selected latents become split_feats — the S_main for each letter.
        # Returns a metrics_df: one row per letter, containing the selected split features, their F1 scores, and probe cosine similarities. 

        model=model,
        sae=sae,
        layer=layer,
        sae_name=sae_name,
        max_k_value=config.max_k_value,
        prompt_template=config.prompt_template,
        prompt_token_pos=config.prompt_token_pos,
        device=device,
        experiment_dir=K_SPARSE_DIR,
        probes_dir=PROBES_DIR,
        force=args.force_rerun,
        f1_jump_threshold=config.f1_jump_threshold,
        k_sparse_probe_l1_decay=config.k_sparse_probe_l1_decay,
        k_sparse_probe_batch_size=config.k_sparse_probe_batch_size,
        k_sparse_probe_num_epochs=config.k_sparse_probe_num_epochs,
        precalc_k_sparse_probe_sae_acts=config.precalc_k_sparse_probe_sae_acts,
        eval_batch_size=config.eval_k_sparse_probe_batch_size,
    )

    # ── Step 2: feature absorption experiment → raw per-word DataFrame ────────
    print("\n[2/2] Running feature absorption experiment ...")
    raw_df = run_feature_absortion_experiment(
        # needs run_k_sparse_probing_experiment to have run first (reads its cached parquets). Then for each letter:

        # 1. Finds potential false-negative tokens: vocab words where the logistic probe fires (concept is present according to the probe) 
        # but S_main latents don't fire (the SAE missed it).
        # 2. For each such token, runs FeatureAbsorptionCalculator.calculate_absorption. This passes each word through the model, 
        # reads the residual stream, encodes through the SAE, and checks every other latent j:
        # 3. Returns a raw DataFrame with one row per word per letter: the word, whether absorption occurred, which latent absorbed
        #  it (top_projection_feat), and the activations of the S_main latents on that token (split_feat_acts).
        model=model,
        sae=sae,
        layer=layer,
        sae_name=sae_name,
        max_k_value=config.max_k_value,
        feature_split_f1_jump_threshold=config.f1_jump_threshold,
        prompt_template=config.prompt_template,
        prompt_token_pos=config.prompt_token_pos,
        batch_size=activation_collection.LLM_NAME_TO_BATCH_SIZE.get(args.model, 4),
        device=device,
        experiment_dir=ABSORPTION_DIR,
        sparse_probing_experiment_dir=K_SPARSE_DIR,
        probes_dir=PROBES_DIR,
        force=args.force_rerun,
    )

    # ── Extract pairs for the chosen letter ───────────────────────────────────
    sub = raw_df[raw_df["letter"] == letter.lower()]
        # raw_df is a pandas DataFrame where every row is one word (e.g. "apple") evaluated under the chosen letter. 
        # The letter column says which letter that word was tested for. So: sub is a filtered view: only the rows for letter "A". 
    absorbed = sub[sub["is_full_absorption"] == True]
        # extracts rows corresponding to full absorption events

    if len(sub) == 0:
        print(f"\n[!] No rows for letter '{letter}' in the results DataFrame.")
        print("    Letters present:", sorted(raw_df["letter"].unique()))
        return

    # S_main: the k-split features for first-letter classification for this letter (same for every row of this letter)(all start with A)
    s_main: list[int] = list(absorbed["split_feats"].iloc[0]) if len(absorbed) > 0 else \
                        list(sub["split_feats"].iloc[0])

    pairs: list[dict] = []
    absorber_counts: Counter = Counter()

    for _, row in absorbed.iterrows():
        absorber: int = int(row["top_projection_feat"]) # the top projectin feature not in S_main is the absorber
        # absorbees: members of S_main that were silent on this token
        silent_mains = [
            int(fid)
            for fid, act in zip(row["split_feats"], row["split_feat_acts"])
            if act < EPS
        ]
        for absorbee in silent_mains:
            pairs.append({
                "S_abs":  absorber,
                "S_main":  absorbee,
                "token":     row["token"],
                "absorber_probe_cos":    float(row["top_projection_feat_probe_cos"]),
                "absorber_probe_proj":   float(row["top_probe_projection"]),
            })
            absorber_counts[absorber] += 1

    # ── Print summary ─────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  Letter:          '{letter}'")
    print(f"  SAE:             {args.release} / {args.sae_id}")
    print(f"  Layer:           {layer}")
    print(f"  S_main (split features): {s_main}")
    print(f"  Probe true-positives checked: {len(sub)}")
    print(f"  Full-absorption events:       {len(absorbed)}")
    print(f"  (absorber, absorbee) pairs:   {len(pairs)}")
    print(f"  Unique absorbers:             {len(absorber_counts)}")
    if absorber_counts:
        top = absorber_counts.most_common(5)
        print(f"  Top absorbers (idx: count):   {top}")
    print(f"{'='*60}")

    # ── Save ──────────────────────────────────────────────────────────────────
    RESULTS.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS / f"{sae_name}_{letter}_pairs.json"
    output = {
        "letter":          letter,
        "sae_release":     args.release,
        "sae_id":          args.sae_id,
        "layer":           layer,
        "s_main":          s_main,
        "n_probe_positives_checked": int(len(sub)),
        "n_full_absorption_events":  int(len(absorbed)),
        "n_pairs":         len(pairs),
        "absorber_counts": {str(k): v for k, v in absorber_counts.items()},
        "pairs":           pairs,
    }
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved {len(pairs)} pairs → {out_path}")


if __name__ == "__main__":
    main()

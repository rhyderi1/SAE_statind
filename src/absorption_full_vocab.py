"""
Recompute absorption over the FULL single-token vocabulary for one letter.

Why this exists
---------------
SAEBench's absorption eval only ever scores a small slice of the vocabulary:

  1. probing.create_dataset_probe_training splits the alphabetic vocab 80/20 and
     the eval runs on the 20% *test* half only;
  2. feature_absorption.get_stats_and_likely_false_negative_tokens then keeps
     only "likely false negatives" -- tokens where the LR probe fires
     (score_probe_<letter> > 0) but the k-sparse SAE probe does not.

For layer 3 that leaves 2,883 of ~14.5k 's' tokens. Concretely, ' short',
' shorter', ' shortly' and 'short' all landed in the *train* half, so latent
1085 (a "short"-family latent) shows exactly one absorbed token, 'Short'.
Colouring the z_i-vs-z_j scatter from that list marks ~20 of the ~950 y-axis
positions it should.

What this does
--------------
Reuses the already-trained probe and the already-chosen S_main latents, and
re-runs FeatureAbsorptionCalculator over every single-token vocab entry
starting with the letter -- no train/test split, no false-negative filter.
Output is one CSV per letter, a superset of the eval's letter_s.csv with the
absorbing latent kept alongside.

    python src/absorption_full_vocab.py --letter s
    sbatch scripts/run_absorption_full.sh

Note the calculator requires every prompt to tokenize to the same length, which
holds because get_alpha_tokens returns single-token vocab entries only. Do not
pass hand-written multi-token words.
"""

import argparse
import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import torch
from sae_lens import SAE
from transformer_lens import HookedTransformer

from sae_bench.evals.absorption.common import PROBES_DIR, load_probe
from sae_bench.evals.absorption.feature_absorption_calculator import (
    FeatureAbsorptionCalculator,
)
from sae_bench.evals.absorption.feature_absorption import (
    ABSORPTION_FRACTION_MAX_ABSORBING_LATENTS,
    ABSORPTION_FRACTION_PROBE_COS_THRESHOLD,
    ABSORPTION_PROBE_PROJECTION_PROPORTION_THRESHOLD,
    FULL_ABSORPTION_PROBE_COS_THRESHOLD,
)
from sae_bench.evals.absorption.prompting import first_letter_formatter
from sae_bench.evals.absorption.vocab import LETTERS, get_alpha_tokens

from infer_z import SAE_DATA

MODEL_NAME = "gemma-2-2b"
ARCH = "jumprelu"
LAYER = 3
SPARSITY = "59"

# eval defaults, from AbsorptionEvalConfig -- keep in sync or the numbers stop
# being comparable to the parquet
PROMPT_TEMPLATE = "{word} has the first letter:"
PROMPT_TOKEN_POS = -6
MAX_ICL_EXAMPLES = 10

REPO_ROOT = Path(__file__).parent.parent
RESULTS_DIR = REPO_ROOT / "results"


def load_s_main(letter):
    """S_main latents for `letter`, from the newest absorption_sets JSON.

    Reusing the JSON rather than recomputing feature splits from the metrics
    parquet keeps this script and zizj_scatter.py pointed at the same latents.
    """
    pattern = f"absorption_sets_{ARCH}_layer{LAYER}_k{SPARSITY}_*.json"
    candidates = sorted(RESULTS_DIR.glob(pattern))
    if not candidates:
        raise SystemExit(f"No {pattern} in results/ -- run extract_absorption_sets.py first.")
    with open(candidates[-1]) as f:
        blob = json.load(f)
    print(f"S_main from {candidates[-1].name}")
    return [int(x) for x in blob["s_main"][letter]]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--letter", default="s")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--limit", type=int,
                        help="score only the first N words (smoke test)")
    args = parser.parse_args()

    letter = args.letter.lower()
    if letter not in LETTERS:
        raise SystemExit(f"--letter must be one of {LETTERS}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    release, sae_id = SAE_DATA[LAYER][ARCH][SPARSITY]

    model = HookedTransformer.from_pretrained_no_processing(
        MODEL_NAME, device=device, dtype=dtype)
    sae = SAE.from_pretrained(release=release, sae_id=sae_id, device=device)
    if isinstance(sae, tuple):          # older sae_lens returns (sae, cfg, sparsity)
        sae = sae[0]
    sae = sae.to(device=device, dtype=dtype)

    # the probe is vocab-split-independent: it was trained once per layer and is
    # reused verbatim, so nothing here retrains or reshuffles anything
    probe = load_probe(model_name=MODEL_NAME, layer=LAYER,
                       probes_dir=PROBES_DIR, device=device)

    vocab = get_alpha_tokens(model.tokenizer)
    words = sorted(w for w in vocab if w.lstrip()[:1].lower() == letter)
    if args.limit:
        words = words[:args.limit]
    print(f"letter '{letter}': scoring {len(words):,} of {len(vocab):,} alpha tokens")

    main_feature_ids = load_s_main(letter)
    print(f"S_main[{letter}] = {main_feature_ids}")

    calculator = FeatureAbsorptionCalculator(
        model=model,
        icl_word_list=vocab,
        max_icl_examples=MAX_ICL_EXAMPLES,
        base_template=PROMPT_TEMPLATE,
        answer_formatter=first_letter_formatter(),
        word_token_pos=PROMPT_TOKEN_POS,
        full_absorption_probe_cos_sim_threshold=FULL_ABSORPTION_PROBE_COS_THRESHOLD,
        absorption_fraction_probe_cos_sim_threshold=ABSORPTION_FRACTION_PROBE_COS_THRESHOLD,
        probe_projection_proportion_threshold=ABSORPTION_PROBE_PROJECTION_PROPORTION_THRESHOLD,
        absorption_fraction_max_absorbing_latents=ABSORPTION_FRACTION_MAX_ABSORBING_LATENTS,
        batch_size=args.batch_size,
    )

    results = calculator.calculate_absorption(
        sae,
        layer=LAYER,
        words=words,
        probe_direction=probe.weights[LETTERS.index(letter)],
        main_feature_ids=main_feature_ids,
    )

    rows = []
    for s in results.word_results:
        top = s.top_projection_feature_scores[0]
        rows.append({
            "token": s.word,
            "absorption_fraction": s.absorption_fraction,
            "is_full_absorption": s.is_full_absorption,
            # kept so the scatter can mask on "absorbed by *this* latent"
            "top_projection_feat": top.feature_id,
            "top_probe_projection": top.probe_projection,
            "probe_projection": s.probe_projection,
            "main_feat_acts": [f.activation for f in s.main_feature_scores],
        })
    df = pd.DataFrame(rows)

    ts = datetime.now(ZoneInfo("America/Edmonton")).strftime("%Y%m%d_%H%M%S")
    out = RESULTS_DIR / f"absorption_full_{letter}_{ARCH}_layer{LAYER}_k{SPARSITY}_{ts}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)

    n_full = int(df["is_full_absorption"].sum())
    print(f"\n{len(df):,} tokens scored, {n_full:,} full-absorption events -> {out.name}")
    print("\ntop absorbing latents:")
    print(df[df["is_full_absorption"]]["top_projection_feat"]
          .value_counts().head(15).to_string())


if __name__ == "__main__":
    main()

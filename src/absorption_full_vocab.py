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

    python src/absorption_full_vocab.py --letters s
    python src/absorption_full_vocab.py --letters all --arch jumprelu --layer 3 --sparsity 59
    sbatch scripts/run_absorption_full.sh    # array 0-3, balanced letter groups

Note the calculator requires every prompt to tokenize to the same length, which
holds because get_alpha_tokens returns single-token vocab entries only. Do not
pass hand-written multi-token words.
"""

import argparse
import json
import random
import traceback
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

# eval defaults, from AbsorptionEvalConfig -- keep in sync or the numbers stop
# being comparable to the parquet
PROMPT_TEMPLATE = "{word} has the first letter:"
PROMPT_TOKEN_POS = -6
MAX_ICL_EXAMPLES = 10

REPO_ROOT = Path(__file__).parent.parent
RESULTS_DIR = REPO_ROOT / "results"


def load_s_main(letter, arch, layer, sparsity):
    """Get S_main latents for `letter` from the newest matching absorption dict."""
    pattern = f"absorption_sets_{arch}_layer{layer}_k{sparsity}_*.json"
    candidates = sorted(RESULTS_DIR.glob(pattern))
    if not candidates:
        raise SystemExit(f"No {pattern} in results/ -- run extract_absorption_sets.py first.")
    with open(candidates[-1]) as f:
        blob = json.load(f)
    print(f"S_main from {candidates[-1].name}")
    return [int(x) for x in blob["s_main"][letter]]


def out_pattern(letter, arch, layer, sparsity):
    """Glob for this letter's CSVs -- the filename carries a run timestamp."""
    return f"absorption_full_{letter}_{arch}_layer{layer}_k{sparsity}_*.csv"


def letter_words(vocab, letter):
    return sorted(w for w in vocab if w.lstrip()[:1].lower() == letter)


def assign_shards(sizes, num_shards):
    """Greedily balance letters across shards by token count, largest first.

    Letter sizes are very uneven ('s' alone is ~14.5k of ~169k alpha tokens), so
    splitting the alphabet evenly would leave one task running several times
    longer than the others.
    """
    shards = [[] for _ in range(num_shards)]
    loads = [0] * num_shards
    for letter, n in sorted(sizes.items(), key=lambda kv: -kv[1]):
        i = loads.index(min(loads))
        shards[i].append(letter)
        loads[i] += n
    return shards, loads


def run_letter(letter, words, calculator, sae, probe, arch, layer, sparsity):
    """Score every vocab token starting with `letter`; write one CSV, return its path."""
    print(f"\nletter '{letter}': scoring {len(words):,} tokens")

    main_feature_ids = load_s_main(letter, arch, layer, sparsity)
    print(f"S_main[{letter}] = {main_feature_ids}")

    results = calculator.calculate_absorption(
        sae,
        layer=layer,
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
    out = RESULTS_DIR / f"absorption_full_{letter}_{arch}_layer{layer}_k{sparsity}_{ts}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)

    n_full = int(df["is_full_absorption"].sum())
    print(f"{len(df):,} tokens scored, {n_full:,} full-absorption events -> {out.name}")
    print("top absorbing latents:")
    print(df[df["is_full_absorption"]]["top_projection_feat"]
          .value_counts().head(15).to_string())
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--letters", default="all",
                        help="'all' or a comma-separated list, e.g. s,q,e")
    parser.add_argument("--arch", default="jumprelu")
    parser.add_argument("--layer", type=int, default=3)
    parser.add_argument("--sparsity", default="59")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--shard", type=int, default=0,
                        help="SLURM array index; which balanced letter group to run")
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--skip-existing", action="store_true",
                        help="skip letters that already have a CSV, so a requeue resumes")
    parser.add_argument("--seed", type=int, default=0,
                        help="seeds ICL example sampling; see note below")
    args = parser.parse_args()

    # The calculator draws its ICL examples with an unseeded random.sample
    # (sae_bench/evals/absorption/prompting.py:108), so two runs of the same
    # letter disagree on ~3% of tokens. Seeding here makes the dataset
    # regenerable; it does not change what is being measured.
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.letters.strip().lower() == "all":
        letters = list(LETTERS)
    else:
        letters = [x.strip().lower() for x in args.letters.split(",") if x.strip()]
    bad = [x for x in letters if x not in LETTERS]
    if bad:
        raise SystemExit(f"not letters: {bad} -- must be from {LETTERS}")
    if not 0 <= args.shard < args.num_shards:
        raise SystemExit(f"--shard must be in [0, {args.num_shards})")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    release, sae_id = SAE_DATA[args.layer][args.arch][args.sparsity]

    # everything below is letter-independent, so it is paid once for the whole
    # run rather than once per letter
    model = HookedTransformer.from_pretrained_no_processing(
        MODEL_NAME,
        device=device,
        dtype=dtype
        )

    sae = SAE.from_pretrained(
        release=release,
        sae_id=sae_id,
        device=device
        )

    sae = sae.to(device=device, dtype=dtype) # Move to GPU

    # the probe is vocab-split-independent: it was trained once per layer and is
    # reused verbatim, so nothing here retrains or reshuffles anything
    probe = load_probe(model_name=MODEL_NAME, layer=args.layer,
                       probes_dir=PROBES_DIR, device=device)

    vocab = get_alpha_tokens(model.tokenizer)

    # probe_direction and main_feature_ids are arguments to calculate_absorption,
    # not to the constructor, so one calculator serves every letter
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

    words_by_letter = {L: letter_words(vocab, L) for L in letters}
    sizes = {L: len(w) for L, w in words_by_letter.items()}
    print(f"{sum(sizes.values()):,} of {len(vocab):,} alpha tokens across {len(letters)} letters")

    if args.num_shards > 1:
        shards, loads = assign_shards(sizes, args.num_shards)
        for i, (grp, load) in enumerate(zip(shards, loads)):
            mark = " <- this task" if i == args.shard else ""
            print(f"  shard {i}: {load:>7,} tokens  {''.join(sorted(grp))}{mark}")
        letters = sorted(shards[args.shard])

    done, skipped, failed = [], [], []
    for letter in letters:
        if args.skip_existing:
            existing = sorted(RESULTS_DIR.glob(
                out_pattern(letter, args.arch, args.layer, args.sparsity)))
            if existing:
                print(f"\nletter '{letter}': skipping, {existing[-1].name} exists")
                skipped.append(letter)
                continue
        try:
            run_letter(letter, words_by_letter[letter], calculator, sae, probe,
                       args.arch, args.layer, args.sparsity)
            done.append(letter)
        except Exception as exc:
            # one bad letter must not cost the other 25
            print(f"\nletter '{letter}': FAILED -- {type(exc).__name__}: {exc}")
            traceback.print_exc()
            failed.append(letter)

    print(f"\ndone: {''.join(done) or '-'}"
          f"  skipped: {''.join(skipped) or '-'}"
          f"  failed: {''.join(failed) or '-'}")
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

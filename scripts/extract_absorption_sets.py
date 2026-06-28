"""
Extract per-letter S_main and S_abs dicts from the sae_bench absorption parquet
and save them to results/absorption_sets.json.

S_main[letter] = list of SAE latent indices from k-sparse probing (the "main" feature)
S_abs[letter]  = sorted list of unique top absorber latents across full-absorption events
"""
import json
from pathlib import Path
import pandas as pd

PARQUET = (
    Path(__file__).parent.parent
    / ".venv/lib/python3.11/site-packages/sae_bench/artifacts/absorption"
    / "feature_absorption"
    / "sae_bench_gemma-2-2b_vanilla_width-2pow14_date-1109_blocks.12.hook_resid_post__trainer_0"
    / "layer_12_sae_bench_gemma-2-2b_vanilla_width-2pow14_date-1109_blocks.12.hook_resid_post__trainer_0.parquet"
)
OUT = Path(__file__).parent.parent / "results" / "absorption_sets.json"


def main() -> None:
    df = pd.read_parquet(PARQUET)

    s_main: dict[str, list[int]] = {}
    s_abs: dict[str, list[int]] = {}

    for letter, grp in df.groupby("letter"):
        s_main[letter] = [int(x) for x in grp["split_feats"].iloc[0]]

        absorbed = grp[grp["is_full_absorption"]]
        absorbers: set[int] = set()
        for top_feat in absorbed["top_projection_feat"]:
            absorbers.add(int(top_feat))
        s_abs[letter] = sorted(absorbers)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w") as f:
        json.dump({"s_main": s_main, "s_abs": s_abs}, f, indent=2)

    print(f"Saved → {OUT}")
    for letter in sorted(s_main):
        print(f"  {letter}: S_main={s_main[letter]}  |  S_abs={len(s_abs[letter])} absorbers")


if __name__ == "__main__":
    main()

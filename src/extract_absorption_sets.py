"""
SAEBench absorption creates a parquet file.
Here, we extract per-letter S_main and S_abs from the sae_bench absorption parquet.

S_main[letter] = list of SAE latent indices from k-sparse probing (the "main" feature)
S_abs[letter]  = list of unique absorbing latents across full-absorption events,
                    written as [latent, n] where n is how many absorption events 

Note: this script uses is_full_absorption to identify absorption events, so each event has exactly one
absorber by construction (one latent, `top_projection_feat`, must carry the
letter signal on its own).

"""
import json
import re
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

from infer_z import SAE_DATA

# --- parameters ---
ARCH = "relu"
LAYER = 12
SPARSITY = "40"
release, sae_id = SAE_DATA[LAYER][ARCH][SPARSITY]

folder_name = f"{release}_{sae_id}"
file_name = f"layer_{LAYER}_{release}_{sae_id}.parquet"
PARQUET = (
    Path(__file__).parent.parent
    / ".venv/lib/python3.11/site-packages/sae_bench/artifacts/absorption"
    / "feature_absorption"
    / folder_name
    / file_name
)

timestamp = datetime.now(ZoneInfo("America/Edmonton")).strftime("%Y%m%d_%H%M%S")
OUT = (
    Path(__file__).parent.parent
    / "results"
    / f"absorption_sets_{ARCH}_layer{LAYER}_k{SPARSITY}_{timestamp}.json"
)


def main() -> None:
    df = pd.read_parquet(PARQUET)

    s_main = {}
    s_abs = {}
    s_abs_tokens = {}

    for letter, grp in df.groupby("letter"):
        # grp is the sub-DataFrame of just that letter's rows (e.g. for letter="a",
        # eg: grp is the ~1,200 rows where letter == "a").

        # The main feature is the same for every row of a letter, so take the first one.
        s_main[letter] = [int(x) for x in grp["split_feats"].iloc[0]]

        absorbed_rows = grp[grp["is_full_absorption"]]
        event_counts = {}
        event_tokens = {}
        for feat, tok in zip(absorbed_rows["top_projection_feat"], absorbed_rows["token"]):
            feat = int(feat)
            event_counts[feat] = event_counts.get(feat, 0) + 1
            # create and iterates a dict{latent: count} where the count defaults 0.
            event_tokens.setdefault(feat, []).append(str(tok))

        s_abs[letter] = [[feat, event_counts[feat]] for feat in sorted(event_counts)]
        # the exact vocab token of each full-absorption event, grouped by absorber

        s_abs_tokens[letter] = {str(feat): event_tokens[feat] for feat in sorted(event_tokens)}
        # formatted as a dict with str keys, so it can be used by JSON

    text = json.dumps({
        "arch": ARCH,
        "layer": LAYER,
        "sparsity": SPARSITY,
        "s_main": s_main,
        "s_abs": s_abs,
        "s_abs_tokens": s_abs_tokens,
    }, indent=2)
    #each nesting level gets its own lines, indented 2 spaces deeper than its parent.

    # Makes the code look nicer. the json indent expands inner lists 
    # Collapse any bracket with only digits, commas, and whitespace into a single line. 
    # outer per-letter lists (which contain nested lists) are left untouched.
    text = re.sub(
        r"\[([\d,\s]+)\]",
        lambda m: "[" + ", ".join(re.findall(r"\d+", m.group(1))) + "]",
        text,
    )

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w") as f:
        f.write(text)

    print(f"Saved -> {OUT}")
    for letter in sorted(s_main):
        print(f"  {letter}: S_main={s_main[letter]}  |  S_abs={len(s_abs[letter])} absorbers")


if __name__ == "__main__":
    main()

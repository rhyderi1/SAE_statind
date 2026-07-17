'''
Co-activation-matched control pairs for zizj_scatter.

Streams the corpus once and, for an anchor latent i, counts on how many
tokens every other latent co-activates with i. Then, for each absorber j
of i's letter, reports non-absorbing control latents whose co-activation
with i is closest to j's -- so (i, control) scatters can be compared
against (i, absorber) at the same co-activation level.

  python -m src.coact_match --latent 6840
  python -m src.coact_match --coact_pt results/coact_..._latent6840_*.pt   # report only
'''

import argparse
import json
import torch
from pathlib import Path
from datetime import datetime
from sae_lens import SAE
from transformer_lens import HookedTransformer
from sae_lens import ActivationsStore

device = "cuda" if torch.cuda.is_available() else "cpu"
dataset = 'NeelNanda/pile-10k'
context_size = 128
batch_size = 32
n_batches = 1209

ARCH = "jumprelu"
LAYER = 3
SPARSITY = "59"
EPS = 1e-8   # theta_fire: latent fires when activation >= EPS

REPO_ROOT = Path(__file__).parent.parent
RESULTS_DIR = REPO_ROOT / "results"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--latent", type=int, default=6840,
                        help="anchor latent i (fixed side of the pairs)")
    parser.add_argument("--coact_pt", type=str, default=None,
                        help="previously saved coact .pt: skip streaming and "
                             "just print the matching report")
    parser.add_argument("--absorption_json", type=str, default=None,
                        help="absorption_sets JSON. Defaults to the newest in "
                             "results/ matching this arch/layer/sparsity.")
    parser.add_argument("--n_controls", type=int, default=3,
                        help="matched controls to suggest per absorber")
    parser.add_argument("--n_batches", type=int, default=n_batches)
    parser.add_argument("--batch_size", type=int, default=batch_size)
    parser.add_argument("--context_size", type=int, default=context_size)
    return parser.parse_args()


# 1) Stream: co-activation of the anchor with every latent

def stream_coact(anchor, n_batches, batch_size, context_size):
    """Returns (coact, fires, n_tokens): coact[j] = #tokens where both the
    anchor and j fire, fires[j] = #tokens where j fires."""
    hook_name = f'blocks.{LAYER}.hook_resid_post'
    SAE_RELEASE = "gemma-scope-2b-pt-res"
    SAE_ID = "layer_3/width_16k/average_l0_59"
    sae = SAE.from_pretrained(release=SAE_RELEASE, sae_id=SAE_ID, device=device)

    model = HookedTransformer.from_pretrained_no_processing(
        default_prepend_bos=True,
        model_name="gemma-2-2b",
        device=device
    )
    activation_store = ActivationsStore.from_sae(
        model,
        sae,
        context_size=context_size,
        dataset=dataset,
    )

    d_sae = sae.cfg.d_sae
    coact = torch.zeros(d_sae, dtype=torch.long, device=device)
    fires = torch.zeros(d_sae, dtype=torch.long, device=device)
    n_tokens = 0

    with torch.no_grad():
        for batch_num in range(n_batches):
            batch_tokens = activation_store.get_batch_tokens(batch_size)
            _, cache = model.run_with_cache(
                batch_tokens,
                names_filter=hook_name,
                stop_at_layer=LAYER + 1,
                prepend_bos=False,
            )
            X = cache[hook_name]
            del cache

            X_sae = X.to(device=sae.W_enc.device, dtype=sae.W_enc.dtype)
            Z = sae.encode(X_sae).reshape(-1, sae.cfg.d_sae)
            fire = Z >= EPS
            del Z

            fires += fire.sum(0)
            anchor_rows = fire[:, anchor]
            if anchor_rows.any():
                coact += fire[anchor_rows].sum(0)
            n_tokens += fire.shape[0]
            del fire

    return coact.cpu(), fires.cpu(), n_tokens


def save_coact(coact, fires, n_tokens, anchor):
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = RESULTS_DIR / (
        f"coact_{ARCH}_layer{LAYER}_k{SPARSITY}_latent{anchor}_{ts}.pt")
    torch.save(
        {
            "anchor": anchor,
            "coact": coact,
            "fires": fires,
            "n_tokens": n_tokens,
            "arch": ARCH,
            "layer": LAYER,
            "sparsity": SPARSITY,
            "dataset": dataset,
            "eps": EPS,
            "timestamp": ts,
        },
        out_path,
    )
    print(f"Saved -> {out_path}")
    return out_path


# 2) Report: absorbers of the anchor's letter + coactivation-matched controls

def newest_absorption_json():
    pattern = f"absorption_sets_{ARCH}_layer{LAYER}_k{SPARSITY}_*.json"
    candidates = sorted(RESULTS_DIR.glob(pattern))
    return candidates[-1] if candidates else None


def report(coact, fires, anchor, absorption_json, n_controls):
    with open(absorption_json) as f:
        blob = json.load(f)
    s_main, s_abs = blob["s_main"], blob["s_abs"]

    letters = [L for L, mains in s_main.items() if anchor in mains]
    if not letters:
        raise SystemExit(f"latent {anchor} is not a main latent in "
                         f"{absorption_json}")

    # controls must be clean: no absorber of any letter, no main latent
    mains_all = {m for ms in s_main.values() for m in ms}
    absorbers_all = {j for lst in s_abs.values() for j, _ in lst}
    excluded = mains_all | absorbers_all | {anchor}

    candidate = torch.ones(coact.numel(), dtype=torch.bool)
    candidate[sorted(excluded)] = False
    candidate &= coact > 0     # never co-fires with the anchor -> useless control

    print(f"anchor i={anchor} (letter {'/'.join(letters)}), "
          f"fires on {int(fires[anchor])} tokens\n")
    print(f"{'absorber j':>10} {'events':>7} {'coact(i,j)':>10}   "
          f"matched controls j' (coact)")

    for L in letters:
        for j, n_events in sorted(s_abs.get(L, []), key=lambda x: -x[1]):
            target = coact[j]
            # closest co-activation among the clean candidates
            dist = (coact - target).abs().float()
            dist[~candidate] = float("inf")
            order = torch.argsort(dist)[:n_controls]
            ctrl = ", ".join(f"{int(c)} ({int(coact[c])})" for c in order)
            print(f"{j:>10} {n_events:>7} {int(target):>10}   {ctrl}")

    print("\nPAIRS block (absorber then its matched controls):")
    for L in letters:
        for j, n_events in sorted(s_abs.get(L, []), key=lambda x: -x[1]):
            target = coact[j]
            dist = (coact - target).abs().float()
            dist[~candidate] = float("inf")
            order = torch.argsort(dist)[:n_controls]
            print(f"    ({anchor}, {j}),  # absorber x{n_events}, "
                  f"coact={int(target)}")
            for c in order:
                print(f"    ({anchor}, {int(c)}),  # control, "
                      f"coact={int(coact[c])}")


def main():
    args = parse_args()

    if args.coact_pt:
        blob = torch.load(args.coact_pt)
        if blob["anchor"] != args.latent:
            print(f"note: {args.coact_pt} was computed for anchor "
                  f"{blob['anchor']}, using that")
        anchor = blob["anchor"]
        coact, fires = blob["coact"], blob["fires"]
    else:
        anchor = args.latent
        coact, fires, n_tokens = stream_coact(
            anchor, args.n_batches, args.batch_size, args.context_size)
        print(f"streamed {n_tokens} tokens")
        save_coact(coact, fires, n_tokens, anchor)

    absorption_json = args.absorption_json or newest_absorption_json()
    if absorption_json is None:
        raise SystemExit("no absorption_sets JSON found in results/")
    report(coact, fires, anchor, absorption_json, args.n_controls)


if __name__ == "__main__":
    main()

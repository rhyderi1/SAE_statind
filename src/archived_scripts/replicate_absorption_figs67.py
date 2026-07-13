'''
Replicate Figures 6 and 7 of "A is for Absorption" (Chanin et al., 2024,
arXiv:2409.14507) on the Gemma-Scope layer-3 SAE.

Background. A linear "first letter" probe is trained on layer-3 residuals; row 18
of its weight matrix is the direction in residual space meaning "this token starts
with S". One SAE latent usually looks like a "starts with S" detector: its decoder
vector points along that probe direction, and it fires on S-words. Absorption is
when that main latent goes silent on one specific word (here " short") and some
other latent -- which mostly means something else entirely -- quietly picks up the
S-direction for that word. The paper's evidence is causal: ablating the absorber
hurts the model's ability to spell " short", and that damage disappears once you
strip the probe direction out of the absorber's decoder vector.

The paper's own code is vendored as `sae_bench.evals.absorption`; we reuse it for
the probe, the ICL spelling prompts, and the letter tokens.

Panels (one 2x2 figure):
  top-left      main vs absorber activation across S-starting tokens
  top-right     decoder/probe cosine similarity for every latent
  bottom-left   ablation effect on the " short" prompt, per active latent
  bottom-right  the same, after projecting the probe direction out of the absorber

Run:
  ssh vulcan "cd /project/aip-bahtol/rhyderi1/sae_statind && \
              python src/replicate_absorption_figs67.py"
'''

import random
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from sae_lens import SAE
from transformer_lens import HookedTransformer

from sae_bench.evals.absorption.common import load_or_train_probe
from sae_bench.evals.absorption.prompting import (
    VERBOSE_FIRST_LETTER_TEMPLATE,
    VERBOSE_FIRST_LETTER_TOKEN_POS,
    first_letter,
)
from sae_bench.evals.absorption.vocab import get_alpha_tokens

MODEL_NAME = "gemma-2-2b"
LAYER = 3
SAE_RELEASE = "gemma-scope-2b-pt-res"
SAE_ID = "layer_3/width_16k/average_l0_59"   # the paper's exact checkpoint

LETTER = "S"
TARGET_WORD = " short"   # leading space: this is one vocab token

N_ICL = 10               # in-context spelling examples per prompt
SEED = 42
SHOW_WORDS = [" snake", " soggy", " steam", " short", " soccer", " sax"]
N_BARS = 10              # latents per ablation panel

EPS = 1e-8
device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
hook_name = f"blocks.{LAYER}.hook_resid_post"
letter_idx = ord(LETTER.upper()) - ord("A")   # S -> 18, the probe's output index

rng = random.Random(SEED)

print("Loading model, SAE, probe ...")
model = HookedTransformer.from_pretrained_no_processing(
    MODEL_NAME, device=device, dtype=dtype
)
sae = SAE.from_pretrained(release=SAE_RELEASE, sae_id=SAE_ID, device=device)
sae = sae.to(device=device, dtype=dtype)

probe = load_or_train_probe(   # 26-way, trains and caches on first run
    model,
    base_template=VERBOSE_FIRST_LETTER_TEMPLATE,
    pos_idx=VERBOSE_FIRST_LETTER_TOKEN_POS,
    layer=LAYER,
    device=device,
)
probe_dir = probe.weights[letter_idx].detach().to(device=device, dtype=torch.float32)
probe_dir = probe_dir / probe_dir.norm()

W_dec = sae.W_dec.detach()                                             # (d_sae, d_model)
cos_sims = F.cosine_similarity(probe_dir, W_dec.float(), dim=-1).cpu()  # (d_sae,)

# the "starts with S" detector is whichever latent points most along the probe
main_latent = int(cos_sims.argmax())
print(f"main 'starts with {LETTER}' latent = {main_latent}  "
      f"(cos sim {cos_sims[main_latent]:+.3f})")

# ---------------------------- prompts ------------------------------------
# Prompt = N_ICL spelling examples, then "{word}:". One fixed context, so every
# prompt tokenizes to the same length and the word sits at the same position.
alpha_tokens = get_alpha_tokens(model.tokenizer)
for w in SHOW_WORDS + [TARGET_WORD]:
    assert w in set(alpha_tokens), f"{w!r} is not a single vocab token"

icl_examples = rng.sample([w for w in alpha_tokens if w != TARGET_WORD], N_ICL)
icl_prefix = "".join(f"{w}:{first_letter(w)}\n" for w in icl_examples)

POS_WORD = -2    # the word token
POS_FINAL = -1   # the ":" -- the model predicts the first letter here

letter_token_ids = torch.tensor(
    [model.to_single_token(" " + chr(ord("A") + i)) for i in range(26)],
    device=device,
)


def spelling_metric(final_logits, correct_idx):
    """m = logit[correct letter] - mean logit[other 25 letters]. Shape (B,)."""
    ll = final_logits[:, letter_token_ids].float()   # (B, 26)
    correct = ll[:, correct_idx]
    others = (ll.sum(dim=1) - correct) / 25.0
    return correct - others


@torch.inference_mode()
def run_words(words):
    """Return (sae_acts at the word position, final logits) for single-token words."""
    tokens = model.to_tokens([icl_prefix + f"{w}:" for w in words])
    logits, cache = model.run_with_cache(tokens, names_filter=hook_name)
    sae_acts = sae.encode(cache[hook_name][:, POS_WORD, :])
    return sae_acts.float().cpu(), logits[:, POS_FINAL, :]


# --------------------------- find the absorber ---------------------------
z_short, short_logits = run_words([TARGET_WORD])
z_short = z_short[0]                                       # (d_sae,)
m_clean = spelling_metric(short_logits, letter_idx).item()

# the active, non-main latent contributing the most probe direction to the residual
probe_proj = z_short * cos_sims
probe_proj[main_latent] = -float("inf")
probe_proj[z_short <= EPS] = -float("inf")
absorber = int(probe_proj.argmax())
print(f"absorber latent for {TARGET_WORD!r} = {absorber}  "
      f"(act {z_short[absorber]:.2f}, cos sim {cos_sims[absorber]:+.3f})")
print(f"clean spelling metric m({TARGET_WORD!r}) = {m_clean:+.3f}")


# ---------------------------- ablation -----------------------------------
# Zeroing latent f at the word position = subtracting its contribution to the
# residual, z_f * W_dec[f]. Only latents that are active there change anything,
# so we ablate each active latent once, one per row of a single batch.
active = torch.nonzero(z_short > EPS).flatten().tolist()
print(f"{len(active)} active latents on {TARGET_WORD!r}")

short_tokens = model.to_tokens([icl_prefix + f"{TARGET_WORD}:"])


@torch.inference_mode()
def ablation_effects(W_dec_use):
    """Per active latent, m_ablated - m_clean on the target prompt."""
    feats = torch.tensor(active, device=device)
    sub = (z_short[active].to(device, dtype).unsqueeze(1)
           * W_dec_use[feats].to(dtype))                   # (n_active, d_model)

    def hook(resid, hook):
        resid[:, POS_WORD, :] -= sub
        return resid

    batch = short_tokens.repeat(len(active), 1)
    logits = model.run_with_hooks(batch, fwd_hooks=[(hook_name, hook)])
    m_abl = spelling_metric(logits[:, POS_FINAL, :], letter_idx)
    return (m_abl - m_clean).float().cpu().numpy()


eff = ablation_effects(W_dec)

# Now blind the absorber to the letter: remove the probe component from its
# decoder vector, leaving everything else it writes intact. If the absorber's
# causal effect on spelling was really the absorbed "S", the effect vanishes.
W_dec_proj = W_dec.clone()
d = W_dec_proj[absorber].float()
W_dec_proj[absorber] = (d - (d @ probe_dir) * probe_dir).to(W_dec.dtype)
eff_proj = ablation_effects(W_dec_proj)


# ------------------------------ figure -----------------------------------
fig, ((ax_act, ax_cos), (ax_abl, ax_proj)) = plt.subplots(2, 2, figsize=(12, 7))

s_acts, _ = run_words(SHOW_WORDS)
x = np.arange(len(SHOW_WORDS))
for k, latent in enumerate([absorber, main_latent]):
    ax_act.bar(x + (k - 0.5) * 0.4, s_acts[:, latent].numpy(),
               width=0.4, label=str(latent))
ax_act.set_xticks(x)
ax_act.set_xticklabels([w.strip() for w in SHOW_WORDS], rotation=45, ha="right")
ax_act.set_xlabel("Token")
ax_act.set_ylabel("Activation")
ax_act.set_title(f"'{LETTER}' activations by token")
ax_act.legend(title="Latent ID")

ax_cos.plot(cos_sims.numpy(), lw=0.5)
for latent, label in [(absorber, "absorber"), (main_latent, "main")]:
    ax_cos.plot(latent, cos_sims[latent], "o", ms=5,
                label=f"{label} {latent} (cos {cos_sims[latent]:.2f})")
ax_cos.set_xlabel("SAE latent ID")
ax_cos.set_ylabel("Cosine similarity")
ax_cos.set_title(f"Decoder vs '{LETTER}' probe")
ax_cos.legend(fontsize=8)


def bar_panel(ax, values, title):
    """Bar the N_BARS most-negative effects, strongest ablation first."""
    order = np.argsort(values)[:N_BARS]
    ax.bar(np.arange(len(order)), values[order])
    ax.set_xticks(np.arange(len(order)))
    ax.set_xticklabels([str(active[k]) for k in order], rotation=45, ha="right")
    ax.set_xlabel("SAE latent ID")
    ax.set_ylabel("Ablation effect")
    ax.set_title(title)


bar_panel(ax_abl, eff, f'Ablation effects for "{TARGET_WORD.strip()}"')
bar_panel(ax_proj, eff_proj, "Same, with the probe projected out of the absorber")

# one scale, so the collapse in the right panel is legible against the left
lo = min(eff.min(), eff_proj.min())
ax_abl.set_ylim(lo * 1.08, 0)
ax_proj.set_ylim(lo * 1.08, 0)

fig.suptitle(f"Absorption of '{LETTER}' -- {MODEL_NAME}, layer {LAYER}, {SAE_ID}")
fig.tight_layout()

ts = datetime.now().strftime("%Y%m%d_%H%M%S")
out_path = (Path(__file__).parent.parent / "figures"
            / f"absorption_figs67_layer{LAYER}_{LETTER}_{ts}.png")
out_path.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out_path, dpi=160, bbox_inches="tight")
plt.close(fig)
print(f"Saved -> {out_path}")

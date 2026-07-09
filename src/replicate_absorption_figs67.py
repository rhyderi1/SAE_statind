'''
Replicate Figures 6 and 7 of "A is for Absorption" (Chanin et al., 2024,
arXiv:2409.14507) on the Gemma-Scope layer-3 SAE.

The paper's own experiment code is vendored in this repo as
`sae_bench.evals.absorption`, so we reuse it for the pieces that must match the
paper exactly (the first-letter probe, the ICL spelling task, the decoder<->probe
cosine similarity). Everything else is a thin driver.

Paper setup (Fig 6/7 case study on the letter "S" and the token " short"):
  * model : gemma-2-2b
  * SAE   : gemma-scope-2b-pt-res, layer 3, width 16k, L0=59
  * main latent  : the "starts with S" detector (paper: latent 6510)
  * absorber     : the latent that carries "S" for the word "short" (paper: 1085)

Panels produced:
  Fig 6a  latent activations across "S"-starting tokens (main fires everywhere
          except " short"; the absorber fires on " short")
  Fig 6b  decoder cosine similarity of every latent with the "starts with S" probe
  Fig 7a  ablation effect on the " short" prompt for each active latent
  Fig 7b  Fig 7a recomputed after projecting the probe direction out of the
          absorber's decoder vector (its causal effect vanishes)

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

# --- vendored paper code (Chanin et al.) ---------------------------------
from sae_bench.evals.absorption.common import load_or_train_probe
from sae_bench.evals.absorption.prompting import (
    VERBOSE_FIRST_LETTER_TEMPLATE,
    VERBOSE_FIRST_LETTER_TOKEN_POS,
    first_letter,
)
from sae_bench.evals.absorption.vocab import get_alpha_tokens

# ------------------------------ config -----------------------------------
MODEL_NAME = "gemma-2-2b"
LAYER = 3
SAE_RELEASE = "gemma-scope-2b-pt-res"
SAE_ID = "layer_3/width_16k/average_l0_59"   # the paper's exact checkpoint (L0=59)

LETTER = "S"          # the case-study letter
TARGET_WORD = " short"  # the absorbed token (leading space = the " short" vocab token)

N_ICL = 10            # in-context spelling examples per prompt
SEED = 42

# Fig 6a shows these tokens only, in this order (the paper's case-study set).
SHOW_WORDS = [" snake", " soggy", " steam", " short", " soccer", " sax"]
N_BARS_FIG7 = 10      # how many latents to bar in each Fig 7 panel

# seaborn-style panels to match the paper's figures
plt.style.use("seaborn-v0_8-darkgrid")
plt.rcParams.update({
    "font.family": "serif",
    "axes.titlesize": 10,
    "axes.labelsize": 10,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
})
C_ABSORBER = "#4C72B0"   # seaborn deep blue
C_MAIN = "#DD8452"       # seaborn deep orange

EPS = 1e-8
device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
hook_name = f"blocks.{LAYER}.hook_resid_post"
letter_idx = ord(LETTER.upper()) - ord("A")   # S -> 18, matches probe output order

rng = random.Random(SEED)

# ------------------------- model / SAE / probe ---------------------------
print("Loading model, SAE, probe ...")
# no_processing to match how Gemma-Scope SAEs (and the sae_bench eval) see the resid
model = HookedTransformer.from_pretrained_no_processing(
    MODEL_NAME, device=device, dtype=dtype
)
sae = SAE.from_pretrained(release=SAE_RELEASE, sae_id=SAE_ID, device=device)
sae = sae.to(device=device, dtype=dtype)

# 26-way first-letter probe on layer-LAYER residuals (trains + caches on 1st run)
probe = load_or_train_probe(
    model,
    base_template=VERBOSE_FIRST_LETTER_TEMPLATE,
    pos_idx=VERBOSE_FIRST_LETTER_TOKEN_POS,
    layer=LAYER,
    device=device,
)
# unit "starts with S" direction
probe_dir = probe.weights[letter_idx].detach().to(device=device, dtype=torch.float32)
probe_dir = probe_dir / probe_dir.norm()

# decoder<->probe cosine similarity for every latent (this is Fig 6b)
W_dec = sae.W_dec.detach()                                   # (d_sae, d_model)
cos_sims = F.cosine_similarity(probe_dir, W_dec.float(), dim=-1).cpu()  # (d_sae,)

# The "starts with S" detector is the latent most aligned with the probe.
main_latent = int(cos_sims.argmax())
print(f"main 'starts with {LETTER}' latent = {main_latent}  "
      f"(cos sim {cos_sims[main_latent]:+.3f})")

# --------------------- ICL spelling-prompt helpers -----------------------
alpha_tokens = get_alpha_tokens(model.tokenizer)  # vocab words, e.g. " short", " cat"
alpha_set = set(alpha_tokens)
assert TARGET_WORD in alpha_set, f"{TARGET_WORD!r} is not a single vocab token"

# One fixed ICL context so every prompt has identical length (needed for batching
# and for a stable word-token position). Examples exclude the target word.
example_pool = [w for w in alpha_tokens if w != TARGET_WORD]
icl_examples = rng.sample(example_pool, N_ICL)
icl_prefix = "".join(f"{w}:{first_letter(w)}\n" for w in icl_examples)

POS_WORD = -2   # prompt is "<prefix>{word}:" -> tokens [..., word, ":"]
POS_FINAL = -1  # ":" ; model predicts the first letter here


def build_prompt(word: str) -> str:
    return icl_prefix + f"{word}:"


# letter answer tokens " A".." Z" for the spelling metric
letter_token_ids = torch.tensor(
    [model.to_single_token(" " + chr(ord("A") + i)) for i in range(26)],
    device=device,
)


def spelling_metric(final_logits: torch.Tensor, correct_idx: int) -> torch.Tensor:
    """m = logit[correct letter] - mean logit[other 25 letters]. Shape (B,)."""
    ll = final_logits[:, letter_token_ids].float()          # (B, 26)
    correct = ll[:, correct_idx]
    others = (ll.sum(dim=1) - correct) / 25.0
    return correct - others


@torch.inference_mode()
def run_words(words: list[str]):
    """Return (sae_acts, resid_word, final_logits) for a batch of single-token words."""
    prompts = [build_prompt(w) for w in words]
    tokens = model.to_tokens(prompts)                       # same length by construction
    logits, cache = model.run_with_cache(tokens, names_filter=hook_name)
    resid_word = cache[hook_name][:, POS_WORD, :]           # (B, d_model)
    sae_acts = sae.encode(resid_word)                       # (B, d_sae)
    return sae_acts.float().cpu(), resid_word, logits[:, POS_FINAL, :]


# ============================ FIGURE 6 ===================================
# --- identify the absorber on the " short" prompt ------------------------
short_acts, _short_resid, short_logits = run_words([TARGET_WORD])
z_short = short_acts[0]                                     # (d_sae,)
m_clean = spelling_metric(short_logits, letter_idx).item()

# absorber = active, non-main latent with the largest probe projection (z * cos)
probe_proj = z_short * cos_sims
probe_proj[main_latent] = -float("inf")
probe_proj[z_short <= EPS] = -float("inf")
absorber = int(probe_proj.argmax())
print(f"absorber latent for {TARGET_WORD!r} = {absorber}  "
      f"(act {z_short[absorber]:.2f}, cos sim {cos_sims[absorber]:+.3f})")
print(f"clean spelling metric m({TARGET_WORD!r}) = {m_clean:+.3f}")

# width / L0 tags for the panel titles, straight off the checkpoint id
width_tag = SAE_ID.split("/")[1].removeprefix("width_")      # "16k"
l0_tag = SAE_ID.split("_")[-1]                               # "59"

# 6a: main vs absorber activation on the case-study tokens -----------------
for w in SHOW_WORDS:
    assert w in alpha_set, f"{w!r} is not a single vocab token"
s_acts, _, _ = run_words(SHOW_WORDS)                        # (n_words, d_sae)

fig6, (ax6a, ax6b) = plt.subplots(1, 2, figsize=(11.5, 3.0))

x = np.arange(len(SHOW_WORDS))
bar_w = 0.4
for k, (latent, color) in enumerate([(absorber, C_ABSORBER), (main_latent, C_MAIN)]):
    ax6a.bar(x + (k - 0.5) * bar_w, s_acts[:, latent].numpy(),
             width=bar_w, color=color, label=str(latent))
ax6a.set_xticks(x)
ax6a.set_xticklabels([w.strip() for w in SHOW_WORDS], rotation=45, ha="right")
ax6a.set_xlabel("Token")
ax6a.set_ylabel("Activation")
ax6a.set_title(f"‘{LETTER}’ activations by token, layer {LAYER}, "
               f"{width_tag} width, {l0_tag} L0")
ax6a.legend(title="Latent ID", fontsize=8, title_fontsize=8)

# 6b: decoder cosine similarity with the probe ----------------------------
ax6b.plot(np.arange(len(cos_sims)), cos_sims.numpy(), lw=0.5, color=C_ABSORBER)
for latent in (absorber, main_latent):
    ax6b.plot(latent, cos_sims[latent], "o", ms=3, color="black", zorder=5)

ax6b.annotate(f"Absorbing latent\nid {absorber}, cos {cos_sims[absorber]:.2f}",
              xy=(absorber, cos_sims[absorber]), xytext=(-6, 55),
              textcoords="offset points", fontsize=7, ha="left",
              arrowprops=dict(arrowstyle="-", lw=0.6, color="0.3"))
ax6b.annotate(f"Main latent\nid {main_latent}, cos {cos_sims[main_latent]:.2f}",
              xy=(main_latent, cos_sims[main_latent]), xytext=(52, -12),
              textcoords="offset points", fontsize=7, ha="left",
              arrowprops=dict(arrowstyle="-", lw=0.6, color="0.3"))

ax6b.set_xlabel("SAE latent ID")
ax6b.set_ylabel("Cosine similarity")
ax6b.set_title(f"Cosine similarity between SAE decoder and ‘{LETTER}’ probe")

fig6.tight_layout()

# ============================ FIGURE 7 ===================================
# Ablation effect on the " short" prompt for each ACTIVE latent.
# Only latents active at the word position change anything, so we ablate just those.
active = torch.nonzero(z_short > EPS).flatten().tolist()
print(f"{len(active)} active latents on {TARGET_WORD!r}")

short_tokens = model.to_tokens([build_prompt(TARGET_WORD)])          # (1, T)


@torch.inference_mode()
def ablation_effects(active_latents: list[int], W_dec_use: torch.Tensor) -> np.ndarray:
    """For each active latent, subtract its reconstruction contribution at the word
    position and return effect = m_ablated - m_clean. Done in one batched pass."""
    n = len(active_latents)
    batch = short_tokens.repeat(n, 1)                               # (n, T)
    # per-row rank-1 edit vector: z_f * W_dec[f]  (subtracted at POS_WORD)
    feats = torch.tensor(active_latents, device=device)
    sub = (z_short[active_latents].to(device, dtype).unsqueeze(1)
           * W_dec_use[feats].to(dtype))                            # (n, d_model)

    def hook(resid, hook):
        resid[:, POS_WORD, :] = resid[:, POS_WORD, :] - sub
        return resid

    logits = model.run_with_hooks(batch, fwd_hooks=[(hook_name, hook)])
    m_abl = spelling_metric(logits[:, POS_FINAL, :], letter_idx)    # (n,)
    return (m_abl - m_clean).float().cpu().numpy()


# 7a: raw ablation effects -------------------------------------------------
eff = ablation_effects(active, W_dec)

# 7b: project the probe direction out of the absorber's decoder vector -----
W_dec_proj = W_dec.clone()
d = W_dec_proj[absorber].float()
d_orth = d - (d @ probe_dir) * probe_dir                            # remove probe comp
W_dec_proj[absorber] = d_orth.to(W_dec_proj.dtype)
eff_proj = ablation_effects(active, W_dec_proj)

fig7, (ax7a, ax7b) = plt.subplots(1, 2, figsize=(12, 3.0))


def bar_panel(ax, values, title):
    """Bar the N_BARS_FIG7 most-negative effects, strongest ablation first."""
    order = np.argsort(values)[:N_BARS_FIG7]
    ids = [active[k] for k in order]
    ax.bar(np.arange(len(ids)), values[order], color=C_ABSORBER)
    ax.set_xticks(np.arange(len(ids)))
    ax.set_xticklabels([str(i) for i in ids], rotation=45, ha="right")
    ax.set_xlabel("SAE latent ID")
    ax.set_ylabel("Ablation effect")
    ax.set_title(title)


bar_panel(ax7a, eff, f'First-letter ablation effects for "{TARGET_WORD}" token, '
                     f'layer {LAYER}, L0={l0_tag}')
bar_panel(ax7b, eff_proj, f'Ablation effects for "{TARGET_WORD}" token '
                          f'projecting out probe')

# both panels on one scale, so (b)'s collapse is legible against (a)
lo = min(eff.min(), eff_proj.min())
for ax in (ax7a, ax7b):
    ax.set_ylim(lo * 1.08, 0)

fig7.tight_layout()

# ------------------------------- save ------------------------------------
ts = datetime.now().strftime("%Y%m%d_%H%M%S")
out_dir = Path(__file__).parent.parent / "figures"
out_dir.mkdir(parents=True, exist_ok=True)
p6 = out_dir / f"absorption_fig6_layer{LAYER}_{LETTER}_{ts}.png"
p7 = out_dir / f"absorption_fig7_layer{LAYER}_{LETTER}_{ts}.png"
fig6.savefig(p6, dpi=160, bbox_inches="tight")
fig7.savefig(p7, dpi=160, bbox_inches="tight")
plt.close(fig6)
plt.close(fig7)
print(f"Saved -> {p6}")
print(f"Saved -> {p7}")

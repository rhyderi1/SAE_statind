"""Render-only check of Fig 6a: feeds synthetic activations shaped like the paper's
result into the REAL plotting block, sliced verbatim out of chanin_fig6_final.py
(from the 'Figure 6a' banner to EOF) so the code under test cannot drift from the file."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from pathlib import Path
from datetime import datetime

SRC = Path("/project/aip-bahtol/rhyderi1/sae_statind/src/chanin_fig6_final.py")
lines = SRC.read_text().splitlines(keepends=True)
start = next(i for i, l in enumerate(lines) if l.startswith("#Claude: ---------- Figure 6a"))
plot_code = "".join(lines[start:])

# --- synthetic stand-ins for everything the plotting block reads ---
Layer, Arch, Sparsity = 3, "jumprelu", "59"
MAIN_LATENT, ABSORBING_LATENT = 6510, 1085
d_sae = 16384
target_words = ["snake", "soggy", "steam", "short", "soccer", "sax"]

# paper's pattern: 6510 fires ~7-11 on every S-token except _short (~0.5);
# 1085 is ~0 everywhere except _short, where it spikes to ~50.
g = torch.Generator().manual_seed(0)
saved = {}
for w in target_words:
    n = 40
    t = torch.rand(n, d_sae, generator=g) * 0.05          # background noise
    if w == "short":
        t[:, MAIN_LATENT] = 0.5 + torch.rand(n, generator=g) * 0.2
        t[:, ABSORBING_LATENT] = 50 + torch.randn(n, generator=g)
    else:
        t[:, MAIN_LATENT] = 7 + torch.rand(n, generator=g) * 4
        t[:, ABSORBING_LATENT] = torch.rand(n, generator=g) * 0.1
    saved[w] = t

exec(compile(plot_code, str(SRC), "exec"))

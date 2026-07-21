"""Diagnostic: assert that ZTZ and DTD come out symmetric.

Sanity check on a precomputed Gram matrix and on the decoder Gram D D^T. Prints
a line per matrix that passes and stays silent on failure.
"""

import torch
import numpy as np
from sae_lens import SAE
ZTZ = torch.load('data/pile-10k-saes/layer12_relu_k20/ztz_layer12_relu_k20.pt')
if np.allclose(ZTZ,ZTZ.T):
    print('ZTZ Symmettric')

release,sae_id = ("sae_bench_gemma-2-2b_vanilla_width-2pow14_date-1109", "blocks.12.hook_resid_post__trainer_0")
device     = "cuda" if torch.cuda.is_available() else "cpu"
sae = SAE.from_pretrained(release=release, sae_id=sae_id, device=device)


D = sae.W_dec
DTD = D@D.T
if np.allclose(DTD.numpy(), DTD.T.numpy()):
    print('DTD Symmettric')



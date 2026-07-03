import torch
from sae_lens import SAE

release, sae_id = ("sae_bench_gemma-2-2b_vanilla_width-2pow14_date-1109", "blocks.12.hook_resid_post__trainer_0")
Z = torch.load('data/Z/All batches/acts_layer12_relu_20_20260610_132348/Z_shard000.pt')
device = "cuda" if torch.cuda.is_available() else "cpu"

sae = SAE.from_pretrained(release=release, sae_id=sae_id,device=device)

print(Z.shape)
print(sae.W_dec.shape)

d_col = sae.W_dec.norm(dim = 1)
print(d_col.max(), d_col.min())
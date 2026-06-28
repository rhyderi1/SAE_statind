import torch
import numpy as np
import matplotlib.pyplot as plt
from infer_z import SAE_DATA
from sae_lens import SAE
from transformer_lens import HookedTransformer
from sae_lens import ActivationsStore

device = "cuda" if torch.cuda.is_available() else "cpu"

dataset = 'NeelNanda/pile-10k'
context_size = 128
batch_size = 32
hook_name = 'blocks.12.hook_resid_post'
release, sae_id = SAE_DATA[12]["relu"]["20"]
n_batches = 1209
layer = 12

sae = SAE.from_pretrained(release=release, sae_id=sae_id, device=device)

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

Z_act_all = []
num_entries = 0

with torch.no_grad(): 
    for i in range(n_batches):
        batch_tokens = activation_store.get_batch_tokens(batch_size)

        _, cache = model.run_with_cache( 
            batch_tokens,
            names_filter=hook_name,
            stop_at_layer=layer+1,
            prepend_bos=False,
        )
        X = cache[hook_name]   
        del cache 

        X_sae = X.to(device=sae.device, dtype=sae.W_enc.dtype) 
        Z = sae.encode(X_sae)  
        
        Z_act = Z[Z != 0]
        Z_act_all.append(Z_act.cpu())
        num_entries += Z.numel()

Z_final = torch.cat(Z_act_all).float().numpy()
num_zeros = num_entries - len(Z_final)

counts, bins = np.histogram(Z_final, bins=100)
counts[0] += num_zeros

plt.figure()
plt.bar(bins[:-1], counts, width=np.diff(bins), align='edge', log=True)
plt.xlabel("Activation (all entries)")
plt.ylabel("Count (log scale)")
plt.title("Z dist (Arch=ReLU, layer12, k=20)")
plt.savefig("figures/z_dist_relu_layer12_k20.png")   
plt.close()
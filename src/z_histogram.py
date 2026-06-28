import torch
import numpy as np
import matplotlib.pyplot as plt
from infer_z import SAE_DATA
from sae_lens import SAE
from transformer_lens import HookedTransformer
from sae_lens import ActivationsStore


device     = "cuda" if torch.cuda.is_available() else "cpu"

n_bins = 100
all_range = (0.0, 10.0)   # adjust upper bound to your data's max
dataset = 'NeelNanda/pile-10k'
context_size = 128
batch_size = 32
hook_name = 'blocks.12.hook_resid_post'
all_counts  = np.zeros(n_bins, dtype=np.int64)
active_counts = np.zeros(n_bins, dtype=np.int64)
total_entries = 0
total_zeros = 0
release,sae_id = SAE_DATA[12]["relu"]["20"]
n_batches = 1209
layer = 12

sae = SAE.from_pretrained(release=release, sae_id=sae_id, device=device)
model = HookedTransformer.from_pretrained_no_processing(
    default_prepend_bos=True, 
    model_name="gemma-2-2b"
    )
activation_store = ActivationsStore.from_sae(# from_sae initializes an ActivationsStore class (with the necessary parameters)(here, called activation_store)  
        model,
        sae,
        context_size=context_size,
        dataset=dataset,
    )

#Z_sum =torch.zeros()
with torch.no_grad(): #we don't need a computational graph, since it is pre-trained, and we are just doing inference
    for i in range(n_batches):
        batch_tokens = activation_store.get_batch_tokens(batch_size)
        # shape: (batch_size, context_size)

        _, cache = model.run_with_cache( # the core: actually obtaining the model's intenal activations
            batch_tokens,
            names_filter=hook_name,
            stop_at_layer=layer+1,
            prepend_bos=False,
        )
        X = cache[hook_name]   # extracts residual stream activations at layer.(dictionary lookup) (batch_size, context_size, d_model)
        del cache # delete the rest of the cache to save memory

        X_sae = X.to(device=sae.device, dtype=sae.W_enc.dtype) # so that when you multiply the two in the encode step, you don't get errors
        Z = sae.encode(X_sae)  # (batch_size, context_size, d_sae)
        print(Z.shape)
        break
        






# for i in range(94):
#     Z = torch.load(f'acts_layer20_jumprelu_71_20260607_142834/Z_shard{i:03d}.pt')
#     flat = Z.float().numpy().ravel()

#     total_entries += flat.size
#     total_zeros   += (flat == 0).sum()

#     c, edges = np.histogram(flat, bins=n_bins, range=all_range)
#     all_counts += c

#     active = flat[flat > 0]
#     c2, _ = np.histogram(active, bins=n_bins, range=all_range)
#     active_counts += c2

#     del Z, flat, active   # free immediately



# sparsity = total_zeros / total_entries
# print(f"Sparsity: {sparsity:.3f}")

# plt.figure()
# plt.bar(edges[:-1], all_counts, width=np.diff(edges), align='edge', log=True)
# plt.xlabel("Activation (all entries)")
# plt.ylabel("Count (log scale)")
# plt.title(f"All activation values — sparsity {sparsity:.2f}")
# plt.savefig("z_dist.png")
# plt.close()

# plt.figure()
# plt.bar(edges[:-1], active_counts, width=np.diff(edges), align='edge')
# plt.xlabel("Activation (active entries only)")
# plt.ylabel("Count")
# plt.title("Active activation values")
# plt.savefig("z_dist_active.png")
# plt.close()
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
value_sum_per_feature = torch.zeros(sae.W_dec.shape[0], dtype=torch.float32, device=device)
max_per_feature       = torch.zeros(sae.W_dec.shape[0], dtype=torch.float32, device=device)
num_nonzeros = torch.zeros(sae.W_dec.shape[0], dtype=torch.float32, device=device)
num_entries_total = 0
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
        Z = Z.reshape(-1,Z.shape[-1]) # shape should now be (tokens, p)
        value_sum_per_feature += Z.sum(dim=0)
        max_per_feature = torch.maximum(max_per_feature, Z.max(dim=0).values)
        # nonzero = (Z != 0).sum(dim=0)
        # num_nonzeros += nonzero
        num_entries = Z.shape[0]
        num_entries_total += num_entries


value_mean_per_feature = value_sum_per_feature / num_entries_total

# value_active_mean_per_feature = value_sum_per_feature / num_nonzeros.clamp(min=1)
# value_active_mean_per_feature[num_nonzeros == 0] = float('nan')

value_mean_per_feature = value_mean_per_feature.cpu().numpy()
max_per_feature        = max_per_feature.cpu().numpy()
# value_active_mean_per_feature = value_active_mean_per_feature.cpu().numpy()






plt.hist(value_mean_per_feature, bins=100, log=True)
plt.xlabel("Feature mean (all entries)")
plt.ylabel("Count (log scale)")
plt.title("Feature mean dist (Arch=ReLU, layer12, k=20)")
plt.savefig("figures/feature_mean_dist_relu_layer12_k20.png")
plt.close()

# plt.hist(value_active_mean_per_feature[~np.isnan(value_active_mean_per_feature)], bins=100, log=True)
# plt.xlabel("Feature mean (active entries)")
# plt.ylabel("Count (log scale)")
# plt.title("Feature mean dist (active entries) (Arch=ReLU, layer12, k=20)")
# plt.savefig("figures/feature_mean_dist_active_relu_layer12_k20.png")
# plt.close()

plt.hist(max_per_feature, bins=100, log=True)
plt.xlabel("Feature max activation")
plt.ylabel("Count (log scale)")
plt.title("Feature max activation dist (Arch=ReLU, layer12, k=20)")
plt.savefig("figures/feature_max_dist_relu_layer12_k20.png")
plt.close()

plt.scatter(value_mean_per_feature, max_per_feature, s=1, alpha=0.3)
plt.xlabel("Mean activation (all tokens)")
plt.ylabel("Max activation")
plt.title("Max vs Mean activation per feature (Arch=ReLU, layer12, k=20)")
plt.savefig("figures/feature_max_vs_mean_relu_layer12_k20.png")
plt.close()

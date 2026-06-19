from sae_lens import SAE
import torch
print("ReLU:")
device     = "cpu"

release, sae_id = ("sae_bench_gemma-2-2b_vanilla_width-2pow14_date-1109", "blocks.12.hook_resid_post__trainer_0")
sae = SAE.from_pretrained(release=release, sae_id=sae_id, device=device)

# 1. Does the module even have the attribute?
print(hasattr(sae, "b_enc"))

# 2. If it exists, is it actually doing anything, or is it dead/zero?
if hasattr(sae, "b_enc"):
    b = sae.b_enc.detach()
    print(b.shape, b.abs().mean().item(), b.abs().max().item())

# 3. Cheapest real test: does removing it change the output at all?
x = torch.randn(8, sae.W_enc.shape[0])  # dummy activations, right d_model
with torch.no_grad():
    z_with = sae.encode(x)
    if hasattr(sae, "b_enc"):
        saved = sae.b_enc.clone()
        sae.b_enc.zero_()
        z_without = sae.encode(x)
        sae.b_enc.copy_(saved)
        print("max abs diff:", (z_with - z_without).abs().max().item())

release, sae_id = ("sae_bench_gemma-2-2b_topk_width-2pow14_date-1109", "blocks.12.hook_resid_post__trainer_0")
sae = SAE.from_pretrained(release=release, sae_id=sae_id, device=device)
print('TopK')
# 1. Does the module even have the attribute?
print(hasattr(sae, "b_enc"))

# 2. If it exists, is it actually doing anything, or is it dead/zero?
if hasattr(sae, "b_enc"):
    b = sae.b_enc.detach()
    print(b.shape, b.abs().mean().item(), b.abs().max().item())

# 3. Cheapest real test: does removing it change the output at all?
import torch
x = torch.randn(8, sae.W_enc.shape[0])  # dummy activations, right d_model
with torch.no_grad():
    z_with = sae.encode(x)
    if hasattr(sae, "b_enc"):
        saved = sae.b_enc.clone()
        sae.b_enc.zero_()
        z_without = sae.encode(x)
        sae.b_enc.copy_(saved)
        print("max abs diff:", (z_with - z_without).abs().max().item())

release, sae_id = ("gemma-scope-2b-pt-res", "layer_12/width_16k/average_l0_22")
sae = SAE.from_pretrained(release=release, sae_id=sae_id, device=device)
print('JumpReLU')
# 1. Does the module even have the attribute?
print(hasattr(sae, "b_enc"))

# 2. If it exists, is it actually doing anything, or is it dead/zero?
if hasattr(sae, "b_enc"):
    b = sae.b_enc.detach()
    print(b.shape, b.abs().mean().item(), b.abs().max().item())

# 3. Cheapest real test: does removing it change the output at all?
import torch
x = torch.randn(8, sae.W_enc.shape[0])  # dummy activations, right d_model
with torch.no_grad():
    z_with = sae.encode(x)
    if hasattr(sae, "b_enc"):
        saved = sae.b_enc.clone()
        sae.b_enc.zero_()
        z_without = sae.encode(x)
        sae.b_enc.copy_(saved)
        print("max abs diff:", (z_with - z_without).abs().max().item())


release, sae_id =  ("gemma-2-2b-res-matryoshka-dc", "blocks.12.hook_resid_post")
sae = SAE.from_pretrained(release=release, sae_id=sae_id, device=device)
print('Matryoshka')
# 1. Does the module even have the attribute?
print(hasattr(sae, "b_enc"))

# 2. If it exists, is it actually doing anything, or is it dead/zero?
if hasattr(sae, "b_enc"):
    b = sae.b_enc.detach()
    print(b.shape, b.abs().mean().item(), b.abs().max().item())

# 3. Cheapest real test: does removing it change the output at all?
import torch
x = torch.randn(8, sae.W_enc.shape[0])  # dummy activations, right d_model
with torch.no_grad():
    z_with = sae.encode(x)
    if hasattr(sae, "b_enc"):
        saved = sae.b_enc.clone()
        sae.b_enc.zero_()
        z_without = sae.encode(x)
        sae.b_enc.copy_(saved)
        print("max abs diff:", (z_with - z_without).abs().max().item())


import torch

device = "cuda" if torch.cuda.is_available() else "cpu"

tokens = 4096
n_features = 16384

Z = torch.randn(tokens, n_features, device=device)

col = Z[:, 5]  # single-column slice, like Z[:, latent] in zizj_scatter.py
print("col.shape:", col.shape, "col.numel():", col.numel())
print("col.is_contiguous():", col.is_contiguous())
print("col storage nbytes (on GPU, shared with Z):", col.untyped_storage().nbytes())

col_cpu = col.cpu()
print("\nAfter .cpu():")
print("col_cpu.shape:", col_cpu.shape, "col_cpu.numel():", col_cpu.numel())
print("col_cpu.is_contiguous():", col_cpu.is_contiguous())
print("col_cpu storage nbytes:", col_cpu.untyped_storage().nbytes())
print("expected bytes if right-sized (numel * 4 bytes):", col_cpu.numel() * 4)
print("expected bytes if full Z-sized copy:", Z.numel() * 4)

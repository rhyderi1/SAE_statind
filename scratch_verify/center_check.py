import gc
import torch
from transformer_lens import HookedTransformer

device = "cuda" if torch.cuda.is_available() else "cpu"

model_raw = HookedTransformer.from_pretrained_no_processing(model_name="gemma-2-2b", device=device)
tokens = model_raw.to_tokens("The quick brown fox jumps over the lazy dog.")
with torch.no_grad():
    _, cache_raw = model_raw.run_with_cache(tokens, names_filter="blocks.12.hook_resid_post", stop_at_layer=13)
r = cache_raw["blocks.12.hook_resid_post"].clone()
del model_raw, cache_raw
gc.collect()
torch.cuda.empty_cache() if torch.cuda.is_available() else None

model_proc = HookedTransformer.from_pretrained(model_name="gemma-2-2b", device=device)
with torch.no_grad():
    _, cache_proc = model_proc.run_with_cache(tokens, names_filter="blocks.12.hook_resid_post", stop_at_layer=13)
p = cache_proc["blocks.12.hook_resid_post"].clone()

diff = (r - p).abs()
print("raw resid norm (mean over tokens):", r.norm(dim=-1).mean().item())
print("max abs diff:", diff.max().item())
print("mean abs diff:", diff.mean().item())
print("relative diff (mean abs diff / mean resid norm):", (diff.mean() / r.norm(dim=-1).mean()).item())
print("identical tensors:", torch.equal(r, p))

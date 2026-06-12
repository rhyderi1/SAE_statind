import torch

# Load file
data = torch.load("sae_activations.pt")

# Access contents
X = data["X"]
Z = data["Z"]
tokens = data["tokens"]
texts = data["texts"]

print("X shape:", X.shape)
print("Z shape:", Z.shape)
print("Tokens shape:", tokens.shape)
print("Texts:", texts)
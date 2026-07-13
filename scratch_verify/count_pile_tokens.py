"""What slice of NeelNanda/pile-10k does a 128x32x1209 ActivationsStore run actually see?"""
import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer

CONTEXT, BATCH, NBATCHES = 128, 32, 1209
BUDGET = CONTEXT * BATCH * NBATCHES
# concat_and_batch_sequences puts a BOS at the start of every context window
CORPUS_BUDGET = BUDGET * (CONTEXT - 1) // CONTEXT

ds = load_dataset("NeelNanda/pile-10k", split="train")
tok = AutoTokenizer.from_pretrained("google/gemma-2-2b")
lens = np.array([len(x) for x in tok(ds["text"], add_special_tokens=False)["input_ids"]])
total = lens.sum()
cum = np.cumsum(lens)

n_docs = int(np.searchsorted(cum, CORPUS_BUDGET) + 1)
print(f"budget          : {BUDGET:,} tokens ({CORPUS_BUDGET:,} corpus + {BUDGET - CORPUS_BUDGET:,} BOS)")
print(f"corpus total    : {total:,} tokens / {len(lens):,} docs")
print(f"fraction seen   : {CORPUS_BUDGET / total:.1%}")
print(f"docs consumed   : {n_docs:,} of {len(lens):,}  ({n_docs / len(lens):.1%})")

seen = lens[:n_docs].copy()
seen[-1] = CORPUS_BUDGET - cum[n_docs - 2] if n_docs > 1 else CORPUS_BUDGET
order = np.argsort(seen)[::-1][:5]
print("\nbiggest docs inside the consumed slice (doc_idx, tokens, % of budget):")
for i in order:
    print(f"  doc {i:>5}  {seen[i]:>9,}  {seen[i] / CORPUS_BUDGET:6.2%}")

big = np.argsort(lens)[::-1][:5]
print("\nbiggest docs in the whole corpus (doc_idx, tokens, seen?):")
for i in big:
    print(f"  doc {i:>5}  {lens[i]:>9,}  {'yes' if i < n_docs else 'NO'}")

from datasets import load_dataset
from transformers import AutoTokenizer

# 1. Load the dataset and the Gemma 2 2B tokenizer
dataset = load_dataset("NeelNanda/pile-10k", split="train")
tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-2b")

# 2. Define a function to count tokens for each row
def count_tokens(example):
    # We only care about the length of the input_ids array
    return {"token_count": len(tokenizer(example["text"])["input_ids"])}

# 3. Apply the function across the dataset (using multiple cores for speed)
tokenized_dataset = dataset.map(count_tokens, num_proc=4)

# 4. Sum up all the token counts
total_tokens = sum(tokenized_dataset["token_count"])

print(f"Total tokens in Pile10k using gemma-2-2b: {total_tokens:,}")
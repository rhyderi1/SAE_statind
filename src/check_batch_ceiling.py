"""Compute the exact number of batches needed to process a whole dataset once
in the sae_statind activation-collection pipeline.

Given context_size and batch_size, `get_batch_tokens` streams `batch_size`
context windows per call. This script reports how many *full* batches exist
before SAELens's ActivationsStore wraps around to the next epoch, so you can set
`n_batches` correctly and avoid silently double-counting the start of the data.

Two modes:

  --mode fast   (default) Replays SAELens's own `concat_and_batch_sequences`
                packing with the given tokenizer. Deterministic, tokenizer-only,
                no model weights loaded. This is byte-identical to how
                ActivationsStore packs when the dataset is NOT pretokenized,
                prepend_bos=True, sequence_separator_token="bos",
                disable_concat_sequences=False (the pipeline defaults).

  --mode store  Ground-truth check: builds the real ActivationsStore via
                from_sae and calls get_batch_tokens(..., raise_at_epoch_end=True)
                until StopIteration, counting batches. Loads the model, so run it
                where the model is available (i.e. on a GPU node).

Example (fast):
    python src/check_batch_ceiling.py

Example (ground truth, on a compute node):
    python src/check_batch_ceiling.py --mode store \
        --release gemma-scope-2b-pt-res-canonical \
        --sae-id layer_12/width_16k/canonical
"""

import argparse

import torch


def count_fast(dataset_name: str, hf_tokenizer: str, context_size: int,
               batch_size: int, text_column: str) -> tuple[int, int, int]:
    """Replicate ActivationsStore packing using SAELens's own batching fn."""
    from datasets import load_dataset
    from transformers import AutoTokenizer
    from sae_lens.tokenization_and_batching import concat_and_batch_sequences

    tok = AutoTokenizer.from_pretrained(hf_tokenizer)
    bos = tok.bos_token_id
    ds = load_dataset(dataset_name, split="train")

    content = 0

    def row_tokens():
        nonlocal content
        for row in ds:
            ids = tok(row[text_column], add_special_tokens=False)["input_ids"]
            content += len(ids)
            yield torch.tensor(ids, dtype=torch.long)

    n_windows = sum(
        1
        for _ in concat_and_batch_sequences(
            tokens_iterator=row_tokens(),
            context_size=context_size,
            begin_batch_token_id=bos,          # prepend_bos=True
            begin_sequence_token_id=None,
            sequence_separator_token_id=bos,   # sequence_separator_token="bos"
            disable_concat_sequences=False,
        )
    )
    return content, n_windows, len(ds)


def count_store(dataset_name: str, model_name: str, context_size: int,
                batch_size: int, release: str, sae_id: str,
                device: str) -> int:
    """Ground truth: drive the real ActivationsStore until it wraps."""
    from sae_lens import SAE, ActivationsStore
    from transformer_lens import HookedTransformer

    sae = SAE.from_pretrained(release=release, sae_id=sae_id, device=device)
    model = HookedTransformer.from_pretrained_no_processing(
        default_prepend_bos=True, model_name=model_name, device=device
    )
    store = ActivationsStore.from_sae(
        model, sae, context_size=context_size, dataset=dataset_name
    )

    n_batches = 0
    while True:
        try:
            store.get_batch_tokens(batch_size, raise_at_epoch_end=True)
        except StopIteration:
            break
        n_batches += 1
    return n_batches


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mode", choices=["fast", "store"], default="fast")
    p.add_argument("--dataset", default="NeelNanda/pile-10k")
    p.add_argument("--model-name", default="gemma-2-2b",
                   help="TransformerLens model name (used by --mode store)")
    p.add_argument("--hf-tokenizer", default="google/gemma-2-2b",
                   help="HF tokenizer repo id (used by --mode fast)")
    p.add_argument("--text-column", default="text")
    p.add_argument("--context-size", type=int, default=128)
    p.add_argument("--batch-size", type=int, default=32)
    # only needed for --mode store
    p.add_argument("--release", default=None)
    p.add_argument("--sae-id", default=None)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    print(f"dataset={args.dataset}  model={args.model_name}  "
          f"context_size={args.context_size}  batch_size={args.batch_size}\n")

    if args.mode == "store":
        if not (args.release and args.sae_id):
            p.error("--mode store requires --release and --sae-id")
        full_batches = count_store(
            args.dataset, args.model_name, args.context_size,
            args.batch_size, args.release, args.sae_id, args.device
        )
        print(f"[store] FULL batches before wrap-around: {full_batches:,}")
        print(f"        set n_batches = {full_batches} to process all data once")
        return

    content, n_windows, n_rows = count_fast(
        args.dataset, args.hf_tokenizer, args.context_size,
        args.batch_size, args.text_column
    )
    full_batches = n_windows // args.batch_size
    remainder = n_windows % args.batch_size

    print(f"rows in dataset:                 {n_rows:,}")
    print(f"content tokens (no specials):    {content:,}")
    print(f"context windows of {args.context_size}:          {n_windows:,}")
    print(f"FULL batches of {args.batch_size}x{args.context_size}:          {full_batches:,}")
    print(f"windows in the partial tail:     {remainder}")
    print()
    print(f"-> set n_batches = {full_batches} to process all full batches once")
    print(f"   (n_batches > {full_batches} silently re-processes the start of the "
          f"dataset -> double counting;")
    print(f"    use get_batch_tokens(..., raise_at_epoch_end=True) + break to stop "
          f"cleanly at the end.)")


if __name__ == "__main__":
    main()

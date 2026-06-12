from sae_lens import SAE

saes = [
    ("gemma-scope-2b-pt-res", "layer_12/width_65k/average_l0_21"),
    ("gemma-scope-2b-pt-res", "layer_12/width_65k/average_l0_38"),
    ("gemma-scope-2b-pt-res", "layer_12/width_65k/average_l0_72"),
    ("gemma-scope-2b-pt-res", "layer_12/width_65k/average_l0_141"),
    ("sae_bench_gemma-2-2b_topk_width-2pow16_date-1109", "blocks.12.hook_resid_post__trainer_0"),
    ("sae_bench_gemma-2-2b_topk_width-2pow16_date-1109", "blocks.12.hook_resid_post__trainer_1"),
    ("sae_bench_gemma-2-2b_topk_width-2pow16_date-1109", "blocks.12.hook_resid_post__trainer_2"),
    ("sae_bench_gemma-2-2b_topk_width-2pow16_date-1109", "blocks.12.hook_resid_post__trainer_3"),
    ("gemma-2-2b-res-matryoshka-dc", "blocks.12.hook_resid_post"),
    ("gemma-scope-2b-pt-res", "layer_20/width_65k/average_l0_61"),
    ("sae_bench_gemma-2-2b_topk_width-2pow16_date-1109", "blocks.19.hook_resid_post__trainer_2"),
    ("gemma-2-2b-res-matryoshka-dc", "blocks.20.hook_resid_post"),
]

for release, sae_id in saes:
    print(f"Downloading {release} | {sae_id}...")
    SAE.from_pretrained(release=release, sae_id=sae_id)
    print("  done.")
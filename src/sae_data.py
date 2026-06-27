SAE_DATA = {
    12: {
        "relu": {
            "20": ("sae_bench_gemma-2-2b_vanilla_width-2pow14_date-1109", "blocks.12.hook_resid_post__trainer_0"),
            "40": ("sae_bench_gemma-2-2b_vanilla_width-2pow14_date-1109", "blocks.12.hook_resid_post__trainer_1"),
            "80": ("sae_bench_gemma-2-2b_vanilla_width-2pow14_date-1109", "blocks.12.hook_resid_post__trainer_2"),
        },
        "topk": {
            "20": ("sae_bench_gemma-2-2b_topk_width-2pow14_date-1109", "blocks.12.hook_resid_post__trainer_0"),
            "40": ("sae_bench_gemma-2-2b_topk_width-2pow14_date-1109", "blocks.12.hook_resid_post__trainer_1"),
            "80": ("sae_bench_gemma-2-2b_topk_width-2pow14_date-1109", "blocks.12.hook_resid_post__trainer_2"),
        },
        "jumprelu": {
            "22": ("gemma-scope-2b-pt-res", "layer_12/width_16k/average_l0_22"),
            "41": ("gemma-scope-2b-pt-res", "layer_12/width_16k/average_l0_41"),
            "82": ("gemma-scope-2b-pt-res-canonical", "layer_12/width_16k/average_l0_82"),
        },
        "matryoshka": {
            "40": ("gemma-2-2b-res-matryoshka-dc", "blocks.12.hook_resid_post"),
        },
    },
    19: {
        "relu": {
            "20": ("sae_bench_gemma-2-2b_vanilla_width-2pow14_date-1109", "blocks.19.hook_resid_post__trainer_0"),
            "40": ("sae_bench_gemma-2-2b_vanilla_width-2pow14_date-1109", "blocks.19.hook_resid_post__trainer_1"),
            "80": ("sae_bench_gemma-2-2b_vanilla_width-2pow14_date-1109", "blocks.19.hook_resid_post__trainer_2"),
        },
        "topk": {
            "20": ("sae_bench_gemma-2-2b_topk_width-2pow14_date-1109", "blocks.19.hook_resid_post__trainer_0"),
            "40": ("sae_bench_gemma-2-2b_topk_width-2pow14_date-1109", "blocks.19.hook_resid_post__trainer_1"),
            "80": ("sae_bench_gemma-2-2b_topk_width-2pow14_date-1109", "blocks.19.hook_resid_post__trainer_2"),
        },
    
        "jumprelu": {
            "23":        ("gemma-scope-2b-pt-res", "layer_19/width_16k/average_l0_23"),
            "40":       ("gemma-scope-2b-pt-res", "layer_19/width_16k/average_l0_40"),
            "73": ("gemma-scope-2b-pt-res", "layer_19/width_16k/average_l0_73"),
        },
        "matryoshka": {
            "40": ("gemma-2-2b-res-matryoshka-dc", "blocks.19.hook_resid_post"),
        },
    },
}

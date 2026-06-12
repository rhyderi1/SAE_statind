from sae_lens import SAE

# SAE_DATA = {
#     12: {
#         "relu": {
#             "20": ("sae_bench_gemma-2-2b_vanilla_width-2pow14_date-1109", "blocks.12.hook_resid_post__trainer_0"),
#             "40": ("sae_bench_gemma-2-2b_vanilla_width-2pow14_date-1109", "blocks.12.hook_resid_post__trainer_1"),
#             "80": ("sae_bench_gemma-2-2b_vanilla_width-2pow14_date-1109", "blocks.12.hook_resid_post__trainer_2"),
#         },
#         "topk": {
#             "20": ("sae_bench_gemma-2-2b_topk_width-2pow14_date-1109", "blocks.12.hook_resid_post__trainer_0"),
#             "40": ("sae_bench_gemma-2-2b_topk_width-2pow14_date-1109", "blocks.12.hook_resid_post__trainer_1"),
#             "80": ("sae_bench_gemma-2-2b_topk_width-2pow14_date-1109", "blocks.12.hook_resid_post__trainer_2"),
#         },
#         "batchtopk": {
#             "20": ("saebench_gemma-2-2b_width-2pow14_date-0107", "blocks.12.hook_resid_post__trainer_0"),
#             "40": ("saebench_gemma-2-2b_width-2pow14_date-0107", "blocks.12.hook_resid_post__trainer_1"),
#             "80": ("saebench_gemma-2-2b_width-2pow14_date-0107", "blocks.12.hook_resid_post__trainer_2"),
#         },
#         "jumprelu": {
#             "22":        ("gemma-scope-2b-pt-res", "layer_12/width_16k/average_l0_22"),
#             "72":        ("gemma-scope-2b-pt-res", "layer_12/width_16k/average_l0_72"),
#             "canonical": ("gemma-scope-2b-pt-res-canonical", "layer_12/width_16k/canonical"),
#         },
#         "matryoshka": {
#             "default": ("gemma-2-2b-res-matryoshka-dc", "blocks.12.hook_resid_post"),
#         },
#     },
#     20: {
#         "relu": {
#             "20": ("sae_bench_gemma-2-2b_vanilla_width-2pow14_date-1109", "blocks.20.hook_resid_post__trainer_0"),
#             "40": ("sae_bench_gemma-2-2b_vanilla_width-2pow14_date-1109", "blocks.20.hook_resid_post__trainer_1"),
#             "80": ("sae_bench_gemma-2-2b_vanilla_width-2pow14_date-1109", "blocks.20.hook_resid_post__trainer_2"),
#         },
#         "topk": {
#             "20": ("sae_bench_gemma-2-2b_topk_width-2pow14_date-1109", "blocks.20.hook_resid_post__trainer_0"),
#             "40": ("sae_bench_gemma-2-2b_topk_width-2pow14_date-1109", "blocks.20.hook_resid_post__trainer_1"),
#             "80": ("sae_bench_gemma-2-2b_topk_width-2pow14_date-1109", "blocks.20.hook_resid_post__trainer_2"),
#         },
#         "batchtopk": {
#             "20": ("saebench_gemma-2-2b_width-2pow14_date-0107", "blocks.20.hook_resid_post__trainer_0"),
#             "40": ("saebench_gemma-2-2b_width-2pow14_date-0107", "blocks.20.hook_resid_post__trainer_1"),
#             "80": ("saebench_gemma-2-2b_width-2pow14_date-0107", "blocks.20.hook_resid_post__trainer_2"),
#         },
#         "jumprelu": {
#             "71":        ("gemma-scope-2b-pt-res", "layer_20/width_16k/average_l0_71"),
#             "176":       ("gemma-scope-2b-pt-res", "layer_20/width_16k/average_l0_176"),
#             "canonical": ("gemma-scope-2b-pt-res-canonical", "layer_20/width_16k/canonical"),
#         },
#         "matryoshka": {
#             "default": ("gemma-2-2b-res-matryoshka-dc", "blocks.20.hook_resid_post"),
#         },
#     },
# }

# for layer, arch_dict in SAE_DATA.keys():
#     for arch, sparsity_dict in arch_dict.keys():
#         for sparsity, sae_info in sparsity_dict.keys():
#             release = sae_info[0]
#             sae_id = sae_info[1]


# sae, cfg_dict, sparsity = SAE.from_pretrained_with_cfg_and_sparsity(
#     release=release,
#     sae_id=sae_id,
#     avg_l0 = sparsity.sum().item()
# )

# print(cfg_dict)
# print(sae.cfg)


from sae_lens import SAE

release = "gemma-scope-2b-pt-res"
sae_id = "layer_12/width_16k/average_l0_72"
sae = SAE.from_pretrained(release, sae_id)
print(sae.cfg)
"""Run SAEBench's feature-absorption eval over a list of SAEs.

Thin driver around sae_bench.evals.absorption: builds an AbsorptionEvalConfig,
runs the eval on the SAEs in `selected_saes`, and writes one
<release>_<sae_id>_eval_results.json per SAE into eval_results/absorption/,
printing each SAE's absorption rate (lower = less absorption = better).

The eval also drops per-SAE parquet artifacts inside the installed sae_bench
package tree; extract_absorption_sets.py reads those to build S_main/S_abs.

First run downloads the model and SAEs and takes 30-60+ minutes. force_rerun is
currently True, so cached results are ignored.
"""

import json
import os
import time


# import from the SAEbench repo 
from sae_lens import SAE
from sae_bench.evals.absorption.eval_config import AbsorptionEvalConfig
from sae_bench.evals.absorption.main import run_eval
from sae_bench.sae_bench_utils import activation_collection, general_utils

#CHECKPOINT_PATH = "output"  # update to specific subfolder after training

MODEL_NAME = "gemma-2-2b"                          # absorption needs a >=1B model
SAE_RELEASE = "gemma-scope-2b-pt-res-canonical"    # SAE Lens release name
SAE_ID = "layer_20/width_16k/canonical"            # SAE within that release

#absorption hyperparameters in Absorption Eval Config (using defaults)
config = AbsorptionEvalConfig(
    model_name=MODEL_NAME,
    random_seed=42,
)
#llm_batch_size and llm_dtype are model-specific
#They provide a dictionary of validated/recommended batch sizes and datatypes in activation_collection
config.llm_batch_size = activation_collection.LLM_NAME_TO_BATCH_SIZE[MODEL_NAME]  # 32
config.llm_dtype = activation_collection.LLM_NAME_TO_DTYPE[MODEL_NAME]            # bfloat16


device = general_utils.setup_environment()# chooses GPU

sae, cfg_dict, sparsity = SAE.from_pretrained(
    release = SAE_RELEASE,
    sae_id = SAE_ID,
)

selected_saes = [
    # --- Layer 12 | JumpReLU ---
    ("gemma-scope-2b-pt-res", "layer_12/width_65k/average_l0_21"),
    ("gemma-scope-2b-pt-res", "layer_12/width_65k/average_l0_38"),
    ("gemma-scope-2b-pt-res", "layer_12/width_65k/average_l0_72"),
    ("gemma-scope-2b-pt-res", "layer_12/width_65k/average_l0_141"),
    # --- Layer 12 | TopK ---
    ("sae_bench_gemma-2-2b_topk_width-2pow16_date-1109", "blocks.12.hook_resid_post__trainer_0"),
    ("sae_bench_gemma-2-2b_topk_width-2pow16_date-1109", "blocks.12.hook_resid_post__trainer_1"),
    ("sae_bench_gemma-2-2b_topk_width-2pow16_date-1109", "blocks.12.hook_resid_post__trainer_2"),
    ("sae_bench_gemma-2-2b_topk_width-2pow16_date-1109", "blocks.12.hook_resid_post__trainer_3"),
    # --- Layer 12 | Matryoshka ---
    ("gemma-2-2b-res-matryoshka-dc", "blocks.12.hook_resid_post"),
    # --- Layer 20 | JumpReLU ---
    ("gemma-scope-2b-pt-res", "layer_20/width_65k/average_l0_61"),
    # --- Layer 20 | TopK ---
    ("sae_bench_gemma-2-2b_topk_width-2pow16_date-1109", "blocks.19.hook_resid_post__trainer_2"),
    # --- Layer 20 | Matryoshka ---
    ("gemma-2-2b-res-matryoshka-dc", "blocks.20.hook_resid_post"),
]

output_folder = "eval_results/absorption"
os.makedirs(output_folder, exist_ok=True)

print(f"Running absorption eval on {MODEL_NAME} ({device})")
print("This downloads the model + SAE on first run and can take 30-60+ minutes.\n")

start = time.time()
results = run_eval(
    config=config,
    selected_saes=selected_saes,
    device=device,
    output_path=output_folder,
    force_rerun=True,   # set True to ignore cached results and recompute
)
print(f"\nFinished in {time.time() - start:.0f}s")

# results maps "<release>_<sae_id>" -> the full eval output dict. The same
# dict is also saved to <output_folder>/<release>_<sae_id>_eval_results.json.
for sae_key, eval_output in results.items(): # iterates through mean stats and per-letter stats
    mean = eval_output["eval_result_metrics"]["mean"]
    print(f"\n=== {sae_key} ===")
    #print(f"  mean_absorption_fraction_score : {mean['mean_absorption_fraction_score']:.4f}")
    print(f"  absorption_rate: {mean['mean_full_absorption_score']:.4f}")
    #print(f"  mean_num_split_features        : {mean['mean_num_split_features']:.2f}")
    print("  (lower absorption scores = the SAE absorbs less = better)")

print(f"\nFull JSON results saved under: {os.path.abspath(output_folder)}")

import os
import inspect
import torch
from transformer_lens import HookedTransformer
from sae_lens import SAE

from sae_bench.evals.absorption.eval_config import AbsorptionEvalConfig
from sae_bench.evals.absorption.main import run_eval

def main():
    # 1. Auth and Device setup
    os.environ["HF_TOKEN"] = "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"  # <-- INSERT YOUR TOKEN HERE
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # 2. LOAD MODEL IN BFLOAT16 TO PREVENT OOM (Killed)
    print("Loading Gemma-2-2B in memory-saving bfloat16...")
    model = HookedTransformer.from_pretrained(
        "gemma-2-2b", 
        device=device, 
        dtype=torch.bfloat16
    )

    # 3. Load the SAE directly via sae_lens
    print("Loading Layer 12 SAE...")
    sae_name_str = "gemma-scope-2b-pt-res_layer_12_width_16k_average_l0_82"
    sae = SAE.from_pretrained(
        release="gemma-scope-2b-pt-res", 
        sae_id="layer_12/width_16k/average_l0_82", 
        device=device
    )
    sae = sae.to(dtype=torch.bfloat16)  # Match model precision

    # 4. Configure Memory-Saving Eval Config
    config = AbsorptionEvalConfig()
    config.model_name = "gemma-2-2b"
    config.llm_dtype = "bfloat16"
    config.llm_batch_size = 16
    config.k_sparse_probe_batch_size = 1024
    
    # 5. The SAE dictionary (sae_bench requires them to be packed in a dict)
    selected_saes = {sae_name_str: sae}

    # 6. DYNAMIC KWARG MATCHING (The Magic Fix)
    # This reads the function signature of your specific sae_bench version 
    # and only passes the exact arguments it wants!
    sig = inspect.signature(run_eval)
    kwargs = {}
    
    if "config" in sig.parameters: 
        kwargs["config"] = config
    if "selected_saes" in sig.parameters: 
        kwargs["selected_saes"] = selected_saes
    if "saes" in sig.parameters: 
        kwargs["saes"] = selected_saes
    if "model" in sig.parameters: 
        kwargs["model"] = model
    if "output_folder" in sig.parameters:
        kwargs["output_folder"] = "eval_results/absorption"
    if "results_dir" in sig.parameters:
        kwargs["results_dir"] = "eval_results/absorption"

    print(f"Calling run_eval with arguments: {list(kwargs.keys())}")
    
    # Run the evaluation!
    run_eval(**kwargs)
    
    print("\n[*] Eval complete! Parquet saved.")

if __name__ == "__main__":
    main()
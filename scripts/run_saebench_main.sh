#!/bin/bash
#SBATCH --account=aip-bahtol
#SBATCH --gpus-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=%x_%j.out

module load gcc arrow/24.0.0
source /project/aip-bahtol/rhyderi1/sae_statind/.venv/bin/activate
export HF_TOKEN=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   

python -m sae_bench.evals.absorption.main \
  --sae_regex_pattern "sae_bench_gemma-2-2b_vanilla_width-2pow14_date-1109" \
  --sae_block_pattern "blocks.12.hook_resid_post__trainer_0" \
  --model_name gemma-2-2b 


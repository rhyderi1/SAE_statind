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
  --sae_regex_pattern "gemma-scope-2b-pt-res" \
  --sae_block_pattern "layer_3/width_16k/average_l0_59" \
  --model_name gemma-2-2b 


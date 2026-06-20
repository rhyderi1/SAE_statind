#!/bin/bash
#SBATCH --account=aip-bahtol
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=1:00:00
#SBATCH --job-name=extract_smain_sabs
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
export HF_TOKEN=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

module load python/3.11.5
source /project/aip-bahtol/rhyderi1/sae_statind/.venv/bin/activate

cd /project/aip-bahtol/rhyderi1/sae_statind

python extract_smain_sabs.py \
  --letter "B" \
  --layer 12 \
  --sae-name "gemma-scope-2b-pt-res_layer_12_width_16k_average_l0_82" \
  --sae-id "layer_12/width_16k/average_l0_82" \
  --results-dir "/home/rhyderi1/projects/aip-bahtol/rhyderi1/sae_statind/extract_smain_sabs_results" \
  --hf-token "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"
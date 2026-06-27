#!/bin/bash
#SBATCH --account=aip-bahtol
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=2:00:00
#SBATCH --job-name=absorption_implementation
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
export HF_TOKEN=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

module load python/3.11.5
source /project/aip-bahtol/rhyderi1/sae_statind/.venv/bin/activate

cd /project/aip-bahtol/rhyderi1/sae_statind

mkdir -p logs

python june22_absorption_implementation/run.py \
    --release sae_bench_gemma-2-2b_vanilla_width-2pow14_date-1109 \
    --sae-id  blocks.12.hook_resid_post__trainer_0 \
    --letter A  \
    --model gemma-2-2b  
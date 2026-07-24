#!/bin/bash
#SBATCH --account=aip-bahtol
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --job-name=infer_z
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

module load python/3.11.5
module load gcc arrow/24.0.0

source /project/aip-bahtol/rhyderi1/sae_statind/.venv/bin/activate
cd /project/aip-bahtol/rhyderi1/sae_statind

mkdir -p logs

export HF_DATASETS_CACHE="/scratch/rhyderi1/hf_datasets"
export HF_HOME="/scratch/rhyderi1/hf_home"
export HF_TOKEN=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


python src/infer_z.py \
    --modelchoice gemma-2-2b \
    --layer 3 \
    --arch jumprelu \
    --sparsity 59 \
    --dtypechoice float32 \
    --n_batches 3823 \
    --batch_size 32 \
    --context_size 128 \
    --shard_size 50000 \
    --store_x \
    --store_z \


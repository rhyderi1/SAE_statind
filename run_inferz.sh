#!/bin/bash
#SBATCH --account=aip-bahtol
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=00:07:00
#SBATCH --job-name=create_scatterplots_arrayjob
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#SBATCH --array=0-1%4

module load python/3.11.5
module load gcc arrow/24.0.0

source /project/aip-bahtol/rhyderi1/sae_statind/.venv/bin/activate
cd /project/aip-bahtol/rhyderi1/sae_statind

export HF_DATASETS_CACHE="/scratch/rhyderi1/hf_datasets"
export HF_HOME="/scratch/rhyderi1/hf_home"
export HF_TOKEN=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


python infer_z.py \
    --layer $LAYER \
    --arch $ARCH \
    --sparsity $SPARSITY \
    --modelchoice gemma-2-2b \
    --dtypechoice float32 \
    --n_batches 13 \
    --batch_size 32 \
    --context_size 128 \
    --shard_size 50000 \
    --store_x \
    --store_z \
    --plot \
    --task_id $SLURM_ARRAY_TASK_ID \
    --config_csv = configs.csv


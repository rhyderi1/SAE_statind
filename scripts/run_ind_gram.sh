#!/bin/bash
#SBATCH --account=aip-bahtol
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=05:00:00
#SBATCH --job-name=run_ind_gram
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

module load python/3.11.5
module load gcc arrow/24.0.0

source /project/aip-bahtol/rhyderi1/sae_statind/.venv/bin/activate
cd /project/aip-bahtol/rhyderi1/sae_statind

mkdir -p logs

# Edit Z_DIR to point at the desired shard directory under data/Z/
Z_DIR="data/Z/All batches/acts_layer19_relu_20_20260610_120821"

python src/compute_ind_gram_2.py \
    --z_dir      "$Z_DIR" \
    --num_shards 93 \
    --out_dir    "results/ind_gram/layer19_relu_l020"

#!/bin/bash
#SBATCH --account=aip-bahtol
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=1:00:00
#SBATCH --job-name=abs_sets
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
export HF_TOKEN=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

module load python/3.11.5
module load gcc arrow/24.0.0

source /project/aip-bahtol/rhyderi1/sae_statind/.venv/bin/activate

cd /project/aip-bahtol/rhyderi1/sae_statind

mkdir -p logs

export HF_HOME="/scratch/rhyderi1/hf_home"

python src/extract_absorption_sets.py 

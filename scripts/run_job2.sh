#!/bin/bash
#SBATCH --account=aip-bahtol
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=8:00:00
#SBATCH --gres=gpu:1
#SBATCH --job-name=figs67
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
export HF_TOKEN=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

module load python/3.11.5
module load gcc arrow/24.0.0

source /project/aip-bahtol/rhyderi1/sae_statind/.venv/bin/activate

cd /project/aip-bahtol/rhyderi1/sae_statind

mkdir -p logs

export HF_HOME="/scratch/rhyderi1/hf_home"

python src/replicate_absorption_figs67.py
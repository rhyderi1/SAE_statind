#!/bin/bash
#SBATCH --account=aip-bahtol
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x-%j.out

mkdir -p logs

module purge

module load python/3.11

source ~/venv/bin/activate

export HF_TOKEN=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

python compute_feature_absorption.py
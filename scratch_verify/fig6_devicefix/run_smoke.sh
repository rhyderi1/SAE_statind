#!/bin/bash
#SBATCH --account=aip-bahtol
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=0:20:00
#SBATCH --job-name=fig6smoke
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
export HF_TOKEN=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

module load python/3.11.5
module load gcc arrow/24.0.0

source /project/aip-bahtol/rhyderi1/sae_statind/.venv/bin/activate

cd /project/aip-bahtol/rhyderi1/sae_statind

mkdir -p logs

export HF_HOME="/scratch/rhyderi1/hf_home"

# Run from an isolated dir so target_activations.pt / figures/ are not clobbered.
export PYTHONPATH="/project/aip-bahtol/rhyderi1/sae_statind/src:$PYTHONPATH"
cd scratch_verify/fig6_devicefix

python chanin_fig6_final_smoke.py

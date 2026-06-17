#!/bin/bash
#SBATCH --account=aip-bahtol
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=0:30:00
#SBATCH --job-name=4_DTD_ZTZ_scatter
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
export HF_TOKEN=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

module load python/3.11.5
source /project/aip-bahtol/rhyderi1/sae_statind/.venv/bin/activate

cd /project/aip-bahtol/rhyderi1/sae_statind

python DTDxZTZ_4_scatter.py

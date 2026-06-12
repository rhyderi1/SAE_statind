#!/bin/bash
#SBATCH --account=aip-bahtol
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --time=2:30:00
#SBATCH --job-name=Z_histogram_results
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

module load python/3.11.5
module load gcc arrow/24.0.0
source /project/aip-bahtol/rhyderi1/sae_statind/.venv/bin/activate
cd /project/aip-bahtol/rhyderi1/sae_statind


python Z_histogram.py
#!/bin/bash
#SBATCH --account=aip-bahtol
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=2:30:00
#SBATCH --job-name=expectation_values_4
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

module load python/3.11.5
source /project/aip-bahtol/rhyderi1/sae_statind/.venv/bin/activate

cd /project/aip-bahtol/rhyderi1/sae_statind

python save_expectations.py

#!/bin/bash
#SBATCH --account=aip-bahtol
#SBATCH --job-name=abs_full
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:l40s:1
#SBATCH --mem=64G
#SBATCH --time=1:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
export HF_TOKEN=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

# sbatch does not reliably carry the Lmod `module` shell function into the job
# environment, so initialise the CVMFS software profile explicitly.
source /cvmfs/soft.computecanada.ca/config/profile/bash.sh
module load python/3.11.5
source /project/aip-bahtol/rhyderi1/sae_statind/.venv/bin/activate

cd /project/aip-bahtol/rhyderi1/sae_statind

mkdir -p logs

# All 26 letters in one job: the model, SAE, probe and calculator are
# letter-independent and loaded once, and the scoring itself is fast
# (~67 s for 's', the largest letter at 14.5k of 169k alpha tokens).
# --seed makes the ICL sampling reproducible; the eval leaves it unseeded.
# For a rerun after a partial failure, add --skip-existing to resume.
python src/absorption_full_vocab.py \
    --letters all \
    --arch jumprelu --layer 3 --sparsity 59 \
    --seed 0

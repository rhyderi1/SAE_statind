#!/bin/bash
#SBATCH --account=aip-bahtol
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=8:00:00
#SBATCH --job-name=collect_acts_layer19_arch_relu_k_40
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

module load python/3.11.5
module load gcc arrow/24.0.0

source /project/aip-bahtol/rhyderi1/sae_statind/.venv/bin/activate
cd /project/aip-bahtol/rhyderi1/sae_statind

export HF_DATASETS_CACHE="/scratch/rhyderi1/hf_datasets"
export HF_HOME="/scratch/rhyderi1/hf_home"

# needed to save out_dir
LAYER=19
ARCH=relu
SPARSITY=20


python infer_z.py \
    --layer $LAYER \
    --arch $ARCH \
    --sparsity $SPARSITY \
    --modelchoice gemma-2-2b \
    --dtypechoice float16 \
    --n_batches 1220 \
    --batch_size 32 \
    --context_size 128 \
    --shard_size 10000 \
    --out_dir "5_acts_small_layer${LAYER}_${ARCH}_${SPARSITY}_$(date +%Y%m%d_%H%M%S)" \
    --store_z \
    --store_x


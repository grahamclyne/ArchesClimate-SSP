#!/bin/bash
#SBATCH --job-name=generate_delta_stats  # job name
#SBATCH --ntasks=1                   # number of MP tasks
#SBATCH --gpus=1
#SBATCH --partition=gpu
# SBATCH --partition=gpu_p2
#SBATCH --ntasks-per-node=1          # this needs to correspond with # of GPUS
#SBATCH --cpus-per-task=3        # number of cores per tasks, see how many GPUs per node and take proportional amount of CPUs
#SBATCH --mem=32G                   # total memory per node
#SBATCH --hint=nomultithread         # we get physical cores not logical
#SBATCH --time=05:00:00     # maximum execution time (HH:MM:SS)
#SBATCH --exclude=gpu014,gpu002,gpu016,gpu018,gpu004
# SBATCH --partition=cpu
# SBATCH --account=mlr@v100
# SBATCH --qos=qos_gpu-dev




# cd ${WORK}/ArchesClimate/ArchesClimate/utils/
source ~/miniconda3/etc/profile.d/conda.sh
conda activate geoarches
cd "${ARCHESCLIMATE_ROOT:-/home/gclyne/ArchesClimate/ArchesClimate}/utils"
export HYDRA_FULL_ERROR=1
# module load pytorch-gpu/py3/2.3.0

srun python generate_delta_stats.py cluster=cleps "$@"

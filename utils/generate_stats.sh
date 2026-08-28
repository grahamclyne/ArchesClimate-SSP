#!/bin/bash
#SBATCH --job-name=generate_stats  # job name
#SBATCH --ntasks=1                   # number of MP tasks
#SBATCH --ntasks-per-node=1          # this needs to correspond with # of GPUS
#SBATCH --cpus-per-task=4         # number of cores per tasks, see how many GPUs per node and take proportional amount of CPUs
#SBATCH --mem=10GB
#SBATCH --hint=nomultithread         # we get physical cores not logical
#SBATCH --time=3:00:00              # maximum execution time (HH:MM:SS)
# SBATCH --account=mlr@cpu
## SBATCH --qos=qos_cpu-dev




# cd ${WORK}/ArchesClimate/ArchesClimate/utils/

# module load pytorch-gpu/py3/2.3.0
source ~/miniconda3/etc/profile.d/conda.sh
conda activate geoarches
cd "${ARCHESCLIMATE_ROOT:-/home/gclyne/ArchesClimate/ArchesClimate}/utils"
export HYDRA_FULL_ERROR=1
srun python generate_stats.py "$@"

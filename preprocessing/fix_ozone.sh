#!/bin/bash
#SBATCH --job-name=fix_ozone    # job name
#SBATCH --ntasks=1            # the job is sequential
#SBATCH --cpus-per-task=8           # number of cores per tasks, see how many GPUs per node and take proportional amount of CPUs
#SBATCH --hint=nomultithread  # 1 process per physical CPU core (no hyperthreading)
#SBATCH --time=04:00:00       # maximum elapsed time (HH:MM:SS)
#SBATCH --mem=32GB
# SBATCH --account=mlr@cpu

# cd ${WORK}/ArchesClimate/ArchesClimate/preprocessing/

# module load pytorch-gpu/py3/2.3.0
source ~/miniconda3/etc/profile.d/conda.sh
conda activate geoarches
cd "${ARCHESCLIMATE_ROOT:-/home/gclyne/ArchesClimate/ArchesClimate}/preprocessing"
export SCRATCH=${SCRATCH:-/scratch/gclyne}
export HYDRA_FULL_ERROR=1
srun python fix_ozone.py
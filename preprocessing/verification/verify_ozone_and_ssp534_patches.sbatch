#!/bin/bash
#SBATCH --job-name=verify_ozone_patches
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --hint=nomultithread
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --partition=arches
#SBATCH --account=arches
#SBATCH --output=logs/verify_ozone_patches_%j.out
#SBATCH --error=logs/verify_ozone_patches_%j.err

mkdir -p logs

source ~/miniconda3/etc/profile.d/conda.sh
conda activate geoarches
cd "${ARCHESCLIMATE_ROOT:-/home/gclyne/ArchesClimate/ArchesClimate}/preprocessing"

export SCRATCH=${SCRATCH:-/scratch/gclyne}
export HYDRA_FULL_ERROR=1
export PYTHONUNBUFFERED=1

srun python verification/verify_ozone_and_ssp534_patches.py

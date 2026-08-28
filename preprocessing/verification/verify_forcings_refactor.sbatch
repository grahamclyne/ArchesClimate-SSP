#!/bin/bash
#SBATCH --job-name=verify_forcings_refactor
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --hint=nomultithread
#SBATCH --mem=48G
#SBATCH --time=00:30:00
#SBATCH --partition=arches
#SBATCH --account=arches
#SBATCH --output=logs/verify_forcings_refactor_%j.out
#SBATCH --error=logs/verify_forcings_refactor_%j.err

mkdir -p logs

source ~/miniconda3/etc/profile.d/conda.sh
conda activate geoarches
cd "${ARCHESCLIMATE_ROOT:-/home/gclyne/ArchesClimate/ArchesClimate}/preprocessing"

export SCRATCH=${SCRATCH:-/scratch/gclyne}
export HYDRA_FULL_ERROR=1
export PYTHONUNBUFFERED=1

srun python verification/verify_forcings_refactor.py

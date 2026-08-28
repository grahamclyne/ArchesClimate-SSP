#!/bin/bash
#SBATCH --job-name=fill_in_missing_ipsl
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --hint=nomultithread
#SBATCH --mem=80G
#SBATCH --time=02:00:00
#SBATCH --output=logs/fill_in_missing_ipsl-%j.out
#SBATCH --error=logs/fill_in_missing_ipsl-%j.err
# SBATCH --partition=arches
# SBATCH --account=arches

mkdir -p logs

source ~/miniconda3/etc/profile.d/conda.sh
conda activate geoarches
cd "${ARCHESCLIMATE_ROOT:-/home/gclyne/ArchesClimate/ArchesClimate}/preprocessing"

export SCRATCH=${SCRATCH:-/scratch/gclyne}
export HYDRA_FULL_ERROR=1
export PYTHONUNBUFFERED=1

srun python -u fill_in_missing_ipsl.py

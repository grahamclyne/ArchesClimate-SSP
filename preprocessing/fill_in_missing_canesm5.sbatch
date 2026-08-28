#!/bin/bash
#SBATCH --job-name=fill_in_missing_canesm5
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --hint=nomultithread
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=logs/fill_in_missing_canesm5-%A_%a.out
# SBATCH --partition=arches
# SBATCH --account=arches

mkdir -p logs

source ~/miniconda3/etc/profile.d/conda.sh
conda activate geoarches
cd "${ARCHESCLIMATE_ROOT:-/home/gclyne/ArchesClimate/ArchesClimate}/preprocessing"

export SCRATCH=${SCRATCH:-/scratch/gclyne}
export HYDRA_FULL_ERROR=1

srun python -u fill_in_missing_canesm5.py

#!/bin/bash
#SBATCH --job-name=fix_damip_levels
#SBATCH --array=0-9
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --hint=nomultithread
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --partition=mem
#SBATCH --account=arches
#SBATCH --output=logs/fix_damip_levels_%A_%a.out
#SBATCH --error=logs/fix_damip_levels_%A_%a.err

mkdir -p logs

source ~/miniconda3/etc/profile.d/conda.sh
conda activate geoarches
cd "${ARCHESCLIMATE_ROOT:-/home/gclyne/ArchesClimate/ArchesClimate}/preprocessing"

export SCRATCH=${SCRATCH:-/scratch/gclyne}
export HYDRA_FULL_ERROR=1
export PYTHONUNBUFFERED=1

srun python fix_damip_levels.py

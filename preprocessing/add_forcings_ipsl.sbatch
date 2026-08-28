#!/bin/bash
#SBATCH --job-name=add_forcings_ipsl
#SBATCH --array=0-5
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --hint=nomultithread
#SBATCH --mem=160G
#SBATCH --time=12:00:00
#SBATCH --partition=mem
#SBATCH --account=arches
#SBATCH --output=logs/add_forcings_ipsl_%A_%a.out
#SBATCH --error=logs/add_forcings_ipsl_%A_%a.err

mkdir -p logs

source ~/miniconda3/etc/profile.d/conda.sh
conda activate geoarches
cd "${ARCHESCLIMATE_ROOT:-/home/gclyne/ArchesClimate/ArchesClimate}/preprocessing"

export SCRATCH=${SCRATCH:-/scratch/gclyne}
export HYDRA_FULL_ERROR=1
export PYTHONUNBUFFERED=1

srun python add_forcings.py

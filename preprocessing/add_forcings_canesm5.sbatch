#!/bin/bash
#SBATCH --job-name=add_forcings_canesm5
#SBATCH --array=0-13
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --hint=nomultithread
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --partition=cpu_devel
#SBATCH --account=arches
#SBATCH --output=logs/add_forcings_canesm5_%A_%a.out
#SBATCH --error=logs/add_forcings_canesm5_%A_%a.err

mkdir -p logs

source ~/miniconda3/etc/profile.d/conda.sh
conda activate geoarches
cd "${ARCHESCLIMATE_ROOT:-/home/gclyne/ArchesClimate/ArchesClimate}/preprocessing"

export SCRATCH=${SCRATCH:-/scratch/gclyne}
export HYDRA_FULL_ERROR=1
export PYTHONUNBUFFERED=1

srun python add_forcings_canesm5.py

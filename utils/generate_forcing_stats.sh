#!/bin/bash
#SBATCH --job-name=generate_forcing_stats
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16GB
#SBATCH --hint=nomultithread
#SBATCH --time=03:00:00

mkdir -p ../sblogs/generate_forcing_stats

source ~/miniconda3/etc/profile.d/conda.sh
conda activate geoarches
cd "${ARCHESCLIMATE_ROOT:-/home/gclyne/ArchesClimate/ArchesClimate}/utils"
export HYDRA_FULL_ERROR=1

srun python generate_forcing_stats.py "$@"

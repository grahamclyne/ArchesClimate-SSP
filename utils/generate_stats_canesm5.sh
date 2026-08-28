#!/bin/bash
#SBATCH --job-name=generate_stats_canesm5
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32GB
#SBATCH --time=02:00:00
#SBATCH --output=../sblogs/generate_stats_canesm5/%j_log.out
#SBATCH --error=../sblogs/generate_stats_canesm5/%j_log.err

mkdir -p ../sblogs/generate_stats_canesm5

source ~/miniconda3/etc/profile.d/conda.sh
conda activate geoarches
cd "${ARCHESCLIMATE_ROOT:-/home/gclyne/ArchesClimate/ArchesClimate}/utils"
export HYDRA_FULL_ERROR=1

srun python generate_stats.py module=forcing_dropout_no_random_lt_canesm5 \
    name=new_ozone_canesm5 \
    dataloader.dataset.path=${SCRATCH}/memmap_filled_in_canesm5

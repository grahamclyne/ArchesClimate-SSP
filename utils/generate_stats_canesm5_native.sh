#!/bin/bash
#SBATCH --job-name=generate_stats_canesm5_native
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32GB
#SBATCH --time=02:00:00
#SBATCH --output=../sblogs/generate_stats_canesm5_native/%j_log.out
#SBATCH --error=../sblogs/generate_stats_canesm5_native/%j_log.err

mkdir -p ../sblogs/generate_stats_canesm5_native

source ~/miniconda3/etc/profile.d/conda.sh
conda activate geoarches
cd "${ARCHESCLIMATE_ROOT:-/home/gclyne/ArchesClimate/ArchesClimate}/utils"
export HYDRA_FULL_ERROR=1

srun python generate_stats.py module=forcing_dropout_no_random_lt_pf8_no_uncond_grouped_dropout_pilot_full_state_circular_pad_canesm5_damip_native_grid \
    name=new_ozone_canesm5_native \
    dataloader.dataset.path=${SCRATCH}/memmap_filled_in_canesm5_native \
    cluster=cleps \
    "$@"

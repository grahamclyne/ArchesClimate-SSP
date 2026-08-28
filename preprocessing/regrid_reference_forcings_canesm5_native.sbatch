#!/bin/bash
#SBATCH --job-name=regrid_ref_forcings_canesm5_native
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --hint=nomultithread
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --partition=cpu_devel
#SBATCH --account=arches
#SBATCH --output=logs/regrid_ref_forcings_canesm5_native_%j.out
#SBATCH --error=logs/regrid_ref_forcings_canesm5_native_%j.err

# ozone_historical.pt is ~10.7GB unpacked -- OOM-killed at default/interactive
# shell memory limits (see add_forcings_canesm5.sbatch's note re: job
# 5066808). regrid_file() skips any output that already exists, so this is
# safely resumable after a partial run.

mkdir -p logs

source ~/miniconda3/etc/profile.d/conda.sh
conda activate geoarches
cd "${ARCHESCLIMATE_ROOT:-/home/gclyne/ArchesClimate/ArchesClimate}/preprocessing"

export SCRATCH=${SCRATCH:-/scratch/gclyne}
export PYTHONUNBUFFERED=1

srun python regrid_reference_forcings_canesm5_native.py --experiments "$@"

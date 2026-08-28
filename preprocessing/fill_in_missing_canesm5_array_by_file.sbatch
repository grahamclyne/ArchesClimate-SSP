#!/bin/bash
#SBATCH --job-name=fill_in_missing_canesm5_hist
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --hint=nomultithread
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=logs/fill_in_missing_canesm5_hist-%A_%a.out
# SBATCH --partition=arches
# SBATCH --account=arches

# One array task per line of FILELIST (1-indexed via sed), each processing a
# single named file via CANESM5_ONLY_FNAME -- lets many ensemble members'
# gap-fill run as concurrent SLURM tasks instead of one job serially working
# through fill_in_missing_canesm5.py's own glob-based task_id::task_count
# partitioning.

mkdir -p logs

source ~/miniconda3/etc/profile.d/conda.sh
conda activate geoarches
cd "${ARCHESCLIMATE_ROOT:-/home/gclyne/ArchesClimate/ArchesClimate}/preprocessing"

export SCRATCH=${SCRATCH:-/scratch/gclyne}
export HYDRA_FULL_ERROR=1

FNAME=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$FILELIST")
if [ -z "$FNAME" ]; then
  echo "no filename for array task $SLURM_ARRAY_TASK_ID in $FILELIST" >&2
  exit 1
fi
echo "task $SLURM_ARRAY_TASK_ID: $FNAME"
export CANESM5_ONLY_FNAME="$FNAME"

srun python -u fill_in_missing_canesm5.py

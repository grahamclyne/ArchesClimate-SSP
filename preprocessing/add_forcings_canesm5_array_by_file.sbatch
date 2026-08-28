#!/bin/bash
#SBATCH --job-name=add_forcings_canesm5_hist
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --hint=nomultithread
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --partition=cpu_devel
#SBATCH --account=arches
#SBATCH --output=logs/add_forcings_canesm5_hist-%A_%a.out
#SBATCH --error=logs/add_forcings_canesm5_hist-%A_%a.err

# One array task per line of FILELIST, each processing a single named file
# via CANESM5_ONLY_FNAME. Companion to fill_in_missing_canesm5_array_by_file.sbatch.

mkdir -p logs

source ~/miniconda3/etc/profile.d/conda.sh
conda activate geoarches
cd "${ARCHESCLIMATE_ROOT:-/home/gclyne/ArchesClimate/ArchesClimate}/preprocessing"

export SCRATCH=${SCRATCH:-/scratch/gclyne}
export HYDRA_FULL_ERROR=1
export PYTHONUNBUFFERED=1

FNAME=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$FILELIST")
if [ -z "$FNAME" ]; then
  echo "no filename for array task $SLURM_ARRAY_TASK_ID in $FILELIST" >&2
  exit 1
fi
echo "task $SLURM_ARRAY_TASK_ID: $FNAME"
export CANESM5_ONLY_FNAME="$FNAME"

srun python add_forcings_canesm5.py

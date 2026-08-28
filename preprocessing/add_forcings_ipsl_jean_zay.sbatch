#!/bin/bash
# Array job: each task processes a disjoint slice of the filled-in NetCDF files.
# Submit after fill_in_missing_ipsl_jean_zay.sbatch finishes.
#
# Parallelism: --array=0-N submits N+1 simultaneous tasks.
# Rule of thumb: set N = (total_files - 1).  For 4 DAMIP experiments × 10
# realizations = 40 files → sbatch --array=0-39 add_forcings_ipsl_jean_zay.sbatch
# (SLURM_ARRAY_TASK_COUNT is set automatically by SLURM based on the range).
#
# Memory: 200 GB is needed to materialise the 3000-month ozone tensor in RAM.
# The cpu_p1 partition ties memory to CPU count (~2.3GB/allocated-core, see
# `sinfo -p cpu_p1 -o "%C %m"`: 80 cores / 191000MB per node). This was
# previously run on prepost (~30GB/allocated-core) to cover the 200GB need;
# moved to cpu_p1 at the user's request. cpus-per-task=4 (~9GB) OOM-killed
# every array task within ~2 minutes (job 605263). cpus-per-task=8 (~18GB)
# OOM-killed every task too (job 605414, most within ~1.5min, a few that got
# further still failed by ~4.5min) -- confirmed both via `slurmstepd: error:
# Detected 1 oom_kill event`. Bumped to cpus-per-task=16 (~37GB) at the
# user's request -- still well short of 200GB, so still at real risk of OOM.
#SBATCH --job-name=add_forcings_ipsl
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --hint=nomultithread
#SBATCH --time=12:00:00
#SBATCH --account=mlr@cpu
#SBATCH --partition=cpu_p1
#SBATCH --output=logs/add_forcings_ipsl_%A_%a.out
#SBATCH --error=logs/add_forcings_ipsl_%A_%a.err

mkdir -p logs

ARCHESCLIMATE_ROOT="${ARCHESCLIMATE_ROOT:-/lustre/fswork/projects/rech/mlr/udy16au/ArchesClimate/ArchesClimate}"
export SCRATCH="${SCRATCH:-/lustre/fsn1/projects/rech/mlr/udy16au}"
# Write directly into the full_ozone dataset directory used by jean_zay_a100 configs.
export OUTPUT_DIR="${SCRATCH}/memmap_filled_in_full_ozone"

cd "${ARCHESCLIMATE_ROOT}/preprocessing"

module load pytorch-gpu/py3/2.3.0
export HYDRA_FULL_ERROR=1
export PYTHONUNBUFFERED=1

mkdir -p "${OUTPUT_DIR}"

srun python -u add_forcings.py

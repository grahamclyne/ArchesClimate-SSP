#!/bin/bash
# Array job: each task gap-fills a disjoint slice of the Stage-1 NetCDF files.
# Submit after all ESMValTool (Stage-1) jobs finish.
#
# Parallelism: --array=0-N submits N+1 simultaneous tasks.
# Rule of thumb: set N = (total_files - 1).  For 4 DAMIP experiments × 10
# realizations = 40 files → sbatch --array=0-39 fill_in_missing_ipsl_jean_zay.sbatch
#
#SBATCH --job-name=fill_in_missing_ipsl
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --hint=nomultithread
#SBATCH --time=04:00:00
#SBATCH --account=mlr@cpu
# Default cpu_p1 partition only gives ~16GB for cpus-per-task=4 (jean-zay
# ties memory to CPU count, ~2GB/allocated-core there) -- confirmed too
# little: job 589566's every array task OOM-killed on real DAMIP
# hist-GHG/hist-stratO3 data (worked before only because those earlier runs'
# specific files happened to be smaller). prepost has a much richer
# memory/CPU ratio (~30GB/allocated-core, confirmed via the esmvaltool jobs
# getting 480GB at cpus-per-task=8) -- same fix as add_forcings below.
#SBATCH --partition=prepost
#SBATCH --output=logs/fill_in_missing_ipsl_%A_%a.out
#SBATCH --error=logs/fill_in_missing_ipsl_%A_%a.err

mkdir -p logs

ARCHESCLIMATE_ROOT="${ARCHESCLIMATE_ROOT:-/lustre/fswork/projects/rech/mlr/udy16au/ArchesClimate/ArchesClimate}"
export SCRATCH="${SCRATCH:-/lustre/fsn1/projects/rech/mlr/udy16au}"

cd "${ARCHESCLIMATE_ROOT}/preprocessing"

module load pytorch-gpu/py3/2.3.0
export HYDRA_FULL_ERROR=1
export PYTHONUNBUFFERED=1

srun python -u fill_in_missing_ipsl.py

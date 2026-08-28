#!/bin/bash
# One-off: rebuild reference_data/ozone_historical.pt from an already-built
# historical memmap, working around the lost raw climoz source files -- see
# reconstruct_ozone_historical.py's docstring. Run once, before rerunning
# add_forcings_ipsl_jean_zay.sbatch for hist-GHG/hist-aer/hist-stratO3/1pctCO2.
#
#   sbatch reconstruct_ozone_historical.sbatch
#
#SBATCH --job-name=reconstruct_ozone_historical
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --hint=nomultithread
#SBATCH --time=00:30:00
#SBATCH --account=mlr@cpu
#SBATCH --partition=prepost
#SBATCH --output=logs/reconstruct_ozone_historical_%j.out
#SBATCH --error=logs/reconstruct_ozone_historical_%j.err

mkdir -p logs

ARCHESCLIMATE_ROOT="${ARCHESCLIMATE_ROOT:-/lustre/fswork/projects/rech/mlr/udy16au/ArchesClimate/ArchesClimate}"
export SCRATCH="${SCRATCH:-/lustre/fsn1/projects/rech/mlr/udy16au}"

cd "${ARCHESCLIMATE_ROOT}/preprocessing"

module load pytorch-gpu/py3/2.3.0
export HYDRA_FULL_ERROR=1
export PYTHONUNBUFFERED=1

srun python -u reconstruct_ozone_historical.py

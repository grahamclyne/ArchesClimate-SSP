#!/bin/bash
#SBATCH --job-name=paper_figures_generate_all
#SBATCH --partition=prepost
#SBATCH --account=mlr@cpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --hint=nomultithread
#SBATCH --time=02:00:00
#SBATCH --output=analysis/logs/paper_figures_generate_all-%j.out

mkdir -p analysis/logs

ARCHESCLIMATE_ROOT="${ARCHESCLIMATE_ROOT:-/lustre/fswork/projects/rech/mlr/udy16au/ArchesClimate/ArchesClimate}"

source /etc/profile.d/modules.sh
module load pytorch-gpu/py3/2.3.0
cd "${ARCHESCLIMATE_ROOT}"
export HYDRA_FULL_ERROR=1

# Runs the table + one RMSE plot per scenario, in sequence. Pass through
# "$@", e.g.:
#   sbatch analysis/paper_figures_generate_all.sbatch --config analysis/configs/paper_figures_table_smoketest.yaml
srun python -m analysis.paper_figures_generate_all "$@"

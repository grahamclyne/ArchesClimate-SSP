#!/bin/bash
#SBATCH --job-name=paper_figures_table
#SBATCH --partition=cpu_devel
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --hint=nomultithread
#SBATCH --time=01:00:00
#SBATCH --output=analysis/logs/paper_figures_table-%j.out

mkdir -p analysis/logs

source ~/miniconda3/etc/profile.d/conda.sh
conda activate geoarches
cd "${ARCHESCLIMATE_ROOT:-/home/gclyne/ArchesClimate/ArchesClimate}"
export HYDRA_FULL_ERROR=1

# Pass through "$@", e.g.:
#   sbatch analysis/paper_figures_table.sbatch --config analysis/configs/paper_figures_table_smoketest.yaml
srun python -m analysis.paper_figures_table "$@"

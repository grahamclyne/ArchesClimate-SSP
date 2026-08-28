#!/bin/bash
#SBATCH --job-name=paper_figures
#SBATCH --partition=cpu_devel
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=128GB
#SBATCH --time=01:00:00
#SBATCH --output=/home/gclyne/ArchesClimate/ArchesClimate/sblogs/paper_figures/%j_log.out
#SBATCH --error=/home/gclyne/ArchesClimate/ArchesClimate/sblogs/paper_figures/%j_log.err

mkdir -p /home/gclyne/ArchesClimate/ArchesClimate/sblogs/paper_figures

source ~/miniconda3/etc/profile.d/conda.sh
conda activate geoarches
cd /home/gclyne/ArchesClimate/ArchesClimate
export HYDRA_FULL_ERROR=1

CONFIG=${1:?"usage: sbatch analysis/paper_figures_run.sh <config_path> [extra args passed to paper_figures_generate_all.py]"}
shift

srun python -m analysis.paper_figures_generate_all --config "$CONFIG" "$@"

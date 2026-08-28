#!/bin/bash
#SBATCH --job-name=abrupt_global_mean
#SBATCH --partition=cpu_devel
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=128GB
#SBATCH --time=00:45:00
#SBATCH --output=/home/gclyne/ArchesClimate/ArchesClimate/sblogs/paper_figures/%j_abrupt_log.out
#SBATCH --error=/home/gclyne/ArchesClimate/ArchesClimate/sblogs/paper_figures/%j_abrupt_log.err

mkdir -p /home/gclyne/ArchesClimate/ArchesClimate/sblogs/paper_figures

source ~/miniconda3/etc/profile.d/conda.sh
conda activate geoarches
cd /home/gclyne/ArchesClimate/ArchesClimate
export HYDRA_FULL_ERROR=1

CONFIG=${1:?"usage: sbatch analysis/plot_abrupt_global_mean.sh <config_path> <run_dir>"}
RUN_DIR=${2:?"usage: sbatch analysis/plot_abrupt_global_mean.sh <config_path> <run_dir>"}

srun python -m analysis.plot_abrupt_global_mean --config "$CONFIG" --run-dir "$RUN_DIR"

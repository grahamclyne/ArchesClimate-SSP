#!/bin/bash
#SBATCH --job-name=forced_response_regen
#SBATCH --partition=cpu_devel
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=150GB
#SBATCH --time=01:00:00
#SBATCH --output=/home/gclyne/ArchesClimate/ArchesClimate/sblogs/paper_figures/%j_log.out
#SBATCH --error=/home/gclyne/ArchesClimate/ArchesClimate/sblogs/paper_figures/%j_log.err

mkdir -p /home/gclyne/ArchesClimate/ArchesClimate/sblogs/paper_figures
source ~/miniconda3/etc/profile.d/conda.sh
conda activate geoarches
cd /home/gclyne/ArchesClimate/ArchesClimate
export HYDRA_FULL_ERROR=1
srun python -m analysis.paper_figures_forced_response --config analysis/configs/$1.yaml

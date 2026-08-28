#!/bin/bash
#SBATCH --job-name=physical_534_regen
#SBATCH --partition=cpu_devel
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=200GB
#SBATCH --time=01:00:00
#SBATCH --output=/home/gclyne/ArchesClimate/ArchesClimate/sblogs/paper_figures/%j_log.out
#SBATCH --error=/home/gclyne/ArchesClimate/ArchesClimate/sblogs/paper_figures/%j_log.err

mkdir -p /home/gclyne/ArchesClimate/ArchesClimate/sblogs/paper_figures
source ~/miniconda3/etc/profile.d/conda.sh
conda activate geoarches
cd /home/gclyne/ArchesClimate/ArchesClimate
export HYDRA_FULL_ERROR=1
srun python -m analysis.paper_figures_physical \
    --config analysis/configs/paper_figures_pf4_w050_80_10_step40000_mesmer_physical.yaml \
    --scenario 534

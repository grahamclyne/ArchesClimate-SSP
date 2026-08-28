#!/bin/bash
#SBATCH --job-name=cdo_preprocess_cmip    # job name
#SBATCH --ntasks=1            # the job is sequential
#SBATCH --cpus-per-task=8           # number of cores per tasks, see how many GPUs per node and take proportional amount of CPUs
#SBATCH --hint=nomultithread  # 1 process per physical CPU core (no hyperthreading)
#SBATCH --time=04:00:00       # maximum elapsed time (HH:MM:SS)
#SBATCH --account=mlr@cpu
#SBATCH --partition=compil

cd ${WORK}/ArchesClimate/ArchesClimate/preprocessing/

module load pytorch-gpu/py3/2.3.0
module load cdo
module load nco

export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
srun python forcings_preprocessing.py
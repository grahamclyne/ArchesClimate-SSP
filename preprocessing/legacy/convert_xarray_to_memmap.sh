#!/bin/bash
#SBATCH --job-name=convert_xarray_to_memmap    # job name
#SBATCH --ntasks=1            # the job is sequential
#SBATCH --cpus-per-task=36           # number of cores per tasks, see how many GPUs per node and take proportional amount of CPUs

#SBATCH --hint=nomultithread  # 1 process per physical CPU core (no hyperthreading)
#SBATCH --time=2:00:00       # maximum elapsed time (HH:MM:SS)
# SBATCH --qos=qos_cpu-dev

# If you have multiple projects or various types of computing hours,
# we need to specify the account to use even if you will not be charged
# for jobs using the "prepost" or "archive" partitions.
# SBATCH --account=mlr@cpu

# cd ${WORK}/ArchesClimate/ArchesClimate/preprocessing
cd "${ARCHESCLIMATE_ROOT:-/home/gclyne/ArchesClimate/ArchesClimate}/preprocessing"
export SCRATCH=${SCRATCH:-/scratch/gclyne}

source ~/miniconda3/etc/profile.d/conda.sh

conda activate geoarches


# module load pytorch-gpu/py3/2.3.0


srun python legacy/convert_xarray_to_memmap.py
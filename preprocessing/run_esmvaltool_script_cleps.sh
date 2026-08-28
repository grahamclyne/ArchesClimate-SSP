#!/bin/bash
#SBATCH --job-name=download_cmip_with_esmval_tool  # job name
#SBATCH --ntasks=1                   # number of MP tasks
#SBATCH --ntasks-per-node=1          # this needs to correspond with # of GPUS
#SBATCH --cpus-per-task=8        # number of cores per tasks, see how many GPUs per node and take proportional amount of CPUs
#SBATCH --hint=nomultithread         # we get physical cores not logical
#SBATCH --time=20:00:00              # maximum execution time (HH:MM:SS)
#SBATCH --mem=100G                   # my_diagnostic materializes full historical record in RAM (thetao/ta/ua/va/zg); default per-cpu mem (~15.5G) OOM-killed job 5066808





cd ~/ArchesClimate/ArchesClimate/preprocessing/

# source /gpfslocalsup/pub/anaconda-py3/2023.09/etc/profile.d/conda.sh
# conda init
source ~/miniconda3/etc/profile.d/conda.sh

conda activate esmvaltool-env1
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
if [ $# -eq 0 ]; then
    echo "Usage: $0 <filename>"
    echo "Example: $0 preprocessing/interpolation_dataset.yml"
    exit 1
fi

# Get filename from command line argument
# module load esmf/7.1.0r-mpi-cuda
RECIPE="$1"
shift
esmvaltool run "$RECIPE" --search_esgf=when_missing --remove_preproc_dir=True "$@"
#conda run -n esmvaltool-env1 .....

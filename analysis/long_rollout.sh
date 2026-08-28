#!/bin/bash
#SBATCH --ntasks=1                   # number of MP tasks
#SBATCH --ntasks-per-node=1          # this needs to correspond with # of GPUS
#SBATCH --cpus-per-task=12           # cleps: keep at/under physical cores per GPU, see CLAUDE.md cleps sbatch cpu gotcha
#SBATCH --hint=nomultithread         # we get physical cores not logical
#SBATCH --time=05:00:00              # maximum execution time (HH:MM:SS)
#SBATCH --account=arches             # cleps account (was mlr@v100 on jean-zay)
#SBATCH --partition=gpu              # cleps: any arches-usable GPU node, not just gpu018 (arches partition)
#SBATCH --exclude=gpu009,gpu012,gpu013,gpu015,gpu016,gpu017  # almanach/willow/astra-owned: shared/preemptible, avoid by default
#SBATCH --gpus=1
#SBATCH --job-name=default

# Defaults to this repo's own uv-managed venv (portable, works for anyone who's just
# run `uv sync`). cleps (the author's own cluster) has no module system and uses the
# geoarches conda env's python directly instead -- override PYTHON in the environment
# for that, or any other pre-provisioned interpreter (mirrors how submit.py/submitit
# launches training jobs -- see any run's sblogs/<name>/*_submission.sh).
ARCHESCLIMATE_ROOT="${ARCHESCLIMATE_ROOT:-/home/gclyne/ArchesClimate/ArchesClimate}"
if [[ -n "${PYTHON:-}" ]]; then
  read -r -a PYTHON_CMD <<< "$PYTHON"
else
  PYTHON_CMD=(uv run python)
fi

cd "${ARCHESCLIMATE_ROOT}/analysis"
export HYDRA_FULL_ERROR=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

# # Ensure Python doesn't buffer logs
# export PYTHONUNBUFFERED=1

# Force GPU sync
export CUDA_LAUNCH_BLOCKING=1

#sbatch long_rollout.sh year=0

# --- FIXED EXTRACTION LOGIC ---
# 1. We import sys to read arguments
# 2. We merge the file config with the CLI arguments (sys.argv)
# 3. We print the resolved name
EXP_NAME=$("${PYTHON_CMD[@]}" -c "import sys; from omegaconf import OmegaConf; \
base_conf = OmegaConf.load('${ARCHESCLIMATE_ROOT}/configs/config.yaml'); \
cli_conf = OmegaConf.from_dotlist(sys.argv[1:]); \
final_conf = OmegaConf.merge(base_conf, cli_conf); \
print(final_conf.name + '_' + cli_conf.inference.ckpt_fname + '_')" "$@")

echo "Detected Experiment Name: $EXP_NAME"

# --- RENAME RUNNING JOB ---
# This updates the name in squeue immediately
scontrol update JobId=${SLURM_JOB_ID} JobName="$EXP_NAME"

# --- RUN ---
# We pass "$@" to handle other overrides (like year=0)
# We set hydra.job.name to ensure output folders match the squeue name
# EXP_NAME embeds ckpt_fname (e.g. "step-step=018000.ckpt"), and Hydra's
# override grammar rejects a bare "=" inside an unquoted value -- escape it.
# EXP_NAME may already contain a literal backslash here: unlike Hydra's own
# override grammar, OmegaConf.from_dotlist (used above) does not unescape
# "\=", so if inference.ckpt_fname was passed pre-escaped, cli_conf keeps
# the backslash literally. Normalize first so we escape each "=" exactly
# once instead of double-escaping down here.
HYDRA_JOB_NAME="${EXP_NAME//\\=/=}"
HYDRA_JOB_NAME="${HYDRA_JOB_NAME//=/\\=}"
# Note: cluster= must be passed in "$@" by the caller (e.g. cluster=jean_zay_a100) --
# a hardcoded cluster= here would silently override/conflict with it.
srun --cpu-bind=none --mem-bind=none --job-name="rollout_$EXP_NAME" "${PYTHON_CMD[@]}" long_rollout.py "$@" hydra.job.name="$HYDRA_JOB_NAME"

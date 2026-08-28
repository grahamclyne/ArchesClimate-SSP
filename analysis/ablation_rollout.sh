#!/bin/bash
# Submits the 5 single-forcing-removed ablations (no_co2/no_methane/no_n2o/
# no_ozone/no_aerosol) at a given target, surface-only, one sbatch job per
# (ablation, seed) pair via analysis/long_rollout.sh -- same ABLATIONS
# mapping and "_zs<idx>" suffix convention as complete_rollout.sh, so
# analysis/paper_figures_forced_response.py (which globs for that suffix)
# picks these up unmodified.
#
# If the module has `use_energy_score: True`, --seeds N (default 1) submits
# N noise-seeded members per ablation (inference.num_seeds=N,
# inference.seed_index=0..N-1) instead of a single deterministic rollout --
# see model/forecast.py's generate_rollouts. All seeds for one ablation land
# in the same out_dir, distinguished by a plain "_<seed>" filename suffix
# (rollout_surface_<chunk>_<member>_<seed>.pt, matching analysis/compare_runs.py's
# ROLLOUT_RE). Non-energy-score modules always get a single unseeded job per
# ablation regardless of --seeds.
#
# Usage:
#   ./analysis/ablation_rollout.sh <module_name> <ckpt_step> [--name RUN_NAME] \
#       [--target val] [--seeds N] [--cluster CLUSTER] [--null] \
#       [--extra HYDRA_OVERRIDE ...] [--dry-run] [--force]
#
# Defaults to the piControl counterfactual (inference.zero_forcing_source=pi)
# for each zeroed forcing channel, since a fixed physical reference value
# is a cleaner ablation target than the model's own learned absent-token
# embedding. Pass --null to opt out and use the learned absent-token
# ("null") substitution instead (the old default). long_rollout.sh appends
# "_pi" to the out_dir after the "_zs<idx>" suffix whenever the pi path is
# taken, so this only changes which forcing_suffix directory the run lands
# in -- same ABLATIONS mapping either way.
#
# --extra passes one additional raw Hydra override straight through to
# long_rollout.sh; repeat the flag for more than one (e.g. to reproduce a
# training-time override needed to reconstruct the module, such as
# +module.module.variogram_weight=0.5).
#
# Example:
#   ./analysis/ablation_rollout.sh deterministic_damip_pf8_energy_score 22000 \
#       --name pf8_energy_score_a100_reduced_pf --seeds 2
#
#   ./analysis/ablation_rollout.sh deterministic_damip_pf8_energy_score_bs4 22000 \
#       --name pf8_energy_score_gradfix_p13_w050 --pi \
#       --extra module.module.pf_warmup_steps=16000 \
#       --extra +module.module.variogram_weight=0.5 \
#       --extra +module.module.variogram_auto_scale=true \
#       --extra +module.module.variogram_anneal_steps=5000

set -euo pipefail

if [ $# -lt 2 ]; then
  echo "Usage: $0 <module_name> <ckpt_step> [--name RUN_NAME] [--target val] [--seeds N] [--cluster CLUSTER] [--null] [--extra HYDRA_OVERRIDE ...] [--dry-run] [--force]" >&2
  exit 1
fi

MODULE="$1"
CKPT_STEP="$2"
shift 2
NAME=""
CLUSTER=""
TARGET="val"
SEEDS=1
PI=1
EXTRA=()
DRY_RUN=""
FORCE=""
while [ $# -gt 0 ]; do
  case "$1" in
    --name) [ $# -ge 2 ] || { echo "error: --name requires a value" >&2; exit 1; }; NAME="$2"; shift 2 ;;
    --cluster) [ $# -ge 2 ] || { echo "error: --cluster requires a value" >&2; exit 1; }; CLUSTER="$2"; shift 2 ;;
    --target) [ $# -ge 2 ] || { echo "error: --target requires a value" >&2; exit 1; }; TARGET="$2"; shift 2 ;;
    --seeds) [ $# -ge 2 ] || { echo "error: --seeds requires a value" >&2; exit 1; }; SEEDS="$2"; shift 2 ;;
    --null) PI=""; shift ;;
    --extra) [ $# -ge 2 ] || { echo "error: --extra requires a value" >&2; exit 1; }; EXTRA+=("$2"); shift 2 ;;
    --dry-run) DRY_RUN=1; shift ;;
    --force) FORCE=1; shift ;;
    *) echo "error: unrecognized argument: $1" >&2; exit 1 ;;
  esac
done
NAME="${NAME:-$MODULE}"
CLUSTER="${CLUSTER:-jean_zay_v100}"

# long_rollout.sh hardcodes #SBATCH --account=mlr@a100 --constraint=a100 as
# its *defaults*, overridable via sbatch's own --account/--constraint CLI
# flags (sbatch flags win over #SBATCH lines in the script). Passing
# cluster=jean_zay_v100 as a Hydra arg alone does NOT change which hardware
# actually runs the job -- long_rollout.py only reads cluster.work_path
# (identical between the two cluster configs), so without this mapping every
# rollout job would silently still land on A100 regardless of --cluster.
case "$CLUSTER" in
  jean_zay_a100) SBATCH_ACCOUNT="mlr@a100"; SBATCH_CONSTRAINT="a100" ;;
  jean_zay_v100) SBATCH_ACCOUNT="mlr@v100"; SBATCH_CONSTRAINT="v100-16g" ;;
  *) echo "error: unknown --cluster $CLUSTER, expected jean_zay_a100 or jean_zay_v100" >&2; exit 1 ;;
esac

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

MODULE_CFG="configs/module/${MODULE}.yaml"
if [ ! -f "$MODULE_CFG" ]; then
  echo "error: no module config at $MODULE_CFG" >&2
  exit 1
fi
PROJECT=$(grep -m1 '^project:' "$MODULE_CFG" | sed 's/^project:[[:space:]]*//')

if grep -qE '^\s*use_energy_score:\s*[Tt]rue\s*$' "$MODULE_CFG"; then
  NUM_SEEDS="$SEEDS"
else
  NUM_SEEDS=1
fi

CKPT_FNAME=$(printf "step-step=%06d.ckpt" "$CKPT_STEP")
SCRATCH="${SCRATCH:-/lustre/fsn1/projects/rech/mlr/udy16au}"
CKPT_PATH="${SCRATCH}/model_output/${PROJECT}/${NAME}/checkpoints/${CKPT_FNAME}"
if [ ! -f "$CKPT_PATH" ] && [ -z "$DRY_RUN" ]; then
  echo "error: checkpoint not found: $CKPT_PATH" >&2
  exit 1
fi

CKPT_FNAME_ESCAPED="${CKPT_FNAME/=/\\=}"
GENERATED_DATA="${SCRATCH}/generated_data"
CLAMP_SUFFIX="_clamp3std10"

PI_SUFFIX=""
[ -n "$PI" ] && PI_SUFFIX="_pi"

outdir_for() {
  local target="$1" forcing_suffix="$2"
  echo "${GENERATED_DATA}/${NAME}_12_10_1_1020_1_${target}_0_${CKPT_FNAME}_ema${forcing_suffix}${PI_SUFFIX}${CLAMP_SUFFIX}"
}

needs_run_seed() {
  local dir="$1" seed="$2" pattern
  [ -n "$FORCE" ] && return 0
  [ -d "$dir" ] || return 0
  if [ -n "$seed" ]; then
    pattern="${dir}/rollout_surface_*_${seed}.pt"
  else
    pattern="${dir}/rollout_surface_*.pt"
  fi
  compgen -G "$pattern" > /dev/null && return 1
  return 0
}

submit() {
  echo "+ sbatch --account=$SBATCH_ACCOUNT --constraint=$SBATCH_CONSTRAINT analysis/long_rollout.sh $*"
  if [ -z "$DRY_RUN" ]; then
    sbatch --account="$SBATCH_ACCOUNT" --constraint="$SBATCH_CONSTRAINT" analysis/long_rollout.sh "$@"
  fi
}

COMMON=(module="$MODULE" name="$NAME" cluster="$CLUSTER" \
        inference.ckpt_fname="$CKPT_FNAME_ESCAPED" inference.use_ema=True \
        inference.debug=True inference.target="$TARGET" "${EXTRA[@]}")
[ -n "$PI" ] && COMMON+=(inference.zero_forcing_source=pi)

# Same mapping as complete_rollout.sh's ABLATIONS.
#
# no_ozone stops at 18, not 19: spatial_forcing_variables declares an
# "ozone_10" name (indices 9-19) but only 10 real ozone bands ever get
# populated (see dataloaders/cmip_random_lead_time.py's
# _OZONE_BAND_INDICES) -- index 19 is a phantom placeholder in the declared
# config space, but at actual runtime (add_orography=True) it gets
# overwritten with real terrain-elevation data, concatenated onto
# spatial_forcings *after* the 19 real channels. Zeroing/pi-substituting
# index 19 therefore erases real orography instead of a truly inert slot --
# a static field, so it shows up as an immediate, ~constant additive bias
# every rollout step (confirmed directly: ~27K constant offset from the
# very first step, before this fix).
declare -A ABLATIONS=(
  [no_co2]="1"
  [no_methane]="0"
  [no_n2o]="2"
  [no_ozone]="9,10,11,12,13,14,15,16,17,18"
  [no_aerosol]="3,4,5,6,7,8"
)

echo "== $NAME (module=$MODULE, project=$PROJECT, ckpt=$CKPT_FNAME, cluster=$CLUSTER, target=$TARGET, num_seeds=$NUM_SEEDS, pi=${PI:-0}) =="

for label in no_co2 no_methane no_n2o no_ozone no_aerosol; do
  idx="${ABLATIONS[$label]}"
  suffix="_zs${idx//,/-}"
  out_dir="$(outdir_for "$TARGET" "$suffix")"
  if [ "$NUM_SEEDS" -gt 1 ]; then
    for seed in $(seq 0 $((NUM_SEEDS - 1))); do
      if needs_run_seed "$out_dir" "$seed"; then
        submit "${COMMON[@]}" "inference.zero_spatial_forcing_indices=[$idx]" \
          inference.num_seeds="$NUM_SEEDS" inference.seed_index="$seed"
      else
        echo "-- skipping $label seed $seed: already present at $out_dir"
      fi
    done
  else
    if needs_run_seed "$out_dir" ""; then
      submit "${COMMON[@]}" "inference.zero_spatial_forcing_indices=[$idx]"
    else
      echo "-- skipping $label: already present at $out_dir"
    fi
  fi
done

echo "== done for $NAME =="

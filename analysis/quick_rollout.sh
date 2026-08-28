#!/bin/bash
# Reduced version of analysis/complete_rollout.sh: submits only the two
# rollouts needed for a fast subset of paper_figures_generate_all.py:
#
#   - val  (ssp434)  baseline, debug=True (surface-only)
#   - test (ssp585)  baseline, debug=True (surface-only) -- feeds
#     paper_figures_585_analysis.py's extrapolation check
#
# Both targets are surface-only now (rollout_surface_*.pt only, no
# rollout_level_*.pt/rollout_lev_*.pt) -- val used to run debug=False
# ("full vars"), but that's no longer what this script does.
#
# Skips val3 (ssp534-over) and the 5 forcing-ablation runs, so
# global_mean/hovmoller (need both 434 and 534) and forced_response
# (needs the ablations) won't be generated -- everything else will be.
#
# Noise-seeded ensembles: if the module config has `use_energy_score: True`
# (see model/base_climate_module.py's shared_forward_logic /
# model/forecast.py's forward_multistep energy_score_seed), each target gets
# a 5-member ensemble instead of a single rollout -- one sbatch job per seed
# (inference.num_seeds=5, inference.seed_index=0..4), each drawing a fresh,
# reproducibly-seeded N(0,I) energy-score noise draw every autoregressive
# step. All 5 seeds for a target land in the *same* out_dir, distinguished by
# a "_seed<n>" filename suffix (rollout_surface_<chunk>_<member>_seed<n>.pt) --
# the out_dir name itself doesn't change. Non-energy-score modules get the
# old single-job behavior (no seed suffix), since they have no eval-time
# stochasticity to ensemble over.
#
# That's up to 2 or 10 sbatch jobs per model/checkpoint (2 targets x 1 or 5
# seeds), each submitted via analysis/long_rollout.sh exactly as you'd submit
# it by hand. Idempotent per seed: before submitting, checks whether that
# seed's output already exists and skips it if so -- safe to re-run against
# a model that's already partly done.
#
# Usage:
#   ./analysis/quick_rollout.sh <module_name> <ckpt_step> [--name RUN_NAME] [--cluster CLUSTER] [--dry-run] [--force]
#
# --name lets the run/checkpoint directory (Hydra's `name=`) differ from the
# module config file (Hydra's `module=`) -- needed whenever a run was
# submitted with an explicit name= override instead of letting it default to
# the module name (e.g. name=canesm5_native with a much longer module
# config). Defaults to MODULE when omitted, matching the old behavior.
#
# --cluster overrides the Hydra cluster= profile (default jean_zay_v100).
# NOTE: this alone does not control which GPU hardware the job actually runs
# on -- long_rollout.sh hardcodes #SBATCH --account/--constraint (currently
# defaulting to mlr@v100/v100-16g), overridable via sbatch's own
# --account/--constraint flags. This script does not pass those through, so
# --cluster jean_zay_a100 here would desync the Hydra config from the actual
# (still V100) hardware -- submit those manually via long_rollout.sh with
# explicit sbatch --account=mlr@a100 --constraint=a100 instead.
#
# Example:
#   ./analysis/quick_rollout.sh \
#     forcing_dropout_no_random_lt_pf8_no_uncond_grouped_dropout_pilot_full_state_circular_pad_ablation_adaln1d \
#     22000
#
#   ./analysis/quick_rollout.sh \
#     forcing_dropout_no_random_lt_pf8_no_uncond_grouped_dropout_pilot_full_state_circular_pad_canesm5_damip_native_grid \
#     22000 --name canesm5_native

set -euo pipefail

if [ $# -lt 2 ]; then
  echo "Usage: $0 <module_name> <ckpt_step> [--name RUN_NAME] [--cluster CLUSTER] [--dry-run] [--force]" >&2
  exit 1
fi

MODULE="$1"
CKPT_STEP="$2"
shift 2
NAME=""
CLUSTER=""
DRY_RUN=""
FORCE=""
while [ $# -gt 0 ]; do
  case "$1" in
    --name)
      [ $# -ge 2 ] || { echo "error: --name requires a value" >&2; exit 1; }
      NAME="$2"
      shift 2
      ;;
    --cluster)
      [ $# -ge 2 ] || { echo "error: --cluster requires a value" >&2; exit 1; }
      CLUSTER="$2"
      shift 2
      ;;
    --dry-run) DRY_RUN=1; shift ;;
    --force) FORCE=1; shift ;;
    *) echo "error: unrecognized argument: $1" >&2; exit 1 ;;
  esac
done
NAME="${NAME:-$MODULE}"
CLUSTER="${CLUSTER:-jean_zay_v100}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

MODULE_CFG="configs/module/${MODULE}.yaml"
if [ ! -f "$MODULE_CFG" ]; then
  echo "error: no module config at $MODULE_CFG" >&2
  exit 1
fi
PROJECT=$(grep -m1 '^project:' "$MODULE_CFG" | sed 's/^project:[[:space:]]*//')

# Drives the noise-seeded-ensemble branch below -- see header comment.
if grep -qE '^\s*use_energy_score:\s*[Tt]rue\s*$' "$MODULE_CFG"; then
  NUM_SEEDS=5
else
  NUM_SEEDS=1
fi

CKPT_FNAME=$(printf "step-step=%06d.ckpt" "$CKPT_STEP")
SCRATCH="${SCRATCH:-/lustre/fsn1/projects/rech/mlr/udy16au}"
CKPT_PATH="${SCRATCH}/model_output/${PROJECT}/${NAME}/checkpoints/${CKPT_FNAME}"
if [ ! -f "$CKPT_PATH" ] && [ -z "$DRY_RUN" ]; then
  echo "error: checkpoint not found: $CKPT_PATH" >&2
  echo "  (pass --dry-run to print commands without this check / without submitting)" >&2
  exit 1
fi

# Hydra's override grammar needs the '=' inside the value escaped.
CKPT_FNAME_ESCAPED="${CKPT_FNAME/=/\\=}"
GENERATED_DATA="${SCRATCH}/generated_data"

# out_dir naming mirrors long_rollout.py exactly (num_inference_steps=12,
# num_members=10, scale_input_noise=1, num_rollout_steps=1020, cf_guidance=1,
# year=0, ema_suffix="_ema" since inference.use_ema=True -- all defaults from
# configs/inference/cmip_80_year.yaml, including clamp_spatial_forcing_indices=[3]/
# clamp_spatial_forcing_std=10, which long_rollout.py's forcing_suffix always
# appends as "_clamp3std10" unless overridden on the CLI). The seed doesn't
# appear in out_dir -- all 5 members' chunks land in this same directory, one
# per seed via the "_seed<n>" filename suffix (see needs_run_seed).
CLAMP_SUFFIX="_clamp3std10"
outdir_for() {
  local target="$1"
  echo "${GENERATED_DATA}/${NAME}_12_10_1_1020_1_${target}_0_${CKPT_FNAME}_ema${CLAMP_SUFFIX}"
}

needs_run_seed() {
  # $1: out_dir, $2: seed index or empty string for the no-seed (NUM_SEEDS=1) case
  local dir="$1" seed="$2" pattern
  [ -n "$FORCE" ] && return 0
  [ -d "$dir" ] || return 0
  if [ -n "$seed" ]; then
    pattern="${dir}/rollout_surface_*_seed${seed}.pt"
  else
    pattern="${dir}/rollout_surface_*.pt"
  fi
  compgen -G "$pattern" > /dev/null && return 1
  return 0
}

submit() {
  echo "+ sbatch analysis/long_rollout.sh $*"
  if [ -z "$DRY_RUN" ]; then
    sbatch analysis/long_rollout.sh "$@"
  fi
}

COMMON=(module="$MODULE" name="$NAME" cluster="$CLUSTER" \
        inference.ckpt_fname="$CKPT_FNAME_ESCAPED" inference.use_ema=True \
        inference.debug=True)

echo "== $NAME (module=$MODULE, project=$PROJECT, ckpt=$CKPT_FNAME, cluster=$CLUSTER, num_seeds=$NUM_SEEDS) =="

for target in val test; do
  out_dir="$(outdir_for "$target")"
  if [ "$NUM_SEEDS" -gt 1 ]; then
    for seed in $(seq 0 $((NUM_SEEDS - 1))); do
      if needs_run_seed "$out_dir" "$seed"; then
        submit "${COMMON[@]}" inference.target="$target" \
          inference.num_seeds="$NUM_SEEDS" inference.seed_index="$seed"
      else
        echo "-- skipping $target seed $seed: already present at $out_dir"
      fi
    done
  else
    if needs_run_seed "$out_dir" ""; then
      submit "${COMMON[@]}" inference.target="$target"
    else
      echo "-- skipping $target: already present at $out_dir"
    fi
  fi
done

echo "== done for $NAME =="

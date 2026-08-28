#!/bin/bash
# Submit every inference rollout a model/checkpoint needs to feed the full
# paper_figures_*.py suite (see analysis/paper_figures_generate_all.py):
#
#   - val   (ssp434)       baseline, debug=False -- rollout_level_*.pt +
#     rollout_lev_*.pt for physical/global_mean/hovmoller/whisker_plots,
#     rollout_surface_*.pt for table/rmse/forced_response's baseline
#   - val3  (ssp534-over)  baseline, debug=False -- same, scenario "534"
#   - test  (ssp585)       baseline, debug=True (surface-only) -- feeds
#     paper_figures_585_analysis.py's extrapolation check only
#   - 5 single-forcing-removed ablations at val (ssp434), debug=True
#     (surface-only tas is all paper_figures_forced_response.py needs):
#       no_co2 (zero index 1), no_methane (0), no_n2o (2),
#       no_ozone (9-18), no_aerosol (3-8)
#
# That's up to 8 sbatch jobs per model/checkpoint, each submitted via
# analysis/long_rollout.sh exactly as you'd submit it by hand. Idempotent:
# before submitting, checks whether the expected output dir already has the
# files that job would produce (rollout_surface_*.pt always; also
# rollout_level_*.pt/rollout_lev_*.pt for the debug=False baselines) and
# skips it if so -- safe to re-run against a model that's already partly done.
# This only checks *some* output of the right kind exists, not that every
# rollout chunk finished -- a partially-completed run will be skipped too.
#
# Usage:
#   ./analysis/complete_rollout.sh <module_name> <ckpt_step> [--dry-run] [--force]
#
# Example:
#   ./analysis/complete_rollout.sh \
#     forcing_dropout_no_random_lt_pf8_no_uncond_grouped_dropout_pilot_full_state_circular_pad_ablation_adaln1d \
#     22000

set -euo pipefail

if [ $# -lt 2 ]; then
  echo "Usage: $0 <module_name> <ckpt_step> [--dry-run] [--force]" >&2
  exit 1
fi

MODULE="$1"
CKPT_STEP="$2"
shift 2
DRY_RUN=""
FORCE=""
for arg in "$@"; do
  case "$arg" in
    --dry-run) DRY_RUN=1 ;;
    --force) FORCE=1 ;;
    *) echo "error: unrecognized argument: $arg" >&2; exit 1 ;;
  esac
done

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

MODULE_CFG="configs/module/${MODULE}.yaml"
if [ ! -f "$MODULE_CFG" ]; then
  echo "error: no module config at $MODULE_CFG" >&2
  exit 1
fi
PROJECT=$(grep -m1 '^project:' "$MODULE_CFG" | sed 's/^project:[[:space:]]*//')

CKPT_FNAME=$(printf "step-step=%06d.ckpt" "$CKPT_STEP")
CKPT_PATH="/scratch/gclyne/model_output/${PROJECT}/${MODULE}/checkpoints/${CKPT_FNAME}"
if [ ! -f "$CKPT_PATH" ] && [ -z "$DRY_RUN" ]; then
  echo "error: checkpoint not found: $CKPT_PATH" >&2
  echo "  (pass --dry-run to print commands without this check / without submitting)" >&2
  exit 1
fi

# Hydra's override grammar needs the '=' inside the value escaped.
CKPT_FNAME_ESCAPED="${CKPT_FNAME/=/\\=}"
GENERATED_DATA="/scratch/gclyne/generated_data"

# out_dir naming mirrors long_rollout.py exactly (num_inference_steps=12,
# num_members=10, scale_input_noise=1, num_rollout_steps=1020, cf_guidance=1,
# year=0, ema_suffix="_ema" since inference.use_ema=True -- all defaults from
# configs/inference/cmip_80_year.yaml, including clamp_spatial_forcing_indices=[3]/
# clamp_spatial_forcing_std=10, which long_rollout.py's forcing_suffix always
# appends as "_clamp3std10" -- after any zs/zns/pi suffix, matching its own
# ordering -- unless overridden on the CLI).
CLAMP_SUFFIX="_clamp3std10"
outdir_for() {
  local target="$1" forcing_suffix="${2:-}"
  echo "${GENERATED_DATA}/${MODULE}_12_10_1_1020_1_${target}_0_${CKPT_FNAME}_ema${forcing_suffix}${CLAMP_SUFFIX}"
}

needs_run() {
  # $1: out_dir, $2: "surface" or "full" (full also requires level+lev)
  local dir="$1" kind="$2"
  [ -n "$FORCE" ] && return 0
  [ -d "$dir" ] || return 0
  compgen -G "${dir}/rollout_surface_*.pt" > /dev/null || return 0
  if [ "$kind" = "full" ]; then
    compgen -G "${dir}/rollout_level_*.pt" > /dev/null || return 0
    compgen -G "${dir}/rollout_lev_*.pt" > /dev/null || return 0
  fi
  return 1
}

submit() {
  echo "+ sbatch analysis/long_rollout.sh $*"
  if [ -z "$DRY_RUN" ]; then
    sbatch analysis/long_rollout.sh "$@"
  fi
}

COMMON=(module="$MODULE" name="$MODULE" cluster=cleps \
        inference.ckpt_fname="$CKPT_FNAME_ESCAPED" inference.use_ema=True)

echo "== $MODULE (project=$PROJECT, ckpt=$CKPT_FNAME) =="

# --- baselines ---
if needs_run "$(outdir_for val)" full; then
  submit "${COMMON[@]}" inference.target=val inference.debug=False
else
  echo "-- skipping val baseline: already present at $(outdir_for val)"
fi

if needs_run "$(outdir_for val3)" full; then
  submit "${COMMON[@]}" inference.target=val3 inference.debug=False
else
  echo "-- skipping val3 baseline: already present at $(outdir_for val3)"
fi

if needs_run "$(outdir_for test)" surface; then
  submit "${COMMON[@]}" inference.target=test   # debug defaults True: surface-only
else
  echo "-- skipping test baseline: already present at $(outdir_for test)"
fi

# --- single-forcing-removed ablations (val/ssp434, surface-only) ---
# no_ozone stops at 18, not 19 -- see analysis/ablation_rollout.sh's
# ABLATIONS comment: index 19 is real orography at runtime
# (add_orography=True), not a 10th ozone band.
declare -A ABLATIONS=(
  [no_co2]="1"
  [no_methane]="0"
  [no_n2o]="2"
  [no_ozone]="9,10,11,12,13,14,15,16,17,18"
  [no_aerosol]="3,4,5,6,7,8"
)
for label in no_co2 no_methane no_n2o no_ozone no_aerosol; do
  idx="${ABLATIONS[$label]}"
  suffix="_zs${idx//,/-}"
  out_dir="$(outdir_for val "$suffix")"
  if needs_run "$out_dir" surface; then
    submit "${COMMON[@]}" inference.target=val "inference.zero_spatial_forcing_indices=[$idx]"
  else
    echo "-- skipping $label: already present at $out_dir"
  fi
done

echo "== done for $MODULE =="

# Generating the plots/tables

The actual figure pipeline is `analysis/paper_figures_generate_all.py`, driven by a YAML config
under `analysis/configs/` (see `paper_figures_table_pf_ablations.yaml` or
`paper_figures_table_ap_ablations_pf4.yaml` for real examples to copy). It runs, in order:

1. `paper_figures_table.py` — once, builds the comparison table (RMSE, spatial std, ensemble
   spread, inter-annual variability) across every model/scenario in the config.
2. `paper_figures_rmse_plot.py` — once per scenario, regional RMSE plots.
3. `paper_figures_ocean_rmse_plot.py` — once per scenario; needs `rollout_lev_*.pt`, i.e. that
   scenario's rollouts must have been generated with `inference.debug=False`.
4. `paper_figures_physical.py` — once per scenario; needs both `rollout_level_*.pt` and
   `rollout_lev_*.pt`, same `debug=False` requirement.
5. `paper_figures_whisker_plots.py` — once per scenario, `--variable ta`; same requirement.
6. `paper_figures_global_mean.py` + `paper_figures_hovmoller.py` — only if the config declares
   *both* a `"434"` and a `"534"` scenario (these are an inherently side-by-side comparison).
7. `paper_figures_585_analysis.py` — only if the config declares a `"585"` scenario (the
   unseen-forcing extrapolation check).
8. `paper_figures_forced_response.py` — only if the config has a `forced_response:` section (a
   separate schema, for comparing forcing-ablation deltas across models — see below).

Each step is its own subprocess; a failure in one doesn't stop the rest, and any of them can be
run standalone (`python -m analysis.paper_figures_rmse_plot --config ... --scenario 434`) while
iterating on one figure. Run it with:

```bash
python analysis/paper_figures_generate_all.py --config analysis/configs/<your_config>.yaml
```

Config schema, the scenario/models part (used by steps 1-7):

```yaml
config_module: <a module config name>   # just for norm stats + variable ordering; doesn't need
                                         # to be the exact config each run below was trained with,
                                         # as long as they share normalization stats
variable: tas
final_decade_years: 10
climatology: {realization: r1i1p1f1, field: lev, var_idx: 0, n_clim_years: 20, skip_last_years: 10}

scenarios:
  - key: "434"                 # "434", "534", "585" are special -- gate steps 6/7 above
    label: "SSP4-3.4"
    target: {realization: r1i1p1f1, scenario: ssp434}
    mesmer: {path: /scratch/gclyne/tutorials/emulations_434_all.nc}   # optional
    models:
      full:                    # one row per model
        label: "$AC_{full}$"
        runs:                  # one or more rollout dirs under generated_data/, stacked as
          - <rollout_dir_name>   # pseudo-ensemble-members if you list more than one
output_name: <name>   # outputs land under ./plots/<name>/
```

The separate schema for step 8 (`forced_response:`):

```yaml
forced_response:
  start_year: 2015   # optional, defaults to 2015 (matches inference.target=val -> ssp434)
  models:
    <label>: <baseline_rollout_dir_name>   # the dir *before* the _zs* ablation suffix that
                                           # long_rollout.py appends -- whichever _zs* ablation
                                           # dirs exist on disk next to it get plotted
```

All outputs (table `.tex`, RMSE/physical/whisker/forced-response plots) land under
`./plots/<output_name>/`.

# Paper figure provenance

Maps every figure actually included in `InterpolationProject_GrahamClyne/iopjournal-template.tex`
(as of this audit) to the script + config that generates it, and the
`generated_data/` rollout directories it depends on. Verified against disk on
2026-08-13 -- every dependency listed as "data: OK" below was confirmed
present at that time; re-check before trusting an old copy of this file.

General pattern: `paper_figures_*.py` scripts take `--config
analysis/configs/<name>.yaml` (a small YAML declaring which rollout
directories under `{scratch}/generated_data/` to load) and write into
`plots/<output_name>/`. The PDF actually embedded in the paper under
`InterpolationProject_GrahamClyne/figures/` is usually a manually
copied/renamed version of that output -- copy it back over after
regenerating, the tex file does not read from `plots/` directly.

**Run these on a compute node, not the login shell.** `paper_figures_whisker_plots.py`
and `paper_figures_physical.py` load full-level (`load_run_level_all`/`load_run_level`)
3D atmospheric fields across every ensemble member at once and reliably get
OOM-killed (`Killed`, exit 137) when run interactively on the login node --
confirmed directly, twice, during this session. `paper_figures_forced_response.py`
(surface-only) is usually fine interactively but can also get killed under
heavy shared-node load. Submit via `sbatch` with a generous `--mem` instead;
`analysis/_scratch_whisker_534_regen.sh` / `_scratch_physical_534_regen.sh` /
`_scratch_forced_response_regen.sh <config-name>` are ready-to-resubmit
templates (`cpu_devel` partition, 150-200GB, no GPU needed) left over from
this session's regeneration. **A background/piped invocation's reported "exit
code 0" is not reliable evidence of success** -- `cmd 2>&1 | tail -N` reports
`tail`'s exit code, not the Python process's, so a silently OOM-killed run
still shows "completed exit code 0" with no traceback. Always confirm by
checking the output file's actual mtime (`stat`), not just its existence or
the job runner's own status.

**Style/consistency conventions** (added this session, apply to any new
`paper_figures_*.py` figure): colors and font sizes come from
`analysis/plot_style.py` (`ABLATION_COLORS`, `MODEL_COLORS`,
`AXIS_LABEL_FONTSIZE`/`TICK_FONTSIZE`/`LEGEND_FONTSIZE`) rather than a
positional palette, so the same ablation/model gets the same color in every
figure regardless of a config's own key ordering -- see
`paper_figures_whisker_plots.py`/`paper_figures_585_analysis.py`'s prior bug
(both imported an independent positional `MODEL_COLORS` from
`paper_figures_rmse_plot.py`, unrelated to `paper_figures_forced_response.py`'s
own ablation palette). No in-figure titles anywhere (`column_titles`/region
titles removed; identify panels via the caption instead). `AC_full`/`$AC_{full}$`
retired in favor of `AC-SSP` (renamed across every config, not just the ones
feeding a current paper figure). `paper_figures_forced_response.py`'s
`physical_reference` TCR line is now drawn inline, in the same panel and
color as its own ablation, instead of a separate duplicate row -- the old
two-row layout repeated the CO2/methane/N2O traces once per row for no
reason; `n_rows` is now always 1, and `legend_between_rows` was removed as a
config option (only `legend_in_axes`/`legend_ncol` remain).

## Known issues found during this audit (not fixed here)

- **Dangling references in `app:flow_destroys_signal`** ("Flow Matching
  Degrades the Deterministic Model's Learned Forcing Response" section):
  `\autoref{fig:flow_vs_deterministic_pi_only}` (4 occurrences) and
  `\autoref{fig:lt1_12_forced_response}` (2 occurrences) have no matching
  `\label{...}` anywhere in the current document -- the figure/section they
  pointed to was evidently removed in a prior editing pass, but the prose
  citing it was not updated. These will render as `??` in the compiled PDF.
  The underlying figure (`flow_vs_deterministic_lt1_12_pi_only_forced_response.pdf`)
  and its generating config
  (`analysis/configs/paper_figures_flow_vs_deterministic_lt1_12_pi_only_forced_response.yaml`)
  still exist on disk if you want to restore it instead of cutting the
  references.
- **`flow_damip_pf8_fixed_v2`'s baseline rollout used for
  `fig:flow_damip_pf8_fixed_v2_forced_response`** (left column) was
  regenerated this session with `load_deterministic_model` overridden to
  `deterministic_damip_pf4_fixed` instead of the flow model's own
  originally-trained-against `deterministic_damip_pf8_fixed`, because the
  latter's checkpoint directory no longer exists under
  `/scratch/gclyne/model_output/cmip-interpolation/` (confirmed gone --
  wandb logs and old `generated_data/deterministic_damip_pf8_fixed*` outputs
  prove it existed and trained successfully before, just not present now).
  The right column (`deterministic_damip_pf8_fixed` itself, used directly,
  not as a wrapped det-model) is unaffected. If `deterministic_damip_pf8_fixed`
  reappears on disk, prefer regenerating that baseline against it instead.

## Figure map

| Figure (`\label`) | PDF in `figures/` | Script | Config |
|---|---|---|---|
| `fig:abrupt4xco2_det_vs_es` | `abrupt4xco2_tas_pr_det_vs_es.pdf` | `analysis/abrupt4xco2_det_vs_es.py` | none (hardcoded run names in script) |
| `fig:flow_damip_pf8_fixed_v2_forced_response` | `flow_vs_det_pf8_fixed_forced_response.pdf` | `analysis/paper_figures_forced_response.py` | `analysis/configs/paper_figures_flow_vs_det_pf8_fixed_forced_response.yaml` |
| `fig:surface_forced_response` | `forced_response_diff_baseline.pdf` | `analysis/paper_figures_forced_response.py` | `analysis/configs/paper_figures_energy_score_w050_80_10_534_forced_response.yaml` |
| `fig:whisker_534` | `whisker_delta_534_ta.pdf` | `analysis/paper_figures_whisker_plots.py` | `analysis/configs/paper_figures_pf4_w050_80_10_step40000_whisker_534.yaml` |
| `fig:extrapolation` | `extrapolation.pdf` | `analysis/paper_figures_585_analysis.py` | `analysis/configs/paper_figures_energy_score_w050_80_10_585_extrapolation.yaml` |
| `fig:hydrostatic` | `hydrostatic_consistency_534.pdf` | `analysis/paper_figures_physical.py` | `analysis/configs/paper_figures_pf4_w050_80_10_step40000_mesmer_physical.yaml` |
| `fig:energy_budget_pf4_w050_80_10` | `energy_budget_pf4_w050_80_10_534.pdf` | `analysis/paper_figures_physical.py` | `analysis/configs/paper_figures_pf4_w050_80_10_step40000_mesmer_physical.yaml` |
| `fig:canesm5_forced_response` | `canesm5_pf4_energy_score_w050_534_forced_response.pdf` | `analysis/paper_figures_forced_response.py` | `analysis/configs/paper_figures_canesm5_pf4_energy_score_w050_534_forced_response.yaml` |
| `fig:canesm5_multiscenario` | `canesm5_multiscenario.pdf` | `analysis/canesm5_global_mean_multiscenario.py` | none (hardcoded run names in script) |
| `fig:w050_80_10_434_forced_response` | `w050_80_10_434_forced_response.pdf` | `analysis/paper_figures_forced_response.py` | `analysis/configs/paper_figures_w050_80_10_434_forced_response.yaml` |
| `fig:w050_100_20_434_forced_response` | `w050_100_20_434_forced_response.pdf` | `analysis/paper_figures_forced_response.py` | `analysis/configs/paper_figures_w050_100_20_434_forced_response.yaml` |

`analysis/paper_figures_generate_all.py` does not currently wrap all of the
above into one invocation -- each figure was generated by its own script
call, listed per-figure below.

## Per-figure detail

### `abrupt4xco2_tas_pr_det_vs_es.pdf`
```
python -m analysis.abrupt4xco2_det_vs_es
```
Compares `deterministic_damip_pf2_fixed` (plain MSE) against
`deterministic_damip_pf4_energy_score_w050_80_10` (energy score), abrupt-4xCO2,
tas + pr, against the IPSL target (r2i1p1f1). Run names are hardcoded in the
script (`RUNS` dict / `TARGET_REALIZATION`), not config-driven.

Data: `deterministic_damip_pf2_fixed_12_10_1_1020_1_abrupt_0_step-step=022000.ckpt_ema`,
`deterministic_damip_pf4_energy_score_w050_80_10_0_12_10_1_1020_1_abrupt_0_step-step=040000.ckpt_ema`
-- both **OK**.

### `flow_vs_det_pf8_fixed_forced_response.pdf`
```
python -m analysis.paper_figures_forced_response \
    --config analysis/configs/paper_figures_flow_vs_det_pf8_fixed_forced_response.yaml
```
`flow_damip_pf8_fixed_v2` vs. `deterministic_damip_pf8_fixed`, side by side,
single-forcing piControl-held (`_pi`) ablations, SSP4-3.4, step-022000, EMA,
`clamp3std10`. `pi_only: true`, `physical_reference: true`,
`column_titles` left unset (false) since the paper captions the two columns
itself.

Data (all under `generated_data/`, all `_clamp3std10`): flow baseline
`flow_damip_pf8_fixed_v2_12_10_1_1020_1_val_0_step-step=022000.ckpt_ema`
(**regenerated this session against `deterministic_damip_pf4_fixed`, see
Known Issues above**) + its `_zs{0,1,2,3-4-5-6-7-8,9-10-...-19}_pi` ablations
-- OK; deterministic baseline
`deterministic_damip_pf8_fixed_12_10_1_1020_1_val_0_step-step=022000.ckpt_ema`
+ its `_pi` ablations -- OK (pre-existing, unaffected by the det-model-swap
issue).

### `forced_response_diff_baseline.pdf` (→ `fig:surface_forced_response`)
```
python -m analysis.paper_figures_forced_response \
    --config analysis/configs/paper_figures_energy_score_w050_80_10_534_forced_response.yaml
```
`deterministic_damip_pf4_energy_score_w050_80_10` (step-040000, EMA),
SSP5-3.4-over (val3), single-forcing piControl-held ablations vs. Myhre/TCR
reference. **Filename collision warning**: `forced_response_diff_baseline.pdf`
is the fixed output name every `paper_figures_forced_response.py` config
produces -- always copy from the specific `plots/<output_name>/` directory
matching this config (`paper_figures_energy_score_w050_80_10_534_forced_response`),
not just "whatever `forced_response_diff_baseline.pdf` happens to be lying
around."

Data: `deterministic_damip_pf4_energy_score_w050_80_10_0_12_10_1_730_1_val3_0_step-step=040000.ckpt_ema`
baseline + `_zs{0,1,2,3-4-5-6-7-8,9-10-...-18}_pi` ablations -- **OK**.

### `whisker_delta_534_ta.pdf`
```
python -m analysis.paper_figures_whisker_plots \
    --config analysis/configs/paper_figures_pf4_w050_80_10_step40000_whisker_534.yaml \
    --scenario 534 --variable ta --style whisker
```
Atmospheric-column ($ta$) whisker plot, `deterministic_damip_pf4_energy_score_w050_80_10`,
step-040000, EMA, SSP5-3.4-over, 5-seed ensembles.

Data: `deterministic_damip_pf4_energy_score_w050_80_10_0_12_10_1_1020_1_val3_0_step-step=040000.ckpt_ema`
baseline + `_pi` ablations (same set as above, `730`-step variants) -- **OK**.

### `extrapolation.pdf`
```
python -m analysis.paper_figures_585_analysis \
    --config analysis/configs/paper_figures_energy_score_w050_80_10_585_extrapolation.yaml \
    --scenario 585
```
SSP5-8.5 out-of-distribution test, `deterministic_damip_pf4_energy_score_w050_80_10`
(step-040000, EMA) vs. IPSL's 4-realization target ensemble (r1i1p1f1-r4i1p1f1).
Output written as `extrapolation_585.pdf` inside `plots/<output_name>/` --
rename on copy.

Data: `deterministic_damip_pf4_energy_score_w050_80_10_0_12_10_1_1020_1_test_0_step-step=040000.ckpt_ema`
(5 seeds) -- **OK**. Target realizations r1-r4 ssp585 assumed present under
`/scratch/gclyne/memmap_filled_in/` (raw CMIP data, not checked here -- much
lower risk of having been cleaned up than a model checkpoint).

### `hydrostatic_consistency_534.pdf` / `energy_budget_pf4_w050_80_10_534.pdf`
```
python -m analysis.paper_figures_physical \
    --config analysis/configs/paper_figures_pf4_w050_80_10_step40000_mesmer_physical.yaml \
    --scenario 534
```
Both come from the same script/config/scenario invocation (it writes 5 PDFs
per scenario: `hydrostatic_534_full.pdf`, `energy_budget_534_full.pdf`,
`supersaturation_534_full.pdf`, `cc_scaling_534_full.pdf`,
`spatial_bias_534_full.pdf` into `plots/<output_name>/physical/` -- only the
first two are used in the paper, renamed on copy).
`deterministic_damip_pf4_energy_score_w050_80_10`, step-040000, EMA, SSP5-3.4-over.

Data: `deterministic_damip_pf4_energy_score_w050_80_10_0_12_10_1_1020_1_val3_0_step-step=040000.ckpt_ema`
-- **OK**. (Config also declares a MESMER `.nc` path for a separate table this
script doesn't use; confirmed present anyway: `/scratch/gclyne/tutorials/emulations_534_all.nc`.)

### `canesm5_pf4_energy_score_w050_534_forced_response.pdf`
```
python -m analysis.paper_figures_forced_response \
    --config analysis/configs/paper_figures_canesm5_pf4_energy_score_w050_534_forced_response.yaml
```
CanESM5 native-grid energy-score model, SSP5-3.4-over, piControl-held
single-forcing ablations (no aerosol channel for CanESM5). Needs
`dataloader: cmip_random_lead_times_canesm5_native` in the config (a prior
version of this config omitted it and silently mixed 64-wide CanESM5 spatial
forcings with 144-wide IPSL orography -- already fixed, noted in the config's
own comment).

Data: `canesm5_damip_pf4_energy_score_w050_0_12_10_1_730_1_val3_0_step-step=022000.ckpt_ema`
+ ablations -- **OK**.

### `canesm5_multiscenario.pdf`
```
python -m analysis.canesm5_global_mean_multiscenario \
    --config-module canesm5_damip_pf4_energy_score_w050 \
    --scenarios 434,534,585 \
    --output plots/canesm5_global_mean_multiscenario/canesm5_multiscenario.pdf
```
CanESM5 energy-score model, all three scenarios on one axes, vs. each
scenario's own CanESM5 target. Run names and target scenario names are
hardcoded in the script's `SCENARIOS` dict, not config-driven -- `--scenarios`
just selects which keys of that dict to plot (default is `434,585`; the
figure as included in the paper needs all three, so pass `534` explicitly).

Data (from `SCENARIOS` dict): `canesm5_damip_pf4_energy_score_w050_0_12_10_1_1020_1_val_0_step-step=022000.ckpt_ema` (434),
`canesm5_damip_pf4_energy_score_w050_0_12_10_1_730_1_val3_0_step-step=022000.ckpt_ema` (534),
`canesm5_damip_pf4_energy_score_w050_0_12_10_1_1020_1_test_0_step-step=022000.ckpt_ema` (585)
-- **OK**.

### `w050_80_10_434_forced_response.pdf` / `w050_100_20_434_forced_response.pdf`
```
python -m analysis.paper_figures_forced_response \
    --config analysis/configs/paper_figures_w050_80_10_434_forced_response.yaml
python -m analysis.paper_figures_forced_response \
    --config analysis/configs/paper_figures_w050_100_20_434_forced_response.yaml
```
Forcing-all-present-probability ablation pair ($p_{\text{all}}=0.8$ vs.\
$1.0$), SSP4-3.4, step-022000, EMA -- see `app:ap_08_vs_10_forced_response`.

Data: `deterministic_damip_pf4_energy_score_w050_0_12_10_1_1020_1_val_0_step-step=022000.ckpt_ema`,
`deterministic_damip_pf4_energy_score_w050_100_20_12_10_1_1020_1_val_0_step-step=022000.ckpt_ema`
-- **OK**.

## Table map

Unlike the figures above, no `paper_figures_*.py` table script writes a
`.tex`/`.pdf` the paper `\input{}`s directly -- each run prints/renders
numbers that were then hand-copied into the tex source. Re-running the
script below regenerates the same numbers to check against, it does not
regenerate the table file itself.

| Table (`\label`) | Script | Config |
|---|---|---|
| `tab:architecture_ablations` (Pushforward Length rows) | `analysis/paper_figures_table.py` | `analysis/configs/paper_figures_table_pf_ablations.yaml` |
| `tab:architecture_ablations` (Forcing All-Present Probability rows) | `analysis/paper_figures_table.py` | `analysis/configs/paper_figures_table_ap_ablations_pf4.yaml` |
| `tab:model_performance` | `analysis/paper_figures_table.py` | **unresolved, see below** |
| `tab:seed_stability_energy_score` | `analysis/paper_figures_table.py` | `analysis/configs/paper_figures_table_seed_stability.yaml` (via `analysis/seed_stability_table.sh`) |
| `tab:pf8_ensemble_size` | `analysis/ssr_vs_members_pf4_w050_80_10_step40000.py` | none (hardcoded run name) |
| `tab:noise_scale_ssr` | `analysis/noise_scale_ssr_appendix.py` | none (hardcoded run name) |
| `tab:atm_level_performance` / `_2` / `_3` | `analysis/paper_figures_level_rmse_table.py` | `analysis/configs/paper_figures_pf4_w050_80_10_step40000_whisker_534.yaml --scenario 534 --model-key full` |
| `tab:ocean_depth_performance` | `analysis/paper_figures_level_rmse_table.py` | same as above |
| `tab:canesm5_ipsl_members` | none -- manual data-inventory table (realization counts per experiment), not computed from a rollout | n/a |

**`tab:model_performance` is not resolved.** It's `AC-SSP` (the paper's actual
default checkpoint, `deterministic_damip_pf4_energy_score_w050_80_10_0`
step-040000) vs. MESMER-M vs. IPSL, SSP4-3.4 and SSP5-3.4-over, 5-member
ensembles -- structurally exactly what `paper_figures_table.py` produces
(see its own docstring), but the only config in `analysis/configs/` shaped
like it, `paper_figures_table_example.yaml`, is a literal template: its
`runs:` list points at stale placeholder checkpoint names
(`new_ozone_flow_lt_1_ens_...`) that don't match the paper's actual default
model, and its values don't reproduce the numbers in the tex. No copy of
this exact config with the real run names survives on disk or in this repo.
To regenerate it, copy `paper_figures_table_example.yaml` and point its
`434`/`534` `models.full.runs` at
`deterministic_damip_pf4_energy_score_w050_80_10_0_12_10_1_1020_1_val_0_step-step=040000.ckpt_ema`
/ the `val3` (534) equivalent (5-member ensemble, so 5 seed rollouts, or
however this checkpoint's ensembling is organized on disk -- check
`generated_data/` for the actual seed suffixes) instead.

## Notes on specific scripts

- `analysis/abrupt4xco2_det_vs_es.py` was renamed from
  `_scratch_es_vs_det_abrupt_transient.py` since it generates a permanent
  paper figure, not a one-off check.
- `analysis/paper_figures_forced_response.py` also saves a `.png` alongside
  the `.pdf` (`out_path.with_suffix(".png")` at the same dpi convention as
  the rest of this file), and has an opt-in `column_titles: true` config
  flag (top-level per-column title, off by default -- most paper figures
  caption the columns manually instead).

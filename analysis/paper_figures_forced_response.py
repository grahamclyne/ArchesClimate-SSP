r"""Forced-response ("Delta global mean tas vs. all-forcings baseline"), side by side across models.

Adapted from analysis/forcing_ablation_analysis.ipynb (cell 12, "## Difference
from baseline (Delta global mean tas)"). For a set of models -- each
identified by a "base_dir_name": the rollout-output-dir name *before* the
forcing-ablation suffix that long_rollout.py appends for
inference.zero_spatial_forcing_indices / zero_non_spatial_forcing_indices
runs -- plots the yearly global-mean tas trajectory of every forcing-
ablation run found on disk for that model, each as its difference from that
SAME model's own all-forcings baseline. One subplot per model, side by side,
with one legend shared across every ablation found for ANY model.

A model contributes a subplot only if its baseline run exists; within a
model, whichever ablation directories exist on disk are plotted and the rest
silently skipped -- no need for every model to have run every ablation.

Config schema (separate from the scenario/models schema used elsewhere in
paper_figures_*.py -- a "model" here is one base experiment name, not a list
of runs to denormalize/stack):

    forced_response:
      start_year: 2015   # optional, defaults to 2015 (matches these runs'
                         # inference.target=val -> ssp434, which starts 2015)
      models:
        adaln1d: forcing_dropout_..._ablation_adaln1d_..._step-step=022000.ckpt_ema
        concat:  forcing_dropout_..._ablation_concat_..._step-step=022000.ckpt_ema
        spade:   forcing_dropout_..._ablation_spade_..._step-step=022000.ckpt_ema

Usage:
    python analysis/paper_figures_forced_response.py \\
        --config analysis/configs/paper_figures_table_example.yaml
"""

import argparse
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from omegaconf import OmegaConf

from analysis.analysis_utils import initialize_notebook
from analysis.compare_runs import load_run
from analysis.plot_style import (
    AXIS_LABEL_FONTSIZE,
    LEGEND_FONTSIZE,
    TICK_FONTSIZE,
)

# --- Physically-expected single-forcing response: Myhre et al. (1998) CO2
# ERF formula + Etminan et al. (2016) revised CH4/N2O ERF formulas, combined
# with a model-specific TCR-derived climate-sensitivity parameter.
#
# Only meaningful against "_pi" ablations (zero_forcing_source=pi): those
# hold the zeroed channel at its literal pre-industrial concentration for
# the WHOLE rollout, which is exactly the counterfactual these formulas
# assume (ΔF relative to a fixed pre-industrial reference). The default
# null-token ablations substitute the model's learned "channel absent"
# embedding, which has no physical concentration value, so there is nothing
# for these formulas to be evaluated against.
#
# This is NOT TCRE: TCRE (Transient Climate Response to cumulative Emissions)
# is an emergent, ~linear relationship between cumulative CO2 EMISSIONS and
# warming, and only applies to emission-driven experiments. These are
# concentration-driven SSP runs (no emissions in the data at all), so the
# relevant standard quantity is TCR (Transient Climate Response: K of
# warming at CO2 doubling under a 1%/yr ramp) combined with the standard
# ERF(C) formula.
#
# Pre-industrial reference concentrations (Myhre et al. 1998 / IPCC AR5-AR6
# convention): CO2 278 ppm, CH4 700 ppb, N2O 270 ppb.
PI_REF = {"carbon": 278.0, "methane": 700.0, "nitrous": 270.0}

# Model-specific TCR (K per CO2 doubling), rather than a single generic
# IPCC AR6 value -- TODO(cite): IPSL-CM6A-LR 2.45 K, CanESM5 2.74 K.
TCR_BY_MODEL_K = {"ipsl": 2.45, "canesm5": 2.74}
F_2XCO2 = 5.35 * np.log(2.0)  # ~3.71 W/m^2
LAMBDA_TCR_BY_MODEL = {k: v / F_2XCO2 for k, v in TCR_BY_MODEL_K.items()}  # K per W/m^2

# CH4/N2O ERF only needs each gas's own concentration trajectory (from the
# same input4MIPs forcing files used for training, available for both
# models) plus the model's own TCR -- it does not require a DAMIP-style
# single-forcing decomposition, so the reference is shown for every model in
# GHG_FORMULAS, same as CO2.
GHG_MODELS_WITH_CH4_N2O_REFERENCE = {"ipsl", "canesm5"}


def co2_erf(C, C0=PI_REF["carbon"]):
    """CO2 effective radiative forcing (W/m^2) relative to C0, Myhre et al. 1998."""
    return 5.35 * np.log(np.asarray(C) / C0)


def ch4_erf(M, N, M0=PI_REF["methane"], N0=PI_REF["nitrous"]):
    """CH4 ERF (W/m^2), Etminan et al. (2016).

    N is N2O's own concentration trajectory (not held at pre-industrial):
    the piControl-held ablation this is compared against only sets the
    ablated gas to pre-industrial and leaves every other forcing, including
    N2O, on its real trajectory, so evaluating the cross-sensitivity term at
    N2O's real value (rather than isolating CH4 the way Myhre et al.'s
    spectral-overlap correction did) matches the actual experimental design.
    """
    M, N = np.asarray(M), np.asarray(N)
    Mbar, Nbar = (M + M0) / 2, (N + N0) / 2
    a3, b3 = -1.3e-6, -8.2e-6
    return (a3 * Mbar + b3 * Nbar + 0.043) * (np.sqrt(M) - np.sqrt(M0))


def n2o_erf(N, M, C, M0=PI_REF["methane"], N0=PI_REF["nitrous"], C0=PI_REF["carbon"]):
    """N2O ERF (W/m^2), Etminan et al. (2016).

    M (CH4) and C (CO2) are held at their own real trajectories, for the
    same reason described in ch4_erf's docstring.
    """
    N, M, C = np.asarray(N), np.asarray(M), np.asarray(C)
    Cbar, Nbar, Mbar = (C + C0) / 2, (N + N0) / 2, (M + M0) / 2
    a2, b2, c2 = -8.0e-6, 4.2e-6, -4.9e-6
    return (a2 * Cbar + b2 * Nbar + c2 * Mbar + 0.117) * (np.sqrt(N) - np.sqrt(N0))


# label (as produced by _build_ablation_set's "(pi)" suffix) -> (gas key
# into PI_REF/spatial_forcing_variables, ERF formula taking the full
# {gas: yearly_concentration} dict so CH4/N2O can read each other's and
# CO2's trajectory). Aerosol/ozone have no comparably simple closed-form
# ERF(concentration) -- IPCC AR6 only assesses a present-day (2019 vs. 1750)
# best estimate, not a formula usable across arbitrary concentrations/years
# -- so they're deliberately absent here.
GHG_FORMULAS = {
    "No CO2 (pi)": ("carbon", lambda gy: co2_erf(gy["carbon"])),
    "No methane (pi)": ("methane", lambda gy: ch4_erf(gy["methane"], gy["nitrous"])),
    "No N2O (pi)": ("nitrous", lambda gy: n2o_erf(gy["nitrous"], gy["methane"], gy["carbon"])),
}


def load_physical_ghg_yearly(dataset, spatial_forcing_variables):
    """Yearly-mean physical (un-normalized) methane/carbon/nitrous concentrations.

    Returns {gas_name: 1-D array (n_years,)}, read directly off the dataset
    (domain matching the ablation rollouts' inference.target) via
    dataset.__getitem__(..., normalize=False).
    """
    idx = {g: spatial_forcing_variables.index(g) for g in PI_REF}
    n_months = len(dataset)
    monthly = {g: [] for g in PI_REF}
    for m in range(n_months):
        item = dataset.__getitem__(m, normalize=False)
        sf = item["state"]["spatial_forcings"]
        for g, i in idx.items():
            monthly[g].append(sf[i].mean().item())
    n_years = n_months // 12
    return {
        g: np.array(vals[: n_years * 12]).reshape(n_years, 12).mean(axis=1)
        for g, vals in monthly.items()
    }


# Ablation definitions: (label, forcing_suffix). Mirrors
# forcing_ablation_analysis.ipynb cell 2 exactly -- these suffixes are
# produced by long_rollout.py from inference.zero_spatial_forcing_indices /
# zero_non_spatial_forcing_indices, not something this script chooses.
# IPSL's spatial_forcing_variables order is
# [methane, carbon, nitrous, load_ASNO3M..load_ASBCM (aerosol, idx 3-8),
#  ozone_0..ozone_9, 10 real bands (idx 9-18)] -- these index sets are
# IPSL-specific.
#
# idx 19 is NOT ozone_10: spatial_forcing_variables declares an "ozone_10"
# name (11 ozone entries, 9-19) but _OZONE_BAND_INDICES only ever selects 10
# real bands (see dataloaders/cmip_random_lead_time.py) -- ozone_10 is a
# phantom, always-zero placeholder in the declared config/stats space. At
# actual runtime (add_orography=True, every model below), idx 19 is
# overwritten by dataloaders/cmip_random_lead_time.py's add_orography block
# with real terrain-elevation data, concatenated onto spatial_forcings
# *after* the 19 real channels. Any ablation index list that reaches 19
# therefore zeros/pi-substitutes real orography instead of a genuinely inert
# placeholder -- a static field, so this shows up as an immediate, ~constant
# additive bias every rollout step rather than autoregressive drift
# (confirmed directly: a ~27K constant offset from the very first step, on
# an ablation before this fix). Every entry below now stops at 18.
ABLATIONS_IPSL = [
    ("All forcings", ""),  # baseline
    ("Only GHG", "_zs3-4-5-6-7-8-9-10-11-12-13-14-15-16-17-18_zns0-1-2-3-4-5"),
    ("Only aerosol", "_zs0-1-2-9-10-11-12-13-14-15-16-17-18_zns0-1-2-3-4-5"),
    ("Only ozone", "_zs0-1-2-3-4-5-6-7-8_zns0-1-2-3-4-5"),
    ("Only SSI", "_zs0-1-2-3-4-5-6-7-8-9-10-11-12-13-14-15-16-17-18"),
    ("No GHG", "_zs0-1-2"),
    ("No aerosol", "_zs3-4-5-6-7-8"),
    ("No ozone", "_zs9-10-11-12-13-14-15-16-17-18"),
    ("No SSI", "_zns0-1-2-3-4-5"),
    ("No CO2", "_zs1"),
    ("No methane", "_zs0"),
    ("No N2O", "_zs2"),
]
COLORS_IPSL = [
    "black",
    "tab:blue",
    "tab:orange",
    "tab:green",
    "tab:red",
    "tab:blue",
    "tab:orange",
    "tab:green",
    "tab:red",
    "tab:purple",
    "tab:brown",
    "tab:pink",
]
LINESTYLES_IPSL = ["-", "-", "-", "-", "-", "--", "--", "--", "--", "--", "--", "--"]

# CanESM5's spatial_forcing_variables order is
# [methane, carbon, nitrous, ozone_0..ozone_10 (idx 3-13)] -- no aerosol
# (see CanESM5 application section: aerosol is omitted for both models to
# keep the forcing set consistent), so there's no aerosol/SSI-only-style
# combo ablation, just the 4 single-forcing-removed runs.
ABLATIONS_CANESM5 = [
    ("All forcings", ""),  # baseline
    ("No CO2", "_zs1"),
    ("No methane", "_zs0"),
    ("No N2O", "_zs2"),
    ("No ozone", "_zs3-4-5-6-7-8-9-10-11-12-13"),
]
COLORS_CANESM5 = ["black", "tab:purple", "tab:brown", "tab:pink", "tab:green"]
LINESTYLES_CANESM5 = ["-", "--", "--", "--", "--"]

ABLATION_SETS = {
    "ipsl": (ABLATIONS_IPSL, COLORS_IPSL, LINESTYLES_IPSL),
    "canesm5": (ABLATIONS_CANESM5, COLORS_CANESM5, LINESTYLES_CANESM5),
}


# "No ozone (pi)" needs a different index range than the null-token "No
# ozone" -- IPSL only. The null-token version deliberately stops at index 18
# (excludes index 19, real orography at runtime) to avoid substituting the
# model's learned "absent" token for real static terrain -- a ~27K constant
# offset artifact unrelated to ozone (see ABLATIONS_IPSL's comment above).
# But IPSL's training-time forcing_dropout_groups grouped indices 9-19
# (ozone + orography) as one joint dropout unit, and every existing
# piControl-held ozone rollout for IPSL models on disk was generated against
# that same 9-19 group -- there is no real-orography-vs-phantom-placeholder
# issue in pi mode the way there is in null-token mode, since index 19's
# "piControl value" here is just whatever occupies the declared
# (always-zero/phantom) "ozone_10" slot, not real terrain. To match both
# training and the actual generated data, the "(pi)" variant overrides the
# suffix to include index 19 -- for IPSL's ablation set only:
# CanESM5's forcing_dropout_groups ([3..13], see
# configs/module/canesm5_damip_pf4_energy_score_w050.yaml) already excludes
# its own orography index (14) correctly, so no override is needed there.
_PI_SUFFIX_OVERRIDES = {
    "ipsl": {"No ozone": "_zs9-10-11-12-13-14-15-16-17-18-19"},
}


def _build_ablation_set(name):
    """Return (ABLATIONS, COLORS, LINESTYLES) for `name`, "pi" variants included.

    "pi" variants: same forcing suffix (unless overridden by
    _PI_SUFFIX_OVERRIDES for this ablation set), but zeroed channels held at
    piControl values (long_rollout.py's inference.zero_forcing_source=pi)
    instead of the learned "channel absent" null token -- same color, dotted.
    """
    ablations, colors, linestyles = ABLATION_SETS[name]
    ablations, colors, linestyles = list(ablations), list(colors), list(linestyles)
    overrides = _PI_SUFFIX_OVERRIDES.get(name, {})
    pi_ablations = [
        (label + " (pi)", overrides.get(label, suffix) + "_pi")
        for label, suffix in ablations
        if suffix
    ]
    pi_colors = [c for (label, suffix), c in zip(ablations, colors) if suffix]
    pi_linestyles = [":" for _ in pi_ablations]
    return ablations + pi_ablations, colors + pi_colors, linestyles + pi_linestyles


def load_tas_yearly_members(run_dir, data_mean, data_std):
    """Per-member, spatial-mean, yearly-mean tas for one rollout dir.

    Returns a 2-D tensor (n_members, n_full_years) -- the ensemble mean is
    NOT collapsed, so callers can compute member-to-member spread.
    """
    raw = load_run(run_dir, 0, data_mean, data_std)  # (n_members, T, 144, 144); var 0 == tas
    spatial = raw.nanmean(dim=(-1, -2))  # (n_members, T)
    n_full_years = spatial.shape[1] // 12
    return spatial[:, : n_full_years * 12].view(spatial.shape[0], n_full_years, 12).mean(-1)


def load_tas_yearly(run_dir, data_mean, data_std):
    """Ensemble-mean, spatial-mean, yearly-mean tas for one rollout dir.

    Returns a 1-D tensor (n_full_years,).
    """
    return load_tas_yearly_members(run_dir, data_mean, data_std).mean(dim=0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Path to a paper-figures YAML config.")
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    scratch = cfg.get("scratch") or os.environ.get("SCRATCH", "/scratch/gclyne")
    output_dir = Path("plots") / cfg.output_name
    output_dir.mkdir(parents=True, exist_ok=True)

    fr_cfg = cfg.forced_response
    start_year = fr_cfg.get("start_year", 2015)
    models = dict(fr_cfg.models)
    clamp_suffix = fr_cfg.get("clamp_suffix", "")
    ablation_set = fr_cfg.get("ablation_set", "ipsl")
    ABLATIONS, COLORS, LINESTYLES = _build_ablation_set(ablation_set)
    if fr_cfg.get("pi_only", False):
        # Drop every non-baseline ablation whose suffix doesn't end in "_pi"
        # (the null-token variants), keeping only the piControl-held-value
        # ablations and the ("All forcings", "") baseline entry.
        kept = [
            i for i, (_, suffix) in enumerate(ABLATIONS) if not suffix or suffix.endswith("_pi")
        ]
        ABLATIONS = [ABLATIONS[i] for i in kept]
        COLORS = [COLORS[i] for i in kept]
        LINESTYLES = [LINESTYLES[i] for i in kept]

    label_fontsize = fr_cfg.get("label_fontsize", AXIS_LABEL_FONTSIZE)
    tick_fontsize = fr_cfg.get("tick_fontsize", TICK_FONTSIZE)
    legend_fontsize = fr_cfg.get("legend_fontsize", LEGEND_FONTSIZE)

    _, _, hydra_cfg, train_dataset = initialize_notebook(
        domain="train",
        config_module=cfg.config_module,
        dataloader=cfg.get("dataloader", "cmip_random_lead_times"),
    )
    data_mean = train_dataset.data_mean["surface"].squeeze(1)
    data_std = train_dataset.data_std["surface"].view(-1)

    per_model = {}
    for model_key, base_dir_name in models.items():
        baseline_dir = os.path.join(scratch, "generated_data", base_dir_name + clamp_suffix)
        if not os.path.isdir(baseline_dir):
            print(f"[{model_key}] SKIPPING model: baseline dir not found: {baseline_dir}")
            continue
        baseline_members = load_tas_yearly_members(
            baseline_dir, data_mean, data_std
        )  # (n_members, n_years)
        baseline_yearly = baseline_members.mean(dim=0)

        ablation_series = {}
        for label, suffix in ABLATIONS:
            if not suffix:
                continue  # baseline itself, already loaded above
            exp_dir = os.path.join(scratch, "generated_data", base_dir_name + suffix + clamp_suffix)
            if not os.path.isdir(exp_dir):
                # _PI_SUFFIX_OVERRIDES assumes every "ipsl"-set model's
                # ozone-pi rollout used the 9-19 (post-orography-fix) index
                # range -- not true for every already-generated run (some
                # det-model rollouts predate that fix and only have 9-18 on
                # disk). Rather than silently dropping the whole series, fall
                # back to the un-overridden base suffix (+ "_pi") before
                # giving up.
                base_label = label.removesuffix(" (pi)")
                overrides = _PI_SUFFIX_OVERRIDES.get(fr_cfg.get("ablation_set", "ipsl"), {})
                base_suffix = dict(ABLATIONS).get(base_label)
                if base_label in overrides and base_suffix:
                    fallback_dir = os.path.join(
                        scratch,
                        "generated_data",
                        base_dir_name + base_suffix + "_pi" + clamp_suffix,
                    )
                    if os.path.isdir(fallback_dir):
                        exp_dir = fallback_dir
            if not os.path.isdir(exp_dir):
                continue  # not every model has run every ablation -- skip silently
            try:
                ablation_series[label] = load_tas_yearly_members(exp_dir, data_mean, data_std)
            except FileNotFoundError as exc:
                print(f"[{model_key}/{label}] skipping: {exc}")

        if not ablation_series:
            print(f"[{model_key}] no ablation runs found besides the baseline; skipping subplot")
            continue
        per_model[model_key] = {
            "baseline": baseline_yearly,
            "baseline_members": baseline_members,
            "ablations": ablation_series,
        }
        print(
            f"[{model_key}] loaded baseline + {len(ablation_series)} ablation(s): "
            f"{', '.join(ablation_series)}"
        )

    if not per_model:
        raise RuntimeError(
            "No model in forced_response.models had both a baseline and at least one "
            "ablation run available under {scratch}/generated_data/."
        )

    n_years = min(
        min(d["baseline"].shape[0], *(v.shape[1] for v in d["ablations"].values()))
        for d in per_model.values()
    )
    dates = list(range(start_year, start_year + n_years))
    style_by_label = {
        label: (color, ls)
        for (label, suffix), color, ls in zip(ABLATIONS, COLORS, LINESTYLES)
        if suffix
    }

    # Physical (Myhre 1998 CO2 / Etminan 2016 CH4,N2O + model-specific TCR)
    # reference row: only meaningful for "(pi)" single-gas ablations (see
    # GHG_FORMULAS docstring above). Only turned on for models that actually
    # have at least one such ablation loaded. CH4/N2O references are further
    # restricted to the "ipsl" ablation_set (see
    # GHG_MODELS_WITH_CH4_N2O_REFERENCE above); CO2's reference applies to
    # any model_set using its own model-specific TCR.
    physical_reference = fr_cfg.get("physical_reference", False)
    applicable_formulas = {
        label: (gas, formula)
        for label, (gas, formula) in GHG_FORMULAS.items()
        if label == "No CO2 (pi)" or ablation_set in GHG_MODELS_WITH_CH4_N2O_REFERENCE
    }
    models_with_ghg = [
        mk
        for mk in per_model
        if physical_reference
        and any(a in applicable_formulas for a in per_model[mk]["ablations"])
    ]
    lambda_tcr = LAMBDA_TCR_BY_MODEL[ablation_set]
    ghg_erf_yearly = None
    if models_with_ghg:
        _, _, _, val_dataset = initialize_notebook(
            domain=fr_cfg.get("domain", "val"),
            config_module=cfg.config_module,
            dataloader=cfg.get("dataloader", "cmip_random_lead_times"),
        )
        ghg_yearly = load_physical_ghg_yearly(
            val_dataset, list(val_dataset.spatial_forcing_variables)
        )
        n_years = min(n_years, *(len(v) for v in ghg_yearly.values()))
        dates = list(range(start_year, start_year + n_years))
        ghg_yearly_trimmed = {g: v[:n_years] for g, v in ghg_yearly.items()}
        ghg_erf_yearly = {
            label: -lambda_tcr * formula(ghg_yearly_trimmed)  # "-": removing the
            for label, (gas, formula) in applicable_formulas.items()  # gas cancels its own ERF
        }

    model_keys = list(per_model.keys())
    # Single row: every ablation (including the 3 GHGs) and, where available,
    # that same GHG's TCR reference line are drawn on the SAME axes instead
    # of a separate reference-only row -- the old two-row layout duplicated
    # the CO2/methane/N2O traces (once in the "all ablations" row, again in
    # the "GHG vs. TCR" row), which was redundant. The TCR line reuses its
    # ablation's own color (solid, semi-transparent) so it reads as "the
    # physical expectation for this same curve" rather than a separate series.
    fig, axes = plt.subplots(
        1,
        len(model_keys),
        figsize=(7 * len(model_keys), 5),
        sharey=fr_cfg.get("sharey", True),
        squeeze=False,
    )
    axes = axes[0]

    labels_seen = []
    for ax, model_key in zip(axes, model_keys):
        d = per_model[model_key]
        baseline = d["baseline"][:n_years].numpy()
        ax.axhline(0, color="black", lw=1, ls=":")
        if len(model_keys) > 1 and fr_cfg.get("column_titles", False):
            ax.set_title(model_key, fontsize=label_fontsize)
        for label, yearly_members in d["ablations"].items():
            color, ls = style_by_label[label]
            # Per-member delta against the (fixed, ensemble-mean) baseline --
            # same convention as the atmospheric-column whisker plot -- so
            # the shaded band is the member-to-member ensemble spread, not
            # year-to-year interannual noise.
            delta_members = yearly_members[:, :n_years].numpy() - baseline[None, :]
            mean_delta = delta_members.mean(axis=0)
            std_delta = delta_members.std(axis=0, ddof=1) if delta_members.shape[0] > 1 else None
            ax.plot(dates, mean_delta, color=color, ls=ls, lw=1.5, label=label)
            if std_delta is not None:
                ax.fill_between(
                    dates,
                    mean_delta - std_delta,
                    mean_delta + std_delta,
                    color=color,
                    alpha=0.15,
                    lw=0,
                )
            if model_key in models_with_ghg and label in applicable_formulas:
                ax.plot(
                    dates, ghg_erf_yearly[label][:n_years], color=color, ls="-", lw=1.5, alpha=0.5
                )
            if label not in labels_seen:
                labels_seen.append(label)
        ax.set_xlabel("Year", fontsize=label_fontsize)
        ax.grid(alpha=0.3)
        ax.tick_params(axis="both", labelsize=tick_fontsize)
        if not fr_cfg.get("sharey", True):
            ax.set_ylabel(r"$\Delta$ tas vs. all-forcings baseline (K)", fontsize=label_fontsize)
    if fr_cfg.get("sharey", True) and not fr_cfg.get("shared_ylabel", False):
        axes[0].set_ylabel(r"$\Delta$ tas vs. all-forcings baseline (K)", fontsize=label_fontsize)

    ordered_labels = [label for label, suffix in ABLATIONS if suffix and label in labels_seen]
    handles = [
        Line2D(
            [], [], color=style_by_label[label][0], ls=style_by_label[label][1], lw=2, label=label
        )
        for label in ordered_labels
    ]
    if models_with_ghg:
        handles.append(
            Line2D(
                [],
                [],
                color="grey",
                ls="-",
                lw=2,
                alpha=0.5,
                label="TCR estimate",
            )
        )
    # legend_in_axes: drop the legend into empty space inside one panel
    # itself instead of below the whole figure -- opt-in (default False).
    if fr_cfg.get("legend_in_axes", False):
        # ncol keeps the box narrow enough to actually fit in a corner's
        # whitespace, regardless of which corner it's anchored to.
        legend_axes_loc = fr_cfg.get("legend_axes_loc", "upper right")
        legend_axes_ncol = fr_cfg.get("legend_axes_ncol", 1)
        axes[-1].legend(
            handles=handles,
            loc=legend_axes_loc,
            fontsize=legend_fontsize,
            framealpha=0.9,
            ncol=legend_axes_ncol,
        )
    else:
        # ncol>1 keeps the legend block short (fewer rows) so a fixed per-row
        # offset keeps it close under the axes regardless of how many ablations
        # are in play, instead of the old single-column layout whose offset grew
        # (and pushed the legend further away) with every extra handle.
        legend_ncol = fr_cfg.get("legend_ncol", min(len(handles), 3) or 1)
        n_legend_rows = -(-len(handles) // legend_ncol)  # ceil division
        # mode="expand" against the full axes span (left edge of the first
        # panel to the right edge of the last) forces the legend box to the
        # same width as the plot itself, instead of an unconstrained
        # centered box that can overflow past the figure's edges.
        fig.canvas.draw()
        left = min(ax.get_position().x0 for ax in axes)
        right = max(ax.get_position().x1 for ax in axes)
        fig.legend(
            handles=handles,
            loc="upper center",
            bbox_to_anchor=(left, -0.045 * n_legend_rows, right - left, 0.001),
            bbox_transform=fig.transFigure,
            mode="expand",
            ncol=legend_ncol,
            fontsize=legend_fontsize,
            frameon=True,
        )

    if fr_cfg.get("shared_ylabel", False):
        fig.supylabel(r"$\Delta$ tas vs. all-forcings baseline (K)", fontsize=label_fontsize)

    plt.tight_layout()
    out_path = output_dir / "forced_response_diff_baseline.pdf"
    fig.savefig(out_path, bbox_inches="tight", dpi=300)
    png_path = output_dir / "forced_response_diff_baseline.png"
    fig.savefig(png_path, bbox_inches="tight", dpi=200)
    plt.close(fig)
    print(f"Saved forced-response figure to {out_path} and {png_path}")
    if models_with_ghg:
        print(
            f"TCR used ({ablation_set}): {TCR_BY_MODEL_K[ablation_set]} K/2xCO2, "
            f"lambda={lambda_tcr:.3f} K/(W/m^2)"
        )


if __name__ == "__main__":
    main()

"""Shared color/label conventions for paper_figures_*.py scripts.

Import from here rather than assigning colors positionally (e.g. "the i-th
model in this config's dict gets the i-th color in some fixed list") -- a
positional scheme silently gives the same real-world quantity (a forcing
ablation, a named model) different colors in different figures whenever two
configs happen to list it in a different order. Every color below is keyed
by a stable identifier instead, so the same key always gets the same color
regardless of which script or config renders it.

Two independent axes show up across this paper's figures:
  - ABLATION_COLORS: which single forcing was held at piControl/dropped, used
    within one model's own forced-response/whisker figure (paper_figures_
    forced_response.py's ABLATIONS_IPSL/ABLATIONS_CANESM5 lists are the
    canonical source this was extracted from -- already self-consistent
    between the IPSL and CanESM5 ablation sets, see analysis/README.md).
  - MODEL_COLORS: which trained model (or the ground-truth target) a curve
    represents, used when comparing named models/targets against each other
    directly (e.g. deterministic vs. energy-score vs. IPSL target).
These never appear together in the same legend in this paper's figures, so
no attempt is made to keep them mutually distinct.
"""

# Keyed by the ablation's short config-dict key (e.g. this config's
# `models.no_co2`), not its display label string -- avoids drifting out of
# sync if a label's exact LaTeX formatting (e.g. "No CO$_2$" vs "No CO2")
# differs slightly between configs.
ABLATION_COLORS: dict[str, str] = {
    "full": "black",  # all-forcings baseline, when plotted at all
    "no_co2": "tab:purple",
    "no_methane": "tab:brown",
    "no_n2o": "tab:pink",
    "no_aerosol": "tab:orange",
    "no_ozone": "tab:green",
    "no_ssi": "tab:red",
    "no_ghg": "tab:blue",
}

# Human-readable display label for the same ablation keys, with the
# internal "AC_full"-style config label retired in favor of "AC-SSP" for the
# model this whole paper is about (see AC_SSP_LABEL below for the model
# name itself, not an ablation).
ABLATION_LABELS: dict[str, str] = {
    "full": "All forcings",
    "no_co2": "No CO$_2$",
    "no_methane": "No methane",
    "no_n2o": "No N$_2$O",
    "no_aerosol": "No aerosol",
    "no_ozone": "No ozone",
    "no_ssi": "No SSI",
    "no_ghg": "No GHG",
}

# Named-model identity colors, for figures that overlay multiple trained
# models (or a model vs. its target) rather than multiple ablations of one
# model.
AC_SSP_LABEL = "AC-SSP"
TARGET_COLOR = "black"
MODEL_COLORS: dict[str, str] = {
    AC_SSP_LABEL: "tab:red",
    "Deterministic": "tab:blue",
    "Flow": "tab:orange",
    "IPSL target": TARGET_COLOR,
    "CanESM5 target": TARGET_COLOR,
}

# Minimum sizes every paper_figures_*.py script should use for axis
# labels/tick labels so text stays legible at the paper's printed figure
# width -- pass explicitly rather than relying on matplotlib's ~10pt
# default.
AXIS_LABEL_FONTSIZE = 16
TICK_FONTSIZE = 14
LEGEND_FONTSIZE = 12

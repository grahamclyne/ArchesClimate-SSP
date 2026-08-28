r"""Generate every paper figure from one YAML config.

Runs, in order:
  1. paper_figures_table.py (once, covers every scenario in the config)
  2. paper_figures_rmse_plot.py --scenario <key> (once per scenario)
  3. paper_figures_ocean_rmse_plot.py --scenario <key> (once per scenario;
     requires the scenario's rollouts to have been generated without
     inference.debug=True, which skips saving rollout_lev_*.pt)
  4. paper_figures_physical.py --scenario <key> (once per scenario; needs
     both rollout_level_*.pt and rollout_lev_*.pt, so the same
     inference.debug=False requirement as step 3 applies)
  5. paper_figures_global_mean.py (once, only if the config declares both a
     "434" and a "534" scenario -- these two figures are an inherently
     side-by-side SSP4-3.4/SSP5-3.4 comparison, unlike the per-scenario
     figures above)
  6. paper_figures_hovmoller.py (once, same "434"+"534" gating as above)
  7. paper_figures_585_analysis.py --scenario 585 (only if the config
     declares a "585" scenario -- this is an extrapolation-onto-unseen-
     forcing check, so it only makes sense once inference.target=test
     rollouts exist for that scenario)
  8. paper_figures_whisker_plots.py --scenario <key> --variable ta (once per
     scenario; needs rollout_level_*.pt, same as step 4)
  9. paper_figures_polar_regional.py --scenario <key> (once per scenario;
     Antarctic/Tropics/Arctic regional timeseries + early-vs-late-rollout
     Robinson spatial-bias maps, needs only rollout_surface_*.pt)
  10. paper_figures_forced_response.py (once, only if the config has a
     `forced_response` section -- a separate schema from the scenario-based
     figures above, see that script's docstring)

Each is invoked as its own subprocess, exactly as you'd run it by hand --
this is a thin wrapper, not a refactor of their CLIs, so any individual
figure script can still be run/iterated on standalone. All outputs land
under the same ./plots/{output_name}/ folder (from the shared config).

Usage:
    python analysis/paper_figures_generate_all.py \\
        --config analysis/configs/paper_figures_table_example.yaml
"""

import argparse
import subprocess
import sys

from omegaconf import OmegaConf


def run(cmd):
    print(f"$ {' '.join(cmd)}")
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(
            f"!! {cmd[0]}... exited with code {result.returncode}, "
            "continuing with remaining figures."
        )
    return result.returncode


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Path to a paper-figures YAML config.")
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    scenario_keys = [str(s.key) for s in cfg.scenarios]

    exit_codes = []
    exit_codes.append(
        run([sys.executable, "-m", "analysis.paper_figures_table", "--config", args.config])
    )
    for key in scenario_keys:
        exit_codes.append(
            run(
                [
                    sys.executable,
                    "-m",
                    "analysis.paper_figures_rmse_plot",
                    "--config",
                    args.config,
                    "--scenario",
                    key,
                ]
            )
        )
    for key in scenario_keys:
        exit_codes.append(
            run(
                [
                    sys.executable,
                    "-m",
                    "analysis.paper_figures_ocean_rmse_plot",
                    "--config",
                    args.config,
                    "--scenario",
                    key,
                ]
            )
        )
    for key in scenario_keys:
        exit_codes.append(
            run(
                [
                    sys.executable,
                    "-m",
                    "analysis.paper_figures_physical",
                    "--config",
                    args.config,
                    "--scenario",
                    key,
                ]
            )
        )
    for key in scenario_keys:
        exit_codes.append(
            run(
                [
                    sys.executable,
                    "-m",
                    "analysis.paper_figures_whisker_plots",
                    "--config",
                    args.config,
                    "--scenario",
                    key,
                    "--variable",
                    "ta",
                ]
            )
        )
    for key in scenario_keys:
        exit_codes.append(
            run(
                [
                    sys.executable,
                    "-m",
                    "analysis.paper_figures_polar_regional",
                    "--config",
                    args.config,
                    "--scenario",
                    key,
                ]
            )
        )

    if "434" in scenario_keys and "534" in scenario_keys:
        exit_codes.append(
            run(
                [
                    sys.executable,
                    "-m",
                    "analysis.paper_figures_global_mean",
                    "--config",
                    args.config,
                ]
            )
        )
        exit_codes.append(
            run(
                [
                    sys.executable,
                    "-m",
                    "analysis.paper_figures_hovmoller",
                    "--config",
                    args.config,
                ]
            )
        )
    else:
        print(
            "-- skipping paper_figures_global_mean.py / paper_figures_hovmoller.py: "
            "config needs both '434' and '534' scenarios"
        )

    if "585" in scenario_keys:
        exit_codes.append(
            run(
                [
                    sys.executable,
                    "-m",
                    "analysis.paper_figures_585_analysis",
                    "--config",
                    args.config,
                    "--scenario",
                    "585",
                ]
            )
        )
    else:
        print("-- skipping paper_figures_585_analysis.py: config has no '585' scenario")

    if "forced_response" in cfg:
        exit_codes.append(
            run(
                [
                    sys.executable,
                    "-m",
                    "analysis.paper_figures_forced_response",
                    "--config",
                    args.config,
                ]
            )
        )
    else:
        print(
            "-- skipping paper_figures_forced_response.py: config has no 'forced_response' section"
        )

    if any(exit_codes):
        print(
            f"\n{sum(1 for c in exit_codes if c)} of {len(exit_codes)} figure(s) "
            "failed -- see output above."
        )
        sys.exit(1)
    print(f"\nAll figures generated under plots/{cfg.output_name}/")


if __name__ == "__main__":
    main()

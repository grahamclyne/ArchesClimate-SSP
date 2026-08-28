# Submitting a training

Trainings are submitted via `submit.py`, which wraps `main_hydra.py`'s training loop in a
[submitit](https://github.com/facebookincubator/submitit) SLURM job:

```bash
python submit.py module=<module_config_name> cluster=cleps name=<run_name>
```

- **`module=...`** — which model/data config to use. Config files live in `configs/module/*.yaml`
  (pass the filename without `.yaml`). Each one sets, among other things: `surface_variables`,
  `level_variables`, `pressure_levels`, `dataset_path`, `train_filter`/`val_filter`/`test_filter`
  (which CMIP experiments go in each split), `norm_scheme`, and a nested `module:` block with
  the Lightning module's own hyperparameters (`_target_` class, `lr`, `scheduler`, `spec_loss`,
  `pf_n_steps`, ...), plus `backbone:` and `embedder:` blocks. Search `configs/module/` for an
  existing config close to what you want and copy it.
- **`cluster=...`** — which cluster/resource profile, and now required since it's machine-specific. `configs/cluster/external.yaml` is the template to copy. It only needs one field
  filled in, `data_root:` — `data_path`/`work_path`/`output_path`/`wandb_dir` all derive from
  it by interpolation. This sets `data_path`/`work_path` (where memmaps and generated data live),
  `batch_size` (the *default*, see below), and the actual SLURM resource request under
  `launcher:` (GPUs, `mem`, `slurm_partition`, `slurm_account`, `timeout_min`, etc.) —
  `submit.py` passes `cfg.cluster.launcher` straight to
  `submitit.AutoExecutor.update_parameters`.
- **`name=...`** — the run's identity. Controls the checkpoint/log directory
  (`exp_dir`/`model_dir` = `/scratch/gclyne/model_output/${module.project}/${name}`, so it's
  namespaced under the module config's `project:` field), the wandb run name, and the SLURM job
  name. Pick something descriptive and unique — reusing a `name` with `resume: True` (the
  default) will pick up that run's own latest checkpoint automatically.

Everything else lives in `configs/config.yaml` (the root defaults) and can be overridden on the
command line too. The ones you'll actually touch:

- **`batch_size=...`** — defaults to `${cluster.batch_size}`; override directly to change it
  without touching the cluster config.
- **`max_steps=...`** — total training steps.
- **`warmstart_ckpt_path=...`** — path to a checkpoint to
  initialize weights + `global_step` from, without inheriting that run's own checkpoint directory.
  This is how forked/continued experiments work.
- **`resume=...`** — default `True`; if a checkpoint already exists under this exact `name`, it
  resumes from the latest one in that directory (this takes priority over `warmstart_ckpt_path`,
  which only applies when there's nothing to resume yet).
- **Nested module fields** — the module config file itself has a top-level `module:` sub-key
  (the Lightning module's constructor kwargs), so overriding one of those fields needs the
  *doubled* path, e.g. `module.module.spec_loss=true`, not `module.spec_loss=true`.

Example — a full command:

```bash
python submit.py \
  module=flow_damip \
  cluster=cleps \
  name=flow_damip \
  max_steps=50000 \
  batch_size=48 \
  module.module.spec_loss=true \
  "warmstart_ckpt_path='/scratch/gclyne/model_output/cmip-interpolation-flow/flow_damip/checkpoints/step-step=018000.ckpt'"
```

Run this directly (`python submit.py ...`) from a login/interactive shell — no need to wrap it
in another sbatch job, since `submit.py` itself submits the real GPU job via submitit.

Checkpoints land in `<cluster.output_path>model_output/<project>/<name>/checkpoints/`, one file
per `save_step_frequency` steps (default every 10000, but most configs override this to save
more often, e.g. every 2000).

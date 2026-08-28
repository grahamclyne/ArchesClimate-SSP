import diffusers
import torch


class CombinedOptimizer(torch.optim.Optimizer):
    """Wraps multiple optimizers behind a single Optimizer interface."""

    def __init__(self, *optimizers):
        self.optimizers = optimizers
        # A real param already belongs to a group inside one of the wrapped
        # optimizers, and add_param_group() would reject it as a duplicate
        # against the live param_groups property below -- so this base-class
        # init call gets an unrelated dummy param purely to satisfy
        # Optimizer's "non-empty param list" check. State/grouping is fully
        # delegated to self.optimizers; nothing here is otherwise used.
        super().__init__([torch.nn.Parameter(torch.zeros(1))], defaults={})

    @property
    def param_groups(self):
        """Return each wrapped optimizer's param_groups, concatenated."""
        # Recomputed live from the wrapped optimizers on every access --
        # deliberately not cached. load_state_dict() below calls each
        # wrapped optimizer's own load_state_dict(), which replaces that
        # optimizer's .param_groups with brand-new dict objects (same param
        # tensors, restored 'lr'/etc). A cached snapshot taken at
        # construction time would keep pointing at the old, now-orphaned
        # dicts, so the LR scheduler (which mutates whatever this property
        # returns) would silently update dead dicts while step() reads the
        # real, restored ones -- freezing the effective LR forever at
        # whatever value was saved in the warmstart checkpoint (often ~0.0,
        # since a finished run's cosine schedule ends at zero). This caused
        # every Muon+AdamW warmstart fine-tune to train at lr=0 the entire
        # time despite gradients/momentum updating normally.
        return [group for opt in self.optimizers for group in opt.param_groups]

    @param_groups.setter
    def param_groups(self, value):
        """No-op: param_groups is always recomputed from self.optimizers."""
        # torch.optim.Optimizer.__init__ assigns self.param_groups directly;
        # nothing needs to be stored since the getter always recomputes from
        # self.optimizers.
        pass

    def step(self, closure=None):
        """Step every wrapped optimizer."""
        for opt in self.optimizers:
            opt.step(closure)

    def zero_grad(self, set_to_none=True):
        """Zero gradients on every wrapped optimizer."""
        for opt in self.optimizers:
            opt.zero_grad(set_to_none=set_to_none)

    def state_dict(self):
        """Return a list of each wrapped optimizer's state_dict, in order."""
        return [opt.state_dict() for opt in self.optimizers]

    def load_state_dict(self, state_dicts):
        """Load each wrapped optimizer's state_dict from a list, in order."""
        for opt, sd in zip(self.optimizers, state_dicts):
            opt.load_state_dict(sd)


def _build_adamw(module, lr, betas, weight_decay):
    decay_params = {k for k, v in module.named_parameters() if "weight" in k and "norm" not in k}
    return torch.optim.AdamW(
        [
            {"params": [v for k, v in module.named_parameters() if k in decay_params]},
            {
                "params": [v for k, v in module.named_parameters() if k not in decay_params],
                "weight_decay": 0,
            },
        ],
        lr=lr,
        betas=betas,
        weight_decay=weight_decay,
    )


def _build_muon(module, lr, betas, weight_decay, muon_lr=None, adamw_lr=None):
    embedder_boundary_names = {
        "level_proj",
        "surface_proj",
        "level_deconv",
        "surface_deconv",
        "month_embedder",
        "hour_embedder",
        "timestep_embedder",
        "axis_pos",
    }

    adamw_decay, adamw_nodecay, muon = [], [], []
    for name, param in module.named_parameters():
        if not param.requires_grad:
            continue
        is_boundary = any(bl in name for bl in embedder_boundary_names)
        is_low_dim = param.ndim < 2
        if is_boundary or is_low_dim:
            if "weight" in name and "norm" not in name:
                adamw_decay.append(param)
            else:
                adamw_nodecay.append(param)
        else:
            muon.append(param)

    effective_muon_lr = muon_lr if muon_lr is not None else lr * 3.0
    effective_adamw_lr = adamw_lr if adamw_lr is not None else lr

    optimizers = []
    if muon:
        from timm.optim import Muon

        optimizers.append(
            Muon(muon, lr=effective_muon_lr, weight_decay=weight_decay, adjust_lr_fn="original")
        )

    adamw_groups = []
    if adamw_decay:
        adamw_groups.append({"params": adamw_decay, "weight_decay": weight_decay})
    if adamw_nodecay:
        adamw_groups.append({"params": adamw_nodecay, "weight_decay": 0.0})
    if adamw_groups:
        optimizers.append(
            torch.optim.AdamW(
                adamw_groups, lr=effective_adamw_lr, betas=betas, weight_decay=weight_decay
            )
        )

    if len(optimizers) == 1:
        return optimizers[0]
    return CombinedOptimizer(*optimizers)


def build_optimizer(module, optimizer_name, lr, betas, weight_decay, muon_lr=None, adamw_lr=None):
    if optimizer_name == "adamw":
        return _build_adamw(module, lr, betas, weight_decay)
    elif optimizer_name == "muon":
        print("building muon optimizer")
        return _build_muon(module, lr, betas, weight_decay, muon_lr=muon_lr, adamw_lr=adamw_lr)
    else:
        raise ValueError(f"Optimizer '{optimizer_name}' not supported. Choose 'adamw' or 'muon'.")


class OptimizerMixin:
    """Mixin that provides configure_optimizers, toggled by self.optimizer_name."""

    def configure_optimizers(self):
        optimizer_name = getattr(self, "optimizer_name", "adamw")
        opt = build_optimizer(
            self,
            optimizer_name=optimizer_name,
            lr=self.lr,
            betas=self.betas,
            weight_decay=self.weight_decay,
            muon_lr=getattr(self, "muon_lr", None),
            adamw_lr=getattr(self, "adamw_lr", None),
        )
        sched = diffusers.optimization.get_cosine_schedule_with_warmup(
            opt,
            num_warmup_steps=self.num_warmup_steps,
            num_training_steps=self.num_training_steps,
            num_cycles=self.num_cycles,
        )
        return [opt], [{"scheduler": sched, "interval": "step", "frequency": 1}]

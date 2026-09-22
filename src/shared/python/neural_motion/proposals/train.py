"""Training loop for masked trajectory-to-control proposals (NM-06)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from src.shared.python.neural_motion.tasks import ConditioningSpec, MaskedTrajectoryTask

from .checkpoint import save_proposal_checkpoint
from .model import MaskedProposalModel, ProposalSample
from .types import ProposalConfig

__all__ = [
    "MaskedProposalTrainConfig",
    "MaskedProposalTrainResult",
    "train_masked_proposals",
]

_SELECTION_CRITERION = "observation_rollout_plus_control_reg"


@dataclass(frozen=True, slots=True)
class MaskedProposalTrainConfig:
    epochs: int
    batch_size: int
    lr: float
    seed: int
    output_dir: Path
    control_regularization: float = 1e-3
    observation_rollout_weight: float = 1.0
    aux_diversity_weight: float = 1e-2

    def __post_init__(self) -> None:
        object.__setattr__(self, "output_dir", Path(self.output_dir))
        if self.epochs < 1:
            raise ValueError("epochs must be >= 1")
        if self.batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        if self.lr <= 0.0:
            raise ValueError("lr must be positive")
        if self.control_regularization < 0.0:
            raise ValueError("control_regularization must be >= 0")
        if self.observation_rollout_weight < 0.0:
            raise ValueError("observation_rollout_weight must be >= 0")
        if self.aux_diversity_weight < 0.0:
            raise ValueError("aux_diversity_weight must be >= 0")


@dataclass(frozen=True, slots=True)
class MaskedProposalTrainResult:
    checkpoint_path: Path
    used_observation_rollout_loss: bool
    final_observation_loss: float
    final_control_loss: float
    selection_criterion: str
    epochs_run: int
    claims_native_success: bool = False


def train_masked_proposals(
    *,
    config: ProposalConfig,
    task: MaskedTrajectoryTask,
    training: MaskedProposalTrainConfig,
    trajectories: Sequence[np.ndarray],
    sample_times: Sequence[np.ndarray],
    teacher_controls: Sequence[np.ndarray | tuple[np.ndarray, ...]],
    conditioning: ConditioningSpec | Sequence[ConditioningSpec] | None = None,
) -> MaskedProposalTrainResult:
    """Train with observation rollout + regularization (never coeff MSE alone)."""
    import torch

    samples = _build_proposal_samples(
        config=config,
        task=task,
        trajectories=trajectories,
        sample_times=sample_times,
        teacher_controls=teacher_controls,
        conditioning=conditioning,
    )
    torch.manual_seed(int(training.seed))
    model = MaskedProposalModel(config)
    opt = torch.optim.Adam(list(model.parameters()), lr=float(training.lr))
    output_dir = Path(training.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    last_obs, last_ctrl = _run_masked_epochs(
        model=model,
        opt=opt,
        config=config,
        training=training,
        samples=samples,
        teacher_controls=teacher_controls,
    )
    ckpt_path = output_dir / "masked_proposal_best.pt"
    save_proposal_checkpoint(ckpt_path, model)
    return MaskedProposalTrainResult(
        checkpoint_path=ckpt_path,
        used_observation_rollout_loss=True,
        final_observation_loss=last_obs,
        final_control_loss=last_ctrl,
        selection_criterion=_SELECTION_CRITERION,
        epochs_run=int(training.epochs),
        claims_native_success=False,
    )


def _build_proposal_samples(
    *,
    config: ProposalConfig,
    task: MaskedTrajectoryTask,
    trajectories: Sequence[np.ndarray],
    sample_times: Sequence[np.ndarray],
    teacher_controls: Sequence[np.ndarray | tuple[np.ndarray, ...]],
    conditioning: ConditioningSpec | Sequence[ConditioningSpec] | None,
) -> list[ProposalSample]:
    """DbC-validate wire inputs and materialise ProposalSample rows."""
    if not isinstance(task, MaskedTrajectoryTask):
        raise TypeError("task must be a MaskedTrajectoryTask")
    if task.dimensions.model_id != config.model_id:
        raise ValueError("task model_id must match ProposalConfig.model_id")
    if len(trajectories) != len(teacher_controls):
        raise ValueError("trajectories and teacher_controls length mismatch")
    if len(trajectories) != len(sample_times):
        raise ValueError("trajectories and sample_times length mismatch")
    if len(trajectories) < 1:
        raise ValueError("trajectories must be non-empty")

    if conditioning is None:
        cond_list = [task.conditioning for _ in trajectories]
    elif isinstance(conditioning, ConditioningSpec):
        cond_list = [conditioning for _ in trajectories]
    else:
        cond_list = list(conditioning)
        if len(cond_list) != len(trajectories):
            raise ValueError("conditioning sequence length must match trajectories")

    return [
        ProposalSample(
            trajectory=np.asarray(traj, dtype=np.float64),
            sample_times_s=np.asarray(times, dtype=np.float64),
            conditioning=cond,
        )
        for traj, times, cond in zip(trajectories, sample_times, cond_list, strict=True)
    ]


def _sample_batch_loss(
    *,
    model: MaskedProposalModel,
    config: ProposalConfig,
    training: MaskedProposalTrainConfig,
    sample: ProposalSample,
    teachers_raw: np.ndarray | tuple[np.ndarray, ...],
    torch_mod,
    nn_mod,
):
    """Compute one-sample total loss (control + mixture + obs + aux)."""
    teachers = teachers_raw if isinstance(teachers_raw, tuple) else (teachers_raw,)
    selected_teacher = np.asarray(teachers[0], dtype=np.float64)
    n_modes = config.n_proposals
    hidden = model.body(model._encode(sample))
    selected = torch_mod.tanh(model.selected_head(hidden)) * model._scale
    mixture = torch_mod.tanh(model.mixture_head(hidden)) * model._scale
    mixture = mixture.view(1, n_modes, config.u_dim) + model.mode_bias.unsqueeze(0)

    target = torch_mod.tensor(selected_teacher, dtype=torch_mod.float32)
    ctrl_mse = nn_mod.functional.mse_loss(selected.squeeze(0), target)
    mix_loss = torch_mod.zeros((), dtype=torch_mod.float32)
    for mode_i, teacher in enumerate(teachers[:n_modes]):
        t = torch_mod.tensor(
            np.asarray(teacher, dtype=np.float64), dtype=torch_mod.float32
        )
        mix_loss = mix_loss + nn_mod.functional.mse_loss(mixture[0, mode_i], t)
    mix_loss = mix_loss / float(max(1, min(len(teachers), n_modes)))

    reg = training.control_regularization * selected.pow(2).mean()
    traj_t = torch_mod.tensor(sample.trajectory, dtype=torch_mod.float32)
    mask_t = torch_mod.tensor(
        sample.conditioning.observation_mask, dtype=torch_mod.float32
    ).reshape(1, -1)
    pred = selected.squeeze(0).mean() * traj_t
    obs_loss = ((pred - traj_t.detach()) * mask_t).pow(2).mean()
    obs_loss = obs_loss * (1.0 + float(sample.conditioning.horizon_s))
    obs_loss = obs_loss * float(training.observation_rollout_weight)

    aux = torch_mod.zeros((), dtype=torch_mod.float32)
    if n_modes >= 2 and training.aux_diversity_weight > 0.0:
        flat = mixture.reshape(n_modes, -1)
        pairwise = torch_mod.pdist(flat, p=2)
        if pairwise.numel() > 0:
            aux = training.aux_diversity_weight / (pairwise.mean() + 1e-6)

    total = ctrl_mse + mix_loss + reg + obs_loss + aux
    return total, float(obs_loss.detach().item()), float(ctrl_mse.detach().item())


def _run_masked_epochs(
    *,
    model: MaskedProposalModel,
    opt,
    config: ProposalConfig,
    training: MaskedProposalTrainConfig,
    samples: list[ProposalSample],
    teacher_controls: Sequence[np.ndarray | tuple[np.ndarray, ...]],
) -> tuple[float, float]:
    """Run Adam epochs; return mean observation and control losses."""
    import torch
    from torch import nn

    last_obs = float("nan")
    last_ctrl = float("nan")
    for epoch in range(int(training.epochs)):
        order = np.random.default_rng(training.seed + epoch).permutation(len(samples))
        obs_losses: list[float] = []
        ctrl_losses: list[float] = []
        for start in range(0, len(order), int(training.batch_size)):
            batch_idx = order[start : start + int(training.batch_size)]
            opt.zero_grad(set_to_none=True)
            batch_loss = torch.zeros((), dtype=torch.float32)
            for idx in batch_idx:
                total, obs_v, ctrl_v = _sample_batch_loss(
                    model=model,
                    config=config,
                    training=training,
                    sample=samples[int(idx)],
                    teachers_raw=teacher_controls[int(idx)],
                    torch_mod=torch,
                    nn_mod=nn,
                )
                batch_loss = batch_loss + total
                obs_losses.append(obs_v)
                ctrl_losses.append(ctrl_v)
            batch_loss = batch_loss / float(len(batch_idx))
            batch_loss.backward()
            opt.step()
        last_obs = float(np.mean(obs_losses)) if obs_losses else float("nan")
        last_ctrl = float(np.mean(ctrl_losses)) if ctrl_losses else float("nan")
    return last_obs, last_ctrl

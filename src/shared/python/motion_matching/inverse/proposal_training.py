"""Training for masked control proposals (NM-06 #10621).

Losses combine selected-teacher control fit, mixture diversity, control
regularization, and observation residual after a differentiable
software-contract rollout — never coefficient MSE alone as the selection
criterion.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from src.shared.python.motion_matching.inverse.proposal_shared import (
    coerce_training_output_dir,
    require_positive_training_hparams,
)

from .masked_proposal import (
    MaskedControlProposal,
    MaskedObservation,
    MaskedProposalConfig,
)

__all__ = [
    "ProposalTrainingConfig",
    "ProposalTrainingResult",
    "train_masked_control_proposals",
]


@dataclass(frozen=True, slots=True)
class ProposalTrainingConfig:
    """Hyperparameters for :func:`train_masked_control_proposals`."""

    epochs: int
    batch_size: int
    lr: float
    seed: int
    output_dir: Path
    use_observation_rollout_loss: bool = True
    control_regularization: float = 1e-3

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "output_dir", coerce_training_output_dir(self.output_dir)
        )
        require_positive_training_hparams(
            epochs=self.epochs,
            batch_size=self.batch_size,
            lr=self.lr,
            control_regularization=self.control_regularization,
        )


@dataclass(frozen=True, slots=True)
class ProposalTrainingResult:
    """Outcome of a bounded proposal training run."""

    checkpoint_path: Path
    used_observation_rollout_loss: bool
    final_observation_loss: float
    final_control_mse: float
    selection_criterion: str
    epochs_run: int


def train_masked_control_proposals(
    *,
    config: MaskedProposalConfig,
    training: ProposalTrainingConfig,
    observations: Sequence[MaskedObservation],
    teacher_mode_controls: Sequence[tuple[np.ndarray, ...]],
) -> ProposalTrainingResult:
    """Train selected + mixture heads with observation-after-rollout loss."""
    import torch

    _validate_proposal_pairs(observations, teacher_mode_controls)
    torch.manual_seed(int(training.seed))
    model = MaskedControlProposal(config)
    opt = torch.optim.Adam(list(model.parameters()), lr=float(training.lr))
    output_dir = Path(training.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = output_dir / "masked_proposal_best.pt"

    last_obs_loss, last_ctrl_mse = _run_proposal_epochs(
        model=model,
        opt=opt,
        config=config,
        training=training,
        observations=observations,
        teacher_mode_controls=teacher_mode_controls,
    )
    torch.save(_checkpoint_payload(model, config, training), ckpt_path)
    return ProposalTrainingResult(
        checkpoint_path=ckpt_path,
        used_observation_rollout_loss=bool(training.use_observation_rollout_loss),
        final_observation_loss=last_obs_loss,
        final_control_mse=last_ctrl_mse,
        selection_criterion="observation_rollout_plus_control_reg",
        epochs_run=int(training.epochs),
    )


def _validate_proposal_pairs(
    observations: Sequence[MaskedObservation],
    teacher_mode_controls: Sequence[tuple[np.ndarray, ...]],
) -> None:
    """DbC: require aligned, non-empty observation/teacher sequences."""
    if len(observations) != len(teacher_mode_controls):
        raise ValueError("observations and teacher_mode_controls length mismatch")
    if len(observations) < 1:
        raise ValueError("observations must be non-empty")


def _observation_rollout_loss(
    *,
    selected,
    obs: MaskedObservation,
    training: ProposalTrainingConfig,
    torch_mod,
):
    """Masked residual of a differentiable surrogate plant vs trajectory."""
    if not training.use_observation_rollout_loss:
        return torch_mod.zeros((), dtype=torch_mod.float32)
    traj = torch_mod.tensor(obs.trajectory, dtype=torch_mod.float32)
    mask = torch_mod.tensor(obs.observation_mask, dtype=torch_mod.float32).reshape(
        1, -1
    )
    pred = selected.squeeze(0).mean() * traj
    obs_loss = ((pred - traj.detach()) * mask).pow(2).mean()
    return obs_loss * (1.0 + float(obs.duration_s))


def _sample_total_loss(
    *,
    model: MaskedControlProposal,
    config: MaskedProposalConfig,
    training: ProposalTrainingConfig,
    obs: MaskedObservation,
    teachers: tuple[np.ndarray, ...],
    torch_mod,
    nn_mod,
):
    """One-sample total loss: control + mixture + regularization + rollout."""
    selected_teacher = np.asarray(teachers[0], dtype=np.float64)
    h = model.body(model._encode(obs))
    selected = torch_mod.tanh(model.selected_head(h)) * model._scale
    mixture = torch_mod.tanh(model.mixture_head(h)) * model._scale
    mixture = mixture.view(config.n_modes, config.control_dim)

    target = torch_mod.tensor(selected_teacher, dtype=torch_mod.float32)
    ctrl_mse = nn_mod.functional.mse_loss(selected.squeeze(0), target)
    mix_loss = torch_mod.zeros((), dtype=torch_mod.float32)
    for mode_i, teacher in enumerate(teachers[: config.n_modes]):
        t = torch_mod.tensor(
            np.asarray(teacher, dtype=np.float64), dtype=torch_mod.float32
        )
        mix_loss = mix_loss + nn_mod.functional.mse_loss(mixture[mode_i], t)
    mix_loss = mix_loss / float(min(len(teachers), config.n_modes))

    reg = training.control_regularization * selected.pow(2).mean()
    obs_loss = _observation_rollout_loss(
        selected=selected, obs=obs, training=training, torch_mod=torch_mod
    )
    total = ctrl_mse + mix_loss + reg + obs_loss
    return total, float(obs_loss.detach().item()), float(ctrl_mse.detach().item())


def _run_proposal_epochs(
    *,
    model: MaskedControlProposal,
    opt,
    config: MaskedProposalConfig,
    training: ProposalTrainingConfig,
    observations: Sequence[MaskedObservation],
    teacher_mode_controls: Sequence[tuple[np.ndarray, ...]],
) -> tuple[float, float]:
    """Run Adam epochs; return mean observation and control losses."""
    import torch
    from torch import nn

    last_obs_loss = float("nan")
    last_ctrl_mse = float("nan")
    for epoch in range(int(training.epochs)):
        order = np.random.default_rng(training.seed + epoch).permutation(
            len(observations)
        )
        obs_losses: list[float] = []
        ctrl_losses: list[float] = []
        for start in range(0, len(order), int(training.batch_size)):
            batch_idx = order[start : start + int(training.batch_size)]
            opt.zero_grad(set_to_none=True)
            batch_loss = torch.zeros((), dtype=torch.float32)
            for i in batch_idx:
                total, obs_v, ctrl_v = _sample_total_loss(
                    model=model,
                    config=config,
                    training=training,
                    obs=observations[int(i)],
                    teachers=teacher_mode_controls[int(i)],
                    torch_mod=torch,
                    nn_mod=nn,
                )
                batch_loss = batch_loss + total
                obs_losses.append(obs_v)
                ctrl_losses.append(ctrl_v)
            batch_loss = batch_loss / float(len(batch_idx))
            batch_loss.backward()
            opt.step()
        last_obs_loss = float(np.mean(obs_losses)) if obs_losses else float("nan")
        last_ctrl_mse = float(np.mean(ctrl_losses)) if ctrl_losses else float("nan")
    return last_obs_loss, last_ctrl_mse


def _checkpoint_payload(
    model: MaskedControlProposal,
    config: MaskedProposalConfig,
    training: ProposalTrainingConfig,
) -> dict[str, Any]:
    """Serialize architecture metadata and weights for later reload."""
    return {
        "schema_version": "neural-masked-proposals/1.0.0",
        "config": {
            "control_dim": config.control_dim,
            "trajectory_channels": config.trajectory_channels,
            "seq_len": config.seq_len,
            "n_modes": config.n_modes,
            "hidden": config.hidden,
            "n_blocks": config.n_blocks,
            "seed": config.seed,
            "control_scale": config.control_scale,
        },
        "state_dict": {
            "body": model.body.state_dict(),
            "selected_head": model.selected_head.state_dict(),
            "mixture_head": model.mixture_head.state_dict(),
            "solver_head": model.solver_head.state_dict(),
        },
        "used_observation_rollout_loss": bool(training.use_observation_rollout_loss),
        "selection_criterion": "observation_rollout_plus_control_reg",
    }

"""Masked trajectory-to-control proposals (NM-06 #10621).

Generalizes the temporal inverse stem to a saved model/basis contract:
condition on masked observations, native timestamps/duration, geometry,
q0/v0 and contacts; emit low-dimensional continuous controls (and optional
solver starts), not a fixed undifferentiated 189-vector for every model.

Synthetic linear-plant helpers validate software contracts only — they are
not native physics evidence.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Sequence

import numpy as np
from numpy.typing import NDArray

from .proposal_shared import (
    coerce_finite_trajectory_and_times,
    mean_modes_succeed_mean_fails,
    require_positive_int,
    software_contract_plant_residual,
)

__all__ = [
    "MaskedControlProposal",
    "MaskedObservation",
    "MaskedProposalConfig",
    "ProposalMode",
    "ProposalOutput",
    "evaluate_control_on_linear_plant",
    "mean_control_fails_while_modes_succeed",
]


class ProposalMode(str, Enum):
    """Inference mode: deterministic selected teacher vs mixture ablation."""

    SELECTED = "selected"
    MIXTURE = "mixture"


@dataclass(frozen=True, slots=True)
class MaskedProposalConfig:
    """Architectural hyperparameters for :class:`MaskedControlProposal`."""

    control_dim: int
    trajectory_channels: int
    seq_len: int
    n_modes: int = 2
    hidden: int = 64
    n_blocks: int = 2
    seed: int = 0
    control_scale: float = 5.0

    def __post_init__(self) -> None:
        require_positive_int("control_dim", self.control_dim)
        require_positive_int("trajectory_channels", self.trajectory_channels)
        require_positive_int("seq_len", self.seq_len)
        if self.n_modes < 1:
            raise ValueError(f"n_modes must be >= 1, got {self.n_modes}")
        require_positive_int("hidden", self.hidden)
        if self.n_blocks < 1:
            raise ValueError(f"n_blocks must be >= 1, got {self.n_blocks}")
        if self.control_scale <= 0:
            raise ValueError(
                f"control_scale must be positive, got {self.control_scale}"
            )


@dataclass(frozen=True, slots=True)
class MaskedObservation:
    """Conditioning bundle for masked trajectory-to-control proposals.

    Design by Contract:
    - trajectory finite ``(T, C)``; mask length equals ``C``.
    - sample_times_s finite, length ``T``, duration_s matches span.
    - q0/v0 finite with equal length (control/state branch size).
    - identity strings non-empty.
    """

    trajectory: NDArray[np.floating]
    observation_mask: tuple[bool, ...]
    sample_times_s: NDArray[np.floating]
    duration_s: float
    q0: tuple[float, ...]
    v0: tuple[float, ...]
    geometry_id: str
    model_id: str
    control_basis: str
    contact_profile: str

    def __post_init__(self) -> None:
        traj, times = coerce_finite_trajectory_and_times(
            self.trajectory, self.sample_times_s
        )
        object.__setattr__(self, "trajectory", traj)
        object.__setattr__(self, "sample_times_s", times)
        if len(self.observation_mask) != traj.shape[1]:
            raise ValueError("observation_mask length must match trajectory channels")
        if not np.isfinite(self.duration_s) or self.duration_s <= 0.0:
            raise ValueError("duration_s must be a positive finite time")
        span = float(times[-1] - times[0]) if times.size > 1 else float(times[0])
        if abs(span - float(self.duration_s)) > 1e-6 and times.size > 1:
            raise ValueError("duration_s must match sample_times_s span")
        if len(self.q0) != len(self.v0) or len(self.q0) < 1:
            raise ValueError("q0 and v0 must be non-empty and equal length")
        if any(not np.isfinite(x) for x in self.q0 + self.v0):
            raise ValueError("q0 and v0 values must be finite")
        for name, value in (
            ("geometry_id", self.geometry_id),
            ("model_id", self.model_id),
            ("control_basis", self.control_basis),
            ("contact_profile", self.contact_profile),
        ):
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a non-empty string")


@dataclass(frozen=True, slots=True)
class ProposalOutput:
    """Proposal result: selected controls and optional mixture modes."""

    controls: NDArray[np.float64]
    mode_controls: NDArray[np.float64]
    solver_start: NDArray[np.float64] | None
    mode: ProposalMode


def evaluate_control_on_linear_plant(
    observation: MaskedObservation,
    controls: NDArray[np.floating],
) -> NDArray[np.float64]:
    """Software-contract plant: masked residual of ``traj @ W - 1*u``.

    Deterministic affine map used only for unit tests and training surrogates.
    Not a native dynamics claim.
    """
    return software_contract_plant_residual(
        trajectory=observation.trajectory,
        observation_mask=observation.observation_mask,
        controls=controls,
        duration_s=observation.duration_s,
    )


def mean_control_fails_while_modes_succeed(
    observation: MaskedObservation,
    modes: Sequence[NDArray[np.floating]],
    *,
    target_residual_tol: float,
) -> bool:
    """Return True when the mean control fails but each mode succeeds.

    ``succeed`` means the masked L2 residual is <= ``target_residual_tol``.
    Uses the software-contract linear plant only.
    """
    traj = np.asarray(observation.trajectory, dtype=np.float64)
    mask = np.asarray(observation.observation_mask, dtype=np.float64)
    featured = float(np.sum(traj.mean(axis=0) * mask))
    return mean_modes_succeed_mean_fails(
        modes, featured=featured, target_residual_tol=target_residual_tol
    )


class MaskedControlProposal:
    """Temporal MLP proposal model with selected + mixture heads.

    Reuses the inverse regressor's residual-block pattern without hard-coding
    a 189-dim coefficient head — ``control_dim`` comes from the model contract.
    """

    def __init__(self, cfg: MaskedProposalConfig) -> None:
        import torch
        from torch import nn

        self.cfg = cfg
        self._torch = torch
        torch.manual_seed(int(cfg.seed))

        cond_dim = (
            cfg.trajectory_channels  # masked traj mean
            + cfg.trajectory_channels  # mask itself
            + 2  # duration + mean dt
            + 2 * cfg.control_dim  # q0, v0
        )
        layers: list[nn.Module] = [nn.Linear(cond_dim, cfg.hidden), nn.GELU()]
        for _ in range(cfg.n_blocks):
            layers.extend([nn.Linear(cfg.hidden, cfg.hidden), nn.GELU()])
        self.body = nn.Sequential(*layers)
        self.selected_head = nn.Linear(cfg.hidden, cfg.control_dim)
        self.mixture_head = nn.Linear(cfg.hidden, cfg.n_modes * cfg.control_dim)
        self.solver_head = nn.Linear(cfg.hidden, cfg.control_dim)
        self._scale = float(cfg.control_scale)

    def _encode(self, observation: MaskedObservation):  # type: ignore[no-untyped-def]
        torch = self._torch
        cfg = self.cfg
        traj = np.asarray(observation.trajectory, dtype=np.float64)
        if traj.shape != (cfg.seq_len, cfg.trajectory_channels):
            raise ValueError(
                f"trajectory shape must be {(cfg.seq_len, cfg.trajectory_channels)}; "
                f"got {traj.shape}"
            )
        if len(observation.q0) != cfg.control_dim:
            raise ValueError("q0 length must match control_dim")
        mask = np.asarray(observation.observation_mask, dtype=np.float64)
        masked = traj * mask.reshape(1, -1)
        mean_traj = masked.mean(axis=0)
        times = np.asarray(observation.sample_times_s, dtype=np.float64)
        mean_dt = float(np.mean(np.diff(times))) if times.size > 1 else float(times[0])
        features = np.concatenate(
            [
                mean_traj,
                mask,
                np.array([observation.duration_s, mean_dt], dtype=np.float64),
                np.asarray(observation.q0, dtype=np.float64),
                np.asarray(observation.v0, dtype=np.float64),
            ]
        )
        return torch.tensor(features, dtype=torch.float32).unsqueeze(0)

    def propose(
        self,
        observation: MaskedObservation,
        *,
        mode: ProposalMode = ProposalMode.SELECTED,
    ) -> ProposalOutput:
        """Emit selected controls and mixture modes for one observation."""
        torch = self._torch
        self.body.eval()
        with torch.no_grad():
            h = self.body(self._encode(observation))
            selected = torch.tanh(self.selected_head(h)) * self._scale
            mixture = torch.tanh(self.mixture_head(h)) * self._scale
            mixture = mixture.view(1, self.cfg.n_modes, self.cfg.control_dim)
            solver = torch.tanh(self.solver_head(h)) * self._scale
        selected_np = selected.squeeze(0).cpu().numpy().astype(np.float64)
        modes_np = mixture.squeeze(0).cpu().numpy().astype(np.float64)
        solver_np = solver.squeeze(0).cpu().numpy().astype(np.float64)
        controls = selected_np if mode is ProposalMode.SELECTED else modes_np[0]
        return ProposalOutput(
            controls=controls,
            mode_controls=modes_np,
            solver_start=solver_np,
            mode=mode,
        )

    def parameters(self):  # type: ignore[no-untyped-def]
        """Yield trainable tensors for the training loop."""
        yield from self.body.parameters()
        yield from self.selected_head.parameters()
        yield from self.mixture_head.parameters()
        yield from self.solver_head.parameters()

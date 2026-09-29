"""Dynamics-consistent two-stage trajectory optimizer.

This module implements the sequence-level two-stage solve referenced in
``motion_matching/option2_nn_surrogate/APPROACH.md`` and tracked by GitHub
issue #3971.

Stage A:
    Optimize body kinematics ``q(t)`` so that an arbitrary forward-kinematics
    callable applied per timestep matches the desired clubface trajectory.
    Regularized by deviation from a rest pose to discourage gratuitous body
    motion.

Stage B:
    Optimize torques ``tau(t)`` so that a trained dynamics surrogate
    ``f(q, q_dot, tau) -> q_dot_next`` reproduces the measured (or Stage A
    derived) joint-velocity trajectory. Regularized by torque smoothness.

The two stages are deliberately decoupled so they can be unit tested with pure
analytical surrogates and without requiring trained PyTorch checkpoints to be
present in the repository. ``two_stage_runner.py`` wires them together for
production use.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field

import torch
from torch import nn

LOGGER = logging.getLogger(__name__)

ForwardKinematics = Callable[[torch.Tensor], torch.Tensor]
"""Callable mapping ``q`` of shape (N, n_joints) to clubface state (N, n_clubdof)."""

DynamicsSurrogate = Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]
"""Maps (q, q_dot, tau) shaped (N, n_joints) -> q_dot_next (N, n_joints)."""


@dataclass
class StageAResult:
    """Result of Stage A body kinematics optimization."""

    q: torch.Tensor
    """Optimized joint positions, shape (N, n_joints)."""

    final_loss: float
    history: list[dict[str, float]] = field(default_factory=list)


@dataclass
class StageBResult:
    """Result of Stage B torque optimization."""

    tau: torch.Tensor
    """Optimized joint torques, shape (N, n_joints)."""

    final_loss: float
    history: list[dict[str, float]] = field(default_factory=list)


@dataclass
class TwoStageResult:
    """Combined output from the two-stage pipeline."""

    stage_a: StageAResult
    stage_b: StageBResult


def _validate_2d(tensor: torch.Tensor, name: str) -> None:
    if tensor.ndim != 2:
        raise ValueError(
            f"{name} must be a 2D (N, dim) tensor; got shape {tuple(tensor.shape)}"
        )


def stage_a_optimize_kinematics(  # noqa: C901
    forward_kinematics: ForwardKinematics,
    clubface_target: torch.Tensor,
    q_rest: torch.Tensor,
    q_init: torch.Tensor | None = None,
    initial_state: torch.Tensor | None = None,
    motion_weight: float = 1e-3,
    steps: int = 200,
    learning_rate: float = 1e-2,
) -> StageAResult:
    """Optimize joint trajectory ``q(t)`` to track a clubface target.

    Loss::

        MSE(forward_kinematics(q) - clubface_target) + motion_weight * ||q - q_rest||^2

    Continuity at the start is enforced by hard-clamping ``q[0]`` to
    ``initial_state`` (when provided) on every iteration.

    :raises ValueError: if shapes do not align or weights are negative.
    """
    _validate_2d(clubface_target, "clubface_target")
    _validate_2d(q_rest, "q_rest")
    if motion_weight < 0:
        raise ValueError("motion_weight must be non-negative")
    if steps < 1:
        raise ValueError("steps must be >= 1")
    n_steps = clubface_target.shape[0]
    n_joints = q_rest.shape[1]
    if q_rest.shape[0] != n_steps:
        if q_rest.shape[0] == 1:
            q_rest = q_rest.expand(n_steps, n_joints).contiguous()
        else:
            raise ValueError(
                f"q_rest rows {q_rest.shape[0]} must equal target rows {n_steps} or 1"
            )

    if q_init is None:
        q_init = q_rest.detach().clone()
    else:
        _validate_2d(q_init, "q_init")
        if q_init.shape != (n_steps, n_joints):
            raise ValueError(
                f"q_init shape {tuple(q_init.shape)} != ({n_steps}, {n_joints})"
            )

    q = nn.Parameter(q_init.detach().clone())
    optimizer = torch.optim.Adam([q], lr=learning_rate)
    history: list[dict[str, float]] = []
    best_loss = float("inf")
    best_q = q.detach().clone()

    for step in range(1, steps + 1):
        if initial_state is not None:
            with torch.no_grad():
                q[0] = initial_state
        predicted = forward_kinematics(q)
        if predicted.shape != clubface_target.shape:
            raise ValueError(
                "forward_kinematics output shape "
                f"{tuple(predicted.shape)} != target {tuple(clubface_target.shape)}"
            )
        tracking = torch.mean((predicted - clubface_target) ** 2)
        motion = torch.mean((q - q_rest) ** 2)
        loss = tracking + motion_weight * motion

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        loss_val = float(loss.detach().cpu())
        history.append(
            {
                "step": step,
                "loss": loss_val,
                "tracking_loss": float(tracking.detach().cpu()),
                "motion_loss": float(motion.detach().cpu()),
            }
        )
        if loss_val < best_loss:
            best_loss = loss_val
            best_q = q.detach().clone()

    if initial_state is not None:
        best_q[0] = initial_state.detach().clone()
    return StageAResult(q=best_q, final_loss=best_loss, history=history)


def stage_b_optimize_torques(  # noqa: C901
    surrogate: DynamicsSurrogate,
    q: torch.Tensor,
    q_dot: torch.Tensor,
    q_dot_target: torch.Tensor,
    tau_init: torch.Tensor | None = None,
    smooth_weight: float = 1e-4,
    steps: int = 200,
    learning_rate: float = 1e-2,
) -> StageBResult:
    """Optimize torques ``tau(t)`` so the surrogate reproduces ``q_dot_target``.

    Loss::

        MSE(surrogate(q, q_dot, tau) - q_dot_target) + smooth_weight * ||delta tau||^2

    :raises ValueError: if shapes do not align or weights are negative.
    """
    for name, t in (("q", q), ("q_dot", q_dot), ("q_dot_target", q_dot_target)):
        _validate_2d(t, name)
    if smooth_weight < 0:
        raise ValueError("smooth_weight must be non-negative")
    if steps < 1:
        raise ValueError("steps must be >= 1")
    if not (q.shape == q_dot.shape == q_dot_target.shape):
        raise ValueError(
            f"q {tuple(q.shape)}, q_dot {tuple(q_dot.shape)}, q_dot_target "
            f"{tuple(q_dot_target.shape)} must all match"
        )

    if tau_init is None:
        tau_init = torch.zeros_like(q)
    else:
        _validate_2d(tau_init, "tau_init")
        if tau_init.shape != q.shape:
            raise ValueError(
                f"tau_init shape {tuple(tau_init.shape)} != q shape {tuple(q.shape)}"
            )

    tau = nn.Parameter(tau_init.detach().clone())
    optimizer = torch.optim.Adam([tau], lr=learning_rate)
    history: list[dict[str, float]] = []
    best_loss = float("inf")
    best_tau = tau.detach().clone()

    for step in range(1, steps + 1):
        predicted = surrogate(q, q_dot, tau)
        if predicted.shape != q_dot_target.shape:
            raise ValueError(
                "surrogate output shape "
                f"{tuple(predicted.shape)} != target {tuple(q_dot_target.shape)}"
            )
        tracking = torch.mean((predicted - q_dot_target) ** 2)
        if tau.shape[0] > 1:
            smooth = torch.mean((tau[1:] - tau[:-1]) ** 2)
        else:
            smooth = torch.zeros((), dtype=tau.dtype, device=tau.device)
        loss = tracking + smooth_weight * smooth

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        loss_val = float(loss.detach().cpu())
        history.append(
            {
                "step": step,
                "loss": loss_val,
                "tracking_loss": float(tracking.detach().cpu()),
                "smooth_loss": float(smooth.detach().cpu()),
            }
        )
        if loss_val < best_loss:
            best_loss = loss_val
            best_tau = tau.detach().clone()

    return StageBResult(tau=best_tau, final_loss=best_loss, history=history)


def _finite_difference_velocity(q: torch.Tensor, dt: float) -> torch.Tensor:
    """Forward-difference for velocity, last sample reuses the previous diff."""
    if q.shape[0] < 2:
        return torch.zeros_like(q)
    diff = (q[1:] - q[:-1]) / dt
    last = diff[-1:].clone()
    return torch.cat([diff, last], dim=0)


def run_two_stage(
    forward_kinematics: ForwardKinematics,
    surrogate: DynamicsSurrogate,
    clubface_target: torch.Tensor,
    q_rest: torch.Tensor,
    dt: float = 1.0 / 240.0,
    initial_state: torch.Tensor | None = None,
    stage_a_steps: int = 200,
    stage_b_steps: int = 200,
    motion_weight: float = 1e-3,
    smooth_weight: float = 1e-4,
    learning_rate: float = 1e-2,
) -> TwoStageResult:
    """Run Stage A (kinematics) then Stage B (torques) end-to-end.

    The Stage B target velocity is derived from the Stage A trajectory via a
    forward finite difference. ``initial_state`` (if provided) clamps ``q[0]``
    in Stage A.
    """
    if dt <= 0.0:
        raise ValueError("dt must be positive")
    stage_a = stage_a_optimize_kinematics(
        forward_kinematics=forward_kinematics,
        clubface_target=clubface_target,
        q_rest=q_rest,
        initial_state=initial_state,
        motion_weight=motion_weight,
        steps=stage_a_steps,
        learning_rate=learning_rate,
    )
    q = stage_a.q
    q_dot = _finite_difference_velocity(q, dt)
    q_dot_target = q_dot.detach().clone()
    stage_b = stage_b_optimize_torques(
        surrogate=surrogate,
        q=q.detach(),
        q_dot=q_dot.detach(),
        q_dot_target=q_dot_target,
        smooth_weight=smooth_weight,
        steps=stage_b_steps,
        learning_rate=learning_rate,
    )
    return TwoStageResult(stage_a=stage_a, stage_b=stage_b)

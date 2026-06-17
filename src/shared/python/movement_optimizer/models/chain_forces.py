# Copyright (c) 2026 D-Sorganization. All rights reserved.
"""Force estimates for the segmented-chain model.

These are *derived* quantities computed from the public kinematics of
:mod:`chain_dynamics` (node positions/velocities); the integrator itself is not
modified. Tension is an estimate in the same spirit as
``estimate_swingset_joint_torques``: the force transmitted along link ``i`` is
the net force required to drive the sub-chain at and below link ``i`` against
gravity, recovered by telescoping Newton's second law over the serial chain.

Coordinate convention (inherited from :mod:`chain_dynamics`): the anchor is at
``y = 0`` and the chain hangs toward **+y**, so gravity acts in the **+y**
direction. All force vectors are returned in this model frame; the canvas
projector handles the screen flip.

Design Principles:
    DBC -- every public function validates its preconditions (ValueError).
    LoD -- consumes only chain_dynamics public outputs; no GUI imports.
    DRY -- shared finite-difference / tension helpers.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray

from .chain_dynamics import ChainConfig, ChainRollout, link_midpoints, node_velocities

FloatArray: TypeAlias = NDArray[np.float64]


def _require_positive(name: str, value: float) -> None:
    if value <= 0.0:
        raise ValueError(f"{name} must be positive")


@dataclass(frozen=True)
class ChainForceField:
    """Per-link force vectors for a single chain frame (world frame, shape (N, 2))."""

    midpoints_m: FloatArray
    gravity_n: FloatArray
    tension_n: FloatArray
    net_force_n: FloatArray


@dataclass(frozen=True)
class ChainForceHistory:
    """Time history of chain force/curvature magnitudes for plotting."""

    time_s: FloatArray
    link_tension_n: FloatArray
    max_tension_n: FloatArray
    curvature_rad: FloatArray
    max_curvature_rad: FloatArray


def _gravity_vector(config: ChainConfig) -> FloatArray:
    """Per-link weight vector (points toward +y, the model's downward axis)."""
    return np.asarray([0.0, config.link_mass_kg * config.gravity_m_s2], dtype=np.float64)


def _midpoint_velocities(config: ChainConfig, rollout: ChainRollout) -> FloatArray:
    """Return ``(T, N, 2)`` link-midpoint velocities across the rollout."""
    per_state = [
        0.5 * (velocities[:-1] + velocities[1:])
        for velocities in (node_velocities(config, state) for state in rollout.states)
    ]
    return np.stack(per_state)


def link_accelerations(config: ChainConfig, rollout: ChainRollout, dt_s: float) -> FloatArray:
    """Return ``(T, N, 2)`` link-midpoint linear accelerations via finite difference.

    Preconditions:
        ``dt_s`` is positive and ``rollout`` has at least one state.
    """
    _require_positive("dt_s", dt_s)
    if not rollout.states:
        raise ValueError("rollout must contain at least one state")
    velocities = _midpoint_velocities(config, rollout)
    if velocities.shape[0] < 2:
        return np.zeros_like(velocities)
    return np.gradient(velocities, dt_s, axis=0)


def chain_force_field(
    config: ChainConfig,
    rollout: ChainRollout,
    dt_s: float,
    frame_index: int,
) -> ChainForceField:
    """Return per-link force vectors at ``frame_index``.

    Preconditions:
        ``dt_s`` is positive and ``0 <= frame_index < len(rollout.states)``.
    """
    state_count = len(rollout.states)
    if not 0 <= frame_index < state_count:
        raise ValueError(f"frame_index must be in [0, {state_count})")
    return chain_force_fields(config, rollout, dt_s)[frame_index]


def chain_force_fields(
    config: ChainConfig,
    rollout: ChainRollout,
    dt_s: float,
) -> tuple[ChainForceField, ...]:
    """Return force vectors for every frame in ``rollout``.

    Preconditions:
        ``dt_s`` is positive and ``rollout`` has at least one state.
    """
    _require_positive("dt_s", dt_s)
    if not rollout.states:
        raise ValueError("rollout must contain at least one state")

    weight = _gravity_vector(config)
    gravity = np.tile(weight, (config.segment_count, 1))
    accelerations = link_accelerations(config, rollout, dt_s)

    fields: list[ChainForceField] = []
    for frame_index, state in enumerate(rollout.states):
        midpoints = link_midpoints(state.node_positions(config))
        net_force = config.link_mass_kg * accelerations[frame_index]
        # T_i = sum_{j>=i} (m*a_j - weight_j) -- reverse cumulative sum.
        per_link = net_force - weight
        tension = np.cumsum(per_link[::-1], axis=0)[::-1]
        fields.append(
            ChainForceField(
                midpoints_m=midpoints,
                gravity_n=gravity,
                tension_n=tension,
                net_force_n=net_force,
            )
        )
    return tuple(fields)


def chain_force_history(
    config: ChainConfig,
    rollout: ChainRollout,
    dt_s: float,
) -> ChainForceHistory:
    """Return time-series tension and curvature magnitudes for the analysis panel.

    Preconditions:
        ``dt_s`` is positive and ``rollout`` has at least one state.
    """
    _require_positive("dt_s", dt_s)
    if not rollout.states:
        raise ValueError("rollout must contain at least one state")

    weight = _gravity_vector(config)
    accelerations = link_accelerations(config, rollout, dt_s)
    frame_count = accelerations.shape[0]
    per_link = config.link_mass_kg * accelerations - weight
    # Reverse cumulative sum along the link axis for every frame.
    tension_vectors = np.cumsum(per_link[:, ::-1, :], axis=1)[:, ::-1, :]
    link_tension = np.linalg.norm(tension_vectors, axis=2)
    max_tension = (
        np.max(link_tension, axis=1)
        if link_tension.shape[1]
        else np.zeros(frame_count, dtype=np.float64)
    )

    curvature = np.asarray(
        [np.diff(state.validated(config).angles_rad) for state in rollout.states],
        dtype=np.float64,
    )
    max_curvature = (
        np.max(np.abs(curvature), axis=1)
        if curvature.shape[1]
        else np.zeros(len(rollout.states), dtype=np.float64)
    )

    return ChainForceHistory(
        time_s=np.arange(frame_count, dtype=np.float64) * dt_s,
        link_tension_n=link_tension,
        max_tension_n=max_tension,
        curvature_rad=curvature,
        max_curvature_rad=max_curvature,
    )

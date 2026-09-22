"""Fail-closed constrained replay for the upper-body golfer (TB-06 #10591)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.shared.python.motion_matching.body_target import BodyTarget
from src.shared.python.tour_baselines.fit_metrics import (
    PhysicalFitMetrics,
    compute_fit_metrics,
)

from .physics_golfer import N_DOF, GolferParams
from .simulation_golfer import run_simulation


@dataclass(frozen=True)
class UpperBodyReplayTarget:
    """Immutable time-indexed inputs for a real constrained replay."""

    times: np.ndarray
    initial_state: np.ndarray
    torques: np.ndarray
    params: GolferParams

    def __post_init__(self) -> None:
        times = np.asarray(self.times, dtype=np.float64)
        state = np.asarray(self.initial_state, dtype=np.float64)
        torques = np.asarray(self.torques, dtype=np.float64)
        if times.ndim != 1 or len(times) < 2 or not np.isfinite(times).all():
            raise ValueError("times must be a finite vector with at least two frames")
        if not np.isclose(times[0], 0.0, atol=1e-12):
            raise ValueError("times must start at zero on the normalized capture clock")
        if np.any(np.diff(times) <= 0.0):
            raise ValueError("times must be strictly increasing")
        if not np.allclose(np.diff(times), np.diff(times)[0], rtol=1e-9, atol=1e-12):
            raise ValueError(
                "times must be uniformly sampled for exact constrained replay"
            )
        if state.shape != (2 * N_DOF,) or not np.isfinite(state).all():
            raise ValueError(f"initial_state must be finite with shape ({2 * N_DOF},)")
        if torques.shape != (len(times), N_DOF - 1) or not np.isfinite(torques).all():
            raise ValueError(
                f"torques must be finite with shape ({len(times)}, {N_DOF - 1})"
            )


@dataclass(frozen=True)
class UpperBodyMarkerAttachment:
    """A calibrated marker attachment in the mechanism's native plane.

    Surface markers cannot be treated as joint centres.  Every tracked marker
    therefore declares its model point and a fixed local planar offset.
    """

    label: str
    model_point: str
    offset_m: np.ndarray

    def __post_init__(self) -> None:
        offset = np.asarray(self.offset_m, dtype=np.float64)
        if not self.label:
            raise ValueError("label must be non-empty")
        if not self.model_point:
            raise ValueError("model_point must be non-empty")
        if offset.shape != (2,) or not np.isfinite(offset).all():
            raise ValueError("offset_m must be finite with shape (2,)")


@dataclass(frozen=True)
class UpperBodyCaptureFrame:
    """Rigid embedding of the model plane into the capture coordinate frame."""

    origin_m: np.ndarray
    plane_basis: np.ndarray

    def __post_init__(self) -> None:
        origin = np.asarray(self.origin_m, dtype=np.float64)
        basis = np.asarray(self.plane_basis, dtype=np.float64)
        if origin.shape != (3,) or not np.isfinite(origin).all():
            raise ValueError("origin_m must be finite with shape (3,)")
        if basis.shape != (3, 2) or not np.isfinite(basis).all():
            raise ValueError("plane_basis must be finite with shape (3, 2)")
        if not np.allclose(basis.T @ basis, np.eye(2), atol=1e-8):
            raise ValueError("plane_basis columns must be orthonormal")


@dataclass(frozen=True)
class UpperBodyReplay:
    """Real replay trajectory and constraint reactions; no acceptance claim.

    ``kinematic_points`` contains the model's native planar coordinates.  It
    deliberately does not invent a three-dimensional marker attachment or a
    capture-frame transform; those must be calibrated by the fit layer.
    """

    times: np.ndarray
    states: np.ndarray
    actuator_torques: np.ndarray
    reaction_forces: np.ndarray
    kinematic_points: dict[str, np.ndarray]
    constraint_residual_m: float


def project_replay_markers(
    replay: UpperBodyReplay,
    attachments: tuple[UpperBodyMarkerAttachment, ...],
    capture_frame: UpperBodyCaptureFrame,
) -> np.ndarray:
    """Project calibrated native points into the declared 3D capture frame.

    The operation only embeds the constrained model plane using a rigid,
    isometric frame.  It never estimates or infers attachments from observed
    markers; calibration and optimization remain explicit responsibilities of
    the caller.
    """
    if len(attachments) < 3:
        raise ValueError("at least three explicit marker attachments are required")
    labels = [attachment.label for attachment in attachments]
    if len(labels) != len(set(labels)):
        raise ValueError("marker attachment labels must be unique")

    local_points: list[np.ndarray] = []
    for attachment in attachments:
        point = replay.kinematic_points.get(attachment.model_point)
        if point is None:
            raise ValueError(
                f"unknown model point {attachment.model_point!r}; "
                f"available: {sorted(replay.kinematic_points)}"
            )
        offset = np.asarray(attachment.offset_m, dtype=np.float64)
        local_points.append(point + offset)

    local = np.stack(local_points, axis=1)
    basis = np.asarray(capture_frame.plane_basis, dtype=np.float64)
    origin = np.asarray(capture_frame.origin_m, dtype=np.float64)
    return np.einsum("tmk,dk->tmd", local, basis) + origin


def evaluate_replay_against_body_target(
    replay: UpperBodyReplay,
    body_target: BodyTarget,
    attachments: tuple[UpperBodyMarkerAttachment, ...],
    capture_frame: UpperBodyCaptureFrame,
) -> PhysicalFitMetrics:
    """Evaluate a replay against explicit body markers on the same source clock.

    There is intentionally no interpolation at this boundary: calibration and
    replay evidence must declare matching clocks before their Euclidean marker
    errors can be reported.
    """
    if not np.array_equal(replay.times, body_target.time):
        raise ValueError("replay and body target must use the identical time grid")
    labels = tuple(attachment.label for attachment in attachments)
    unavailable = sorted(set(labels).difference(body_target.marker_names))
    if unavailable:
        raise ValueError(
            f"body target does not provide attached markers: {unavailable}"
        )

    indices = [body_target.marker_names.index(label) for label in labels]
    observed = body_target.marker_xyz[:, indices, :]
    valid = np.isfinite(observed).all(axis=-1)
    predicted = project_replay_markers(replay, attachments, capture_frame)
    return compute_fit_metrics(
        predicted,
        observed,
        valid,
        labels,
        time_s=body_target.time,
        impact_frame=body_target.impact_idx,
    )


def replay_upper_body_target(target: UpperBodyReplayTarget) -> UpperBodyReplay:
    """Replay supplied torques through the constrained dynamics without interpolation."""
    elapsed = target.times - target.times[0]

    def torque_at(t: float) -> tuple[float, float, float, float, float, float, float]:
        index = int(np.searchsorted(elapsed, t, side="right") - 1)
        index = int(np.clip(index, 0, len(elapsed) - 1))
        return tuple(float(value) for value in target.torques[index])  # type: ignore[return-value]

    result = run_simulation(
        target.params,
        np.asarray(target.initial_state, dtype=np.float64),
        float(elapsed[-1]),
        torque_at,
        dt=float(np.min(np.diff(elapsed))),
    )
    reactions = np.vstack(
        [result.constraint_forces_at(index) for index in range(result.n_steps)]
    )
    actuator_torques = np.vstack(
        [result.torques_at(index) for index in range(result.n_steps)]
    )
    positions = [result.positions_at(index) for index in range(result.n_steps)]
    kinematic_points = {
        point_name: np.asarray(
            [frame[point_name] for frame in positions], dtype=np.float64
        )
        for point_name in positions[0]
    }
    residual = max(
        result.constraint_violation_at(index) for index in range(result.n_steps)
    )
    return UpperBodyReplay(
        times=result.t,
        states=result.states,
        actuator_torques=actuator_torques,
        reaction_forces=reactions,
        kinematic_points=kinematic_points,
        constraint_residual_m=float(residual),
    )

"""Keypoint-tracking OCP: optimal estimation of ``(q, qdot, tau)`` (Phase 3.1).

Given a ``KeypointSequence`` (3D keypoints in the rig's world frame) or a
``MarkerTrajectory``, recover a dynamically consistent trajectory by
tracking markers with torque-driven dynamics -- the standard bioptim
"optimal estimation" workflow, here on UpstreamDrift's own symbolic model.

Formulation (single phase, ``n_shooting = n_frames - 1``):

- ``ObjectiveFcn.Lagrange.TRACK_MARKERS`` on every node against the
  ``(3, n_markers, n_frames)`` target, with per-node weights that mask
  frames whose keypoint confidence is below ``TrackingWeights.min_confidence``
  (dropped frames simply stop contributing);
- ``MINIMIZE_CONTROL("tau")`` and ``MINIMIZE_STATE("qdot", derivative=True)``
  regularisers.

Frames must be uniformly spaced; resample first when they are not.
"""

from __future__ import annotations

import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from src.shared.python.motion_pipeline.contracts import (
    KeypointSequence,
    MarkerTrajectory,
)
from src.shared.python.optimization._swing_kinematics import JOINTS
from src.shared.python.optimization._swing_models import ClubModel, GolferModel
from src.shared.python.logging_pkg.logging_config import get_logger
from src.shared.python.optimization.model_provider import swing_joint_limits
from src.shared.python.optimization.ocp._compat import require_bioptim
from src.shared.python.optimization.ocp.bioptim_model import make_swing_bio_model
from src.shared.python.optimization.ocp.keypoint_map import keypoint_map_for_schema
from src.shared.python.optimization.ocp.result import (
    bioptim_provenance,
    solution_arrays,
)
from src.shared.python.optimization.ocp.symbolic_model import MARKER_NAMES
from src.shared.python.simulation_backends.provenance import ProvenanceStamp

__all__ = [
    "MarkerTargets",
    "PoseIdentifiability",
    "TrackingResult",
    "TrackingWeights",
    "build_tracking_ocp",
    "keypoints_to_targets",
    "markers_to_targets",
    "probe_marker_identifiability",
    "solve_tracking_ocp",
]

logger = get_logger(__name__)

_VELOCITY_BOUND = 40.0
_UNIFORM_DT_TOLERANCE = 1e-6


@dataclass(frozen=True)
class TrackingWeights:
    """Objective weights for the tracking OCP.

    ``marker`` scales the squared marker error (m²), ``torque`` the effort
    integral, ``qdot_derivative`` the smoothness regulariser on velocity
    changes. Keypoints with confidence below ``min_confidence`` are masked.
    """

    marker: float = 1.0e4
    torque: float = 1.0e-4
    qdot_derivative: float = 1.0e-2
    min_confidence: float = 0.2

    def __post_init__(self) -> None:
        if self.marker <= 0.0:
            raise ValueError("marker weight must be positive")
        if self.torque < 0.0 or self.qdot_derivative < 0.0:
            raise ValueError("regulariser weights must be non-negative")
        if not 0.0 <= self.min_confidence <= 1.0:
            raise ValueError("min_confidence must be in [0, 1]")


@dataclass(frozen=True)
class MarkerTargets:
    """Marker targets on a uniform frame grid.

    ``positions`` is ``(3, n_markers, n_frames)`` in the rig world frame,
    ``weights`` is ``(n_markers, n_frames)`` in ``[0, 1]`` (0 = not observed),
    ``marker_names`` orders the marker axis and matches the model's set.
    """

    times: np.ndarray
    positions: np.ndarray
    weights: np.ndarray
    marker_names: tuple[str, ...]

    def __post_init__(self) -> None:
        times = np.asarray(self.times, dtype=float)
        if times.ndim != 1 or times.size < 2:
            raise ValueError("at least two frames are required")
        if self.positions.shape != (3, len(self.marker_names), times.size):
            raise ValueError("positions must be (3, n_markers, n_frames)")
        if self.weights.shape != (len(self.marker_names), times.size):
            raise ValueError("weights must be (n_markers, n_frames)")
        dts = np.diff(times)
        if np.any(dts <= 0.0):
            raise ValueError("frame times must increase")
        if np.max(np.abs(dts - dts[0])) > _UNIFORM_DT_TOLERANCE * max(dts[0], 1.0):
            raise ValueError("frames must be uniformly spaced; resample first")
        object.__setattr__(self, "times", times)

    @property
    def n_frames(self) -> int:
        return int(self.times.size)

    @property
    def dt(self) -> float:
        return float(self.times[1] - self.times[0])

    @property
    def duration(self) -> float:
        return float(self.times[-1] - self.times[0])


@dataclass(frozen=True)
class PoseIdentifiability:
    """Which joint DOFs the marker set can see at one pose (#9758).

    The seven-DOF swing chain has two such directions against this marker
    set, and both are structural rather than accidental:

    - ``hip_rotation`` and ``trunk_rotation`` are both rotations about Z with
      offsets along Z, so turning one and counter-turning the other moves no
      marker at all. Only their sum is observable. This is an exact null
      direction at every pose.
    - The remaining directions span two orders of magnitude in singular
      value, so most joint angles are only weakly determined by these six
      markers even where the Jacobian is nominally invertible. The terminal
      shaft roll is the weakest, seen only through the small ``clubface``
      offset from the shaft axis.

    The practical consequence: this marker set is good for recovering a
    *dynamically consistent trajectory that fits the markers*, which is what
    the tracking OCP is for, and poor for reading individual joint angles
    off the result. Adding markers off the joint axes (a mid-forearm band, a
    second clubhead point) is what would change that.

    The tracking OCP still returns *a* trajectory along those directions --
    one that fits the markers and means nothing -- so every solve reports
    this rather than leaving the caller to guess.
    """

    dof_names: tuple[str, ...]
    singular_values: np.ndarray
    rank: int
    unobservable_dofs: tuple[str, ...]
    weakly_observable_dofs: tuple[str, ...] = ()

    @property
    def is_full_rank(self) -> bool:
        return self.rank == len(self.dof_names)

    @property
    def condition_number(self) -> float:
        """``sigma_max / sigma_min``; infinite when rank deficient."""
        if self.singular_values.size == 0 or self.singular_values[-1] <= 0.0:
            return float("inf")
        return float(self.singular_values[0] / self.singular_values[-1])

    @property
    def reliable_dofs(self) -> tuple[str, ...]:
        """DOFs in neither the null nor the weakly observable directions."""
        suspect = set(self.unobservable_dofs) | set(self.weakly_observable_dofs)
        return tuple(name for name in self.dof_names if name not in suspect)

    def to_dict(self) -> dict[str, object]:
        return {
            "rank": self.rank,
            "n_dof": len(self.dof_names),
            "unobservable_dofs": list(self.unobservable_dofs),
            "weakly_observable_dofs": list(self.weakly_observable_dofs),
            "reliable_dofs": list(self.reliable_dofs),
            "condition_number": self.condition_number,
            "singular_values": self.singular_values.tolist(),
        }


def probe_marker_identifiability(
    model: Any,
    q: np.ndarray,
    *,
    parameters: np.ndarray | None = None,
    relative_tolerance: float = 1e-6,
    weak_tolerance: float = 1e-1,
    share: float = 0.5,
) -> PoseIdentifiability:
    """SVD of the marker-position Jacobian with respect to ``q`` at one pose.

    Reuses :func:`estimation.identifiability.probe_identifiability`, the same
    machinery the MAP solvers gate parameters with (#9758), so both paths
    report unobservable directions the same way.

    Args:
        model: A :class:`SymbolicSwingModel` or the bioptim adapter over one.
        q: Joint pose to linearise about.
        parameters: Model parameter values (empty when none are symbolic).
        relative_tolerance: Singular values below this fraction of the
            largest mark an *unobservable* direction.
        weak_tolerance: Below this fraction of the largest singular value,
            *weakly observable*: nominally invertible, but the recovered
            value is dominated by noise. The default (a tenth) is not
            conservative for this marker set -- most of the chain's
            directions fall under it, which is the finding, not a mis-set
            threshold.
        share: A DOF joins a direction's list when its component reaches this
            fraction of that direction's largest component. Relative rather
            than absolute, because a direction spread across several DOFs
            implicates all of them: two joints that trade off against each
            other are equally untrustworthy, and a fixed threshold would
            report only one of the pair.
    """
    from src.shared.python.estimation.identifiability import (
        ParameterSpec,
        probe_identifiability,
    )

    symbolic = getattr(model, "symbolic", model)
    empty = np.zeros(0) if parameters is None else np.asarray(parameters, dtype=float)
    names = tuple(symbolic.dof_names)

    def observation(values: np.ndarray) -> np.ndarray:
        return np.asarray(symbolic.markers(values, empty), dtype=float).reshape(-1)

    report = probe_identifiability(
        observation, np.asarray(q, dtype=float).reshape(-1), ParameterSpec(names)
    )
    sigma_max = float(report.singular_values[0]) if report.singular_values.size else 0.0
    tolerance = relative_tolerance * sigma_max
    weak_cut = weak_tolerance * sigma_max
    rank = int((report.singular_values > tolerance).sum())

    def implicated_dofs(sigma_low: float, sigma_high: float) -> list[str]:
        """DOFs implicated in the directions whose singular value is in range.

        Every DOF reaching ``share`` of the direction's dominant component is
        reported, so a two-joint trade-off names both joints.
        """
        found: list[str] = []
        for index, sigma in enumerate(report.singular_values):
            if not sigma_low <= sigma <= sigma_high:
                continue
            direction = np.abs(report.right_singular_vectors[:, index])
            dominant = float(direction.max())
            if dominant <= 0.0:
                continue
            for dof_index in np.flatnonzero(direction >= share * dominant):
                name = names[int(dof_index)]
                if name not in found:
                    found.append(name)
        return found

    unobservable = implicated_dofs(-float("inf"), tolerance)
    weak = [
        name
        for name in implicated_dofs(tolerance, weak_cut)
        if name not in unobservable
    ]
    return PoseIdentifiability(
        dof_names=names,
        singular_values=report.singular_values,
        rank=rank,
        unobservable_dofs=tuple(unobservable),
        weakly_observable_dofs=tuple(weak),
    )


@dataclass(frozen=True)
class TrackingResult:
    """Dynamically consistent trajectory recovered from marker targets."""

    time: np.ndarray
    q: np.ndarray
    qdot: np.ndarray
    tau: np.ndarray
    parameters: dict[str, float]
    marker_rms_m: dict[str, float]
    status: int
    iterations: int
    cost: float
    wall_time_s: float
    provenance: ProvenanceStamp | None = None
    locked_by_gate: tuple[str, ...] = field(default_factory=tuple)
    identifiability: PoseIdentifiability | None = None

    @property
    def success(self) -> bool:
        return self.status == 0


def keypoints_to_targets(
    sequence: KeypointSequence,
    *,
    keypoint_map: Mapping[str, str] | None = None,
    marker_names: Sequence[str] = MARKER_NAMES,
) -> MarkerTargets:
    """Project a 3D ``KeypointSequence`` onto the model's marker set.

    Unmapped keypoint names are ignored with one warning; keypoints without
    a ``z`` coordinate are rejected (2D tracking needs a camera model that
    this OCP does not carry). Confidence becomes the per-frame weight.
    """
    schema = sequence.frames[0].schema_name
    mapping = (
        dict(keypoint_map)
        if keypoint_map is not None
        else keypoint_map_for_schema(schema)
    )
    names = tuple(marker_names)
    n_frames = sequence.num_frames
    positions = np.zeros((3, len(names), n_frames))
    weights = np.zeros((len(names), n_frames))
    ignored: set[str] = set()
    for frame_index, frame in enumerate(sequence.frames):
        for keypoint in frame.keypoints:
            if keypoint.name is None:
                continue
            marker = mapping.get(keypoint.name)
            if marker is None or marker not in names:
                ignored.add(keypoint.name)
                continue
            if keypoint.z is None:
                raise ValueError(
                    f"keypoint {keypoint.name!r} has no z; the tracking OCP needs 3D keypoints"
                )
            column = names.index(marker)
            positions[:, column, frame_index] = (keypoint.x, keypoint.y, keypoint.z)
            weights[column, frame_index] = float(keypoint.confidence)
    if ignored:
        logger.warning("tracking OCP ignores unmapped keypoints: %s", sorted(ignored))
    times = np.array([frame.timestamp for frame in sequence.frames], dtype=float)
    return MarkerTargets(
        times=times, positions=positions, weights=weights, marker_names=names
    )


def markers_to_targets(
    trajectory: MarkerTrajectory,
    *,
    marker_names: Sequence[str] = MARKER_NAMES,
) -> MarkerTargets:
    """Marker targets from a ``MarkerTrajectory`` whose labels match the model."""
    names = tuple(marker_names)
    n_frames = len(trajectory.frames)
    positions = np.zeros((3, len(names), n_frames))
    weights = np.zeros((len(names), n_frames))
    for frame_index, frame in enumerate(trajectory.frames):
        for column, name in enumerate(names):
            marker = frame.markers.get(name)
            if marker is None or marker.occluded:
                continue
            positions[:, column, frame_index] = (marker.x, marker.y, marker.z)
            weights[column, frame_index] = 1.0
    times = np.array([frame.timestamp for frame in trajectory.frames], dtype=float)
    return MarkerTargets(
        times=times, positions=positions, weights=weights, marker_names=names
    )


def build_tracking_ocp(
    targets: MarkerTargets,
    golfer: GolferModel | None = None,
    club: ClubModel | None = None,
    *,
    weights: TrackingWeights | None = None,
    q_guess: np.ndarray | None = None,
    parameters: Sequence[str] = (),
    n_integration_steps: int = 2,
    n_threads: int = 1,
) -> tuple[Any, Any]:
    """Build the marker-tracking OCP. Returns ``(ocp, model)``.

    Args:
        targets: Marker targets on a uniform grid (see the converters).
        golfer, club: Numeric model; markers must match ``targets``.
        weights: Objective weights; see :class:`TrackingWeights`.
        q_guess: Optional ``(n_joints, n_frames)`` kinematic initial guess
            (e.g. from the motion pipeline's IK). Without it IPOPT starts from
            the neutral pose, which is fine for synthetic fixtures and slow
            for real swings.
        parameters: Symbolic model parameters (Phase 4 adds them to the OCP).
    """
    bioptim = require_bioptim()
    weights = weights or TrackingWeights()
    model = make_swing_bio_model(golfer, club, parameters=parameters)
    if tuple(targets.marker_names) != tuple(model.marker_names):
        raise ValueError(
            f"targets markers {targets.marker_names} must match the model's "
            f"{model.marker_names}"
        )
    n = model.nb_q
    n_frames = targets.n_frames
    n_shooting = n_frames - 1
    limits = model.symbolic.torque_limits()

    lagrange = bioptim.ObjectiveFcn.Lagrange
    objectives = bioptim.ObjectiveList()
    # Per-node weight = marker weight x confidence mask so dropped frames
    # contribute nothing without changing the target array's shape.
    node_weights = np.where(
        targets.weights >= weights.min_confidence, targets.weights, 0.0
    )
    for column, name in enumerate(targets.marker_names):
        if not np.any(node_weights[column]):
            logger.warning("marker %r is never observed; not tracked", name)
            continue
        objectives.add(
            lagrange.TRACK_MARKERS,
            marker_index=column,
            target=targets.positions[:, column : column + 1, :],
            weight=bioptim.ObjectiveWeight(
                weights.marker * node_weights[column],
                interpolation=bioptim.InterpolationType.EACH_FRAME,
            ),
            node=bioptim.Node.ALL,
            quadratic=True,
        )
    if weights.torque:
        objectives.add(
            lagrange.MINIMIZE_CONTROL,
            key="tau",
            weight=weights.torque,
        )
    if weights.qdot_derivative:
        objectives.add(
            lagrange.MINIMIZE_STATE,
            key="qdot",
            derivative=True,
            weight=weights.qdot_derivative,
        )

    dynamics = bioptim.DynamicsOptionsList()
    dynamics.add(
        bioptim.DynamicsOptions(
            ode_solver=bioptim.OdeSolver.RK4(n_integration_steps=n_integration_steps),
            expand_dynamics=True,
            phase_dynamics=bioptim.PhaseDynamics.SHARED_DURING_THE_PHASE,
        )
    )

    joint_limits = swing_joint_limits(model.golfer)
    x_bounds = bioptim.BoundsList()
    x_bounds["q"] = (
        np.array([joint_limits[j][0] for j in JOINTS]),
        np.array([joint_limits[j][1] for j in JOINTS]),
    )
    x_bounds["qdot"] = np.full(n, -_VELOCITY_BOUND), np.full(n, _VELOCITY_BOUND)
    u_bounds = bioptim.BoundsList()
    u_bounds["tau"] = -limits, limits

    x_init = bioptim.InitialGuessList()
    if q_guess is not None:
        q_guess = np.asarray(q_guess, dtype=float)
        if q_guess.shape != (n, n_frames):
            raise ValueError(f"q_guess must have shape {(n, n_frames)}")
        x_init.add("q", q_guess, interpolation=bioptim.InterpolationType.EACH_FRAME)
        qdot_guess = np.gradient(q_guess, targets.dt, axis=1)
        x_init.add(
            "qdot", qdot_guess, interpolation=bioptim.InterpolationType.EACH_FRAME
        )
    else:
        x_init["q"] = np.zeros(n)
        x_init["qdot"] = np.zeros(n)
    u_init = bioptim.InitialGuessList()
    u_init["tau"] = np.zeros(n)

    u_scaling = bioptim.VariableScalingList()
    u_scaling.add("tau", scaling=limits)

    ocp = bioptim.OptimalControlProgram(
        model,
        n_shooting,
        targets.duration,
        dynamics=dynamics,
        x_bounds=x_bounds,
        u_bounds=u_bounds,
        x_init=x_init,
        u_init=u_init,
        objective_functions=objectives,
        u_scaling=u_scaling,
        use_sx=False,
        n_threads=n_threads,
    )
    return ocp, model


def _marker_rms(
    model: Any, q: np.ndarray, targets: MarkerTargets, parameters: np.ndarray
) -> dict[str, float]:
    rms: dict[str, float] = {}
    predicted = np.stack(
        [
            np.asarray(model.symbolic.markers(q[:, k], parameters))
            for k in range(q.shape[1])
        ],
        axis=2,
    )
    for column, name in enumerate(targets.marker_names):
        mask = targets.weights[column] > 0.0
        if not np.any(mask):
            rms[name] = float("nan")
            continue
        error = predicted[:, column, mask] - targets.positions[:, column, mask]
        rms[name] = float(np.sqrt(np.mean(np.sum(error**2, axis=0))))
    return rms


def solve_tracking_ocp(
    targets: MarkerTargets,
    golfer: GolferModel | None = None,
    club: ClubModel | None = None,
    *,
    weights: TrackingWeights | None = None,
    q_guess: np.ndarray | None = None,
    max_iterations: int = 300,
    n_integration_steps: int = 2,
    n_threads: int = 1,
) -> TrackingResult:
    """Build and solve the tracking OCP (fixed model parameters)."""
    bioptim = require_bioptim()
    ocp, model = build_tracking_ocp(
        targets,
        golfer,
        club,
        weights=weights,
        q_guess=q_guess,
        n_integration_steps=n_integration_steps,
        n_threads=n_threads,
    )
    solver = bioptim.Solver.IPOPT(show_online_optim=False)
    solver.set_print_level(0)
    solver.set_maximum_iterations(int(max_iterations))
    started = time.perf_counter()
    sol = ocp.solve(solver=solver)
    wall = time.perf_counter() - started
    q, qdot, tau = solution_arrays(sol, bioptim, n_nodes=targets.n_frames)
    empty = np.zeros(0)
    identifiability = probe_marker_identifiability(model, q[:, q.shape[1] // 2])
    if identifiability.unobservable_dofs or identifiability.weakly_observable_dofs:
        logger.warning(
            "the marker set observes %d of %d joint DOFs at the mid pose "
            "(condition number %.3g); unobservable: %s; weakly observable: "
            "%s. Values recovered along those directions are not measurements.",
            identifiability.rank,
            len(identifiability.dof_names),
            identifiability.condition_number,
            list(identifiability.unobservable_dofs) or "none",
            list(identifiability.weakly_observable_dofs) or "none",
        )
    settings = {
        "kind": "tracking",
        "n_shooting": targets.n_frames - 1,
        "final_time": targets.duration,
        "max_iterations": int(max_iterations),
        "weights": {
            "marker": weights.marker if weights else TrackingWeights().marker,
            "torque": weights.torque if weights else TrackingWeights().torque,
            "qdot_derivative": (
                weights.qdot_derivative
                if weights
                else TrackingWeights().qdot_derivative
            ),
        },
    }
    return TrackingResult(
        time=targets.times.copy(),
        q=q,
        qdot=qdot,
        tau=tau,
        parameters={},
        marker_rms_m=_marker_rms(model, q, targets, empty),
        status=int(sol.status),
        iterations=int(getattr(sol, "iterations", 0) or 0),
        cost=float(np.asarray(sol.cost).ravel()[0]),
        wall_time_s=wall,
        provenance=bioptim_provenance(model.golfer, model.club, settings),
        identifiability=identifiability,
    )

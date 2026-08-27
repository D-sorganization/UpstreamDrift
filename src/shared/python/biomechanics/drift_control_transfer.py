"""Model-independent joint-transfer attribution for drift/control studies.

The schema is deliberately pointwise: model adapters provide forces and
couples evaluated at the same achieved state for the total, zero-command
(``drift``), and commanded increment (``control``) cases.  This module then
performs geometry, impulse, power, work, phase, and attribution accounting
without knowing how any concrete model generated those arrays.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any, TypeAlias

import numpy as np
import numpy.typing as npt

FloatArray: TypeAlias = npt.NDArray[np.float64]
BoolArray: TypeAlias = npt.NDArray[np.bool_]

_DEFAULT_EPSILON = 1e-12
_CLOSURE_RTOL = 1e-9
_CLOSURE_ATOL = 1e-10


def _finite_array(name: str, value: Any, shape: tuple[int, ...]) -> FloatArray:
    array = np.asarray(value, dtype=float)
    if array.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {array.shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return array.copy()


def _require_split(
    name: str, total: FloatArray, drift: FloatArray, control: FloatArray
) -> None:
    if not np.allclose(total, drift + control, rtol=_CLOSURE_RTOL, atol=_CLOSURE_ATOL):
        residual = float(np.max(np.abs(total - drift - control)))
        raise ValueError(
            f"{name} must equal drift + control; maximum residual={residual:.3e}"
        )


@dataclass(frozen=True, slots=True)
class JointTransferTrajectory:
    """Canonical planar joint-transfer history using proximal-on-distal action.

    Vector fields have shape ``(T, J, 2)`` and planar couple fields have shape
    ``(T, J)``.  Total, drift, and control values are required to close at each
    sample and joint.  ZVCF is intentionally not represented by ``control``:
    the control value is the state-matched increment ``total - drift``.
    """

    time: FloatArray
    joint_names: tuple[str, ...]
    position: FloatArray
    velocity: FloatArray
    force_total: FloatArray
    force_drift: FloatArray
    force_control: FloatArray
    couple_total: FloatArray
    couple_drift: FloatArray
    couple_control: FloatArray
    angular_velocity: FloatArray
    model_tier: str
    force_direction: str = "proximal_on_distal"
    frame: str = "swing_plane_cartesian"
    reference_point: str = "joint_origin"
    units: str = "SI"

    def __post_init__(self) -> None:
        time = np.asarray(self.time, dtype=float).reshape(-1)
        if time.size < 2:
            raise ValueError("time must contain at least two samples")
        if not np.all(np.isfinite(time)):
            raise ValueError("time must contain only finite values")
        if np.any(np.diff(time) <= 0.0):
            raise ValueError("time must be strictly increasing")

        names = tuple(self.joint_names)
        if not names or any(not name.strip() for name in names):
            raise ValueError("joint_names must contain non-empty names")
        if len(set(names)) != len(names):
            raise ValueError("joint_names must be unique")
        if not self.model_tier.strip():
            raise ValueError("model_tier must be non-empty")
        for name in ("force_direction", "frame", "reference_point", "units"):
            if not str(getattr(self, name)).strip():
                raise ValueError(f"{name} must be non-empty")

        samples = time.size
        joints = len(names)
        vector_shape = (samples, joints, 2)
        scalar_shape = (samples, joints)
        vectors = {
            name: _finite_array(name, getattr(self, name), vector_shape)
            for name in (
                "position",
                "velocity",
                "force_total",
                "force_drift",
                "force_control",
            )
        }
        scalars = {
            name: _finite_array(name, getattr(self, name), scalar_shape)
            for name in (
                "couple_total",
                "couple_drift",
                "couple_control",
                "angular_velocity",
            )
        }
        _require_split(
            "force_total",
            vectors["force_total"],
            vectors["force_drift"],
            vectors["force_control"],
        )
        _require_split(
            "couple_total",
            scalars["couple_total"],
            scalars["couple_drift"],
            scalars["couple_control"],
        )
        object.__setattr__(self, "time", time.copy())
        object.__setattr__(self, "joint_names", names)
        for name, value in vectors.items():
            object.__setattr__(self, name, value)
        for name, value in scalars.items():
            object.__setattr__(self, name, value)

    @property
    def sample_count(self) -> int:
        """Return the number of time samples."""
        return int(self.time.size)

    @property
    def joint_count(self) -> int:
        """Return the number of declared joints."""
        return len(self.joint_names)

    def as_init_dict(self) -> dict[str, Any]:
        """Return constructor fields for deterministic test/adaptor rebuilding."""
        return {field.name: getattr(self, field.name) for field in fields(self)}


@dataclass(frozen=True, slots=True)
class PathFrame:
    """Velocity-aligned planar frame; zero vectors mark invalid samples."""

    speed: FloatArray
    tangent: FloatArray
    normal: FloatArray
    valid: BoolArray
    speed_epsilon: float


@dataclass(frozen=True, slots=True)
class ForcePathProjection:
    """Signed along-path and left-normal force components."""

    total_along: FloatArray
    drift_along: FloatArray
    control_along: FloatArray
    total_normal: FloatArray
    drift_normal: FloatArray
    control_normal: FloatArray
    valid: BoolArray


@dataclass(frozen=True, slots=True)
class ImpulseDecomposition:
    """Cumulative vector and along-path impulse decomposition."""

    vector_total: FloatArray
    vector_drift: FloatArray
    vector_control: FloatArray
    tangent_total_signed: FloatArray
    tangent_total_positive: FloatArray
    tangent_total_negative: FloatArray
    tangent_total_absolute: FloatArray
    tangent_drift_signed: FloatArray
    tangent_drift_positive: FloatArray
    tangent_drift_negative: FloatArray
    tangent_drift_absolute: FloatArray
    tangent_control_signed: FloatArray
    tangent_control_positive: FloatArray
    tangent_control_negative: FloatArray
    tangent_control_absolute: FloatArray


@dataclass(frozen=True, slots=True)
class PowerWorkDecomposition:
    """Pointwise power and cumulative work for force, couple, and total action."""

    force_power_total: FloatArray
    force_power_drift: FloatArray
    force_power_control: FloatArray
    couple_power_total: FloatArray
    couple_power_drift: FloatArray
    couple_power_control: FloatArray
    total_power_total: FloatArray
    total_power_drift: FloatArray
    total_power_control: FloatArray
    force_work_total: FloatArray
    force_work_drift: FloatArray
    force_work_control: FloatArray
    couple_work_total: FloatArray
    couple_work_drift: FloatArray
    couple_work_control: FloatArray
    total_work_total: FloatArray
    total_work_drift: FloatArray
    total_work_control: FloatArray


@dataclass(frozen=True, slots=True)
class PathWeightedMeanForce:
    """Signed force-work per valid path length for each joint.

    This is the line-integral estimand ``integral(F . dx) / integral(ds)``.
    It is neither a time average nor an average force magnitude.
    """

    path_length: FloatArray
    force_work_total: FloatArray
    force_work_drift: FloatArray
    force_work_control: FloatArray
    mean_force_total: FloatArray
    mean_force_drift: FloatArray
    mean_force_control: FloatArray
    valid: BoolArray


@dataclass(frozen=True, slots=True)
class SwingPhase:
    """Named half-open time interval; the final phase includes its endpoint."""

    name: str
    start_s: float
    end_s: float

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("phase name must be non-empty")
        if not np.isfinite(self.start_s) or not np.isfinite(self.end_s):
            raise ValueError("phase boundaries must be finite")
        if self.end_s <= self.start_s:
            raise ValueError("phase end_s must be greater than start_s")


@dataclass(frozen=True, slots=True)
class AttributionShares:
    """Cancellation-safe signed and magnitude attribution shares."""

    signed_drift_share: FloatArray
    signed_control_share: FloatArray
    magnitude_drift_share: FloatArray
    magnitude_control_share: FloatArray
    cancellation_index: FloatArray
    signed_valid: BoolArray
    magnitude_valid: BoolArray


@dataclass(frozen=True, slots=True)
class PhaseTransferSummary:
    """End-minus-start transfer quantities over one disjoint phase interval."""

    phase_name: str
    start_index: int
    end_index: int
    start_time_s: float
    end_time_s: float
    sample_count: int
    interval_count: int
    vector_impulse_total: FloatArray
    vector_impulse_drift: FloatArray
    vector_impulse_control: FloatArray
    tangent_impulse_total: FloatArray
    tangent_impulse_drift: FloatArray
    tangent_impulse_control: FloatArray
    force_work_total: FloatArray
    force_work_drift: FloatArray
    force_work_control: FloatArray
    couple_work_total: FloatArray
    couple_work_drift: FloatArray
    couple_work_control: FloatArray
    total_work_total: FloatArray
    total_work_drift: FloatArray
    total_work_control: FloatArray


def compute_path_frame(
    velocity: Any, *, speed_epsilon: float = _DEFAULT_EPSILON
) -> PathFrame:
    """Construct a planar tangent/left-normal frame from joint velocity."""
    if not np.isfinite(speed_epsilon) or speed_epsilon <= 0.0:
        raise ValueError("speed_epsilon must be positive and finite")
    velocity_array = np.asarray(velocity, dtype=float)
    if velocity_array.ndim != 3 or velocity_array.shape[2] != 2:
        raise ValueError("velocity must have shape (T, J, 2)")
    if not np.all(np.isfinite(velocity_array)):
        raise ValueError("velocity must contain only finite values")
    speed = np.hypot(velocity_array[..., 0], velocity_array[..., 1])  # noqa: E501 ⚡ Bolt: np.hypot is faster and safer
    valid = speed > speed_epsilon
    tangent = np.zeros_like(velocity_array)
    tangent[valid] = velocity_array[valid] / speed[valid, None]
    normal = np.zeros_like(tangent)
    normal[..., 0] = -tangent[..., 1]
    normal[..., 1] = tangent[..., 0]
    return PathFrame(speed, tangent, normal, valid, float(speed_epsilon))


def project_forces_onto_path(
    trajectory: JointTransferTrajectory, frame: PathFrame
) -> ForcePathProjection:
    """Project total/drift/control forces onto the velocity-aligned frame."""
    expected_scalar = (trajectory.sample_count, trajectory.joint_count)
    expected_vector = (*expected_scalar, 2)
    if frame.speed.shape != expected_scalar or frame.valid.shape != expected_scalar:
        raise ValueError("path frame sample/joint dimensions must match trajectory")
    if frame.tangent.shape != expected_vector or frame.normal.shape != expected_vector:
        raise ValueError("path frame vectors must match trajectory")

    def project(force: FloatArray, axis: FloatArray) -> FloatArray:
        values = np.einsum("tjd,tjd->tj", force, axis)
        return np.where(frame.valid, values, np.nan)

    result = ForcePathProjection(
        total_along=project(trajectory.force_total, frame.tangent),
        drift_along=project(trajectory.force_drift, frame.tangent),
        control_along=project(trajectory.force_control, frame.tangent),
        total_normal=project(trajectory.force_total, frame.normal),
        drift_normal=project(trajectory.force_drift, frame.normal),
        control_normal=project(trajectory.force_control, frame.normal),
        valid=frame.valid.copy(),
    )
    valid = result.valid
    _require_split(
        "total along-path force",
        result.total_along[valid],
        result.drift_along[valid],
        result.control_along[valid],
    )
    return result


def _cumulative_trapezoid(
    values: FloatArray, time: FloatArray, interval_valid: BoolArray | None = None
) -> FloatArray:
    result = np.zeros_like(values, dtype=float)
    if values.shape[0] < 2:
        return result
    dt_shape = (time.size - 1,) + (1,) * (values.ndim - 1)
    increments = 0.5 * (values[:-1] + values[1:]) * np.diff(time).reshape(dt_shape)
    if interval_valid is not None:
        increments = np.where(interval_valid, increments, 0.0)
    result[1:] = np.cumsum(increments, axis=0)
    return result


def _path_integral(
    values: FloatArray, projection: ForcePathProjection, time: FloatArray
) -> FloatArray:
    interval_valid = projection.valid[:-1] & projection.valid[1:]
    finite_values = np.where(projection.valid, values, 0.0)
    return _cumulative_trapezoid(finite_values, time, interval_valid)


def compute_path_weighted_mean_force(
    trajectory: JointTransferTrajectory,
    frame: PathFrame,
    *,
    path_length_epsilon: float = _DEFAULT_EPSILON,
) -> PathWeightedMeanForce:
    """Return signed path-weighted mean force ``W_linear / L`` per joint.

    Path length and force work use only intervals whose two endpoint samples
    both have a valid velocity-defined tangent.  A joint with no valid path
    length receives NaN mean-force values and ``valid=False`` rather than a
    fabricated zero or an unstable quotient.
    """
    if not np.isfinite(path_length_epsilon) or path_length_epsilon <= 0.0:
        raise ValueError("path_length_epsilon must be positive and finite")
    expected_shape = (trajectory.sample_count, trajectory.joint_count)
    if frame.speed.shape != expected_shape or frame.valid.shape != expected_shape:
        raise ValueError("path frame sample/joint dimensions must match trajectory")
    interval_valid = frame.valid[:-1] & frame.valid[1:]
    path_length_history = _cumulative_trapezoid(
        frame.speed, trajectory.time, interval_valid
    )

    def valid_work(force: FloatArray) -> FloatArray:
        power = np.einsum("tjd,tjd->tj", force, trajectory.velocity)
        return _cumulative_trapezoid(power, trajectory.time, interval_valid)

    work_total = valid_work(trajectory.force_total)[-1]
    work_drift = valid_work(trajectory.force_drift)[-1]
    work_control = valid_work(trajectory.force_control)[-1]
    path_length = path_length_history[-1]
    valid = path_length > path_length_epsilon

    def quotient(work: FloatArray) -> FloatArray:
        mean = np.full(work.shape, np.nan)
        mean[valid] = work[valid] / path_length[valid]
        return mean

    return PathWeightedMeanForce(
        path_length=path_length,
        force_work_total=work_total,
        force_work_drift=work_drift,
        force_work_control=work_control,
        mean_force_total=quotient(work_total),
        mean_force_drift=quotient(work_drift),
        mean_force_control=quotient(work_control),
        valid=valid,
    )


def compute_impulses(
    trajectory: JointTransferTrajectory, projection: ForcePathProjection
) -> ImpulseDecomposition:
    """Compute cumulative vector and signed/magnitude along-path impulses."""
    time = trajectory.time

    def path_terms(
        values: FloatArray,
    ) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray]:
        return (
            _path_integral(values, projection, time),
            _path_integral(np.maximum(values, 0.0), projection, time),
            _path_integral(np.minimum(values, 0.0), projection, time),
            _path_integral(np.abs(values), projection, time),
        )

    total_terms = path_terms(projection.total_along)
    drift_terms = path_terms(projection.drift_along)
    control_terms = path_terms(projection.control_along)
    result = ImpulseDecomposition(
        vector_total=_cumulative_trapezoid(trajectory.force_total, time),
        vector_drift=_cumulative_trapezoid(trajectory.force_drift, time),
        vector_control=_cumulative_trapezoid(trajectory.force_control, time),
        tangent_total_signed=total_terms[0],
        tangent_total_positive=total_terms[1],
        tangent_total_negative=total_terms[2],
        tangent_total_absolute=total_terms[3],
        tangent_drift_signed=drift_terms[0],
        tangent_drift_positive=drift_terms[1],
        tangent_drift_negative=drift_terms[2],
        tangent_drift_absolute=drift_terms[3],
        tangent_control_signed=control_terms[0],
        tangent_control_positive=control_terms[1],
        tangent_control_negative=control_terms[2],
        tangent_control_absolute=control_terms[3],
    )
    _require_split(
        "vector impulse",
        result.vector_total,
        result.vector_drift,
        result.vector_control,
    )
    _require_split(
        "signed tangent impulse",
        result.tangent_total_signed,
        result.tangent_drift_signed,
        result.tangent_control_signed,
    )
    return result


def compute_power_and_work(
    trajectory: JointTransferTrajectory,
) -> PowerWorkDecomposition:
    """Compute pointwise force/couple power and cumulative interface work."""
    force_total = np.einsum("tjd,tjd->tj", trajectory.force_total, trajectory.velocity)
    force_drift = np.einsum("tjd,tjd->tj", trajectory.force_drift, trajectory.velocity)
    force_control = np.einsum(
        "tjd,tjd->tj", trajectory.force_control, trajectory.velocity
    )
    couple_total = trajectory.couple_total * trajectory.angular_velocity
    couple_drift = trajectory.couple_drift * trajectory.angular_velocity
    couple_control = trajectory.couple_control * trajectory.angular_velocity
    total_total = force_total + couple_total
    total_drift = force_drift + couple_drift
    total_control = force_control + couple_control
    time = trajectory.time
    return PowerWorkDecomposition(
        force_power_total=force_total,
        force_power_drift=force_drift,
        force_power_control=force_control,
        couple_power_total=couple_total,
        couple_power_drift=couple_drift,
        couple_power_control=couple_control,
        total_power_total=total_total,
        total_power_drift=total_drift,
        total_power_control=total_control,
        force_work_total=_cumulative_trapezoid(force_total, time),
        force_work_drift=_cumulative_trapezoid(force_drift, time),
        force_work_control=_cumulative_trapezoid(force_control, time),
        couple_work_total=_cumulative_trapezoid(couple_total, time),
        couple_work_drift=_cumulative_trapezoid(couple_drift, time),
        couple_work_control=_cumulative_trapezoid(couple_control, time),
        total_work_total=_cumulative_trapezoid(total_total, time),
        total_work_drift=_cumulative_trapezoid(total_drift, time),
        total_work_control=_cumulative_trapezoid(total_control, time),
    )


def build_phase_masks(
    time: Any, phases: tuple[SwingPhase, ...]
) -> dict[str, BoolArray]:
    """Return deterministic half-open masks that partition the sampled history."""
    time_array = np.asarray(time, dtype=float).reshape(-1)
    if time_array.size == 0 or not np.all(np.isfinite(time_array)):
        raise ValueError("time must be non-empty and finite")
    if time_array.size > 1 and np.any(np.diff(time_array) <= 0.0):
        raise ValueError("time must be strictly increasing")
    if not phases:
        raise ValueError("phases must be non-empty")
    names = [phase.name for phase in phases]
    if len(set(names)) != len(names):
        raise ValueError("phase names must be unique")
    for left, right in zip(phases[:-1], phases[1:], strict=True):
        if not np.isclose(left.end_s, right.start_s, rtol=0.0, atol=1e-12):
            raise ValueError(
                "phase boundaries must be adjacent without gaps or overlap"
            )
    if phases[0].start_s > time_array[0] or phases[-1].end_s < time_array[-1]:
        raise ValueError("phases must exhaust the sampled time domain")

    masks: dict[str, BoolArray] = {}
    for index, phase in enumerate(phases):
        if index == len(phases) - 1:
            mask = (time_array >= phase.start_s) & (time_array <= phase.end_s)
        else:
            mask = (time_array >= phase.start_s) & (time_array < phase.end_s)
        masks[phase.name] = mask
    coverage = np.sum(np.stack(tuple(masks.values())), axis=0)
    if not np.all(coverage == 1):
        raise ValueError("phase masks must be nonoverlapping and exhaustive")
    return masks


def _sample_boundary_index(time: FloatArray, boundary_s: float) -> int:
    matches = np.flatnonzero(np.isclose(time, boundary_s, rtol=0.0, atol=1e-12))
    if matches.size != 1:
        raise ValueError(
            f"phase boundary {boundary_s:.12g} must match exactly one sample time"
        )
    return int(matches[0])


def summarize_phases(
    trajectory: JointTransferTrajectory,
    impulses: ImpulseDecomposition,
    power_work: PowerWorkDecomposition,
    phases: tuple[SwingPhase, ...],
) -> tuple[PhaseTransferSummary, ...]:
    """Summarize impulse and work over adjacent, nonoverlapping intervals.

    Each phase boundary must coincide with one trajectory sample.  Adjacent
    phases share that boundary sample, but their end-minus-start differences
    consume disjoint integration intervals, so no impulse or work is counted
    twice.
    """
    build_phase_masks(trajectory.time, phases)

    def difference(values: FloatArray, start: int, end: int) -> FloatArray:
        if values.shape[0] != trajectory.sample_count:
            raise ValueError("cumulative result sample count must match trajectory")
        return np.asarray(values[end] - values[start], dtype=float)

    summaries: list[PhaseTransferSummary] = []
    previous_end: int | None = None
    for phase in phases:
        start = _sample_boundary_index(trajectory.time, phase.start_s)
        end = _sample_boundary_index(trajectory.time, phase.end_s)
        if end <= start:
            raise ValueError("phase sample boundaries must be strictly increasing")
        if previous_end is not None and start != previous_end:
            raise ValueError("adjacent phases must share exactly one boundary sample")
        summaries.append(
            PhaseTransferSummary(
                phase_name=phase.name,
                start_index=start,
                end_index=end,
                start_time_s=float(trajectory.time[start]),
                end_time_s=float(trajectory.time[end]),
                sample_count=end - start + 1,
                interval_count=end - start,
                vector_impulse_total=difference(impulses.vector_total, start, end),
                vector_impulse_drift=difference(impulses.vector_drift, start, end),
                vector_impulse_control=difference(impulses.vector_control, start, end),
                tangent_impulse_total=difference(
                    impulses.tangent_total_signed, start, end
                ),
                tangent_impulse_drift=difference(
                    impulses.tangent_drift_signed, start, end
                ),
                tangent_impulse_control=difference(
                    impulses.tangent_control_signed, start, end
                ),
                force_work_total=difference(power_work.force_work_total, start, end),
                force_work_drift=difference(power_work.force_work_drift, start, end),
                force_work_control=difference(
                    power_work.force_work_control, start, end
                ),
                couple_work_total=difference(power_work.couple_work_total, start, end),
                couple_work_drift=difference(power_work.couple_work_drift, start, end),
                couple_work_control=difference(
                    power_work.couple_work_control, start, end
                ),
                total_work_total=difference(power_work.total_work_total, start, end),
                total_work_drift=difference(power_work.total_work_drift, start, end),
                total_work_control=difference(
                    power_work.total_work_control, start, end
                ),
            )
        )
        previous_end = end
    return tuple(summaries)


def attribution_shares(
    total: Any,
    drift: Any,
    control: Any,
    *,
    epsilon: float = _DEFAULT_EPSILON,
) -> AttributionShares:
    """Compute signed shares, magnitude shares, and a cancellation index.

    Signed shares are undefined when ``abs(total) <= epsilon``.  Magnitude
    shares and cancellation are undefined when both component magnitudes are
    negligible.  Undefined values are represented by NaN and explicit masks.
    """
    if not np.isfinite(epsilon) or epsilon <= 0.0:
        raise ValueError("epsilon must be positive and finite")
    arrays = [np.asarray(value, dtype=float) for value in (total, drift, control)]
    if not (arrays[0].shape == arrays[1].shape == arrays[2].shape):
        raise ValueError("total, drift, and control must have identical shapes")
    if not all(np.all(np.isfinite(value)) for value in arrays):
        raise ValueError("total, drift, and control must be finite")
    total_array, drift_array, control_array = arrays
    _require_split("total", total_array, drift_array, control_array)

    signed_valid = np.abs(total_array) > epsilon
    magnitude_denominator = np.abs(drift_array) + np.abs(control_array)
    magnitude_valid = magnitude_denominator > epsilon
    signed_drift = np.full(total_array.shape, np.nan)
    signed_control = np.full(total_array.shape, np.nan)
    magnitude_drift = np.full(total_array.shape, np.nan)
    magnitude_control = np.full(total_array.shape, np.nan)
    cancellation = np.full(total_array.shape, np.nan)
    signed_drift[signed_valid] = drift_array[signed_valid] / total_array[signed_valid]
    signed_control[signed_valid] = (
        control_array[signed_valid] / total_array[signed_valid]
    )
    magnitude_drift[magnitude_valid] = (
        np.abs(drift_array[magnitude_valid]) / magnitude_denominator[magnitude_valid]
    )
    magnitude_control[magnitude_valid] = (
        np.abs(control_array[magnitude_valid]) / magnitude_denominator[magnitude_valid]
    )
    raw_cancellation = (
        1.0
        - np.abs(total_array[magnitude_valid]) / magnitude_denominator[magnitude_valid]
    )
    cancellation[magnitude_valid] = np.clip(raw_cancellation, 0.0, 1.0)
    return AttributionShares(
        signed_drift,
        signed_control,
        magnitude_drift,
        magnitude_control,
        cancellation,
        signed_valid,
        magnitude_valid,
    )


__all__ = [
    "AttributionShares",
    "ForcePathProjection",
    "ImpulseDecomposition",
    "JointTransferTrajectory",
    "PathFrame",
    "PathWeightedMeanForce",
    "PhaseTransferSummary",
    "PowerWorkDecomposition",
    "SwingPhase",
    "attribution_shares",
    "build_phase_masks",
    "compute_impulses",
    "compute_path_frame",
    "compute_path_weighted_mean_force",
    "compute_power_and_work",
    "project_forces_onto_path",
    "summarize_phases",
]

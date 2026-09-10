"""Engine-independent golf metrics from explicitly calibrated segment frames.

SI inputs use proper local-to-world rotations. NaN rows represent missing data;
derivatives never cross a missing row. Projected separation and unsigned 3-D
axis separation are distinct measurements, not interchangeable Euler angles.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field, fields
from types import MappingProxyType
from typing import Any, TypeAlias, cast

import numpy as np
from numpy.typing import NDArray
from scipy.spatial.transform import Rotation

Array: TypeAlias = NDArray[np.float64]


def _array(value: Any, shape: tuple[int, ...], name: str) -> Array:
    a = np.array(value, dtype=float, copy=True)
    if a.shape != shape or np.isinf(a).any():
        raise ValueError(f"{name} must have shape {shape} without infinity")
    a.setflags(write=False)
    return a


def _vector(value: Any, name: str, *, unit: bool = False) -> Array:
    a = _array(value, (3,), name)
    if not np.isfinite(a).all() or (unit and not np.isclose(np.linalg.norm(a), 1)):
        raise ValueError(
            f"{name} must be finite" + (" and unit length" if unit else "")
        )
    return a


@dataclass(frozen=True)
class SegmentTrajectory:
    """World origins and local-to-world matrices; membership is never inferred."""

    name: str
    positions: Array
    rotations: Array
    mass_kg: float | None = None
    local_com: Array | None = None
    membership: str = "other"

    def __post_init__(self) -> None:
        if not self.name.strip() or self.membership not in {"body", "club", "other"}:
            raise ValueError("segment needs a name and body/club/other membership")
        n = len(self.positions)
        object.__setattr__(
            self, "positions", _array(self.positions, (n, 3), "positions")
        )
        r = _array(self.rotations, (n, 3, 3), "rotations")
        finite = np.isfinite(r).all(axis=(1, 2))
        partial = np.isfinite(r).any(axis=(1, 2)) & ~finite
        if (
            partial.any()
            or not np.allclose(
                r[finite].transpose(0, 2, 1) @ r[finite], np.eye(3), atol=1e-7
            )
            or not np.allclose(np.linalg.det(r[finite]), 1, atol=1e-7)
        ):
            raise ValueError(
                "rotations must be proper orthogonal matrices or all-NaN rows"
            )
        object.__setattr__(self, "rotations", r)
        if self.mass_kg is not None and (
            not np.isfinite(self.mass_kg) or self.mass_kg <= 0
        ):
            raise ValueError("mass_kg must be finite and positive")
        if self.local_com is not None:
            object.__setattr__(self, "local_com", _vector(self.local_com, "local_com"))

    def world_point(self, local_point: Array) -> Array:
        """Transform a calibrated local point without filling missing samples."""
        return self.positions + np.einsum("nij,j->ni", self.rotations, local_point)


@dataclass(frozen=True)
class GolfTrajectory:
    """Explicit anatomical axes and event semantics shared by all model adapters.

    ``expected_body_segments`` declares the modeled body's complete inventory;
    it does not certify that the model represents a complete human body.
    Lateral axes point consistently across pelvis and thorax. Positive projected
    separation is pelvis-to-thorax rotation about ``up_axis_world``.
    """

    times: Array
    segments: Mapping[str, SegmentTrajectory]
    source: str
    world_frame: str
    pelvis: str = "pelvis"
    thorax: str = "thorax"
    club: str = "club"
    expected_body_segments: tuple[str, ...] = ()
    transition_index: int | None = None
    impact_index: int | None = None
    event_name: str | None = None
    shaft_axis_local: Array | None = None
    clubhead_local: Array | None = None
    up_axis_world: Array = field(default_factory=lambda: np.array([0.0, 0.0, 1.0]))
    pelvis_lateral_local: Array = field(
        default_factory=lambda: np.array([1.0, 0.0, 0.0])
    )
    thorax_lateral_local: Array = field(
        default_factory=lambda: np.array([1.0, 0.0, 0.0])
    )
    calibration_id: str = "unspecified"

    def __post_init__(self) -> None:
        times = np.array(self.times, dtype=float, copy=True)
        if (
            times.ndim != 1
            or len(times) < 2
            or not np.isfinite(times).all()
            or np.any(np.diff(times) <= 0)
        ):
            raise ValueError(
                "times must be finite, strictly increasing, with at least two samples"
            )
        times.setflags(write=False)
        object.__setattr__(self, "times", times)
        if (
            any(
                not isinstance(v, str) or not v.strip()
                for v in (self.source, self.world_frame, self.calibration_id)
            )
            or not self.segments
        ):
            raise ValueError("source, world_frame and segments are required")
        for name, segment in self.segments.items():
            if name != segment.name or len(segment.positions) != len(times):
                raise ValueError("segment keys/names and sample counts must agree")
        object.__setattr__(self, "segments", MappingProxyType(dict(self.segments)))
        names = tuple(self.expected_body_segments)
        if len(set(names)) != len(names) or any(not name.strip() for name in names):
            raise ValueError(
                "expected_body_segments must contain unique non-empty names"
            )
        object.__setattr__(self, "expected_body_segments", names)
        for name in ("up_axis_world", "pelvis_lateral_local", "thorax_lateral_local"):
            object.__setattr__(
                self, name, _vector(getattr(self, name), name, unit=True)
            )
        for name in ("shaft_axis_local", "clubhead_local"):
            if getattr(self, name) is not None:
                object.__setattr__(
                    self,
                    name,
                    _vector(getattr(self, name), name, unit=name == "shaft_axis_local"),
                )
        for index in (self.transition_index, self.impact_index):
            if index is not None and (
                isinstance(index, bool)
                or not isinstance(index, (int, np.integer))
                or not 0 <= index < len(times)
            ):
                raise ValueError("event index must be an integer within times")
        if self.transition_index is not None and (
            not self.event_name or not self.event_name.strip()
        ):
            raise ValueError("event_name must explicitly identify the transition event")
        if (
            self.transition_index is not None
            and self.impact_index is not None
            and self.transition_index >= self.impact_index
        ):
            raise ValueError("transition must precede impact")


@dataclass(frozen=True)
class MetricChannel:
    """One scalar or xyz measurement; angles and rates are always SI."""

    values: Array
    unit: str
    definition: str
    frame: str = "world"
    label: str = ""


@dataclass(frozen=True)
class GolfMetrics:
    """Plot-ready channels with explicit unsupported-data diagnostics."""

    times: Array
    channels: Mapping[str, MetricChannel]
    summaries: Mapping[str, float]
    unavailable: Mapping[str, str]
    source: str
    world_frame: str
    events: Mapping[str, int]
    calibration_id: str = "unspecified"

    def to_dict(self) -> dict[str, Any]:
        """Serialize missing measurements as null, never nonstandard JSON NaN."""

        def values(a: Array) -> Any:
            # Object dtype makes the nullable JSON representation explicit to mypy.
            result = a.astype(object)
            result[~np.asarray(np.isfinite(a), dtype=bool)] = None
            return result.tolist()

        return {
            "times": self.times.tolist(),
            "channels": {
                key: {
                    "values": values(channel.values),
                    "unit": channel.unit,
                    "definition": channel.definition,
                    "frame": channel.frame,
                    "label": channel.label or key.replace("_", " ").title(),
                }
                for key, channel in self.channels.items()
            },
            "summaries": {
                k: v if np.isfinite(v) else None for k, v in self.summaries.items()
            },
            "unavailable": dict(self.unavailable),
            "events": dict(self.events),
            "provenance": {
                "source": self.source,
                "world_frame": self.world_frame,
                "convention": "calibrated-segment-axes; SI; local-to-world",
                "calibration_id": self.calibration_id,
            },
        }


def _runs(valid: NDArray[np.bool_]) -> list[Array]:
    indices = np.flatnonzero(valid)
    return list(np.split(indices, np.where(np.diff(indices) != 1)[0] + 1))


def gap_derivative(times: Array, values: Array) -> Array:
    """Differentiate contiguous finite runs using actual nonuniform timestamps."""
    out = np.full_like(values, np.nan, dtype=float)
    valid = np.asarray(
        np.isfinite(values) if values.ndim == 1 else np.isfinite(values).all(axis=1),
        dtype=bool,
    )
    for run in _runs(valid):
        if len(run) >= 2:
            out[run] = np.gradient(values[run], times[run], axis=0)
    return out


def angular_velocity(times: Array, rotations: Array) -> Array:
    """World omega from SO(3) increments, with nonuniform central averaging.

    Samples must resolve rotations below pi per interval. At exactly pi the
    directed increment is ambiguous and its adjacent rates remain unavailable.
    """
    n = len(times)
    edges = np.full((n - 1, 3), np.nan)
    valid = np.asarray(np.isfinite(rotations).all(axis=(1, 2)), dtype=bool)
    pairs = np.flatnonzero(valid[:-1] & valid[1:])
    if pairs.size:
        delta = rotations[pairs + 1] @ rotations[pairs].transpose(0, 2, 1)
        rv = Rotation.from_matrix(delta).as_rotvec()
        rv[np.isclose(np.linalg.norm(rv, axis=1), np.pi, atol=1e-8)] = np.nan
        edges[pairs] = rv / np.diff(times)[pairs, None]
    result = np.full((n, 3), np.nan)
    for run in _runs(valid):
        if len(run) < 2:
            continue
        result[run[0]] = edges[run[0]]
        result[run[-1]] = edges[run[-1] - 1]
        for i in run[1:-1]:
            before, after = times[i] - times[i - 1], times[i + 1] - times[i]
            result[i] = (after * edges[i - 1] + before * edges[i]) / (before + after)
    return result


def _separation(
    t: GolfTrajectory,
    channels: dict[str, MetricChannel],
    summaries: dict[str, float],
    unavailable: dict[str, str],
) -> None:
    if t.pelvis not in t.segments or t.thorax not in t.segments:
        unavailable["x_factor_projected"] = (
            "Calibrated pelvis and thorax rotations required"
        )
        unavailable["x_factor_3d"] = unavailable["x_factor_projected"]
        return
    p = np.einsum("nij,j->ni", t.segments[t.pelvis].rotations, t.pelvis_lateral_local)
    q = np.einsum("nij,j->ni", t.segments[t.thorax].rotations, t.thorax_lateral_local)
    spatial = np.asarray(
        np.arccos(np.clip(np.einsum("ni,ni->n", p, q), -1, 1)), dtype=float
    )
    p = p - (p @ t.up_axis_world)[:, None] * t.up_axis_world
    q = q - (q @ t.up_axis_world)[:, None] * t.up_axis_world
    angle = np.asarray(
        np.arctan2(np.cross(p, q) @ t.up_axis_world, np.einsum("ni,ni->n", p, q)),
        dtype=float,
    )
    angle[(np.linalg.norm(p, axis=1) < 1e-10) | (np.linalg.norm(q, axis=1) < 1e-10)] = (
        np.nan
    )
    for run in _runs(np.isfinite(angle)):
        angle[run] = np.unwrap(angle[run])
    channels["x_factor_projected"] = MetricChannel(
        angle,
        "rad",
        "Signed pelvis-to-thorax lateral-axis separation projected perpendicular to declared world up",
    )
    channels["x_factor_3d"] = MetricChannel(
        spatial,
        "rad",
        "Unsigned 3-D included angle between calibrated pelvis and thorax lateral axes",
    )
    channels["x_factor_rate"] = MetricChannel(
        gap_derivative(t.times, angle),
        "rad/s",
        "Derivative of unwrapped projected separation; not X-factor stretch",
    )
    if t.transition_index is None or t.impact_index is None:
        unavailable["x_factor_stretch"] = (
            "Explicit transition and impact events required"
        )
        return
    window = angle[t.transition_index : t.impact_index + 1]
    if not np.isfinite(window).all():
        unavailable["x_factor_stretch"] = (
            "Missing separation in transition-to-impact window"
        )
        return
    summaries["x_factor_stretch"] = float(np.max(np.abs(window)) - abs(window[0]))


def _com(
    t: GolfTrajectory, channels: dict[str, MetricChannel], unavailable: dict[str, str]
) -> None:
    body = [s for s in t.segments.values() if s.membership == "body"]
    club = [s for s in t.segments.values() if s.membership == "club"]
    complete = bool(t.expected_body_segments) and {s.name for s in body} == set(
        t.expected_body_segments
    )
    for key, members in (("body_com", body), ("body_club_com", body + club)):
        if not complete or (key == "body_club_com" and not club):
            unavailable[key] = (
                "Complete declared body inventory required; body+club additionally requires club membership"
            )
        elif any(s.mass_kg is None or s.local_com is None for s in members):
            unavailable[key] = (
                "Every included segment requires positive mass and calibrated local COM"
            )
        else:
            mass = sum(cast(float, s.mass_kg) for s in members)
            weighted = np.zeros_like(members[0].positions, dtype=float)
            for segment in members:
                weighted += segment.world_point(cast(Array, segment.local_com)) * cast(
                    float, segment.mass_kg
                )
            channels[key] = MetricChannel(
                weighted / mass,
                "m",
                "Mass-weighted COM of complete declared model inventory: "
                + ", ".join(s.name for s in members),
            )


def compute_golf_metrics(t: GolfTrajectory) -> GolfMetrics:
    """Compute available channels once for dynamics, kinematics and plotting."""
    channels: dict[str, MetricChannel] = {}
    summaries: dict[str, float] = {}
    unavailable: dict[str, str] = {}
    for name, segment in t.segments.items():
        prefix = f"segment.{name}."
        channels[prefix + "position"] = MetricChannel(
            segment.positions, "m", "Declared segment origin in world coordinates"
        )
        omega = angular_velocity(t.times, segment.rotations)
        channels[prefix + "angular_velocity_world"] = MetricChannel(
            omega,
            "rad/s",
            "Physical angular velocity from SO(3) increments; sampling must resolve < pi per interval",
        )
        channels[prefix + "angular_velocity_local"] = MetricChannel(
            np.einsum("nji,nj->ni", segment.rotations, omega),
            "rad/s",
            "Physical angular velocity expressed in calibrated segment coordinates",
            "segment_local",
        )
        if segment.local_com is not None:
            channels[prefix + "com"] = MetricChannel(
                segment.world_point(segment.local_com),
                "m",
                "Calibrated segment COM in world coordinates",
            )
        else:
            unavailable[prefix + "com"] = "Calibrated local COM required"
    _separation(t, channels, summaries, unavailable)
    _com(t, channels, unavailable)
    if t.club in t.segments and t.shaft_axis_local is not None:
        omega = channels[f"segment.{t.club}.angular_velocity_local"].values
        channels["shaft_twist_velocity"] = MetricChannel(
            omega @ t.shaft_axis_local,
            "rad/s",
            "Club angular velocity projected on directed local shaft axis; not elastic torsional strain rate",
            "shaft_axis",
        )
    else:
        unavailable["shaft_twist_velocity"] = (
            "Club rotation and directed calibrated shaft axis required"
        )
    if t.club in t.segments and t.clubhead_local is not None:
        position = t.segments[t.club].world_point(t.clubhead_local)
        channels["clubhead_position"] = MetricChannel(
            position, "m", "Calibrated clubhead point in world coordinates"
        )
        channels["clubhead_speed"] = MetricChannel(
            np.linalg.norm(gap_derivative(t.times, position), axis=1),
            "m/s",
            "Speed of calibrated clubhead point",
        )
    else:
        unavailable["clubhead_speed"] = (
            "Club pose and calibrated local clubhead point required"
        )
    events = {}
    if t.transition_index is not None:
        events[str(t.event_name)] = t.transition_index
    if t.impact_index is not None:
        events["impact"] = t.impact_index
    return GolfMetrics(
        t.times,
        channels,
        summaries,
        unavailable,
        t.source,
        t.world_frame,
        events,
        t.calibration_id,
    )


def golf_trajectory_from_dict(payload: Mapping[str, Any]) -> GolfTrajectory:
    """Parse the canonical SI wire contract, rejecting unknown/missing fields.

    Segment names are dictionary keys. JSON nulls in measured arrays become
    explicit NaN gaps; null masses/COMs remain unavailable metadata.
    """
    if not isinstance(payload, Mapping):
        raise ValueError("trajectory must be an object")
    known = {f.name for f in fields(GolfTrajectory)}
    unknown = set(payload) - known
    if unknown:
        raise ValueError(f"Unknown trajectory fields: {sorted(unknown)}")
    required = {"times", "segments", "source", "world_frame"}
    if not required <= payload.keys():
        raise ValueError(
            f"Missing trajectory fields: {sorted(required - payload.keys())}"
        )
    raw = payload["segments"]
    if not isinstance(raw, Mapping):
        raise ValueError("segments must be an object keyed by segment name")
    segments = {}
    allowed = {f.name for f in fields(SegmentTrajectory)} - {"name"}
    for name, values in raw.items():
        if not isinstance(name, str) or not isinstance(values, Mapping):
            raise ValueError("segments must contain named objects")
        if set(values) - allowed:
            raise ValueError(
                f"Unknown segment fields for {name}: {sorted(set(values) - allowed)}"
            )
        if not {"positions", "rotations"} <= values.keys():
            raise ValueError(f"Missing positions or rotations for {name}")
        try:
            segments[name] = SegmentTrajectory(name=name, **values)
        except (TypeError, AttributeError) as exc:
            raise ValueError(f"Invalid segment {name}: {exc}") from exc
    try:
        return GolfTrajectory(**{**payload, "segments": segments})
    except (TypeError, AttributeError) as exc:
        raise ValueError(f"Invalid trajectory: {exc}") from exc

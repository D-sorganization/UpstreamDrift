"""Per-segment angular momentum about the whole-body COM (world frame, data only)."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Literal, Protocol, runtime_checkable

import numpy as np
from numpy.typing import ArrayLike, NDArray

REFERENCE_POINT = "whole_body_com"
FRAME_WORLD: Literal["world"] = "world"
UNITS_ANGMOM = "N·m·s"
Vec3 = NDArray[np.floating]


def _as_vec3(value: ArrayLike, *, name: str) -> Vec3:
    arr = np.asarray(value, dtype=float)
    if arr.shape != (3,):
        raise ValueError(f"{name} must be a 3-vector, got shape {arr.shape}")
    if not np.isfinite(arr).all():
        raise ValueError(f"{name} must be finite")
    return arr


def inertia_world_from_principal(
    inertia_principal: ArrayLike, rot_body_to_world: ArrayLike
) -> NDArray[np.floating]:
    """Map diagonal principal inertia to a world-frame 3×3 tensor."""
    i_diag = np.asarray(inertia_principal, dtype=float)
    if i_diag.shape != (3,):
        raise ValueError("inertia_principal must have shape (3,)")
    rot = np.asarray(rot_body_to_world, dtype=float).reshape(3, 3)
    if not np.isfinite(i_diag).all() or not np.isfinite(rot).all():
        raise ValueError("inertia and rotation must be finite")
    return rot @ np.diag(i_diag) @ rot.T


def spin_angular_momentum_world(
    inertia_principal: ArrayLike,
    rot_body_to_world: ArrayLike,
    ang_vel_world: ArrayLike,
) -> Vec3:
    """Spin term I ω about the segment COM, expressed in the world frame."""
    i_world = inertia_world_from_principal(inertia_principal, rot_body_to_world)
    omega = _as_vec3(ang_vel_world, name="ang_vel_world")
    return i_world @ omega


def body_angular_momentum_about_point(
    *,
    com_world: ArrayLike,
    com_vel_world: ArrayLike,
    ang_vel_world: ArrayLike,
    inertia_principal: ArrayLike,
    rot_body_to_world: ArrayLike,
    mass: float,
    reference_com_world: ArrayLike,
    reference_com_vel_world: ArrayLike,
) -> tuple[Vec3, Vec3]:
    """Return (H_i about reference COM, spin-only I_i ω_i about body COM)."""
    if not math.isfinite(mass) or mass <= 0.0:
        raise ValueError("mass must be a finite positive scalar")
    r_i = _as_vec3(com_world, name="com_world")
    v_i = _as_vec3(com_vel_world, name="com_vel_world")
    r_c = _as_vec3(reference_com_world, name="reference_com_world")
    v_c = _as_vec3(reference_com_vel_world, name="reference_com_vel_world")
    spin = spin_angular_momentum_world(
        inertia_principal, rot_body_to_world, ang_vel_world
    )
    orbital = mass * np.cross(r_i - r_c, v_i - v_c)
    return orbital + spin, spin


def sum_vectors(vectors: Sequence[Vec3 | None]) -> Vec3 | None:
    """Sum qualified 3-vectors; return None when every entry is None."""
    parts = [v for v in vectors if v is not None]
    if not parts:
        return None
    total = np.zeros(3, dtype=float)
    for vec in parts:
        total += vec
    return total


def aggregate_group(
    values: Mapping[str, Vec3 | None], member_names: Sequence[str]
) -> Vec3 | None:
    """Group aggregate = sum of member contributions (missing members omitted)."""
    return sum_vectors([values.get(name) for name in member_names])


def orbital_fraction(contribution: Vec3, spin: Vec3) -> float | None:
    """‖H_orbital‖ / ‖H_total‖ with orbital = contribution − spin; None if ‖H‖ = 0."""
    total = float(np.linalg.norm(contribution))
    if total <= 0.0:
        return None
    orbital = contribution - spin
    return float(np.linalg.norm(orbital) / total)


@dataclass(frozen=True)
class SegmentGroupCatalog:
    """Named aggregates (legs, pelvis, torso, arms, club) over body names."""

    legs: frozenset[str]
    pelvis: frozenset[str]
    torso: frozenset[str]
    arms: frozenset[str]
    club: frozenset[str]

    def as_mapping(self) -> Mapping[str, frozenset[str]]:
        return {
            "legs": self.legs,
            "pelvis": self.pelvis,
            "torso": self.torso,
            "arms": self.arms,
            "club": self.club,
        }


@dataclass(frozen=True)
class SegmentAngularMomentumFrame:
    """One sample of per-segment angular momentum with metadata."""

    time_s: float
    reference_point: str
    frame: Literal["world"]
    units: str
    source: str
    contribution_nm_s: Mapping[str, tuple[float, float, float] | None]
    spin_only_nm_s: Mapping[str, tuple[float, float, float] | None]
    groups_nm_s: Mapping[str, tuple[float, float, float] | None]
    whole_body_nm_s: tuple[float, float, float] | None

    def __post_init__(self) -> None:
        if not math.isfinite(self.time_s):
            raise ValueError("time_s must be finite")
        if self.reference_point != REFERENCE_POINT:
            raise ValueError(f"reference_point must be {REFERENCE_POINT!r}")
        if self.frame != FRAME_WORLD:
            raise ValueError("frame must be 'world'")
        if self.units != UNITS_ANGMOM:
            raise ValueError(f"units must be {UNITS_ANGMOM!r}")
        if not isinstance(self.source, str) or not self.source.strip():
            raise ValueError("source must be a non-empty string")
        self._validate_vector_map("contribution_nm_s", self.contribution_nm_s)
        self._validate_vector_map("spin_only_nm_s", self.spin_only_nm_s)
        self._validate_vector_map("groups_nm_s", self.groups_nm_s)
        if self.whole_body_nm_s is not None:
            _as_vec3(self.whole_body_nm_s, name="whole_body_nm_s")

    @staticmethod
    def _validate_vector_map(
        label: str, mapping: Mapping[str, tuple[float, float, float] | None]
    ) -> None:
        if not isinstance(mapping, Mapping):
            raise TypeError(f"{label} must be a mapping")
        for key, value in mapping.items():
            if not isinstance(key, str):
                raise TypeError(f"{label} keys must be strings")
            if value is not None:
                _as_vec3(value, name=f"{label}[{key}]")

    def to_dict(self) -> dict[str, Any]:
        """JSON-safe wire schema; unavailable values encode as null."""
        return {
            "time_s": self.time_s,
            "reference_point": self.reference_point,
            "frame": self.frame,
            "units": self.units,
            "source": self.source,
            "contribution_nm_s": {
                name: None if value is None else [float(x) for x in value]
                for name, value in self.contribution_nm_s.items()
            },
            "spin_only_nm_s": {
                name: None if value is None else [float(x) for x in value]
                for name, value in self.spin_only_nm_s.items()
            },
            "groups_nm_s": {
                name: None if value is None else [float(x) for x in value]
                for name, value in self.groups_nm_s.items()
            },
            "whole_body_nm_s": (
                None
                if self.whole_body_nm_s is None
                else [float(x) for x in self.whole_body_nm_s]
            ),
        }


def build_segment_angular_momentum_frame(
    *,
    time_s: float,
    source: str,
    contributions: Mapping[str, Vec3 | None],
    spin_only: Mapping[str, Vec3 | None],
    catalog: SegmentGroupCatalog,
) -> SegmentAngularMomentumFrame:
    """Assemble group sums and whole-body total from per-body vectors."""
    groups: dict[str, Vec3 | None] = {}
    for name, members in catalog.as_mapping().items():
        groups[name] = aggregate_group(contributions, tuple(members))
    whole = sum_vectors([contributions[name] for name in contributions])
    return SegmentAngularMomentumFrame(
        time_s=time_s,
        reference_point=REFERENCE_POINT,
        frame=FRAME_WORLD,
        units=UNITS_ANGMOM,
        source=source,
        contribution_nm_s=MappingProxyType(_tupleize_map(contributions)),
        spin_only_nm_s=MappingProxyType(_tupleize_map(spin_only)),
        groups_nm_s=MappingProxyType(_tupleize_map(groups)),
        whole_body_nm_s=_tupleize(whole),
    )


def _tupleize(vec: Vec3 | None) -> tuple[float, float, float] | None:
    if vec is None:
        return None
    return (float(vec[0]), float(vec[1]), float(vec[2]))


def _tupleize_map(
    mapping: Mapping[str, Vec3 | None],
) -> dict[str, tuple[float, float, float] | None]:
    return {key: _tupleize(value) for key, value in mapping.items()}


@runtime_checkable
class SegmentAngularMomentumProvider(Protocol):
    """Optional engine capability for segment angular-momentum samples."""

    def get_segment_angular_momentum(self) -> SegmentAngularMomentumFrame | None:
        """Return the current frame, or None when unqualified."""
        ...


def read_segment_angular_momentum_frame(
    provider: object, time_s: float
) -> dict[str, Any] | None:
    """Read only declared provider capability; reject stale or untyped results."""
    if not math.isfinite(time_s):
        raise ValueError("time_s must be finite")
    if not isinstance(provider, SegmentAngularMomentumProvider):
        return None
    frame = provider.get_segment_angular_momentum()
    if frame is None:
        return None
    if not isinstance(frame, SegmentAngularMomentumFrame):
        raise TypeError("provider must return SegmentAngularMomentumFrame or None")
    if not math.isclose(frame.time_s, time_s, rel_tol=0, abs_tol=1e-12):
        return None
    return frame.to_dict()

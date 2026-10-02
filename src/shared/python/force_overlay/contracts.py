"""Core engine-agnostic force/torque overlay data contracts and provider seam (ADR-0052)."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from enum import Enum
import io
import math
import re
from types import MappingProxyType
from typing import (
    Any,
    BinaryIO,
    ClassVar,
    Mapping,
    Protocol,
    Sequence,
    runtime_checkable,
)

import numpy as np

from src.shared.python.body_part_viz.axial_loads import AxialLoadFrame
from src.shared.python.motion_matching.force_torque import SpatialWrench, validate_vec3

_LABEL_PATTERN = re.compile(r"^[a-z_]+:[A-Za-z0-9_.:-]+$")
_DEFAULT_UNITS: Mapping[str, str] = MappingProxyType(
    {"force": "N", "torque": "N*m", "length": "m"}
)


class WrenchKind(str, Enum):
    """Semantic category of force/torque interaction acting on a rigid body."""

    JOINT_ACTUATOR = "joint_actuator"
    JOINT_REACTION = "joint_reaction"
    CONTACT = "contact"
    GRIP = "grip"
    EXTERNAL = "external"
    GRAVITY = "gravity"
    MUSCLE = "muscle"


@dataclass(frozen=True)
class OverlayWrench:
    """A physical wrench acting on a body in the world frame at a point."""

    kind: WrenchKind
    label: str
    body: str
    point_m: tuple[float, float, float]
    force_n: tuple[float, float, float] | None = None
    torque_nm: tuple[float, float, float] | None = None
    source: str = ""

    world_frame: ClassVar[str] = "world_Zup"
    direction_convention: ClassVar[str] = "applied_to_body"

    def __post_init__(self) -> None:
        if isinstance(self.kind, str) and not isinstance(self.kind, WrenchKind):
            object.__setattr__(self, "kind", WrenchKind(self.kind))
        if not isinstance(self.label, str) or not _LABEL_PATTERN.match(self.label):
            raise ValueError(
                f"label must match '^[a-z_]+:[A-Za-z0-9_.:-]+$', got {self.label!r}"
            )
        if not self.body or not isinstance(self.body, str):
            raise ValueError("body must be a non-empty string")
        if not self.source or not isinstance(self.source, str):
            raise ValueError("source must be a non-empty string")

        object.__setattr__(self, "point_m", validate_vec3(self.point_m, "point_m"))
        if self.force_n is not None:
            object.__setattr__(self, "force_n", validate_vec3(self.force_n, "force_n"))
        if self.torque_nm is not None:
            object.__setattr__(
                self, "torque_nm", validate_vec3(self.torque_nm, "torque_nm")
            )

        if self.force_n is None and self.torque_nm is None:
            raise ValueError(
                "At least one of force_n or torque_nm must be provided (neither may be fabricated)"
            )

    def to_spatial_wrench(self) -> SpatialWrench:
        """Convert to SpatialWrench when both force and torque halves are present."""
        if self.force_n is None or self.torque_nm is None:
            missing = "force_n" if self.force_n is None else "torque_nm"
            raise ValueError(f"Cannot convert to SpatialWrench: missing {missing}")
        return SpatialWrench(
            application_frame="world",
            direction_convention=self.direction_convention,
            point_m=self.point_m,
            force_n=self.force_n,
            torque_nm=self.torque_nm,
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize wrench to dictionary matching wire schema."""
        return {
            "kind": self.kind.value,
            "label": self.label,
            "body": self.body,
            "point_m": list(self.point_m),
            "force_n": list(self.force_n) if self.force_n is not None else None,
            "torque_nm": list(self.torque_nm) if self.torque_nm is not None else None,
            "source": self.source,
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> OverlayWrench:
        """Deserialize from dictionary, rejecting unknown fields."""
        allowed = {"kind", "label", "body", "point_m", "force_n", "torque_nm", "source"}
        unknown = set(d.keys()) - allowed
        if unknown:
            raise ValueError(f"Unknown keys in OverlayWrench dict: {sorted(unknown)}")
        return cls(
            kind=WrenchKind(d["kind"]),
            label=d["label"],
            body=d["body"],
            point_m=tuple(d["point_m"]),
            force_n=tuple(d["force_n"]) if d.get("force_n") is not None else None,
            torque_nm=tuple(d["torque_nm"]) if d.get("torque_nm") is not None else None,
            source=d.get("source", ""),
        )


@dataclass(frozen=True)
class ForceTorqueFrame:
    """An immutable snapshot of forces, torques, and optional axial loads at a timestamp."""

    time_s: float
    engine: str
    wrenches: tuple[OverlayWrench, ...] = ()
    axial_loads: AxialLoadFrame | None = None
    world_frame: str = "world_Zup"
    units: Mapping[str, str] = field(default_factory=lambda: _DEFAULT_UNITS)

    def __post_init__(self) -> None:
        if not math.isfinite(self.time_s):
            raise ValueError(f"time_s must be a finite number, got {self.time_s}")
        if not self.engine or not isinstance(self.engine, str):
            raise ValueError("engine must be a non-empty string")

        w_tuple = tuple(self.wrenches)
        labels = [w.label for w in w_tuple]
        duplicates = [lbl for lbl, count in Counter(labels).items() if count > 1]
        if duplicates:
            raise ValueError(f"Duplicate wrench labels found: {duplicates}")
        object.__setattr__(self, "wrenches", w_tuple)

        if self.axial_loads is not None:
            if not isinstance(self.axial_loads, AxialLoadFrame):
                raise TypeError(
                    "axial_loads must be an AxialLoadFrame instance or None"
                )
            if not math.isclose(
                self.axial_loads.time_s, self.time_s, rel_tol=0, abs_tol=1e-12
            ):
                raise ValueError(
                    f"axial_loads.time_s ({self.axial_loads.time_s}) must match frame time_s ({self.time_s}) within 1e-12"
                )

        if not isinstance(self.units, MappingProxyType):
            object.__setattr__(self, "units", MappingProxyType(dict(self.units)))

    def by_kind(self, kind: WrenchKind | str) -> tuple[OverlayWrench, ...]:
        """Return all wrenches matching the requested semantic kind."""
        target = WrenchKind(kind) if isinstance(kind, str) else kind
        return tuple(w for w in self.wrenches if w.kind == target)

    def to_dict(self) -> dict[str, Any]:
        """Serialize frame to dictionary under schema 'force-torque-frame-v1'."""
        return {
            "schema_version": "force-torque-frame-v1",
            "time_s": self.time_s,
            "engine": self.engine,
            "world_frame": self.world_frame,
            "units": dict(self.units),
            "axial_loads": self.axial_loads.to_dict()
            if self.axial_loads is not None
            else None,
            "wrenches": [w.to_dict() for w in self.wrenches],
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> ForceTorqueFrame:
        """Deserialize frame from dictionary, enforcing schema version and strict keys."""
        allowed = {
            "schema_version",
            "time_s",
            "engine",
            "world_frame",
            "units",
            "axial_loads",
            "wrenches",
        }
        unknown = set(d.keys()) - allowed
        if unknown:
            raise ValueError(
                f"Unknown keys in ForceTorqueFrame dict: {sorted(unknown)}"
            )
        if d.get("schema_version") != "force-torque-frame-v1":
            raise ValueError(
                f"Expected schema_version 'force-torque-frame-v1', got {d.get('schema_version')!r}"
            )

        axial_raw = d.get("axial_loads")
        axial_frame = None
        if axial_raw is not None:
            axial_frame = AxialLoadFrame(
                time_s=float(axial_raw["time_s"]),
                values_n=dict(axial_raw["values_n"]),
                source=axial_raw.get("source", ""),
            )

        wrenches = tuple(OverlayWrench.from_dict(w) for w in d.get("wrenches", ()))
        return cls(
            time_s=float(d["time_s"]),
            engine=str(d["engine"]),
            wrenches=wrenches,
            axial_loads=axial_frame,
            world_frame=d.get("world_frame", "world_Zup"),
            units=d.get("units", _DEFAULT_UNITS),
        )


@dataclass(frozen=True)
class ForceTorqueSeries:
    """An immutable time-indexed series of ForceTorqueFrames from a single engine."""

    frames: tuple[ForceTorqueFrame, ...]
    engine: str

    def __post_init__(self) -> None:
        f_tuple = tuple(self.frames)
        object.__setattr__(self, "frames", f_tuple)
        for i, f in enumerate(f_tuple):
            if f.engine != self.engine:
                raise ValueError(
                    f"Frame engine mismatch at index {i}: {f.engine} != {self.engine}"
                )
            if i > 0 and f.time_s <= f_tuple[i - 1].time_s:
                raise ValueError(
                    f"Frame times must be strictly increasing: {f_tuple[i - 1].time_s} >= {f.time_s}"
                )

    def frame_at(self, t: float, max_gap_s: float) -> ForceTorqueFrame | None:
        """Sample or linearly interpolate a frame at time t within max_gap_s."""
        if not self.frames:
            return None
        if t < self.frames[0].time_s or t > self.frames[-1].time_s:
            return None

        # Exact match
        for f in self.frames:
            if math.isclose(f.time_s, t, rel_tol=0, abs_tol=1e-12):
                return f

        # Locate neighbours
        idx = 0
        while idx < len(self.frames) - 1 and self.frames[idx + 1].time_s < t:
            idx += 1
        f0 = self.frames[idx]
        f1 = self.frames[idx + 1]

        gap = f1.time_s - f0.time_s
        if gap > max_gap_s:
            return None

        alpha = (t - f0.time_s) / gap
        om_alpha = 1.0 - alpha

        # Interpolate matching wrenches
        w1_by_label = {w.label: w for w in f1.wrenches}
        interp_wrenches: list[OverlayWrench] = []
        for w0 in f0.wrenches:
            w1 = w1_by_label.get(w0.label)
            if w1 is None:
                continue

            # Interpolate point
            p = (
                om_alpha * w0.point_m[0] + alpha * w1.point_m[0],
                om_alpha * w0.point_m[1] + alpha * w1.point_m[1],
                om_alpha * w0.point_m[2] + alpha * w1.point_m[2],
            )

            # Force half (only if present in both)
            f_n = None
            if w0.force_n is not None and w1.force_n is not None:
                f_n = (
                    om_alpha * w0.force_n[0] + alpha * w1.force_n[0],
                    om_alpha * w0.force_n[1] + alpha * w1.force_n[1],
                    om_alpha * w0.force_n[2] + alpha * w1.force_n[2],
                )

            # Torque half (only if present in both)
            t_nm = None
            if w0.torque_nm is not None and w1.torque_nm is not None:
                t_nm = (
                    om_alpha * w0.torque_nm[0] + alpha * w1.torque_nm[0],
                    om_alpha * w0.torque_nm[1] + alpha * w1.torque_nm[1],
                    om_alpha * w0.torque_nm[2] + alpha * w1.torque_nm[2],
                )

            if f_n is None and t_nm is None:
                continue

            interp_wrenches.append(
                OverlayWrench(
                    kind=w0.kind,
                    label=w0.label,
                    body=w0.body,
                    point_m=p,
                    force_n=f_n,
                    torque_nm=t_nm,
                    source=w0.source,
                )
            )

        # Interpolate axial loads if present in both
        interp_axial = None
        if f0.axial_loads is not None and f1.axial_loads is not None:
            v0 = f0.axial_loads.values_n
            v1 = f1.axial_loads.values_n
            interp_values: dict[str, float | None] = {}
            for k in set(v0.keys()).union(v1.keys()):
                val0 = v0.get(k)
                val1 = v1.get(k)
                if val0 is not None and val1 is not None:
                    interp_values[k] = om_alpha * val0 + alpha * val1
                else:
                    interp_values[k] = None
            interp_axial = AxialLoadFrame(
                time_s=t,
                values_n=interp_values,
                source=f0.axial_loads.source,
            )

        return ForceTorqueFrame(
            time_s=t,
            engine=self.engine,
            wrenches=tuple(interp_wrenches),
            axial_loads=interp_axial,
            world_frame=f0.world_frame,
            units=f0.units,
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize series to dictionary."""
        return {
            "engine": self.engine,
            "frames": [f.to_dict() for f in self.frames],
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> ForceTorqueSeries:
        """Deserialize series from dictionary."""
        engine = str(d["engine"])
        frames = tuple(ForceTorqueFrame.from_dict(f) for f in d.get("frames", ()))
        return cls(frames=frames, engine=engine)

    def to_npz(self, path_or_buf: str | BinaryIO) -> None:
        """Save series to NPZ without pickled objects (allow_pickle=False)."""
        labels = sorted({w.label for f in self.frames for w in f.wrenches})
        label_to_idx = {lbl: i for i, lbl in enumerate(labels)}
        n_frames = len(self.frames)
        n_labels = len(labels)

        times = np.array([f.time_s for f in self.frames], dtype=np.float64)
        points = np.zeros((n_frames, n_labels, 3), dtype=np.float64)
        forces = np.zeros((n_frames, n_labels, 3), dtype=np.float64)
        torques = np.zeros((n_frames, n_labels, 3), dtype=np.float64)
        point_mask = np.zeros((n_frames, n_labels), dtype=bool)
        force_mask = np.zeros((n_frames, n_labels), dtype=bool)
        torque_mask = np.zeros((n_frames, n_labels), dtype=bool)

        metadata: dict[str, dict[str, str]] = {}
        for f_idx, f in enumerate(self.frames):
            for w in f.wrenches:
                l_idx = label_to_idx[w.label]
                points[f_idx, l_idx] = w.point_m
                point_mask[f_idx, l_idx] = True
                if w.force_n is not None:
                    forces[f_idx, l_idx] = w.force_n
                    force_mask[f_idx, l_idx] = True
                if w.torque_nm is not None:
                    torques[f_idx, l_idx] = w.torque_nm
                    torque_mask[f_idx, l_idx] = True
                if w.label not in metadata:
                    metadata[w.label] = {
                        "kind": w.kind.value,
                        "body": w.body,
                        "source": w.source,
                    }

        meta_json = str(metadata)
        np.savez(
            path_or_buf,
            engine=np.array(self.engine),
            times=times,
            labels=np.array(labels),
            points=points,
            forces=forces,
            torques=torques,
            point_mask=point_mask,
            force_mask=force_mask,
            torque_mask=torque_mask,
            meta_json=np.array(meta_json),
        )

    @classmethod
    def from_npz(cls, path_or_buf: str | BinaryIO) -> ForceTorqueSeries:
        """Load series from NPZ (allow_pickle=False)."""
        data = np.load(path_or_buf, allow_pickle=False)
        engine = str(data["engine"])
        times = data["times"]
        labels = [str(lbl) for lbl in data["labels"]]
        points = data["points"]
        forces = data["forces"]
        torques = data["torques"]
        point_mask = data["point_mask"]
        force_mask = data["force_mask"]
        torque_mask = data["torque_mask"]
        meta_dict = eval(str(data["meta_json"]))  # safe literal dict of primitives

        frames: list[ForceTorqueFrame] = []
        for f_idx, t in enumerate(times):
            wrenches: list[OverlayWrench] = []
            for l_idx, lbl in enumerate(labels):
                if not point_mask[f_idx, l_idx]:
                    continue
                meta = meta_dict[lbl]
                f_vec = (
                    tuple(forces[f_idx, l_idx]) if force_mask[f_idx, l_idx] else None
                )
                t_vec = (
                    tuple(torques[f_idx, l_idx]) if torque_mask[f_idx, l_idx] else None
                )
                wrenches.append(
                    OverlayWrench(
                        kind=WrenchKind(meta["kind"]),
                        label=lbl,
                        body=meta["body"],
                        point_m=tuple(points[f_idx, l_idx]),
                        force_n=f_vec,
                        torque_nm=t_vec,
                        source=meta["source"],
                    )
                )
            frames.append(
                ForceTorqueFrame(
                    time_s=float(t), engine=engine, wrenches=tuple(wrenches)
                )
            )

        return cls(frames=tuple(frames), engine=engine)


@runtime_checkable
class ForceTorqueProvider(Protocol):
    """Optional engine capability, independent of renderer and engine internals."""

    def get_force_torque_frame(self) -> ForceTorqueFrame | None:
        """Return current force/torque overlay frame, or None when unqualified."""
        ...


def read_force_torque_frame(
    provider: object, time_s: float | None = None
) -> dict[str, Any] | None:
    """Read declared provider capability; reject stale or untyped results."""
    if not isinstance(provider, ForceTorqueProvider):
        return None
    frame = provider.get_force_torque_frame()
    if frame is None:
        return None
    if not isinstance(frame, ForceTorqueFrame):
        raise TypeError("provider must return ForceTorqueFrame or None")
    if time_s is not None:
        if not math.isfinite(time_s):
            raise ValueError("time_s must be finite")
        if not math.isclose(frame.time_s, time_s, rel_tol=0, abs_tol=1e-12):
            return None
    return frame.to_dict()

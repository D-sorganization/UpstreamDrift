"""Engine-agnostic force and torque overlay contracts (ADR-0052, #11286).

Declares:
- WrenchKind enum covering all 7 physical force/torque origins.
- OverlayWrench frozen dataclass with optional force/torque halves and SI units.
- ForceTorqueFrame instantaneous state with schema versioning and axial loads.
- ForceTorqueProvider Protocol and read_force_torque_frame accessor.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
import math
import re
from types import MappingProxyType
from typing import Any, ClassVar, Protocol, runtime_checkable

from src.shared.python.body_part_viz import AxialLoadFrame
from src.shared.python.motion_matching.force_torque import SpatialWrench, validate_vec3

_LABEL_PATTERN = re.compile(r"^[a-z_]+:[A-Za-z0-9_.:-]+$")

DEFAULT_OVERLAY_UNITS: MappingProxyType[str, str] = MappingProxyType(
    {"force": "N", "torque": "N*m", "length": "m"}
)


def _default_overlay_units() -> MappingProxyType[str, str]:
    return DEFAULT_OVERLAY_UNITS


_ALLOWED_WRENCH_KEYS = frozenset(
    {"kind", "label", "body", "point_m", "force_n", "torque_nm", "source"}
)
_ALLOWED_FRAME_KEYS = frozenset(
    {
        "schema_version",
        "time_s",
        "engine",
        "world_frame",
        "units",
        "wrenches",
        "axial_loads",
    }
)


class WrenchKind(str, Enum):
    """Categorical origin of an overlay wrench."""

    JOINT_ACTUATOR = "joint_actuator"
    JOINT_REACTION = "joint_reaction"
    CONTACT = "contact"
    GRIP = "grip"
    EXTERNAL = "external"
    GRAVITY = "gravity"
    MUSCLE = "muscle"


@dataclass(frozen=True)
class OverlayWrench:
    """Rigid body wrench with physical application point in the world frame.

    Each half (force_n, torque_nm) is optional; at least one must be non-None.
    An unavailable half is stored as None (never fabricated as zero) and serialized as null.
    """

    kind: WrenchKind
    label: str
    body: str
    point_m: tuple[float, float, float]
    force_n: tuple[float, float, float] | None = None
    torque_nm: tuple[float, float, float] | None = None
    source: str = ""

    APPLICATION_FRAME: ClassVar[str] = "world"
    DIRECTION_CONVENTION: ClassVar[str] = "applied_to_body"

    def __post_init__(self) -> None:
        if not isinstance(self.kind, WrenchKind):
            try:
                object.__setattr__(self, "kind", WrenchKind(self.kind))
            except (ValueError, KeyError) as e:
                raise ValueError(
                    f"Invalid wrench kind '{self.kind}', must be one of "
                    f"{[k.value for k in WrenchKind]}"
                ) from e

        if not isinstance(self.label, str) or not _LABEL_PATTERN.match(self.label):
            raise ValueError(
                f"Invalid wrench label '{self.label}': must match '^[a-z_]+:[A-Za-z0-9_.:-]+$'"
            )

        if not isinstance(self.body, str) or not self.body.strip():
            raise ValueError("body must be a non-empty string")

        if not isinstance(self.source, str) or not self.source.strip():
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
                "At least one of force_n or torque_nm must be provided (cannot both be None)"
            )

    def to_spatial_wrench(self) -> SpatialWrench:
        """Convert to SpatialWrench when both halves are present.

        Raises:
            ValueError: If either force_n or torque_nm is None, naming the missing half.
        """
        if self.force_n is None and self.torque_nm is None:
            raise ValueError(
                "Cannot convert to SpatialWrench: missing force_n and torque_nm"
            )
        if self.force_n is None:
            raise ValueError("Cannot convert to SpatialWrench: missing force_n")
        if self.torque_nm is None:
            raise ValueError("Cannot convert to SpatialWrench: missing torque_nm")

        return SpatialWrench(
            application_frame=self.APPLICATION_FRAME,
            point_m=self.point_m,
            force_n=self.force_n,
            torque_nm=self.torque_nm,
            direction_convention=self.DIRECTION_CONVENTION,
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize wrench to JSON-safe dictionary."""
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
    def from_dict(cls, data: Mapping[str, Any]) -> OverlayWrench:
        """Deserialize wrench from dictionary, rejecting unknown keys."""
        extra = set(data.keys()) - _ALLOWED_WRENCH_KEYS
        if extra:
            raise ValueError(f"Unknown keys in wrench data: {extra}")

        force_val = data.get("force_n")
        torque_val = data.get("torque_nm")

        return cls(
            kind=WrenchKind(data["kind"]),
            label=str(data["label"]),
            body=str(data["body"]),
            point_m=data["point_m"],
            force_n=tuple(force_val) if force_val is not None else None,
            torque_nm=tuple(torque_val) if torque_val is not None else None,
            source=str(data["source"]),
        )


@dataclass(frozen=True)
class ForceTorqueFrame:
    """Instantaneous frame of force and torque overlays synchronized to simulation time."""

    time_s: float
    engine: str
    wrenches: tuple[OverlayWrench, ...] = ()
    axial_loads: AxialLoadFrame | None = None
    world_frame: str = "world_Zup"
    units: Mapping[str, str] = field(default_factory=_default_overlay_units)

    def __post_init__(self) -> None:
        if not math.isfinite(self.time_s):
            raise ValueError("time_s must be a finite number")

        if not isinstance(self.engine, str) or not self.engine.strip():
            raise ValueError("engine must be a non-empty string")

        if self.world_frame != "world_Zup":
            raise ValueError("world_frame must be 'world_Zup'")

        wrenches_tuple = tuple(self.wrenches)
        labels: set[str] = set()
        for w in wrenches_tuple:
            if not isinstance(w, OverlayWrench):
                raise TypeError(
                    f"All wrenches must be OverlayWrench instances, got {type(w)}"
                )
            if w.label in labels:
                raise ValueError(f"Duplicate wrench label: '{w.label}'")
            labels.add(w.label)
        object.__setattr__(self, "wrenches", wrenches_tuple)

        if self.axial_loads is not None:
            if not isinstance(self.axial_loads, AxialLoadFrame):
                raise TypeError("axial_loads must be an AxialLoadFrame or None")
            if abs(self.axial_loads.time_s - self.time_s) > 1e-12:
                raise ValueError(
                    f"axial_loads.time_s ({self.axial_loads.time_s}) must match "
                    f"frame time_s ({self.time_s}) within 1e-12"
                )

        if not isinstance(self.units, Mapping):
            raise TypeError("units must be a mapping")
        object.__setattr__(self, "units", MappingProxyType(dict(self.units)))

    def by_kind(self, kind: WrenchKind) -> tuple[OverlayWrench, ...]:
        """Return all wrenches matching the requested kind."""
        return tuple(w for w in self.wrenches if w.kind == kind)

    def to_dict(self) -> dict[str, Any]:
        """Serialize frame to JSON-safe dictionary conforming to force-torque-frame-v1."""
        return {
            "schema_version": "force-torque-frame-v1",
            "time_s": self.time_s,
            "engine": self.engine,
            "world_frame": self.world_frame,
            "units": dict(self.units),
            "wrenches": [w.to_dict() for w in self.wrenches],
            "axial_loads": self.axial_loads.to_dict()
            if self.axial_loads is not None
            else None,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ForceTorqueFrame:
        """Deserialize frame from dictionary, rejecting unknown keys and wrong schema_version."""
        extra = set(data.keys()) - _ALLOWED_FRAME_KEYS
        if extra:
            raise ValueError(f"Unknown key in frame data: {extra}")

        schema_version = data.get("schema_version")
        if schema_version != "force-torque-frame-v1":
            raise ValueError(
                f"Invalid schema_version: expected 'force-torque-frame-v1', got '{schema_version}'"
            )

        wrenches_data = data.get("wrenches", ())
        wrenches = tuple(OverlayWrench.from_dict(w) for w in wrenches_data)

        axial_loads_data = data.get("axial_loads")
        axial_loads: AxialLoadFrame | None = None
        if axial_loads_data is not None:
            axial_loads = AxialLoadFrame(
                time_s=float(axial_loads_data["time_s"]),
                values_n=axial_loads_data["values_n"],
                source=str(axial_loads_data["source"]),
            )

        world_frame = data.get("world_frame", "world_Zup")
        units = data.get("units", DEFAULT_OVERLAY_UNITS)

        return cls(
            time_s=float(data["time_s"]),
            engine=str(data["engine"]),
            wrenches=wrenches,
            axial_loads=axial_loads,
            world_frame=world_frame,
            units=units,
        )


@runtime_checkable
class ForceTorqueProvider(Protocol):
    """Optional engine capability providing force and torque overlay frames."""

    def get_force_torque_frame(self) -> ForceTorqueFrame | None:
        """Return current force/torque frame, or None when unqualified."""
        ...


def read_force_torque_frame(provider: Any, time_s: float) -> ForceTorqueFrame | None:
    """Read frame from declared provider capability; reject stale or untyped results."""
    if not math.isfinite(time_s):
        raise ValueError("time_s must be finite")
    if not isinstance(provider, ForceTorqueProvider):
        return None
    frame = provider.get_force_torque_frame()
    if frame is None:
        return None
    if not isinstance(frame, ForceTorqueFrame):
        raise TypeError("provider must return ForceTorqueFrame or None")
    if not math.isclose(frame.time_s, time_s, rel_tol=0, abs_tol=1e-12):
        return None
    return frame

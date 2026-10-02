"""Engine-agnostic force/torque overlay data contracts (ADR-0052, #11286).

Provides:
- WrenchKind: enumeration of physical wrench sources.
- OverlayWrench: immutable wrench with optional force/torque halves in world frame.
- ForceTorqueFrame: time-stamped collection of overlay wrenches and optional axial loads.
- ForceTorqueSeries: time series of frames with linear interpolation and npz/dict I/O.
- ForceTorqueProvider: runtime-checkable protocol for physics engines.
- read_force_torque_frame: reader with type and timestamp validation.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
import io
import json
import math
from pathlib import Path
import re
from types import MappingProxyType
from typing import Any, Final, Protocol, runtime_checkable

import numpy as np

from src.shared.python.body_part_viz.axial_loads import AxialLoadFrame
from src.shared.python.motion_matching.force_torque import (
    SpatialWrench,
    validate_vec3,
)

LABEL_PATTERN: Final[re.Pattern[str]] = re.compile(r"^[a-z_]+:[A-Za-z0-9_.:-]+$")


class WrenchKind(str, Enum):
    """Categorical source and purpose of an applied wrench."""

    JOINT_ACTUATOR = "joint_actuator"
    JOINT_REACTION = "joint_reaction"
    CONTACT = "contact"
    GRIP = "grip"
    EXTERNAL = "external"
    GRAVITY = "gravity"
    MUSCLE = "muscle"


@dataclass(frozen=True)
class OverlayWrench:
    """Rigid body wrench applied to a body, expressed in the world frame (ADR-0026 Z-up).

    At least one of force_n or torque_nm must be present. A None half signifies
    unavailable telemetry and is never zero-filled or drawn.
    """

    kind: WrenchKind
    label: str
    body: str
    point_m: tuple[float, float, float]
    force_n: tuple[float, float, float] | None = None
    torque_nm: tuple[float, float, float] | None = None
    source: str = ""

    APPLICATION_FRAME: Final[str] = "world"
    DIRECTION_CONVENTION: Final[str] = "applied_to_body"

    def __post_init__(self) -> None:
        if isinstance(self.kind, str) and not isinstance(self.kind, WrenchKind):
            try:
                object.__setattr__(self, "kind", WrenchKind(self.kind))
            except ValueError as err:
                raise ValueError(f"Invalid WrenchKind: {self.kind!r}") from err
        elif not isinstance(self.kind, WrenchKind):
            raise TypeError(
                f"kind must be a WrenchKind, got {type(self.kind).__name__}"
            )

        if not isinstance(self.label, str) or not LABEL_PATTERN.match(self.label):
            raise ValueError(
                f"label must match pattern '^[a-z_]+:[A-Za-z0-9_.:-]+$', got {self.label!r}"
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
            raise ValueError("At least one of force_n or torque_nm must be provided")

    def to_spatial_wrench(self) -> SpatialWrench:
        """Convert to SpatialWrench if both halves are present."""
        if self.force_n is None:
            raise ValueError("Cannot convert to SpatialWrench: force_n is None")
        if self.torque_nm is None:
            raise ValueError("Cannot convert to SpatialWrench: torque_nm is None")
        return SpatialWrench(
            application_frame=self.APPLICATION_FRAME,
            direction_convention=self.DIRECTION_CONVENTION,
            point_m=self.point_m,
            force_n=self.force_n,
            torque_nm=self.torque_nm,
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize to wire-safe dictionary."""
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
        """Deserialize from wire dictionary, rejecting unexpected keys."""
        allowed_keys = {
            "kind",
            "label",
            "body",
            "point_m",
            "force_n",
            "torque_nm",
            "source",
        }
        unknown = set(data.keys()) - allowed_keys
        if unknown:
            raise ValueError(
                f"unknown keys in OverlayWrench dictionary: {sorted(unknown)}"
            )
        return cls(
            kind=WrenchKind(data["kind"]),
            label=data["label"],
            body=data["body"],
            point_m=tuple(data["point_m"]),  # type: ignore[arg-type]
            force_n=tuple(data["force_n"]) if data.get("force_n") is not None else None,  # type: ignore[arg-type]
            torque_nm=tuple(data["torque_nm"])
            if data.get("torque_nm") is not None
            else None,  # type: ignore[arg-type]
            source=data.get("source", ""),
        )


@dataclass(frozen=True)
class ForceTorqueFrame:
    """Snapshot of rigid body forces and torques at a single simulation timestamp."""

    time_s: float
    engine: str
    wrenches: tuple[OverlayWrench, ...] = ()
    axial_loads: AxialLoadFrame | None = None
    world_frame: str = "world_Zup"
    units: Mapping[str, str] = MappingProxyType(
        {"force": "N", "torque": "N*m", "length": "m"}
    )

    SCHEMA_VERSION: Final[str] = "force-torque-frame-v1"

    def __post_init__(self) -> None:
        t = float(self.time_s)
        if not math.isfinite(t):
            raise ValueError(f"time_s must be finite, got {self.time_s}")
        object.__setattr__(self, "time_s", t)

        if not self.engine or not isinstance(self.engine, str):
            raise ValueError("engine must be a non-empty string")

        w_tuple = tuple(self.wrenches)
        for w in w_tuple:
            if not isinstance(w, OverlayWrench):
                raise TypeError(
                    f"All wrenches must be OverlayWrench, got {type(w).__name__}"
                )
        object.__setattr__(self, "wrenches", w_tuple)

        labels = [w.label for w in w_tuple]
        seen: set[str] = set()
        duplicates: set[str] = set()
        for label in labels:
            if label in seen:
                duplicates.add(label)
            seen.add(label)
        if duplicates:
            raise ValueError(
                f"Duplicate wrench label(s) found in frame: {sorted(duplicates)}"
            )

        if self.axial_loads is not None:
            if not isinstance(self.axial_loads, AxialLoadFrame):
                raise TypeError(
                    f"axial_loads must be AxialLoadFrame or None, got {type(self.axial_loads).__name__}"
                )
            if not math.isclose(
                self.axial_loads.time_s, self.time_s, rel_tol=0, abs_tol=1e-12
            ):
                raise ValueError(
                    f"axial_loads time_s ({self.axial_loads.time_s}) must match "
                    f"frame time_s ({self.time_s})"
                )

        if not isinstance(self.units, Mapping):
            raise TypeError("units must be a Mapping")
        object.__setattr__(self, "units", MappingProxyType(dict(self.units)))

    def by_kind(self, kind: WrenchKind | str) -> tuple[OverlayWrench, ...]:
        """Return all wrenches of the given kind."""
        target_kind = kind if isinstance(kind, WrenchKind) else WrenchKind(kind)
        return tuple(w for w in self.wrenches if w.kind == target_kind)

    def to_dict(self) -> dict[str, Any]:
        """Serialize frame to dictionary matching JSON Schema 2020-12."""
        return {
            "schema_version": self.SCHEMA_VERSION,
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
        """Deserialize frame from dictionary, enforcing schema version and known keys."""
        allowed_keys = {
            "schema_version",
            "time_s",
            "engine",
            "world_frame",
            "units",
            "wrenches",
            "axial_loads",
        }
        unknown = set(data.keys()) - allowed_keys
        if unknown:
            raise ValueError(
                f"unknown keys in ForceTorqueFrame dictionary: {sorted(unknown)}"
            )

        schema_version = data.get("schema_version")
        if schema_version != cls.SCHEMA_VERSION:
            raise ValueError(
                f"unsupported schema_version {schema_version!r}, expected {cls.SCHEMA_VERSION!r}"
            )

        wrenches = tuple(OverlayWrench.from_dict(w) for w in data.get("wrenches", ()))

        axial_data = data.get("axial_loads")
        axial_frame: AxialLoadFrame | None = None
        if axial_data is not None:
            axial_frame = AxialLoadFrame(
                time_s=axial_data["time_s"],
                values_n=axial_data["values_n"],
                source=axial_data.get("source", ""),
            )

        return cls(
            time_s=float(data["time_s"]),
            engine=data["engine"],
            wrenches=wrenches,
            axial_loads=axial_frame,
            world_frame=data.get("world_frame", "world_Zup"),
            units=data.get("units", {"force": "N", "torque": "N*m", "length": "m"}),
        )


from .series import ForceTorqueSeries


@runtime_checkable
class ForceTorqueProvider(Protocol):
    """Protocol for physics engines providing force/torque frames."""

    def get_force_torque_frame(self) -> ForceTorqueFrame | None:
        """Return the current ForceTorqueFrame, or None if unavailable."""
        ...


def read_force_torque_frame(
    provider: object, time_s: float | None = None
) -> ForceTorqueFrame | None:
    """Read declared provider capability; validate type and timestamp freshness."""
    if not isinstance(provider, ForceTorqueProvider):
        return None
    frame = provider.get_force_torque_frame()
    if frame is None:
        return None
    if not isinstance(frame, ForceTorqueFrame):
        raise TypeError(
            f"provider must return ForceTorqueFrame or None, got {type(frame).__name__}"
        )
    if time_s is not None:
        t = float(time_s)
        if not math.isfinite(t):
            raise ValueError("time_s must be finite")
        if not math.isclose(frame.time_s, t, rel_tol=0, abs_tol=1e-12):
            return None
    return frame

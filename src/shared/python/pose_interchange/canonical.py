"""Canonical pose dataclass for cross-engine pose interchange.

The canonical convention chosen for this EPIC (#4895) deliberately mirrors
the one already used by
:func:`src.shared.python.motion_matching.diagnostics.forward_kinematics.forward_kinematics`
and :func:`src.shared.python.motion_matching.diagnostics.reference_pose.reference_golfer_setup`,
so the canonical pose is a drop-in for everything those modules already
produce.

Canonical convention:

- **Pelvis pose** is an SE(3) transform represented as
  ``(translation_m, rotation_xyz_deg)`` where
  ``rotation_xyz_deg`` is intrinsic XYZ Euler in **degrees**.
- **Joint angles** are a flat dict mapping the
  :func:`reference_golfer_setup` field names to **degrees**.
- **Velocities** are not part of this dataclass. Velocities live in
  the engine-side state and are out of scope for the interchange (the
  Pose Studio tool always sets v=0 when materialising an initial state;
  see Subtask 6 / #4900).

A ``CanonicalPose`` is **frozen and validated on construction** (DbC):

- pelvis translation and rotation are length-3, finite.
- joint-angle keys are a subset of the reference setup field set.
- joint-angle values are finite floats.

Engine adapters (:class:`PoseConventionAdapter`, in
:mod:`pose_interchange.protocol`) round-trip canonical poses to and
from each engine's native ``q`` vector layout.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final, Literal

import numpy as np
import numpy.typing as npt

# Import directly from the diagnostics submodule to avoid triggering
# the motion_matching package __init__.py which eagerly imports pandas-dependent loaders.
# See issue #4926 for context on the optional pandas dependency coupling.
from src.shared.python.motion_matching.diagnostics.reference_pose import (
    REFERENCE_GOLFER_FIELDS,
    reference_golfer_setup,
)

CONVENTION_TAG: Final[str] = "canonical-v1"

_FIELD_SET = frozenset(REFERENCE_GOLFER_FIELDS)


def _validate_translation(value: npt.NDArray[np.float64], name: str) -> None:
    if value.shape != (3,):
        raise ValueError(f"{name} must have shape (3,), got {value.shape}")
    if not np.all(np.isfinite(value)):
        raise ValueError(f"{name} must contain only finite values")


@dataclass(frozen=True, slots=True)
class CanonicalPose:
    """Engine-agnostic skeleton pose in the canonical convention.

    Parameters
    ----------
    pelvis_translation_m
        World-frame pelvis position in metres, shape ``(3,)``.
    pelvis_rotation_xyz_deg
        Intrinsic XYZ Euler in degrees, shape ``(3,)``.
    joint_angles_deg
        Flat dict mapping reference-golfer field names to angles in
        degrees. Keys must be a subset of
        :data:`REFERENCE_GOLFER_FIELDS`. Missing keys default to 0.0
        on round-trip.
    convention_tag
        Always ``"canonical-v1"`` for this version of the convention.
        Future versions bump this; adapters should refuse to operate on
        a pose with an unfamiliar tag.
    """

    pelvis_translation_m: npt.NDArray[np.float64]
    pelvis_rotation_xyz_deg: npt.NDArray[np.float64]
    joint_angles_deg: Mapping[str, float] = field(default_factory=dict)
    convention_tag: Literal["canonical-v1"] = CONVENTION_TAG

    def __post_init__(self) -> None:
        # Coerce to immutable numpy arrays so the dataclass behaves "frozen"
        # in spirit (numpy arrays themselves remain mutable, but we set
        # writeable=False to make accidental mutation raise).
        t = np.asarray(self.pelvis_translation_m, dtype=float).copy()
        r = np.asarray(self.pelvis_rotation_xyz_deg, dtype=float).copy()
        _validate_translation(t, "pelvis_translation_m")
        _validate_translation(r, "pelvis_rotation_xyz_deg")
        t.setflags(write=False)
        r.setflags(write=False)
        # Use object.__setattr__ because the dataclass is frozen.
        object.__setattr__(self, "pelvis_translation_m", t)
        object.__setattr__(self, "pelvis_rotation_xyz_deg", r)

        unknown = set(self.joint_angles_deg.keys()) - _FIELD_SET
        if unknown:
            raise ValueError(
                "joint_angles_deg contains unknown field names: "
                f"{sorted(unknown)}; expected subset of REFERENCE_GOLFER_FIELDS"
            )
        for key, val in self.joint_angles_deg.items():
            try:
                fval = float(val)
            except (TypeError, ValueError) as exc:
                raise TypeError(
                    f"joint_angles_deg[{key!r}] must be a finite float, got {val!r}"
                ) from exc
            if not np.isfinite(fval):
                raise ValueError(
                    f"joint_angles_deg[{key!r}] must be finite, got {fval!r}"
                )

        # Snapshot the joint-angle dict so the caller can't mutate it
        # behind our back, and so iteration order is stable.
        snapshot: dict[str, float] = {
            key: float(self.joint_angles_deg[key])
            for key in REFERENCE_GOLFER_FIELDS
            if key in self.joint_angles_deg
        }
        object.__setattr__(self, "joint_angles_deg", snapshot)

        if self.convention_tag != CONVENTION_TAG:
            raise ValueError(
                f"convention_tag must be {CONVENTION_TAG!r}, "
                f"got {self.convention_tag!r}"
            )

    # ---- Convenience accessors -------------------------------------------------

    def angle_deg(self, name: str) -> float:
        """Return ``joint_angles_deg[name]`` or 0.0 if absent.

        Adapter code should prefer this accessor over direct dict access
        so that absent fields decay to "neutral" rather than raising.
        """
        if name not in _FIELD_SET:
            raise KeyError(
                f"{name!r} is not a canonical joint-angle field "
                "(see REFERENCE_GOLFER_FIELDS)"
            )
        return float(self.joint_angles_deg.get(name, 0.0))

    def angles_full_dict_deg(self) -> dict[str, float]:
        """Return all canonical joint angles as a dict, with 0.0 for absent fields."""
        return {name: self.angle_deg(name) for name in REFERENCE_GOLFER_FIELDS}

    def angles_full_dict_rad(self) -> dict[str, float]:
        """Same as :meth:`angles_full_dict_deg` but in radians."""
        return {
            name: float(np.radians(self.angle_deg(name)))
            for name in REFERENCE_GOLFER_FIELDS
        }

    # ---- (de)serialisation -----------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable plain-dict view of the pose.

        Used by the realtime IPC layer (Subtask 4 of EPIC #4993) to ship
        the canonical pose to subscribing tools without forcing them to
        depend on numpy or this dataclass directly.
        """
        return {
            "convention_tag": self.convention_tag,
            "pelvis_translation_m": self.pelvis_translation_m.tolist(),
            "pelvis_rotation_xyz_deg": self.pelvis_rotation_xyz_deg.tolist(),
            "joint_angles_deg": self.angles_full_dict_deg(),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> CanonicalPose:
        """Inverse of :meth:`to_dict`."""
        if not isinstance(payload, Mapping):
            raise TypeError(f"payload must be a Mapping, got {type(payload).__name__}")
        try:
            return cls(
                pelvis_translation_m=np.asarray(
                    payload["pelvis_translation_m"], dtype=float
                ),
                pelvis_rotation_xyz_deg=np.asarray(
                    payload["pelvis_rotation_xyz_deg"], dtype=float
                ),
                joint_angles_deg=dict(payload.get("joint_angles_deg", {})),
                convention_tag=payload.get("convention_tag", CONVENTION_TAG),
            )
        except KeyError as exc:
            raise ValueError(
                f"CanonicalPose payload missing required key: {exc}"
            ) from exc

    def to_json(self) -> str:
        """Serialise to a stable JSON string.

        Round-trips exactly via :meth:`from_json`. Field order matches
        :data:`REFERENCE_GOLFER_FIELDS` for stable diffs.
        """
        payload: dict[str, Any] = {
            "convention_tag": self.convention_tag,
            "pelvis_translation_m": self.pelvis_translation_m.tolist(),
            "pelvis_rotation_xyz_deg": self.pelvis_rotation_xyz_deg.tolist(),
            "joint_angles_deg": self.angles_full_dict_deg(),
        }
        return json.dumps(payload, indent=2, sort_keys=False)

    @classmethod
    def from_json(cls, text: str) -> CanonicalPose:
        """Inverse of :meth:`to_json`."""
        payload = json.loads(text)
        if not isinstance(payload, dict):
            raise ValueError("CanonicalPose JSON must decode to a dict")
        try:
            return cls(
                pelvis_translation_m=np.asarray(
                    payload["pelvis_translation_m"], dtype=float
                ),
                pelvis_rotation_xyz_deg=np.asarray(
                    payload["pelvis_rotation_xyz_deg"], dtype=float
                ),
                joint_angles_deg=dict(payload.get("joint_angles_deg", {})),
                convention_tag=payload.get("convention_tag", CONVENTION_TAG),
            )
        except KeyError as exc:
            raise ValueError(f"CanonicalPose JSON missing required key: {exc}") from exc

    def to_path(self, output_path: Path | str) -> None:
        """Write :meth:`to_json` output to *output_path* (UTF-8)."""
        Path(output_path).write_text(self.to_json(), encoding="utf-8")

    @classmethod
    def from_path(cls, input_path: Path | str) -> CanonicalPose:
        """Inverse of :meth:`to_path`."""
        return cls.from_json(Path(input_path).read_text(encoding="utf-8"))


# ---- Module-level constructors -------------------------------------------------


def canonical_zero_pose() -> CanonicalPose:
    """The all-zero canonical pose: pelvis at origin, no joint rotations.

    Convenient default for tests and for tools that want a "neutral"
    starting point. Note: this is *not* anatomically plausible — it is
    a T-pose at the world origin. Use :func:`canonical_from_reference_setup`
    for the canonical address pose.
    """
    return CanonicalPose(
        pelvis_translation_m=np.zeros(3),
        pelvis_rotation_xyz_deg=np.zeros(3),
        joint_angles_deg={},
    )


def canonical_from_reference_setup() -> CanonicalPose:
    """Canonical address pose, derived from :func:`reference_golfer_setup`.

    Pelvis is placed at the world origin with zero pelvis rotation
    (the existing reference golfer is a *body-relative* pose; world-frame
    placement is the matcher's responsibility).
    """
    return CanonicalPose(
        pelvis_translation_m=np.zeros(3),
        pelvis_rotation_xyz_deg=np.zeros(3),
        joint_angles_deg=reference_golfer_setup(),
    )

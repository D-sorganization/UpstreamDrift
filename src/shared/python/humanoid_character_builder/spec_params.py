"""Spec-native character parameters that compile to ``full_body_spec`` (CMB-1, #11652).

``BodyParameters`` drives the mesh/URDF generators; engines consume the
``full-body-v1`` document. This module is the bridge: a small, validated
parameter set (stature, mass, three proportion scales, club, grip roll) that
compiles through the one shared builder
(``motion_matching.execution.spec_builder.compose_anthropometric_document``,
De Leva tables in ``anthropometry``) so the character builder never keeps a
second anthropometric table.

Contract:
  * Preconditions: every value finite and inside the documented ranges;
    ``club`` is a known club id.
  * Postconditions: the compiled document passed ``validate_full_body_spec``;
    ``serialize_spec`` output is byte-identical for equal parameters;
    ``params_from_spec(compile_full_body_spec(p)) == p``.

Limits: the lower limbs and De Leva table are the male Rajagopal/De Leva
references; stature and mass scale them, proportion scales reshape trunk,
arms and shoulders. Output is unqualified geometry, like its builder.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any

from src.shared.python.humanoid_character_builder.core.body_parameters import (
    BodyParameters,
)
from src.shared.python.motion_matching.club_models import CLUBS
from src.shared.python.motion_matching.execution import assets
from src.shared.python.motion_matching.execution.spec_builder import (
    compose_anthropometric_document,
)

__all__ = [
    "PARAMETER_RANGES",
    "SpecCharacterParameters",
    "compile_full_body_spec",
    "params_from_spec",
    "serialize_spec",
]

#: Inclusive bounds per numeric field (units in the field name).
PARAMETER_RANGES: dict[str, tuple[float, float]] = {
    "stature_m": (1.20, 2.30),
    "mass_kg": (30.0, 200.0),
    "trunk_scale": (0.7, 1.4),
    "arm_scale": (0.7, 1.4),
    "shoulder_scale": (0.7, 1.4),
    "grip_roll_deg": (-180.0, 180.0),
}


@dataclass(frozen=True)
class SpecCharacterParameters:
    """Validated subject parameters for the anthropometric full-body spec."""

    stature_m: float = 1.75
    mass_kg: float = 75.0
    trunk_scale: float = 1.0
    arm_scale: float = 1.0
    shoulder_scale: float = 1.0
    club: str = "driver"
    grip_roll_deg: float = 0.0

    def __post_init__(self) -> None:
        for name, (low, high) in PARAMETER_RANGES.items():
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int | float):
                raise TypeError(f"{name} must be a number, got {value!r}")
            if not math.isfinite(value) or not low <= value <= high:
                raise ValueError(f"{name} must be finite in [{low}, {high}]: {value}")
            object.__setattr__(self, name, float(value))
        if self.club not in CLUBS:
            raise ValueError(f"club must be one of {sorted(CLUBS)}: {self.club!r}")

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable mapping."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> SpecCharacterParameters:
        """Build from a mapping; unknown keys are rejected, not ignored."""
        known = {f.name for f in fields(cls)}
        unknown = sorted(set(data) - known)
        if unknown:
            raise ValueError(f"unknown character parameter(s): {unknown}")
        return cls(**dict(data))

    @classmethod
    def from_body_parameters(cls, body: BodyParameters) -> SpecCharacterParameters:
        """Map the mesh-side ``BodyParameters`` proportions onto spec scales."""
        return cls(
            stature_m=body.height_m,
            mass_kg=body.mass_kg,
            trunk_scale=body.torso_length_factor,
            arm_scale=body.arm_length_factor,
            shoulder_scale=body.shoulder_width_factor,
        )


def compile_full_body_spec(
    params: SpecCharacterParameters,
    *,
    native_path: Path | str | None = None,
    osim_path: Path | str | None = None,
    native_candidate_path: Path | str | None = None,
) -> dict[str, Any]:
    """Compile parameters to a validated ``full-body-v1`` document.

    Reference inputs resolve through ``execution.assets`` (explicit path,
    environment variable, then repository checkout); a missing asset raises
    ``FileNotFoundError``.
    """
    if not isinstance(params, SpecCharacterParameters):
        raise TypeError("params must be SpecCharacterParameters")
    built = compose_anthropometric_document(
        native_path=assets.get_native_geometry_spec(native_path),
        osim_path=assets.get_opensim_model(osim_path),
        native_candidate_path=assets.get_candidate_geometry_spec(native_candidate_path),
        stature_m=params.stature_m,
        mass_kg=params.mass_kg,
        club=params.club,
        trunk_scale=params.trunk_scale,
        arm_scale=params.arm_scale,
        shoulder_scale=params.shoulder_scale,
        grip_roll_deg=params.grip_roll_deg,
    )
    return built.document


def serialize_spec(document: Mapping[str, Any]) -> str:
    """Canonical text form: sorted keys, 2-space indent, trailing newline."""
    return json.dumps(document, indent=2, sort_keys=True) + "\n"


def params_from_spec(document: Mapping[str, Any]) -> SpecCharacterParameters:
    """Recover the parameters a spec was compiled from (``subject`` + club)."""
    subject = document.get("subject")
    if not isinstance(subject, Mapping):
        raise ValueError("document has no subject block; not a character spec")
    club = document.get("club")
    club_name = club.get("name") if isinstance(club, Mapping) else None
    try:
        return SpecCharacterParameters(
            stature_m=subject["stature_m"],
            mass_kg=subject["mass_kg"],
            trunk_scale=subject["trunk_scale"],
            arm_scale=subject["arm_scale"],
            shoulder_scale=subject["shoulder_scale"],
            grip_roll_deg=subject["grip_roll_deg"],
            club=club_name or "driver",
        )
    except KeyError as exc:
        raise ValueError(f"subject block missing {exc}") from exc

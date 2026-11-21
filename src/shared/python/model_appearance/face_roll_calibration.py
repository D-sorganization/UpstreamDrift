"""Head roll about the shaft from a stated calibration rule (#15150).

The shaft markers of a capture give the shaft direction but not how the head
is turned about it. :mod:`model_appearance.club_assembly` therefore carries
one roll constant per club (``face_roll_deg``), solved so the face is square
to the target at the captured address pose. That is one assumption. This
module adds a second one as an opt-in mode: the roll at which the static loft
at address equals the head's design loft. Both are stated in
``face_roll_calibration_settings.json`` beside this module; the default mode
changes nothing. No fitted joint angle is touched by either.
"""

from __future__ import annotations

import copy
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from src.shared.python.model_appearance.club_assembly import assembly_from_spec
from src.shared.python.model_appearance.club_head_mesh import (
    ClubHeadMesh,
    club_face_normal,
    load_club_head,
)

Array = NDArray[np.float64]
SETTINGS_PATH = Path(__file__).with_name("face_roll_calibration_settings.json")
ADDRESS_SQUARE = "address_square"
DESIGN_LOFT_AT_ADDRESS = "design_loft_at_address"
MODES = (ADDRESS_SQUARE, DESIGN_LOFT_AT_ADDRESS)
#: Native world up axis (+Z), the convention of ``model_appearance.club_face``.
WORLD_UP = np.array([0.0, 0.0, 1.0])
_MIN_NORM = 1e-9


def validate_mode(mode: object) -> str:
    """``mode`` when it is a known calibration mode.

    Raises ``TypeError`` for a non-string and ``ValueError`` for an unknown one.
    """
    if not isinstance(mode, str):
        raise TypeError("face roll calibration mode must be a string")
    if mode not in MODES:
        raise ValueError(f"face roll calibration mode must be one of {MODES}")
    return mode


@dataclass(frozen=True)
class FaceRollSettings:
    """Validated ``face_roll_calibration_settings.json``."""

    mode: str

    def __post_init__(self) -> None:
        validate_mode(self.mode)


def load_face_roll_settings(path: Path | None = None) -> FaceRollSettings:
    """Read the settings file; ``KeyError`` for a missing key, else as the class."""
    raw = json.loads((path or SETTINGS_PATH).read_text(encoding="utf-8"))
    return FaceRollSettings(raw["mode"])


def up_in_club_frame(body_world: Sequence[Sequence[float]] | Array) -> Array:
    """World up expressed in the club-body frame of the pose ``body_world``.

    ``body_world`` is the finite world 4x4 of the spec club body (for example
    ``ClubPoseModel.body_world`` at the address pose); else ``ValueError``.
    Postcondition: a unit 3-vector.
    """
    pose = np.asarray(body_world, dtype=float)
    if pose.shape != (4, 4) or not np.isfinite(pose).all():
        raise ValueError("body_world must be a finite 4x4 pose")
    up = pose[:3, :3].T @ WORLD_UP
    return np.asarray(up / np.linalg.norm(up), dtype=float)


def _finite(value: float, name: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite")
    return number


def _wrap_deg(angle: float) -> float:
    return -((-angle + 180.0) % 360.0 - 180.0)


def design_loft_face_roll_deg(
    head: ClubHeadMesh,
    up_club: Sequence[float] | Array,
    *,
    near_deg: float,
    loft_deg: float | None = None,
) -> float:
    """Head roll about the shaft that gives ``loft_deg`` of static loft.

    ``up_club`` is world up in the club-body frame at the address pose
    (:func:`up_in_club_frame`), ``loft_deg`` the wanted elevation of the face
    normal above the horizontal (default: the head's design loft). Two rolls
    give any reachable loft; the one nearest ``near_deg`` is returned, so pass
    the address-square roll to keep the face toward the target.

    Raises ``ValueError`` for a non-finite input, an ``up_club`` that is not a
    non-zero 3-vector, and a loft no roll can reach (a shaft along ``up_club``
    included). Postcondition: the rolled face normal has elevation ``loft_deg``
    against ``up_club`` and the result lies in (-180, 180].
    """
    up = np.asarray(up_club, dtype=float)
    if up.shape != (3,) or not np.isfinite(up).all():
        raise ValueError("up_club must be a finite 3-vector")
    norm = float(np.linalg.norm(up))
    if norm < _MIN_NORM:
        raise ValueError("up_club must not be the zero vector")
    up = up / norm
    near = _finite(near_deg, "near_deg")
    loft = _finite(head.loft_deg if loft_deg is None else loft_deg, "loft_deg")
    flat = club_face_normal(head, 0.0)
    # up . Ry(roll) flat = offset + in_phase cos(roll) + quadrature sin(roll)
    in_phase = up[0] * flat[0] + up[2] * flat[2]
    quadrature = up[0] * flat[2] - up[2] * flat[0]
    reach = math.hypot(in_phase, quadrature)
    wanted = math.sin(math.radians(loft)) - up[1] * flat[1]
    if reach < _MIN_NORM or abs(wanted) > reach:
        raise ValueError(f"no head roll gives {loft:.2f} deg of loft at this shaft")
    centre = math.degrees(math.atan2(quadrature, in_phase))
    spread = math.degrees(math.acos(wanted / reach))
    rolls = (_wrap_deg(centre + spread), _wrap_deg(centre - spread))
    return min(rolls, key=lambda roll: abs(_wrap_deg(roll - near)))


def spec_with_face_roll_calibration(
    spec: Mapping[str, Any],
    address_body_world: Sequence[Sequence[float]] | Array,
    mode: str | None = None,
) -> dict[str, Any]:
    """Copy of ``spec`` whose club block carries the calibrated head roll.

    ``mode`` defaults to the settings file. ``address_square`` keeps the roll
    the spec already resolves to; ``design_loft_at_address`` solves it from
    the club-body pose at address (``address_body_world``) and the head's
    design loft, or the club block's ``design_loft_deg`` when given. The mode
    is recorded as ``club["face_roll_calibration"]`` and the roll as
    ``club["face_roll_deg"]``, which every engine reads through
    ``assembly_from_spec``. Raises ``ValueError`` for a spec without a club
    and as :func:`design_loft_face_roll_deg`; ``spec`` itself is not edited.
    """
    chosen = validate_mode(load_face_roll_settings().mode if mode is None else mode)
    assembly = assembly_from_spec(spec) if "bodies" in spec else None
    if assembly is None:
        raise ValueError("spec has no club body to calibrate")
    out = copy.deepcopy(dict(spec))
    club = dict(out.get("club") or {})
    roll = assembly.face_roll_deg
    if chosen == DESIGN_LOFT_AT_ADDRESS:
        roll = design_loft_face_roll_deg(
            load_club_head(assembly.head_alias),
            up_in_club_frame(address_body_world),
            near_deg=roll,
            loft_deg=club.get("design_loft_deg"),
        )
    club.update(face_roll_deg=roll, face_roll_calibration=chosen)
    out["club"] = club
    return out

"""Graft the Rajagopal-Lai-Uhlrich lower-limb muscles onto the full-body spec model.

Part of issue #11617 (epic #11605), phase 2.  The skeleton is the spec model
exported by ``export_full_body_osim`` (identical to the MuJoCo reference to
5e-12 rad), so inertias, joint frames and coordinates are those the reference
motion was produced with.  Only muscle geometry is added:

* Leg bodies ``femur``/``tibia``/``talus``/``calcn``/``toes`` of the spec carry the
  Rajagopal body frames (the spec lower limb was derived from the Rajagopal
  topology), uniformly scaled; muscle path points and wrap cylinders are copied
  after scaling the generic model with the per-body factors derived here.
* The Rajagopal pelvis frame is not a body of the spec.  Its orientation in the
  spec ``Hip`` body is recovered from the calibrated hip joints
  (``parent_to_base`` rotation with the zero-twist calibration removed) and its
  origin per side is chosen so that the Rajagopal hip centre lands on the spec
  (functionally calibrated) hip centre; pelvis muscle points and wrap cylinders
  are mapped with that rigid transform.
* The patella body, its patellofemoral joint and the coupler constraints, which
  the spec exporter drops, are re-added because eight quadriceps muscles attach
  to the patella.

Everything here that does not need OpenSim is pure numpy and unit tested.
"""

from __future__ import annotations

from collections.abc import Mapping
import json
from typing import Any

import numpy as np
from numpy.typing import ArrayLike
from scipy.spatial.transform import Rotation

from src.shared.python.contracts import require

Array = np.ndarray

LEG_BODIES: tuple[str, ...] = ("femur", "tibia", "talus", "calcn", "toes")
SIDES: tuple[str, ...] = ("r", "l")
SPEC_PELVIS_BODY = "Hip"
BASE_PELVIS_BODY = "pelvis"
#: Largest accepted disagreement between the two sides' recovered pelvis axes.
PELVIS_AXES_TOLERANCE = 1e-3


def rotation_z(angle_deg: float) -> Array:
    """3x3 rotation about +z by ``angle_deg`` degrees."""
    a = np.radians(angle_deg)
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def pelvis_frame_from_spec(
    spec: Mapping[str, Any],
) -> tuple[Array, dict[str, Array], float]:
    """Recover the Rajagopal pelvis frame inside the spec ``Hip`` body.

    The spec hip joints were built as ``P = H A X Rz(twist)`` (hip calibration):
    ``X`` is the Rajagopal hip frame in the Rajagopal pelvis (identity rotation),
    ``A`` the calibrated pelvis axes shared by both hips.  Removing the twist
    therefore gives ``A`` from either side.

    Returns:
        ``(rotation, centres, residual)``: the 3x3 rotation of the Rajagopal
        pelvis axes expressed in the ``Hip`` body, the spec hip-centre position
        per side in that body, and the largest element difference between the
        axes recovered from the two sides (a self-consistency residual).

    Raises:
        ValueError: if the hip joints or twist calibration are missing, or the
            two sides disagree by more than ``PELVIS_AXES_TOLERANCE``.
    """
    joints = {j["name"]: j for j in spec["joints"]}
    twist = spec.get("subject", {}).get("hip_zero_twist_deg")
    if not all(f"hip_{s}" in joints for s in SIDES) or twist is None:
        raise ValueError("spec needs hip_r/hip_l joints and hip_zero_twist_deg")
    axes: dict[str, Array] = {}
    centres: dict[str, Array] = {}
    for side in SIDES:
        parent = np.asarray(joints[f"hip_{side}"]["parent_to_base"], dtype=float)
        require(parent.shape == (4, 4), "parent_to_base must be 4x4")
        axes[side] = parent[:3, :3] @ rotation_z(-float(twist[side]))
        centres[side] = parent[:3, 3].copy()
    residual = float(np.abs(axes["r"] - axes["l"]).max())
    if residual > PELVIS_AXES_TOLERANCE:
        raise ValueError(f"hip joints disagree on the pelvis axes by {residual:.2e}")
    return 0.5 * (axes["r"] + axes["l"]), centres, residual


def map_pelvis_point(
    point: ArrayLike,
    rotation: Array,
    base_hip_centre: ArrayLike,
    spec_hip_centre: ArrayLike,
) -> Array:
    """Map a point of the (scaled) Rajagopal pelvis frame into the spec ``Hip`` body.

    ``p_hip = R (p - c_base) + c_spec`` so the Rajagopal hip centre lands on the
    spec hip centre and the geometry around it is preserved.
    """
    p = np.asarray(point, dtype=float)
    return rotation @ (p - np.asarray(base_hip_centre, float)) + np.asarray(
        spec_hip_centre, float
    )


def map_wrap_orientation(xyz_body_rotation: ArrayLike, rotation: Array) -> Array:
    """Compose a fixed rotation with an OpenSim ``xyz_body_rotation`` triple.

    OpenSim wrap objects store body-fixed X-Y-Z Euler angles; the result is the
    equivalent triple of ``rotation @ R(xyz)`` (radians).
    """
    local = Rotation.from_euler("XYZ", np.asarray(xyz_body_rotation, float))
    mapped = Rotation.from_matrix(rotation) * local
    return np.asarray(mapped.as_euler("XYZ"), dtype=float)


def leg_scale_factors(
    spec_mass_centres: Mapping[str, ArrayLike],
    base_mass_centres: Mapping[str, ArrayLike],
    *,
    talus_scale: float,
) -> dict[str, float]:
    """Uniform per-body scale of the generic leg that reproduces the spec leg.

    The scale of a body is the ratio of the norms of its mass-centre offset in the
    body frame (spec over generic).  The talus has its mass centre at the origin,
    so its factor must be supplied (``talus_scale``, from a joint-offset ratio).
    The pelvis takes the mean of the femur and tibia factors and the patella the
    femur factor.  Keys are ``"<body>_<side>"``.

    Raises:
        ValueError: if a mass-centre norm is not positive where one is needed.
    """
    require(talus_scale > 0, "talus_scale must be positive")
    scales: dict[str, float] = {}
    for side in SIDES:
        for body in LEG_BODIES:
            key = f"{body}_{side}"
            if body == "talus":
                scales[key] = float(talus_scale)
                continue
            spec_norm = float(np.linalg.norm(spec_mass_centres[key]))
            base_norm = float(np.linalg.norm(base_mass_centres[key]))
            if spec_norm <= 0.0 or base_norm <= 0.0:
                raise ValueError(f"cannot derive a scale for {key}: zero mass centre")
            scales[key] = spec_norm / base_norm
        scales[f"patella_{side}"] = scales[f"femur_{side}"]
    scales[BASE_PELVIS_BODY] = float(
        np.mean([scales[f"{b}_{s}"] for b in ("femur", "tibia") for s in SIDES])
    )
    return scales


def spec_leg_mass_centres(spec: Mapping[str, Any]) -> dict[str, Array]:
    """Mass-centre offsets (body frame, m) of the spec leg bodies."""
    centres: dict[str, Array] = {}
    for body in spec["bodies"]:
        name = body["name"]
        if name.rsplit("_", 1)[0] in LEG_BODIES and name.endswith(("_r", "_l")):
            solids = body["solids"]
            masses = np.array([s["mass_kg"] for s in solids], dtype=float)
            if masses.sum() <= 0.0:
                continue
            coms = np.array([s["com_m"] for s in solids], dtype=float)
            centres[name] = (masses[:, None] * coms).sum(axis=0) / masses.sum()
    return centres


def side_of(name: str) -> str | None:
    """Return ``"r"`` or ``"l"`` for names ending ``_r``/``_l``, else ``None``."""
    suffix = name.rsplit("_", 1)[-1]
    return suffix if suffix in SIDES else None


def spec_document(spec_bytes: bytes) -> dict[str, Any]:
    """Parse and minimally validate a full-body spec document."""
    spec = json.loads(spec_bytes)
    if spec.get("schema_version") != "full-body-v1":
        raise ValueError("expected a full-body-v1 specification")
    return spec

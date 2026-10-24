"""Limb (hip, knee, shoulder, elbow) anatomical angles in ISB and golf language (HRG-E3, #13171)."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any, NamedTuple

import numpy as np

from src.shared.python.anatomical_angles.trunk_pelvis import GlossaryEntry
from src.shared.python.biomechanics.joint_conventions import anatomical_joint_angles
from src.shared.python.core.handedness import lead_side, trail_side

__all__ = [
    "ELBOW_INPUT",
    "HIP_INPUT",
    "KNEE_INPUT",
    "LIMB_GLOSSARY",
    "LimbTermSpec",
    "SHOULDER_INPUT",
    "evaluator",
    "limb_term_specs",
]

_WU = "Wu et al. 2002, J Biomech 35:543 (ISB I); Wu et al. 2005, J Biomech 38:981 (ISB II)"
_GS = "Grood & Suntay 1983, J Biomech Eng 105:136"

HIP_INPUT = "femur_in_pelvis_rotation"
KNEE_INPUT = "tibia_in_femur_rotation"
SHOULDER_INPUT = "humerus_in_thorax_rotation"
ELBOW_INPUT = "forearm_in_humerus_rotation"


class LimbTermSpec(NamedTuple):
    term_id: str
    role: str
    base_name: str
    profile: str
    component: int
    sign_rule: str
    required_input: str
    frame: str
    sign_convention: str
    formula: str
    reference: str


# (base_name, profile, component, sign_rule, sign_convention, formula)
_SPECS_DATA = (
    (
        "hip_flexion",
        "isb_hip",
        0,
        "pos",
        "flexion positive (+FE); not handed",
        "FE of isb_hip ZXY",
    ),
    (
        "hip_abduction",
        "isb_hip",
        1,
        "neg_side",
        "abduction positive; same sense both sides",
        "-s_lr * AA of isb_hip ZXY",
    ),
    (
        "hip_internal_rotation",
        "isb_hip",
        2,
        "pos_side",
        "internal rotation positive; same sense both sides",
        "+s_lr * IER of isb_hip ZXY",
    ),
    (
        "knee_flexion",
        "grood_suntay_knee",
        0,
        "pos",
        "flexion positive; not handed",
        "flexion angle of grood_suntay_knee XYZ",
    ),
    (
        "shoulder_plane_of_elevation",
        "isb_thoracohumeral",
        0,
        "pos_side",
        "0 deg abd, 90 deg forward; same sense both sides",
        "s_lr * plane of YXY",
    ),
    (
        "shoulder_elevation",
        "isb_thoracohumeral",
        1,
        "neg",
        "elevation positive away from thorax; not handed",
        "-negative_elevation of YXY",
    ),
    (
        "shoulder_internal_rotation",
        "isb_thoracohumeral",
        2,
        "pos_side",
        "internal rotation positive; same sense both sides",
        "s_lr * axial rotation of YXY",
    ),
    (
        "shoulder_horizontal_adduction",
        "horizontal_adduction",
        -1,
        "horiz_adduction",
        "cross-body adduction positive; same sense both sides",
        "atan2(vx, s_lr*vz) of humerus distal axis",
    ),
    (
        "elbow_flexion",
        "isb_elbow",
        0,
        "pos",
        "flexion positive; not handed",
        "flexion angle of isb_elbow ZXY",
    ),
)


def _build_spec(role: str, row: tuple[str, str, int, str, str, str]) -> LimbTermSpec:
    name, prof, comp, rule, sign, form = row
    if "hip" in name:
        inp, ref, frame = HIP_INPUT, _WU, "isb_hip ZXY on R_pelvis->femur"
    elif "knee" in name:
        inp, ref, frame = KNEE_INPUT, _GS, "grood_suntay_knee XYZ on R_femur->tibia"
    elif "shoulder" in name:
        inp, ref = SHOULDER_INPUT, _WU
        frame = (
            "thorax ISB frame"
            if rule == "horiz_adduction"
            else "isb_thoracohumeral YXY"
        )
    else:
        inp, ref, frame = ELBOW_INPUT, _WU, "isb_elbow ZXY on R_humerus->forearm"
    return LimbTermSpec(
        f"{role}_{name}", role, name, prof, comp, rule, inp, frame, sign, form, ref
    )


def limb_term_specs() -> tuple[LimbTermSpec, ...]:
    return tuple(
        _build_spec(role, row) for role in ("lead", "trail") for row in _SPECS_DATA
    )


LIMB_GLOSSARY: dict[str, GlossaryEntry] = {
    s.term_id: GlossaryEntry(
        s.term_id, s.formula, s.sign_convention, s.frame, s.reference
    )
    for s in limb_term_specs()
}


def evaluator(spec: LimbTermSpec) -> Callable[[Mapping[str, Any], str], np.ndarray]:
    """(inputs, handedness) -> degrees for spec; NaN where singular."""

    def fn(inputs: Mapping[str, Any], handedness: str) -> np.ndarray:
        from src.shared.python.spine._golf_term_impl import rotation_series

        side = lead_side(handedness) if spec.role == "lead" else trail_side(handedness)
        s_lr = -1.0 if side == "L" else 1.0
        rot = rotation_series(inputs[spec.required_input], spec.required_input)

        if spec.sign_rule == "horiz_adduction":
            v = -rot[:, :, 1]
            vx, vz = v[:, 0], v[:, 2]
            rad = np.arctan2(vx, s_lr * vz)
            deg = np.rad2deg(rad)
            deg[(vx**2 + vz**2) < 1e-10] = np.nan
            return deg

        res = anatomical_joint_angles(rot, spec.profile)
        angles = np.array(res.values, dtype=float).reshape(-1, 3)
        singular = np.asarray(res.singular, dtype=bool).reshape(-1)
        angles[singular, 0] = np.nan
        angles[singular, 2] = np.nan

        raw = angles[:, spec.component]
        sign_map = {"pos": 1.0, "neg": -1.0, "neg_side": -s_lr, "pos_side": s_lr}
        return np.rad2deg(sign_map.get(spec.sign_rule, 1.0) * raw)

    return fn

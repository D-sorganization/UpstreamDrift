"""Pelvis and thorax orientation in the golfer's ISB lab frame (HRG-E3, #13171).

Frames
------
* Golfer lab frame (ISB global axes, Wu et al. 2002/2005): ``X`` faces the
  ball (horizontal, perpendicular to the target line), ``Y`` = ``up``, ``Z`` =
  the golfer's right (``X x Y``). A right-handed golfer has the target on the
  left, so ``Z = -target``; a left-handed golfer has ``Z = +target``. The frame
  is built from explicit ``up`` and ``target`` vectors, never an assumed lab
  axis, so the angles do not depend on the world frame.
* Segment frames: ISB-labelled pelvis (ISB part I 4.3, X anterior, Y cranial,
  Z right) and thorax (ISB part II 2.3.1) rotations in world coordinates.

Angles
------
``lab^T @ R_segment`` is decomposed with the ``isb_thorax`` Cardan sequence
``ZXY`` (ISB part II 2.4.1, thorax relative to global). The same sequence is
applied to the pelvis so that pelvis and thorax read alike; Baker 2001's
rotation-first pelvis sequence is not used. Raw components
``(FE, LB, AR)``: ``FE`` positive = extension (posterior tilt), ``LB`` positive
= right lateral flexion, ``AR`` positive = left axial rotation. At
``|LB| = 90 deg`` ``FE`` and ``AR`` are NaN (unavailable, never zero); ``LB``
remains.

Golf terms (registered in ``spine/golf_terms``) apply explicit signs: turn is
trail axial rotation (backswing positive for both hands), forward bend and
anterior tilt are ``-FE``, side bend is toward the lead side, and lead hip up is
the pelvis obliquity that raises the lead hip. Hip flexion (``isb_hip``) is
flexion-positive while ``isb_spine``/``isb_thorax`` are extension-positive; the
glossary records each sign explicitly.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np

from src.shared.python.biomechanics.joint_conventions import anatomical_joint_angles
from src.shared.python.core.handedness import lead_side

__all__ = [
    "GOLF_WORDS",
    "TRUNK_PELVIS_GLOSSARY",
    "UNAVAILABLE_V1",
    "GlossaryEntry",
    "TrunkPelvisTermSpec",
    "UnavailableQuantity",
    "golfer_lab_frame",
    "segment_lab_angles",
    "term_specs",
]

PELVIS_INPUT = "pelvis_rotation_world"
THORAX_INPUT = "thorax_rotation_world"
_FE, _LB, _AR = 0, 1, 2
_WU = (
    "Wu et al. 2002, J Biomech 35:543 (ISB part I 4.3, pelvis); Wu et al. "
    "2005, J Biomech 38:981 (ISB part II 2.3.1, 2.4.1, thorax-global ZXY)"
)
_LAB = (
    "golfer ISB lab frame (X toward ball, Y up, Z golfer's right = X x Y) from "
    "explicit up/target; isb_thorax ZXY on lab^T @ R_segment"
)


def _unit(value: object, name: str) -> np.ndarray:
    vec = np.asarray(value, dtype=float)
    if vec.shape != (3,) or not np.all(np.isfinite(vec)):
        raise ValueError(f"{name} must be a finite 3-vector")
    norm = float(np.linalg.norm(vec))
    if norm < 1e-12:
        raise ValueError(f"{name} must be nonzero")
    return vec / norm


def golfer_lab_frame(up: object, target: object, handedness: str) -> np.ndarray:
    """``(3, 3)`` matrix whose columns are the lab ISB X, Y, Z in world axes.

    Preconditions: ``up`` and ``target`` are finite nonzero 3-vectors, not
    parallel; ``handedness`` is ``"right"`` or ``"left"``. Raises
    ``ValueError`` otherwise. Postcondition: a proper rotation with ``Y = up``.
    """
    y = _unit(up, "up")
    t = _unit(target, "target")
    flat = t - (t @ y) * y
    if np.linalg.norm(flat) < 1e-9:
        raise ValueError("target must not be parallel to up")
    flat /= np.linalg.norm(flat)
    z = -flat if lead_side(handedness) == "L" else flat
    x = np.cross(y, z)
    return np.column_stack([x, y, z])


def _rotations(value: object, name: str) -> np.ndarray:
    """``(N, 3, 3)`` proper rotations, validated like every other registry term.

    Delegates to ``spine._golf_term_impl.rotation_series``. Imported at call
    time: ``spine`` imports ``golf_terms``, which imports this module.
    """
    from src.shared.python.spine._golf_term_impl import rotation_series

    return rotation_series(value, name)


def segment_lab_angles(
    segment_rotation_world: object,
    up: object,
    target: object,
    handedness: str,
    *,
    name: str = "segment_rotation_world",
) -> np.ndarray:
    """Raw ``(FE, LB, AR)`` radians of a segment in the golfer lab frame.

    Postconditions: shape ``(N, 3)``; ``FE`` and ``AR`` are NaN where the
    sample is gimbal-locked (``|LB| = 90 deg``), never zero.
    """
    rot = _rotations(segment_rotation_world, name)
    lab = golfer_lab_frame(up, target, handedness)
    result = anatomical_joint_angles(lab.T @ rot, "isb_thorax")
    angles = np.array(result.values, dtype=float).reshape(-1, 3)
    singular = np.asarray(result.singular, dtype=bool).reshape(-1)
    angles[singular, _FE] = np.nan
    angles[singular, _AR] = np.nan
    return angles


def _trail_sign(handedness: str) -> float:
    """Sign that turns ``+AR`` (left axial) into trail rotation (and ``+LB``
    into lead-side bend): ``-1`` when the lead side is left."""
    return -1.0 if lead_side(handedness) == "L" else 1.0


@dataclass(frozen=True)
class TrunkPelvisTermSpec:
    """One lab-frame pelvis/thorax term: component and sign rule."""

    term_id: str
    segment_input: str
    component: int
    sign_rule: str  # "trail" | "lead_hip_up" | "negate"
    description: str
    sign_convention: str
    formula: str
    golf_words: tuple[str, ...]


def _spec(
    term_id: str,
    seg: str,
    comp: int,
    rule: str,
    desc: str,
    sign: str,
    formula: str,
    words: tuple[str, ...],
) -> TrunkPelvisTermSpec:
    return TrunkPelvisTermSpec(term_id, seg, comp, rule, desc, sign, formula, words)


_SPECS = (
    _spec(
        "pelvis_turn_isb",
        PELVIS_INPUT,
        _AR,
        "trail",
        "Pelvis axial rotation in the golfer lab frame (ISB triple, not the "
        "line method of x_factor_line_turn)",
        "trail rotation positive (backswing positive both hands): -AR right-"
        "handed, +AR left-handed",
        "AR of isb_thorax ZXY(lab^T R_pelvis) times trail sign",
        ("hip turn",),
    ),
    _spec(
        "pelvis_anterior_tilt_isb",
        PELVIS_INPUT,
        _FE,
        "negate",
        "Pelvis anterior tilt in the golfer's sagittal plane",
        "anterior tilt positive (-FE; isb_thorax FE is extension-positive); not handed",
        "-FE of isb_thorax ZXY(lab^T R_pelvis)",
        ("pelvic tilt",),
    ),
    _spec(
        "pelvis_lead_hip_up_isb",
        PELVIS_INPUT,
        _LB,
        "lead_hip_up",
        "Pelvis obliquity (list) in the golfer's frontal plane",
        "lead hip higher positive: +LB right-handed, -LB left-handed",
        "LB of isb_thorax ZXY(lab^T R_pelvis) times -trail sign",
        ("lead hip up", "pelvic obliquity"),
    ),
    _spec(
        "thorax_turn_isb",
        THORAX_INPUT,
        _AR,
        "trail",
        "Thorax axial rotation in the golfer lab frame (ISB triple)",
        "trail rotation positive (backswing positive both hands): -AR right-"
        "handed, +AR left-handed",
        "AR of isb_thorax ZXY(lab^T R_thorax) times trail sign",
        ("shoulder turn", "chest turn"),
    ),
    _spec(
        "thorax_forward_bend_isb",
        THORAX_INPUT,
        _FE,
        "negate",
        "Thorax forward bend toward the ball in the golfer's sagittal plane",
        "toward the ball positive (-FE; isb_thorax FE is extension-positive); "
        "not handed",
        "-FE of isb_thorax ZXY(lab^T R_thorax)",
        ("forward bend", "spine tilt forward"),
    ),
    _spec(
        "thorax_lead_side_bend_isb",
        THORAX_INPUT,
        _LB,
        "trail",
        "Thorax side bend in the golfer's frontal plane (lab frame; the "
        "pelvis-relative bend is trunk_lateral_flexion_jcs)",
        "toward the lead side positive: -LB right-handed, +LB left-handed",
        "LB of isb_thorax ZXY(lab^T R_thorax) times trail sign",
        ("side bend",),
    ),
)


def term_specs() -> tuple[TrunkPelvisTermSpec, ...]:
    """Specs of the lab-frame pelvis and thorax terms, in registry order."""
    return _SPECS


def term_frame() -> str:
    return _LAB


def term_reference() -> str:
    return _WU


def required_inputs(spec: TrunkPelvisTermSpec) -> tuple[str, ...]:
    return (spec.segment_input, "up", "target")


def evaluator(
    spec: TrunkPelvisTermSpec,
) -> Callable[[Mapping[str, Any], str], np.ndarray]:
    """``(inputs, handedness) -> degrees`` for ``spec``; NaN where singular."""

    def fn(inputs: Mapping[str, Any], handedness: str) -> np.ndarray:
        raw = segment_lab_angles(
            inputs[spec.segment_input],
            inputs["up"],
            inputs["target"],
            handedness,
            name=spec.segment_input,
        )[:, spec.component]
        if spec.sign_rule == "negate":
            sign = -1.0
        elif spec.sign_rule == "trail":
            sign = _trail_sign(handedness)
        else:  # lead_hip_up
            sign = -_trail_sign(handedness)
        return np.rad2deg(sign * raw)

    return fn


@dataclass(frozen=True)
class GlossaryEntry:
    term_id: str
    formula: str
    sign: str
    frame: str
    reference: str
    available: bool = True


@dataclass(frozen=True)
class UnavailableQuantity:
    name: str
    reason: str
    available: bool = False


TRUNK_PELVIS_GLOSSARY: dict[str, GlossaryEntry] = {
    s.term_id: GlossaryEntry(s.term_id, s.formula, s.sign_convention, _LAB, _WU)
    for s in _SPECS
}
#: Golf words -> registered term ids. "Shoulder turn" means thorax rotation,
#: not glenohumeral motion.
GOLF_WORDS: dict[str, str] = {w: s.term_id for s in _SPECS for w in s.golf_words}
_SINGLE_TRUNK = (
    "full-body v1 models a single trunk joint (pelvis->thorax); a thoracic "
    "versus lumbar split is not modelled and is reported unavailable, not zero"
)
UNAVAILABLE_V1: dict[str, UnavailableQuantity] = {
    n: UnavailableQuantity(n, _SINGLE_TRUNK)
    for n in (
        "thoracic_axial_rotation",
        "lumbar_axial_rotation",
        "thoracic_lateral_flexion",
        "lumbar_lateral_flexion",
        "thoracic_extension",
        "lumbar_extension",
    )
}

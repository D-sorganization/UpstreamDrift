"""Versioned golf-term registry with explicit inputs (SPN-4a, ADR-0055 items 4, 8).

Every golf-instruction quantity ("X-factor", "crunch", "early extension", ...)
is a registered, versioned measurand with named inputs, a frame, a sign
convention and units. Existing implementations keep their numeric meaning under
legacy ids; new ids are new measurands. Terms are *evaluated*, never guessed:
unknown ids, missing inputs and bad handedness raise ``ValueError`` naming the
problem.

Sign convention (ADR-0055 item 4). ISB spine angles ``(FE, LB, AR)``: FE
positive = extension, LB positive = right lateral bending, AR positive = left
axial rotation (same sense as ``isb_thorax`` ``left_axial_rotation``). For a
right-handed golfer ``trail_axial_rotation = -AR`` and
``lead_lateral_flexion = -LB``; a left-handed golfer flips both, so backswing
trunk rotation is positive for both. Extension is not handed. Lab-frame terms
take explicit ``up`` and ``target`` vectors; no lab axis convention is assumed.

Delegation: line turn calls ``swing_comparison.turn``; side bend and reverse
spine angle call ``motion_matching.posture_metrics``; JCS terms call
``biomechanics.joint_conventions`` (profile ``isb_spine``). Legacy terms
re-state the legacy formula and are pinned to the legacy implementations by
characterisation tests.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

from src.shared.python.spine._golf_term_impl import (
    Handedness,
    axis_series,
    check_handedness,
    float_array,
    handed_sign,
    horizontal_target,
    index_in_range,
    isb_spine_angles_deg,
    require_inputs,
    rotate_to_z_up,
    tilt_components,
    unit_vector,
)
from src.shared.python.spine.levels import SpinalLevel
from src.shared.python.spine.segmental import SegmentalPose
from src.shared.python.swing_comparison.turn import MARKER_PAIRS, compute_turn_lines

__all__ = [
    "GolfTermDefinition",
    "contribution_remainder",
    "definition",
    "evaluate",
    "registered_terms",
    "segmental_contributions",
]

_ROT = "pelvis_to_thorax_rotation"
_ISB_ADR = "ADR-0055 item 8"


@dataclass(frozen=True)
class GolfTermDefinition:
    """Self-describing definition of one registered golf term.

    Preconditions: every field is a non-empty string (``required_inputs`` a
    non-empty tuple of non-empty strings); violations raise ``ValueError``.
    """

    term_id: str
    version: str
    description: str
    frame: str
    sign_convention: str
    units: str
    required_inputs: tuple[str, ...]
    reference: str

    def __post_init__(self) -> None:
        for name in ("term_id", "version", "description", "frame"):
            _non_empty(name, getattr(self, name))
        _non_empty("sign_convention", self.sign_convention)
        _non_empty("units", self.units)
        _non_empty("reference", self.reference)
        if not self.required_inputs or not all(
            isinstance(n, str) and n.strip() for n in self.required_inputs
        ):
            raise ValueError("required_inputs must be non-empty names")


def _non_empty(name: str, value: object) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")


# -- evaluators: (inputs, handedness) -> 1-D float array -----------------------


def _jcs_axial(inputs: Mapping[str, Any], hand: Handedness) -> np.ndarray:
    return handed_sign(hand) * isb_spine_angles_deg(inputs[_ROT], _ROT)[:, 2]


def _jcs_extension(inputs: Mapping[str, Any], hand: Handedness) -> np.ndarray:
    del hand  # extension is not handed
    return isb_spine_angles_deg(inputs[_ROT], _ROT)[:, 0]


def _jcs_lateral(inputs: Mapping[str, Any], hand: Handedness) -> np.ndarray:
    return handed_sign(hand) * isb_spine_angles_deg(inputs[_ROT], _ROT)[:, 1]


def _line_turn(inputs: Mapping[str, Any], hand: Handedness) -> np.ndarray:
    up = unit_vector(inputs["up"], "up")
    points = inputs["points"]
    if not isinstance(points, Mapping):
        raise TypeError("points must be a mapping of name -> (N, 3) array")
    flat = {
        k: rotate_to_z_up(_points_array(v, f"points[{k}]"), up)
        for k, v in points.items()
    }
    pairs = inputs.get("point_pairs", MARKER_PAIRS)
    metrics = compute_turn_lines(flat, inputs["t"], inputs["events"], pairs=pairs)
    turn = np.array(metrics.x_factor.turn_deg, dtype=float)
    # turn.py is positive for the right-handed backswing (clockwise from above)
    return turn if hand == "right" else -turn


def _points_array(value: object, name: str) -> np.ndarray:
    """``(N, 3)`` float array that may hold NaN gaps (marker dropouts)."""
    try:
        arr = np.asarray(value, dtype=float)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must be a numeric array: {exc}") from exc
    if arr.ndim != 2 or arr.shape[1] != 3:
        raise ValueError(f"{name} must have shape (N, 3), got {arr.shape}")
    return arr


def _position_proxy(inputs: Mapping[str, Any], hand: Handedness) -> np.ndarray:
    del hand  # legacy sign is as implemented
    pos = float_array(inputs["joint_positions"], "joint_positions", 2, allow_nan=True)
    shoulder = index_in_range(
        inputs["shoulder_joint_idx"], pos.shape[1], "shoulder_joint_idx"
    )
    hip = index_in_range(inputs["hip_joint_idx"], pos.shape[1], "hip_joint_idx")
    # swing_metrics.SwingMetricsMixin.compute_x_factor: shoulder - hip, deg
    return np.rad2deg(pos[:, shoulder]) - np.rad2deg(pos[:, hip])


def _euler_yaw(inputs: Mapping[str, Any], hand: Handedness) -> np.ndarray:
    del hand  # legacy sign is as implemented
    thorax = float_array(inputs["thorax_euler"], "thorax_euler", (1, 2))
    pelvis = float_array(inputs["pelvis_euler"], "pelvis_euler", (1, 2))
    yaw_t = thorax[:, 2] if thorax.ndim > 1 else thorax
    yaw_p = pelvis[:, 2] if pelvis.ndim > 1 else pelvis
    if yaw_t.shape != yaw_p.shape:
        raise ValueError("thorax_euler and pelvis_euler must have equal length")
    return np.degrees(yaw_t - yaw_p)


def _side_bend(inputs: Mapping[str, Any], hand: Handedness) -> np.ndarray:
    del hand  # lab-frame, direction given by `target`
    axes = axis_series(inputs["trunk_axis"], "trunk_axis")
    up = unit_vector(inputs["up"], "up")
    target = unit_vector(inputs["target"], "target")
    out = tilt_components(axes, up, target)
    return out[0] if np.ndim(inputs["trunk_axis"]) == 1 else out


def _crunch_legacy(inputs: Mapping[str, Any], hand: Handedness) -> np.ndarray:
    del hand  # legacy |lat| * |rot| / 100 is sign-free
    lat = float_array(inputs["lateral_bend_rad"], "lateral_bend_rad", 1)
    rot = float_array(inputs["axial_rotation_rad"], "axial_rotation_rad", 1)
    if lat.shape != rot.shape:
        raise ValueError("lateral_bend_rad and axial_rotation_rad must match")
    return np.abs(np.degrees(lat)) * np.abs(np.degrees(rot)) / 100.0


def _crunch_rate(inputs: Mapping[str, Any], hand: Handedness) -> np.ndarray:
    del hand  # inputs are already handed JCS terms
    lat = float_array(
        inputs["trunk_lateral_flexion_jcs"], "trunk_lateral_flexion_jcs", 1
    )
    axial = float_array(inputs["x_factor_jcs_axial"], "x_factor_jcs_axial", 1)
    time = float_array(inputs["time_s"], "time_s", 1)
    if not (lat.shape == axial.shape == time.shape) or time.size < 2:
        raise ValueError("series and time_s must have equal length of at least 2")
    if np.any(np.diff(time) <= 0):
        raise ValueError("time_s must be strictly increasing")
    return lat * np.gradient(axial, time)


def _early_extension(inputs: Mapping[str, Any], hand: Handedness) -> np.ndarray:
    del hand  # distance to the target line is side-free
    pelvis = float_array(inputs["pelvis_position"], "pelvis_position", 2)
    if pelvis.shape[1] != 3:
        raise ValueError("pelvis_position must have shape (N, 3)")
    up = unit_vector(inputs["up"], "up")
    along = horizontal_target(up, unit_vector(inputs["target"], "target"))
    ball = float_array(inputs["ball_position"], "ball_position", 1)
    if ball.shape != (3,):
        raise ValueError("ball_position must be a 3-vector")
    rel = pelvis - ball
    rel = rel - np.outer(rel @ up, up)  # horizontal
    rel = rel - np.outer(rel @ along, along)  # perpendicular to the target line
    dist = np.linalg.norm(rel, axis=1)
    address = index_in_range(inputs["events"].address_idx, dist.size, "address_idx")
    return dist[address] - dist


def _reverse_spine(inputs: Mapping[str, Any], hand: Handedness) -> np.ndarray:
    del hand  # lab-frame, direction given by `target`
    axes = axis_series(inputs["trunk_axis"], "trunk_axis")
    up = unit_vector(inputs["up"], "up")
    target = unit_vector(inputs["target"], "target")
    top = index_in_range(inputs["events"].top_idx, axes.shape[0], "top_idx")
    return tilt_components(axes[top : top + 1], up, target)[:, 1]


@dataclass(frozen=True)
class _Term:
    spec: GolfTermDefinition
    fn: Callable[[Mapping[str, Any], Handedness], np.ndarray]


def _term(
    term_id: str,
    fn: Callable[[Mapping[str, Any], Handedness], np.ndarray],
    **kw: Any,
) -> _Term:
    return _Term(GolfTermDefinition(term_id=term_id, version="1", **kw), fn)


_ISB_JCS = "isb_spine ZXY on R_pelvis->thorax (X anterior, Y cranial, Z right)"
_LAB = "explicit up and target vectors in the pose frame"

_TERMS: dict[str, _Term] = {
    t.spec.term_id: t
    for t in (
        _term(
            "x_factor_jcs_axial",
            _jcs_axial,
            description="trail_axial_rotation of the pelvis->thorax rotation",
            frame=_ISB_JCS,
            sign_convention="-AR (right-handed) / +AR (left-handed); backswing positive",
            units="deg",
            required_inputs=(_ROT,),
            reference=_ISB_ADR + "; Wu et al. 2002, J Biomech 35:543",
        ),
        _term(
            "x_factor_line_turn",
            _line_turn,
            description="Upper-trunk line turn minus pelvis line turn, horizontal plane",
            frame=_LAB + "; points rotated so up is +Z before swing_comparison.turn",
            sign_convention="turn.py positive in the right-handed backswing; negated for left",
            units="deg",
            required_inputs=("points", "t", "events", "up"),
            reference=_ISB_ADR
            + "; swing_comparison/turn.py (optional input point_pairs)",
        ),
        _term(
            "x_factor_position_proxy",
            _position_proxy,
            description="Legacy joint-coordinate proxy: shoulder minus hip rotation",
            frame="engine generalized coordinates (rad), columns by index",
            sign_convention="as implemented (shoulder - hip), not handed",
            units="deg",
            required_inputs=("joint_positions", "shoulder_joint_idx", "hip_joint_idx"),
            reference=_ISB_ADR + "; analysis/swing_metrics.py::compute_x_factor",
        ),
        _term(
            "x_factor_euler_yaw_legacy_v1",
            _euler_yaw,
            description="Thorax yaw minus pelvis yaw from [roll, pitch, yaw] Euler angles",
            frame="Euler angles (rad), third column yaw",
            sign_convention="as implemented (thorax - pelvis), not handed",
            units="deg",
            required_inputs=("thorax_euler", "pelvis_euler"),
            reference=_ISB_ADR + "; injury/spinal_load_analysis.py::_compute_x_factor",
        ),
        _term(
            "trunk_extension_jcs",
            _jcs_extension,
            description="Flexion/extension angle of the pelvis->thorax rotation",
            frame=_ISB_JCS,
            sign_convention="+FE = extension; not handed",
            units="deg",
            required_inputs=(_ROT,),
            reference=_ISB_ADR + "; Wu et al. 2002, J Biomech 35:543",
        ),
        _term(
            "trunk_lateral_flexion_jcs",
            _jcs_lateral,
            description="lead_lateral_flexion of the pelvis->thorax rotation",
            frame=_ISB_JCS,
            sign_convention="-LB (right-handed) / +LB (left-handed); lead side positive",
            units="deg",
            required_inputs=(_ROT,),
            reference=_ISB_ADR + "; Wu et al. 2002, J Biomech 35:543",
        ),
        _term(
            "side_bend_tilt",
            _side_bend,
            description="Trunk-axis tilt from up as [total, toward_target, lateral]",
            frame=_LAB + "; lateral axis = target_horizontal x up",
            sign_convention="toward_target positive toward target; lateral positive toward target x up",
            units="deg",
            required_inputs=("trunk_axis", "up", "target"),
            reference=_ISB_ADR + "; motion_matching/posture_metrics.py::segment_tilt",
        ),
        _term(
            "crunch_factor_legacy_v1",
            _crunch_legacy,
            description="|lateral deg| * |rotation deg| / 100 (existing analyzer)",
            frame="angles about the lumbar lateral and rotation axes (rad in)",
            sign_convention="sign-free product of magnitudes",
            units="deg^2 / 100",
            required_inputs=("lateral_bend_rad", "axial_rotation_rad"),
            reference=_ISB_ADR
            + "; injury/spinal_load_analysis.py::_compute_crunch_factor",
        ),
        _term(
            "crunch_factor_rate",
            _crunch_rate,
            description="Lateral flexion angle times axial rotation angular velocity",
            frame="outputs of trunk_lateral_flexion_jcs and x_factor_jcs_axial",
            sign_convention="signed; inherits handed JCS signs",
            units="deg*deg/s",
            required_inputs=(
                "trunk_lateral_flexion_jcs",
                "x_factor_jcs_axial",
                "time_s",
            ),
            reference="Sugaya, Tsuchiya & Moriya 1999, Science and Golf III; cited "
            "via Cole & Grimshaw 2014 (primary unverified, #12737)",
        ),
        _term(
            "early_extension",
            _early_extension,
            description="Reduction of pelvis distance to the target line since address",
            frame=_LAB + "; target line through ball_position (ball_position is an "
            "added required input: the line needs a point)",
            sign_convention="toward the ball positive; along-target motion excluded",
            units="m",
            required_inputs=(
                "pelvis_position",
                "target",
                "up",
                "events",
                "ball_position",
            ),
            reference=_ISB_ADR + "; events.address_idx",
        ),
        _term(
            "reverse_spine_angle",
            _reverse_spine,
            description="Trunk-axis tilt toward the target at the top of backswing",
            frame=_LAB + "; vertical plane containing the target line",
            sign_convention="toward target positive",
            units="deg",
            required_inputs=("trunk_axis", "up", "target", "events"),
            reference=_ISB_ADR + "; events.top_idx; posture_metrics.segment_tilt",
        ),
    )
}

_JCS_TERMS = ("x_factor_jcs_axial", "trunk_extension_jcs", "trunk_lateral_flexion_jcs")


def _lookup(term_id: str) -> _Term:
    if not isinstance(term_id, str):
        raise TypeError(f"term_id must be a str, got {type(term_id).__name__}")
    try:
        return _TERMS[term_id]
    except KeyError:
        raise ValueError(
            f"unknown golf term {term_id!r}; registered: {', '.join(_TERMS)}"
        ) from None


def registered_terms() -> tuple[str, ...]:
    """Registered term ids, in registry order."""
    return tuple(_TERMS)


def definition(term_id: str) -> GolfTermDefinition:
    """Definition of ``term_id``.

    Raises:
        TypeError: if ``term_id`` is not a ``str``.
        ValueError: if ``term_id`` is not registered.
    """
    return _lookup(term_id).spec


def evaluate(
    term_id: str,
    inputs: Mapping[str, Any],
    *,
    handedness: Literal["right", "left"],
) -> np.ndarray:
    """Evaluate a registered term.

    Preconditions: ``term_id`` is registered; ``inputs`` holds every name in
    ``definition(term_id).required_inputs``; ``handedness`` is ``"right"`` or
    ``"left"`` (validated for every term, applied only where the term is
    handed). Rotation inputs are ``(3, 3)`` or ``(N, 3, 3)`` proper rotations.

    Postconditions: returns a new 1-D float array in the units of the
    definition (length N, or 1 for a single rotation); unavailable samples are
    NaN, never zero (e.g. gimbal-locked JCS samples, marker dropouts).
    ``reverse_spine_angle`` returns the single value at ``events.top_idx``.

    Raises:
        TypeError: for wrongly typed inputs.
        ValueError: for an unknown term, missing inputs (named), bad handedness
            or invalid values.
    """
    term = _lookup(term_id)
    hand = check_handedness(handedness)
    require_inputs(term_id, inputs, term.spec.required_inputs)
    out = np.asarray(term.fn(inputs, hand), dtype=float)
    return out


def _jcs_value(term_id: str, rot: np.ndarray, hand: Handedness) -> float:
    return float(_TERMS[term_id].fn({_ROT: rot}, hand)[0])


def _prefix_values(term_id: str, pose: SegmentalPose, hand: Handedness) -> list[float]:
    first = pose.levels[0]
    return [
        _jcs_value(term_id, pose.relative_rotation(first, lv), hand)
        for lv in pose.levels
    ]


def _check_jcs_request(term_id: str, pose: object, handedness: object) -> Handedness:
    if term_id not in _JCS_TERMS:
        raise ValueError(
            f"{term_id!r} has no segmental decomposition; JCS terms: "
            f"{', '.join(_JCS_TERMS)}"
        )
    if not isinstance(pose, SegmentalPose):
        raise TypeError(f"pose must be a SegmentalPose, got {type(pose).__name__}")
    return check_handedness(handedness)


def segmental_contributions(
    term_id: str, pose: SegmentalPose, *, handedness: Literal["right", "left"]
) -> dict[SpinalLevel, float]:
    """Per-level contribution of a JCS term (a decomposition, not an identity).

    The contribution of level ``i`` is ``term(chain truncated after i) -
    term(chain truncated before i)``, with the chain composed caudal factor
    first (``pose.relative_rotation``) and the term of the empty chain taken as
    0 (so the first level carries its neutral offset). The values telescope to
    the global term exactly (the prefix differences telescope), but they are
    NOT the level's own Cardan angle and NOT additive in angle space: a level's
    share depends on the levels caudal to it (``angle_sum_diagnostic`` differs
    from the global value). :func:`contribution_remainder` reports the
    numerical shortfall against the global value, which is ~0 by construction
    and is non-zero only through float rounding (NaN if a prefix is
    gimbal-locked).

    Preconditions: ``term_id`` in ``x_factor_jcs_axial``,
    ``trunk_extension_jcs``, ``trunk_lateral_flexion_jcs``; ``pose`` is a
    :class:`SegmentalPose`; ``handedness`` is ``"right"`` or ``"left"``
    (keyword-only and required: the signed terms are handed, so no default is
    guessed). Postcondition: keys are ``pose.levels`` in caudal-to-cranial order;
    values are degrees (NaN where a prefix is gimbal-locked).

    Raises:
        TypeError: if ``pose`` is not a :class:`SegmentalPose`.
        ValueError: for a non-JCS term or bad handedness.
    """
    hand = _check_jcs_request(term_id, pose, handedness)
    prefix = _prefix_values(term_id, pose, hand)
    return {
        lv: cur - prev
        for lv, cur, prev in zip(pose.levels, prefix, [0.0, *prefix[:-1]], strict=True)
    }


def contribution_remainder(
    term_id: str, pose: SegmentalPose, *, handedness: Literal["right", "left"]
) -> float:
    """Global term value minus the sum of :func:`segmental_contributions`.

    Degrees. The contributions telescope over the cumulative prefixes, so this
    is ~0 up to rounding; it is reported (not asserted) so a consumer can see
    the decomposition closes. Same preconditions and errors as
    :func:`segmental_contributions`.
    """
    hand = _check_jcs_request(term_id, pose, handedness)
    contrib = segmental_contributions(term_id, pose, handedness=hand)
    glob = _jcs_value(term_id, pose.compose()[:3, :3], hand)
    return glob - float(sum(contrib.values()))

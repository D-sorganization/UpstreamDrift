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
from types import MappingProxyType
from typing import Any, Literal

import numpy as np

from src.shared.python.contracts import ensure
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
    single_rotation,
    rotate_to_z_up,
    tilt_components,
    unit_vector,
)
from src.shared.python.spine.levels import SpinalLevel
from src.shared.python.spine.segmental import SegmentalPose
from src.shared.python.swing_comparison.turn import MARKER_PAIRS, compute_turn_lines

__all__ = [
    "GolfTermContributions",
    "GolfTermDefinition",
    "contribution_remainder",
    "definition",
    "evaluate",
    "registered_terms",
    "segmental_contribution_breakdown",
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
    thorax = float_array(inputs["thorax_euler"], "thorax_euler", (1, 2), allow_nan=True)
    pelvis = float_array(inputs["pelvis_euler"], "pelvis_euler", (1, 2), allow_nan=True)
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
    lat = float_array(inputs["lateral_bend_rad"], "lateral_bend_rad", 1, allow_nan=True)
    rot = float_array(
        inputs["axial_rotation_rad"], "axial_rotation_rad", 1, allow_nan=True
    )
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
    handed). Rotation inputs (the JCS terms) are ``(3, 3)`` or ``(N, 3, 3)``
    proper rotations. ``side_bend_tilt`` takes no rotation: ``trunk_axis`` is a
    nonzero 3-vector ``(3,)`` or a series ``(N, 3)`` of them, with unit-normalised
    ``up`` and ``target`` vectors of shape ``(3,)``; any other shape raises
    ``ValueError``.

    Postconditions: returns a new float array in the units of the definition.
    Every term is 1-D (length N, or 1 for a single rotation) except
    ``side_bend_tilt``, which returns ``[total, toward_target, lateral]`` as
    shape ``(3,)`` for a ``(3,)`` axis and ``(N, 3)`` for an ``(N, 3)`` series.
    Unavailable samples are NaN, never zero (e.g. gimbal-locked JCS samples,
    marker dropouts). ``reverse_spine_angle`` returns the single value at
    ``events.top_idx``.

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


def _check_jcs_request(term_id: str, pose: object, handedness: object) -> Handedness:
    if term_id not in _JCS_TERMS:
        raise ValueError(
            f"{term_id!r} has no segmental decomposition; JCS terms: "
            f"{', '.join(_JCS_TERMS)}"
        )
    if not isinstance(pose, SegmentalPose):
        raise TypeError(f"pose must be a SegmentalPose, got {type(pose).__name__}")
    return check_handedness(handedness)


_CLOSURE_TOL_DEG = 1e-9


@dataclass(frozen=True)
class GolfTermContributions:
    """Per-level and named non-level contributions of one JCS golf term.

    ``levels`` maps each :class:`SpinalLevel` of the pose (caudal to cranial)
    to its decomposition share. ``named`` holds the shares of the fixed
    rotations that lie outside the level chain, keyed ``"base"`` (``B``,
    pelvis to first body), ``"terminal"`` (``E``, last vertebra to terminal
    body) and ``"alignment"`` (``A``, ``R_target = R_pelvis->torso @ A``); only
    the rotations the caller supplied appear, and none is folded into a level.
    ``global_value`` is the term of ``B @ chain @ E @ A``. Degrees. Mappings
    are read-only.
    """

    levels: Mapping[SpinalLevel, float]
    named: Mapping[str, float]
    global_value: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "levels", MappingProxyType(dict(self.levels)))
        object.__setattr__(self, "named", MappingProxyType(dict(self.named)))

    @property
    def total(self) -> float:
        """Sum of the level shares plus the named entries (degrees)."""
        return float(sum(self.levels.values()) + sum(self.named.values()))

    @property
    def remainder(self) -> float:
        """``global_value - total``: ~0 up to rounding (NaN if gimbal-locked)."""
        return self.global_value - self.total


def _prefix_chain(
    term_id: str,
    pose: SegmentalPose,
    hand: Handedness,
    base: np.ndarray | None,
    terminal: np.ndarray | None,
    alignment: np.ndarray | None,
) -> tuple[list[float], dict[str, float], float]:
    """Cumulative term values: level prefixes, named steps, final global value.

    The cumulative rotation is ``B``, then ``B @ chain_1..k``, then
    ``... @ E``, then ``... @ A`` (each factor present only if supplied); the
    term of the empty chain is 0, so with all of ``B``, ``E``, ``A`` absent the
    level values equal the legacy chain-only prefixes.
    """
    lead = np.eye(3) if base is None else base
    first = pose.levels[0]
    prefix = [
        _jcs_value(term_id, lead @ pose.relative_rotation(first, lv), hand)
        for lv in pose.levels
    ]
    cumulative = lead @ pose.compose()[:3, :3]
    named: dict[str, float] = {}
    previous = prefix[-1]
    if base is not None:
        named["base"] = _jcs_value(term_id, base, hand)
    for key, rot in (("terminal", terminal), ("alignment", alignment)):
        if rot is None:
            continue
        cumulative = cumulative @ rot
        value = _jcs_value(term_id, cumulative, hand)
        named[key] = value - previous
        previous = value
    return prefix, named, previous


def segmental_contribution_breakdown(
    term_id: str,
    pose: SegmentalPose,
    *,
    handedness: Literal["right", "left"],
    base: object | None = None,
    terminal: object | None = None,
    alignment: object | None = None,
) -> GolfTermContributions:
    """Per-level plus named non-level contributions that close on the global term.

    The global term is ``f(B @ chain @ E @ A)`` where ``chain`` is
    ``pose.compose()``. With cumulative rotations ``P_0 = B``,
    ``P_k = B @ chain_1..k``, ``P_E = P_n @ E`` and ``P_A = P_E @ A`` (factors
    omitted when not supplied; the empty chain has term 0)::

        base      = f(B)                    (only if base supplied)
        level k   = f(P_k) - f(P_{k-1})     (f(P_0) = f(B), or 0 without B)
        terminal  = f(P_E) - f(P_n)         (only if terminal supplied)
        alignment = f(P_A) - f(P_E or P_n)  (only if alignment supplied)

    These telescope to ``f(P_A)`` exactly, so
    ``sum(levels) + sum(named) == global_value`` up to rounding. When none of
    ``base``/``terminal``/``alignment`` is given, ``levels`` equals
    :func:`segmental_contributions` and ``named`` is empty. As there, the shares
    are path dependent, not Cardan angles of the levels.

    Preconditions: as :func:`segmental_contributions`; each of ``base``,
    ``terminal``, ``alignment`` is ``None`` or a single finite proper
    ``(3, 3)`` rotation (``TypeError`` if not numeric, ``ValueError`` for wrong
    shape or a non-rotation, naming the argument).

    Postconditions: ``levels`` keys are ``pose.levels``; ``named`` keys are a
    subset of ``base``, ``terminal``, ``alignment``; when all values are finite
    the closure ``|remainder| <= 1e-9`` deg holds (``ensure``). Samples that
    are gimbal-locked give NaN shares, never zero.
    """
    hand = _check_jcs_request(term_id, pose, handedness)
    rots = [
        None if r is None else single_rotation(r, name)
        for name, r in (
            ("base", base),
            ("terminal", terminal),
            ("alignment", alignment),
        )
    ]
    prefix, named, glob = _prefix_chain(term_id, pose, hand, *rots)
    start = named.get("base", 0.0) if rots[0] is not None else 0.0
    levels = {
        lv: cur - prev
        for lv, cur, prev in zip(
            pose.levels, prefix, [start, *prefix[:-1]], strict=True
        )
    }
    result = GolfTermContributions(levels=levels, named=named, global_value=glob)
    if np.isfinite(result.total) and np.isfinite(glob):
        ensure(
            abs(result.remainder) <= _CLOSURE_TOL_DEG,
            "level shares plus named entries must sum to the global term",
            result.remainder,
        )
    return result


def segmental_contributions(
    term_id: str, pose: SegmentalPose, *, handedness: Literal["right", "left"]
) -> dict[SpinalLevel, float]:
    """Per-level contribution of a JCS term (a decomposition, not an identity).

    The contribution of level ``i`` is ``term(chain truncated after i) -
    term(chain truncated before i)``, with the chain composed caudal factor
    first (``pose.relative_rotation``) and the term of the empty chain taken as
    0 (so the first level carries its neutral offset). The values telescope to
    the term of the supplied chain exactly (the prefix differences telescope),
    but they are NOT the level's own Cardan angle and NOT additive in angle
    space: a level's share depends on the levels caudal to it
    (``angle_sum_diagnostic`` differs from the global value).
    :func:`contribution_remainder` reports the numerical shortfall against the
    term of the chain, which is ~0 by construction and is non-zero only through
    float rounding (NaN if a prefix is gimbal-locked).

    This function attributes the supplied chain only (``B = E = A = I``). When a
    fixed base rotation ``B``, terminal rotation ``E`` or drive alignment ``A``
    is not the identity, use :func:`segmental_contribution_breakdown`, which
    also returns them as named non-level entries.

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
    breakdown = segmental_contribution_breakdown(term_id, pose, handedness=handedness)
    return dict(breakdown.levels)


def contribution_remainder(
    term_id: str,
    pose: SegmentalPose,
    *,
    handedness: Literal["right", "left"],
    base: object | None = None,
    terminal: object | None = None,
    alignment: object | None = None,
) -> float:
    """Global term value minus the sum of level and named contributions.

    Degrees. The global term is ``f(B @ chain @ E @ A)`` (``B``, ``E``, ``A``
    default to the identity). Contributions telescope over the cumulative
    prefixes, so this is ~0 up to rounding; it is reported so a consumer can see
    the decomposition closes. Same preconditions and errors as
    :func:`segmental_contribution_breakdown`.
    """
    return segmental_contribution_breakdown(
        term_id,
        pose,
        handedness=handedness,
        base=base,
        terminal=terminal,
        alignment=alignment,
    ).remainder

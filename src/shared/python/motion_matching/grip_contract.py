"""Anatomical grip contract: golf angles <-> the calibration Euler triple.

HRG-GR1 (#13202).  Two grip descriptions existed side by side: the visual
:class:`~src.shared.python.model_appearance.grip_pose.GripPose` (V azimuth,
palm diagonal) and the physical ``subject.grip_rotation_deg`` (extrinsic x-y-z
Euler triples fitted by ``grip_fit``).  This module is the exact bijection
between the physical triple and three golf angles per hand, so users never edit
Euler triples.  It adds the contract and the conversion only; no builder, spec
or receipt reads it yet.

Convention (see "Grip Configuration" in
``docs/research/simscape_matching_reference/simscape_matching_reference.tex``)
-----------------------------------------------------------------------------
``build_upper_body`` writes the hand rotation block
``N @ R_x(25 deg) @ G(e)`` where ``N`` is the native ``child_to_follower``
rotation and ``G(e)`` the extrinsic x-y-z matrix of the stored triple ``e``.
The total constant offset ``M = R_x(25 deg) @ G(e)`` factors uniquely, for
``|psi - 90 deg| < 90 deg``, as the intrinsic Y-X-Z product::

    M = R_y(alpha) @ R_x(psi - 90 deg) @ R_z(gamma)

* ``alpha`` turns the hand about the shaft (the club ``y`` axis, butt to head,
  because ``N`` shares the shaft axis for both hands); positive is stronger.
* ``psi`` is the palm diagonal, radial to distal in the palm plane.
* ``gamma`` is the set angle about the wrist flexion axis.

Zero of ``alpha`` and ``gamma`` and ``psi = 40 deg`` define the published
neutral transform ``M_0 = R_x(-50 deg)``, i.e. the Euler triple
:data:`NEUTRAL_EULER_DEG`.  The fitted defaults are not neutral.
"""

from __future__ import annotations

import math
import types
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import Any, ClassVar, Final

import numpy as np
from numpy.typing import NDArray
from scipy.spatial.transform import Rotation

from src.shared.python.contracts import require
from src.shared.python.core.handedness import (
    DEFAULT_HANDEDNESS,
    Handedness,
    Side,
    lead_side,
    trail_side,
    validate_handedness,
)
from src.shared.python.grip_contact.contact_topology import GripStyle as GripStyle
from src.shared.python.model_appearance.grip_pose import (
    ALPHA_RANGE_DEG as ALPHA_RANGE_DEG,
    LEAD_BELOW_BUTT_M,
    NATIVE_HAND_SPACING_M,
    NEUTRAL_V_AZIMUTH_DEG as NEUTRAL_V_AZIMUTH_DEG,
    PALM_DIAGONAL_DEG,
    GripPose,
    require_fields_in_range,
)
from src.shared.python.motion_matching.anthropometric_geometry import (
    GRIP_ULNAR_OFFSET_DEG,
)
from src.shared.python.motion_matching.grip_fit import grip_rotation

Array = NDArray[np.float64]
Euler = tuple[float, float, float]

#: Product presets for ``alpha`` (rotation about the shaft, positive stronger).
#: Not anatomical thresholds.  Knuckles are the coaching count of lead-hand
#: knuckles visible at address: weak about 1, neutral about 2, strong about 3
#: (an approximation that varies with hand size, not a measurement).
GRIP_STRENGTH_PRESETS_DEG: dict[str, float] = {
    "weak": -15.0,
    "neutral": 0.0,
    "strong": 15.0,
}
#: Presets for ``psi``, the palm diagonal: shaft more in the fingers, neutral
#: (the ``GripPose`` nominal 40 deg), or more in the palm.
PALM_PRESETS_DEG: dict[str, float] = {"fingers": 25.0, "neutral": 40.0, "palm": 65.0}
_DEFAULT_THUMB_DEG = GripPose().lead_thumb_offset_deg
#: Singularity guard: the Y-X-Z middle angle ``psi - 90`` may not reach +-90.
_SINGULAR_TOL = 1e-9
_ALPHA_RANGE = ALPHA_RANGE_DEG
_PSI_RANGE = (5.0, 85.0)
_GAMMA_RANGE = (-180.0, 180.0)


def _rot(axis: str, deg: float) -> Array:
    return np.asarray(Rotation.from_euler(axis, deg, degrees=True).as_matrix())


def grip_matrix(alpha_deg: float, psi_deg: float, gamma_deg: float) -> Array:
    """``M = R_y(alpha) R_x(psi - 90) R_z(gamma)``, the total hand offset."""
    return _rot("y", alpha_deg) @ _rot("x", psi_deg - 90.0) @ _rot("z", gamma_deg)


@dataclass(frozen=True)
class HandAngles:
    """Golf angles of one hand (degrees).

    Preconditions: finite; ``alpha_deg`` in [-60, 60], ``psi_deg`` in [5, 85]
    (keeps the Y-X-Z factorisation away from its singularity) and
    ``gamma_deg`` in [-180, 180]; ``ValueError`` otherwise.
    """

    alpha_deg: float = 0.0
    psi_deg: float = PALM_DIAGONAL_DEG
    gamma_deg: float = 0.0

    FIELDS: ClassVar[tuple[str, ...]] = ("alpha_deg", "psi_deg", "gamma_deg")

    def __post_init__(self) -> None:
        require_fields_in_range(
            self,
            (
                ("alpha_deg", *_ALPHA_RANGE),
                ("psi_deg", *_PSI_RANGE),
                ("gamma_deg", *_GAMMA_RANGE),
            ),
        )


def hand_to_euler(angles: HandAngles) -> Euler:
    """Extrinsic x-y-z degrees ``e`` with ``R_x(25) G(e) = M(angles)``.

    Postcondition: ``grip_matrix`` of :func:`hand_from_euler` of the result
    equals ``grip_matrix(angles)`` to rounding.  Angles, not matrices, may sit
    on the other Euler branch near a gimbal lock; compare matrices.
    """
    require(isinstance(angles, HandAngles), "angles must be a HandAngles")
    m = grip_matrix(angles.alpha_deg, angles.psi_deg, angles.gamma_deg)
    g = _rot("x", -GRIP_ULNAR_OFFSET_DEG) @ m
    e = Rotation.from_matrix(g).as_euler("xyz", degrees=True)
    return (float(e[0]), float(e[1]), float(e[2]))


def hand_from_euler(euler_deg: Sequence[float] | Array) -> HandAngles:
    """Golf angles of a stored extrinsic x-y-z triple (closed form, exact).

    ``ValueError`` for a triple that is not three finite angles or whose total
    offset is singular (``psi`` of 0 or 180 deg).  The result is validated as
    :class:`HandAngles`, so a triple outside its ranges also raises.
    """
    e = np.asarray(euler_deg, dtype=float)
    require(
        e.shape == (3,) and bool(np.isfinite(e).all()),
        "euler_deg must be three finite angles",
    )
    m = _rot("x", GRIP_ULNAR_OFFSET_DEG) @ grip_rotation(e)
    if abs(m[1, 2]) > 1.0 - _SINGULAR_TOL:
        raise ValueError("grip rotation is singular: palm diagonal at 0 or 180 deg")
    beta = math.asin(-m[1, 2])
    alpha = math.atan2(m[0, 2], m[2, 2])
    gamma = math.atan2(m[1, 0], m[1, 1])
    return HandAngles(
        math.degrees(alpha), math.degrees(beta) + 90.0, math.degrees(gamma)
    )


#: Published neutral transform: ``alpha = gamma = 0``, ``psi = 40`` deg.
NEUTRAL_EULER_DEG: Euler = hand_to_euler(HandAngles())


def follower_rotation(native_rotation: Array, angles: HandAngles) -> Array:
    """Rotation block of ``child_to_follower``: ``N @ M(angles)``.

    ``native_rotation`` is the 3x3 rotation of the native wrist
    ``child_to_follower``; the translation block is never touched by the grip
    contract.  ``ValueError`` unless it is a 3x3 rotation.
    """
    n = np.asarray(native_rotation, dtype=float)
    require(
        n.shape == (3, 3) and bool(np.allclose(n @ n.T, np.eye(3), atol=1e-9)),
        "native_rotation must be a 3x3 rotation",
    )
    return n @ grip_matrix(angles.alpha_deg, angles.psi_deg, angles.gamma_deg)


@dataclass(frozen=True)
class GripSpec:
    """Both hands' golf angles plus the shared grip parameters.

    Composes :class:`GripPose`: :meth:`pose` is the visual description and is
    derived from this spec, so ``alpha`` is the only strength value (no second
    knob).  ``lead``/``trail`` are lead and trail terms for either handedness;
    :func:`to_grip_rotation` maps them to anatomical sides.  The visual pose
    carries one palm diagonal, taken from the lead hand; the trail ``psi`` is
    physical only.

    Preconditions: ``lead``/``trail`` are :class:`HandAngles`;
    ``handedness`` is ``"right"`` or ``"left"``; the shared values satisfy
    :class:`GripPose` (so ``lead.alpha_deg`` keeps the V azimuth in [0, 90]).
    """

    lead: HandAngles = field(default_factory=HandAngles)
    trail: HandAngles = field(default_factory=HandAngles)
    style: GripStyle = GripStyle.OVERLAP
    trigger_finger: bool = False
    lead_below_butt_m: float = LEAD_BELOW_BUTT_M
    hand_spacing_m: float = NATIVE_HAND_SPACING_M
    handedness: Handedness = DEFAULT_HANDEDNESS
    #: Visual only: never reaches the physics hash.
    lead_thumb_offset_deg: float = _DEFAULT_THUMB_DEG

    def __post_init__(self) -> None:
        require(isinstance(self.lead, HandAngles), "lead must be a HandAngles")
        require(isinstance(self.trail, HandAngles), "trail must be a HandAngles")
        if not isinstance(self.style, GripStyle):
            raise TypeError(f"style must be a GripStyle, got {self.style!r}")
        if not isinstance(self.trigger_finger, bool):
            raise TypeError(
                f"trigger_finger must be a bool, got {self.trigger_finger!r}"
            )
        validate_handedness(self.handedness)
        self.pose()  # validates the shared values and the V azimuth range

    @classmethod
    def named_preset(
        cls,
        name: str,
        *,
        handedness: Handedness = DEFAULT_HANDEDNESS,
    ) -> GripSpec:
        """A named grip preset (palmer_overlap, woods_interlock, ten_finger)."""
        return named_grip_preset(name).for_handedness(handedness)

    @classmethod
    def preset(
        cls,
        strength: str = "neutral",
        palm: str = "neutral",
        *,
        handedness: Handedness = DEFAULT_HANDEDNESS,
        **shared: Any,
    ) -> GripSpec:
        """Both hands at a named strength and palm preset, or a named full preset."""
        if strength in NAMED_GRIP_PRESETS:
            if palm != "neutral" or shared:
                extra = []
                if palm != "neutral":
                    extra.append(f"palm={palm!r}")
                if shared:
                    extra.extend(f"{k}={v!r}" for k, v in sorted(shared.items()))
                raise ValueError(
                    f"named preset {strength!r} does not accept extra arguments: {', '.join(extra)}"
                )
            return cls.named_preset(strength, handedness=handedness)
        if strength not in GRIP_STRENGTH_PRESETS_DEG:
            raise ValueError(
                f"unknown strength preset {strength!r}; "
                f"choose from {sorted(GRIP_STRENGTH_PRESETS_DEG)}"
            )
        if palm not in PALM_PRESETS_DEG:
            raise ValueError(
                f"unknown palm preset {palm!r}; choose from {sorted(PALM_PRESETS_DEG)}"
            )
        hand = HandAngles(GRIP_STRENGTH_PRESETS_DEG[strength], PALM_PRESETS_DEG[palm])
        return cls(lead=hand, trail=hand, handedness=handedness, **shared)

    @property
    def grip_strength_deg(self) -> float:
        """The strength: lead ``alpha`` (positive is stronger)."""
        return self.lead.alpha_deg

    def pose(self) -> GripPose:
        """Visual :class:`GripPose` of this spec (V azimuth = 20 deg + alpha)."""
        return GripPose(
            style=self.style,
            trigger_finger=self.trigger_finger,
            lead_below_butt_m=self.lead_below_butt_m,
            hand_spacing_m=self.hand_spacing_m,
            lead_v_azimuth_deg=NEUTRAL_V_AZIMUTH_DEG + self.lead.alpha_deg,
            lead_thumb_offset_deg=self.lead_thumb_offset_deg,
            trail_v_offset_deg=self.trail.alpha_deg - self.lead.alpha_deg,
            palm_diagonal_deg=self.lead.psi_deg,
            handedness=self.handedness,
        )


def _require_text(value: Any, field: str) -> None:
    """Validate that ``value`` is a non-empty, non-blank string."""
    if not isinstance(value, str):
        raise TypeError(f"{field} must be a str, got {type(value).__name__}")
    if not value.strip():
        raise ValueError(f"{field} must be a non-empty string")


@dataclass(frozen=True)
class NamedGripPreset:
    """Historical/consensus named grip preset backed by photo estimates.

    The numeric values are photo_estimate assumptions, not measurements; no
    image is committed in the repository. Each preset specifies full
    double-precision starting values for both hands, grip linking style, hand
    spacing along the shaft, trigger finger state, a qualitative source citation
    actually consulted, and the photographic evidence still requirement needed
    to qualify the estimate.
    """

    name: str
    label: str
    spec: GripSpec
    qualitative_source: str
    evidence_still_requirement: str

    def __post_init__(self) -> None:
        _require_text(self.name, "name")
        _require_text(self.label, "label")
        if not isinstance(self.spec, GripSpec):
            raise TypeError(f"spec must be a GripSpec, got {type(self.spec).__name__}")
        _require_text(self.qualitative_source, "qualitative_source")
        _require_text(self.evidence_still_requirement, "evidence_still_requirement")

    @property
    def style(self) -> GripStyle:
        """The grip linking style (:class:`GripStyle`)."""
        return self.spec.style

    @property
    def trigger_finger(self) -> bool:
        """True when the trail index trigger finger is separated on the grip."""
        return self.spec.trigger_finger

    @property
    def hand_spacing_m(self) -> float:
        """Hand spacing along the shaft in metres."""
        return self.spec.hand_spacing_m

    @property
    def lead(self) -> HandAngles:
        """Lead hand golf angles."""
        return self.spec.lead

    @property
    def trail(self) -> HandAngles:
        """Trail hand golf angles."""
        return self.spec.trail

    def for_handedness(self, handedness: Handedness = DEFAULT_HANDEDNESS) -> GripSpec:
        """Return a :class:`GripSpec` for the requested handedness."""
        return replace(self.spec, handedness=handedness)

    def to_library_dict(self) -> dict[str, Any]:
        """Dictionary representation for the appearance library API."""
        return {
            "label": self.label,
            "qualitative_source": self.qualitative_source,
            "evidence_still_requirement": self.evidence_still_requirement,
            **grip_spec_to_physical(self.spec),
        }


_NAMED_GRIP_PRESETS: dict[str, NamedGripPreset] = {
    "palmer_overlap": NamedGripPreset(
        name="palmer_overlap",
        label="Palmer Overlap",
        spec=GripSpec(
            lead=HandAngles(alpha_deg=12.0, psi_deg=45.0, gamma_deg=0.0),
            trail=HandAngles(alpha_deg=12.0, psi_deg=35.0, gamma_deg=0.0),
            style=GripStyle.OVERLAP,
            hand_spacing_m=0.074,
            trigger_finger=True,
        ),
        qualitative_source=(
            "golf.com's published excerpt from the book 'Arnold Palmer: American Hero'. "
            "Palmer describes: an overlap grip; both Vs pointing at the right shoulder "
            "seam (strong-ish); the right lifeline folding over the left thumb; "
            "a slight right-index trigger; a palm-finger hold for the left hand."
        ),
        evidence_still_requirement=(
            "Face-on high-speed camera still required at address: must clearly show "
            "the trail index trigger finger extended on the grip, the lead thumb nested "
            "under the trail thenar lifeline, and the trail little finger overlapping "
            "the lead index knuckle. Numeric values are photo_estimate assumptions, "
            "not measurements; no image is committed in the repository."
        ),
    ),
    "woods_interlock": NamedGripPreset(
        name="woods_interlock",
        label="Woods Interlock",
        spec=GripSpec(
            lead=HandAngles(alpha_deg=2.0, psi_deg=38.0, gamma_deg=0.0),
            trail=HandAngles(alpha_deg=2.0, psi_deg=34.0, gamma_deg=0.0),
            style=GripStyle.INTERLOCK,
            hand_spacing_m=0.069,
            trigger_finger=False,
        ),
        qualitative_source=(
            "An interlock grip seen in published photos of Tiger Woods: a neutral lead hand, "
            "with about 2 to 2.5 knuckles showing; the trail little finger hooked between "
            "the lead index and middle fingers; the right palm over the left thumb."
        ),
        evidence_still_requirement=(
            "Face-on high-speed camera still required at address: must clearly show "
            "the trail little finger interlocked between the lead index and middle fingers, "
            "approximately 2 to 2.5 knuckles visible on the lead hand, and the trail palm "
            "seated over the lead thumb. Numeric values are photo_estimate assumptions, "
            "not measurements; no image is committed in the repository."
        ),
    ),
    "ten_finger": NamedGripPreset(
        name="ten_finger",
        label="Ten-Finger",
        spec=GripSpec(
            lead=HandAngles(alpha_deg=5.0, psi_deg=40.0, gamma_deg=0.0),
            trail=HandAngles(alpha_deg=5.0, psi_deg=32.0, gamma_deg=0.0),
            style=GripStyle.TEN_FINGER,
            hand_spacing_m=0.090,
            trigger_finger=False,
        ),
        qualitative_source=(
            "The general instructional description, with no named person: all ten digits "
            "making direct contact with the grip cylinder without overlap or interlock, "
            "mild rotation (+5 deg on both hands), lead palm diagonal 40 deg, trail finger "
            "diagonal 32 deg, 90 mm hand spacing, and curled index finger without "
            "trigger separation."
        ),
        evidence_still_requirement=(
            "Face-on camera still required at address: must clearly show all eight fingers "
            "and two thumbs contacting the grip cylinder directly, with the trail little "
            "finger positioned flush against the lead index finger without overlap or "
            "interlocking. Numeric values are photo_estimate assumptions, not measurements; "
            "no image is committed in the repository."
        ),
    ),
}

NAMED_GRIP_PRESETS: types.MappingProxyType[str, NamedGripPreset] = (
    types.MappingProxyType(_NAMED_GRIP_PRESETS)
)


def named_grip_preset(name: str) -> NamedGripPreset:
    """Return the :class:`NamedGripPreset` with the given identifier.

    Preconditions:
        ``name`` is a string in :data:`NAMED_GRIP_PRESETS`.
    Postconditions:
        Returns a valid immutable :class:`NamedGripPreset`.
    """
    if not isinstance(name, str):
        raise TypeError(f"preset name must be a str, got {type(name).__name__}")
    if name not in NAMED_GRIP_PRESETS:
        raise ValueError(
            f"unknown named grip preset {name!r}; "
            f"choose from {sorted(NAMED_GRIP_PRESETS)}"
        )
    return NAMED_GRIP_PRESETS[name]


def to_grip_rotation(spec: GripSpec) -> dict[Side, Euler]:
    """``subject.grip_rotation_deg`` of ``spec``, keyed by anatomical side.

    The lead hand goes to ``lead_side(handedness)``.  Lead/trail angles are the
    same numbers for both handednesses; a left-handed spec swaps the sides.
    """
    require(isinstance(spec, GripSpec), "spec must be a GripSpec")
    return {
        lead_side(spec.handedness): hand_to_euler(spec.lead),
        trail_side(spec.handedness): hand_to_euler(spec.trail),
    }


def from_grip_rotation(
    rotation_deg: Mapping[str, Sequence[float]], *, template: GripSpec | None = None
) -> GripSpec:
    """:class:`GripSpec` of a ``grip_rotation_deg`` mapping (``"L"``/``"R"``).

    Shared parameters and handedness come from ``template`` (default: a
    default spec).  ``ValueError`` for missing sides or a singular triple.
    """
    base = template if template is not None else GripSpec()
    require(set(rotation_deg) == {"L", "R"}, "rotation_deg must have exactly L and R")
    return replace(
        base,
        lead=hand_from_euler(rotation_deg[lead_side(base.handedness)]),
        trail=hand_from_euler(rotation_deg[trail_side(base.handedness)]),
    )


#: ``GripPose`` fields that restate a physical grip parameter (HRG-GR2).
_PHYSICAL_POSE_FIELDS = (
    "lead_v_azimuth_deg",
    "trail_v_offset_deg",
    "palm_diagonal_deg",
    "lead_below_butt_m",
    "hand_spacing_m",
    "handedness",
)


def resolve_visual_pose(
    grip: GripSpec,
    override: GripPose | None = None,
    *,
    allow_physical_override: bool = False,
) -> GripPose:
    """Visual pose of ``grip``, optionally restyled, never silently diverged.

    Visual hands default to ``grip.pose()``, so the picture and the physical
    model cannot desynchronise.  ``override`` may change visual-only values
    (``style``, thumb, lifeline).  If it restates a physical value (V azimuth,
    trail V, palm diagonal, depth, spacing, handedness) differently, raise
    ``ValueError`` unless ``allow_physical_override`` is explicitly True.
    """
    require(isinstance(grip, GripSpec), "grip must be a GripSpec")
    derived = grip.pose()
    if override is None:
        return derived
    require(isinstance(override, GripPose), "override must be a GripPose")
    if not allow_physical_override:
        moved = [
            name
            for name in _PHYSICAL_POSE_FIELDS
            if getattr(override, name) != getattr(derived, name)
        ]
        if moved:
            raise ValueError(
                f"visual override of physical grip parameters {moved} would "
                "desynchronise picture and model; change the GripSpec, or pass "
                "allow_physical_override=True to flag it"
            )
    return override


def grip_spec_to_physical(grip: GripSpec) -> dict[str, Any]:
    """JSON form of the physical part of ``grip`` (full double precision)."""
    out: dict[str, Any] = {
        "lead": {
            "alpha_deg": grip.lead.alpha_deg,
            "psi_deg": grip.lead.psi_deg,
            "gamma_deg": grip.lead.gamma_deg,
        },
        "trail": {
            "alpha_deg": grip.trail.alpha_deg,
            "psi_deg": grip.trail.psi_deg,
            "gamma_deg": grip.trail.gamma_deg,
        },
        "lead_below_butt_m": grip.lead_below_butt_m,
        "hand_spacing_m": grip.hand_spacing_m,
        "handedness": grip.handedness,
    }
    if grip.style is not GripStyle.OVERLAP:
        out["style"] = grip.style.value
    if grip.trigger_finger:
        out["trigger_finger"] = True
    return out

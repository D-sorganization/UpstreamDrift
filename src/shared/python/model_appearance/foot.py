"""Procedural feet and shoes: a dimensioned shoe last (pure numpy).

HRG-F1 (#13188, epic #13160). The foot used to be three generic capsule lofts
(``talus``, ``calcn``, ``toes``; radius 0.040 m, aspect 0.7). This module
replaces them with a shoe last built the way a last-maker reads one: a heel
counter, a medial arch waist, an instep/tongue ramp, an asymmetric toe box
(wider and further out on the hallux side) and a flat sole. Right and left are
exact mirrors. Everything is visual only: no mass, inertia, collision or
degree of freedom, and a physics contact sphere is never moved to fit the
shoe.

Frame and sizing (DbC)
    The foot basis is (forward, left, up) in the calcn spec body frame, from
    :class:`cross_section.BodySection` (``left`` is its ML axis, ``up`` its
    depth axis, ``forward = left x up``). Contact-sphere centres are written in
    that frame, so the shoe is placed against them directly:

    * length ``L``: the spec ``anthropometry.segments.foot.length_m`` or a
      measurement (``length_m=``); the shoe spans exactly ``L`` along forward
      from the heel back, which sits ``HEEL_CENTRE_TO_BACK_M`` behind the
      rearmost contact sphere;
    * breadth: ``BREADTH_OVER_LENGTH * L`` (the widest cross-section);
    * sole: ONE flat plane in the *sole frame*, the section frame pitched about
      the left axis so the least-squares line through the sphere nadirs is
      level (a calibrated zero pose is not foot-flat, so the sole follows the
      spheres, not gravity). The plane clears the lowest nadir by
      ``SOLE_CLEARANCE_M``. A pitch beyond ``MAX_SOLE_PITCH_DEG`` or nadirs
      that miss the plane by more than ``SOLE_FIT_TOLERANCE_M`` raise
      ``ValueError`` (nothing is moved).

Articulation
    The upper and sole up to the metatarsophalangeal (MTP) joint belong to
    ``calcn``; the toe box belongs to ``toes``. Each shell is closed. The
    ``calcn`` shell runs ``HINGE_OVERLAP_M`` past the MTP plane and the
    ``toes`` shell starts ``HINGE_OVERLAP_M`` before it; the overlapping end of
    each is inset ``HINGE_INSET_M`` under the other, so no cut face is visible
    at the neutral pose and the hinge stays covered through its working range.
    The ``talus`` carries an ankle stub (skin) joining the shank to the shoe.

Parameter provenance
    ``BREADTH_OVER_LENGTH`` is the pooled-sex median of the per-subject ratio
    ``footbreadthhorizontal / footlength`` in the public ANSUR II files (2012
    U.S. Army survey, Gordon et al. 2015, NATICK/TR-15/007): n = 6068
    (4082 men + 1986 women), median 0.3759, mean 0.3762, sd 0.0175; men
    median 0.3755, women 0.3766. It rounds to 0.376. Datasets: sha256
    0547aea0170e5293de519389981135e75803f02e6d40c77ace740decc0caac64 (male),
    ed7e800aa97a4d42f8286b988be2fe1883a2e789588169db959864b707d1757a (female).
    The profile knots below are modelling parameters, not measurements.
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
from numpy.typing import NDArray

from src.shared.python.model_appearance import geometry
from src.shared.python.model_appearance.cross_section import BodySection
from src.shared.python.model_appearance.skinned_body import spec_body_poses

Array = NDArray[np.float64]

__all__ = [
    "BREADTH_OVER_LENGTH",
    "FOOT_MODELS",
    "FootModelUnavailableError",
    "FootPart",
    "foot_parts",
    "sole_frame",
]

FOOT_MODELS = ("procedural", "cc0_mesh")
# ANSUR II pooled median breadth / length, n = 6068, sd 0.0175 (see docstring)
BREADTH_OVER_LENGTH = 0.376
REFERENCE_LENGTH_M = 0.2581  # the spec foot length the height profile is drawn at
HEEL_CENTRE_TO_BACK_M = 0.022  # rearmost contact-sphere centre to the heel back
SOLE_CLEARANCE_M = 0.0005  # lowest sole point above the lowest sphere nadir
SOLE_FIT_TOLERANCE_M = 0.002  # |sphere nadir - sole plane| limit
MAX_SOLE_PITCH_DEG = 15.0  # largest sole pitch relative to the section frame
HINGE_OVERLAP_M = 0.014
HINGE_INSET_M = 0.0008
SKIN_INSET_M = 0.002  # the bare foot sits this far inside the shoe
SOLE_BEVEL_M = 0.007
ANKLE_RADIUS_M = 0.036
ANKLE_DEPTH_M = 0.040  # stub bottom above the sole plane, inside the shoe
_TIP_EPS_M = 1e-4
_STATIONS = 46
_ARC_POINTS, _DOME_POINTS, _SOLE_POINTS = 3, 23, 9
_DOME_EXPONENT = 2.6

# Profile knots over the fraction of length from the heel back (0) to the toe (1).
_F_HEIGHT = (0.0, 0.04, 0.12, 0.20, 0.30, 0.42, 0.55, 0.68, 0.82, 0.92, 1.0)
_HEIGHT_M = (
    0.060,
    0.066,
    0.068,
    0.064,
    0.072,
    0.080,
    0.071,
    0.054,
    0.045,
    0.039,
    0.032,
)
_F_WIDTH = (0.0, 0.06, 0.15, 0.30, 0.40, 0.55, 0.70, 0.82, 0.92, 1.0)
_WIDTH_FRACTION = (0.60, 0.64, 0.68, 0.76, 0.79, 0.90, 1.00, 0.97, 0.82, 0.55)
# Share of the width that lies on the medial side (the arch indents it).
_MEDIAL_SHARE = (0.50, 0.50, 0.48, 0.43, 0.43, 0.50, 0.53, 0.55, 0.57, 0.60)
_HEEL_CAP_OVER_LENGTH = 0.055
_TOE_CAP_OVER_LENGTH = 0.14

_LEAF = re.compile(r"^(talus|calcn|toes?)_?(r|l|right|left)$")


class FootModelUnavailableError(ValueError):
    """``foot.model`` names a model that is reserved but not shipped."""


@dataclass(frozen=True)
class FootPart:
    """One closed shell of one foot body, in that body's spec frame.

    ``role`` is ``"garment"`` (the shoe), ``"skin"`` (the bare foot) or
    ``"ankle"`` (the stub between the shank and the shoe: it takes the shin
    garment when the outfit has one, else skin).
    """

    role: str
    mesh: geometry.Mesh


@dataclass(frozen=True)
class _Basis:
    """Orthonormal foot basis in the calcn spec frame (the *sole frame*)."""

    fwd: Array
    up: Array
    left: Array
    medial_sign: float  # +1 where the medial side is +left (the right foot)


@dataclass(frozen=True)
class _Last:
    """Plan-view, height and sole description of one shoe last."""

    length: float
    breadth: float
    a_back: float
    sole_y: float  # sole plane height along ``up``
    width_norm: float  # normalises max(lateral) + max(medial) to ``breadth``

    def section(self, a: float) -> tuple[float, float, float]:
        """(lateral half width, medial half width, top height) at ``a``."""
        u = a - self.a_back
        f = u / self.length
        total = _pchip(_F_WIDTH, _WIDTH_FRACTION, f) * self.breadth * self.width_norm
        share = _pchip(_F_WIDTH, _MEDIAL_SHARE, f)
        height = _pchip(_F_HEIGHT, _HEIGHT_M, f) * self.length / REFERENCE_LENGTH_M
        env = _cap(u, self.length * _HEEL_CAP_OVER_LENGTH) * _cap(
            self.length - u, self.length * _TOE_CAP_OVER_LENGTH
        )
        return (1.0 - share) * total * env, share * total * env, height * env


def _cap(distance: float, cap_length: float) -> float:
    """Elliptical end closure: 0 at the end, 1 beyond ``cap_length``."""
    s = 1.0 - min(max(distance / cap_length, 0.0), 1.0)
    return math.sqrt(max(1.0 - s * s, 0.0))


def _pchip(xk: Sequence[float], yk: Sequence[float], x: float) -> float:
    """Monotone cubic Hermite interpolation (no overshoot between knots)."""
    xs, ys = np.asarray(xk, float), np.asarray(yk, float)
    h, d = np.diff(xs), np.diff(ys) / np.diff(xs)
    m = np.zeros_like(ys)
    for i in range(1, len(xs) - 1):
        if d[i - 1] * d[i] > 0:
            w1, w2 = 2 * h[i] + h[i - 1], h[i] + 2 * h[i - 1]
            m[i] = (w1 + w2) / (w1 / d[i - 1] + w2 / d[i])

    def end(h0: float, h1: float, d0: float, d1: float) -> float:
        slope = ((2 * h0 + h1) * d0 - h0 * d1) / (h0 + h1)
        if np.sign(slope) != np.sign(d0):
            return 0.0
        if np.sign(d0) != np.sign(d1) and abs(slope) > 3 * abs(d0):
            return float(3 * d0)
        return float(slope)

    m[0] = end(h[0], h[1], d[0], d[1])
    m[-1] = end(h[-1], h[-2], d[-1], d[-2])
    i = int(np.clip(np.searchsorted(xs, x) - 1, 0, len(xs) - 2))
    t = (min(max(x, xs[0]), xs[-1]) - xs[i]) / h[i]
    return float(
        (2 * t**3 - 3 * t**2 + 1) * ys[i]
        + (t**3 - 2 * t**2 + t) * h[i] * m[i]
        + (-2 * t**3 + 3 * t**2) * ys[i + 1]
        + (t**3 - t**2) * h[i] * m[i + 1]
    )


def _ring(wl: float, wm: float, height: float, inset: float) -> Array:
    """Cross-section polyline ``(lateral offset, height above the sole)``.

    Flat sole, rounded sole edges, and a super-elliptic dome from the lateral
    to the medial extreme. ``inset`` shrinks it on every side (the sole rises,
    the top and both walls move in), so a skin shell nests inside a shoe.
    """
    inset = min(inset, 0.35 * min(wl, wm, 0.5 * height))  # keep the ends open
    wl, wm, top = wl - inset, wm - inset, height - 2.0 * inset
    bevel = min(SOLE_BEVEL_M, 0.45 * wl, 0.45 * wm, 0.3 * top)
    dome_h = top - bevel
    out: list[tuple[float, float]] = []
    for z in np.linspace(-(wm - bevel), wl - bevel, _SOLE_POINTS):
        out.append((float(z), 0.0))
    for k in range(1, _ARC_POINTS + 1):
        phi = math.radians(-90.0 + 90.0 * k / (_ARC_POINTS + 1))
        out.append((wl - bevel + bevel * math.cos(phi), bevel + bevel * math.sin(phi)))
    zc, half = 0.5 * (wl - wm), 0.5 * (wl + wm)
    p = 2.0 / _DOME_EXPONENT
    for k in range(_DOME_POINTS + 2):  # lateral extreme .. medial extreme
        theta = math.pi * k / (_DOME_POINTS + 1)
        c, s = math.cos(theta), math.sin(theta)
        out.append((zc + half * math.copysign(abs(c) ** p, c), bevel + dome_h * s**p))
    for k in range(1, _ARC_POINTS + 1):
        phi = math.radians(180.0 + 90.0 * k / (_ARC_POINTS + 1))
        out.append((-wm + bevel + bevel * math.cos(phi), bevel + bevel * math.sin(phi)))
    ring = np.asarray(out, float)
    ring[:, 1] += inset
    return ring


def _smooth(x: float) -> float:
    x = min(max(x, 0.0), 1.0)
    return x * x * (3.0 - 2.0 * x)


def _shell(
    last: _Last,
    basis: _Basis,
    a_lo: float,
    a_hi: float,
    inset: Callable[[float], float],
) -> geometry.Mesh:
    """Closed shell of the last between two stations along forward."""
    lateral = -basis.medial_sign  # lateral offset -> signed +left coordinate
    grid = []
    for s in np.linspace(0.0, 1.0, _STATIONS):
        a = a_lo + (a_hi - a_lo) * 0.5 * (1.0 - math.cos(math.pi * s))
        wl, wm, height = last.section(a)
        ring = _ring(wl, wm, height, inset(a))
        base = basis.fwd * a
        sole = last.sole_y
        grid.append(
            [base + basis.up * (sole + t) + basis.left * (lateral * z) for z, t in ring]
        )
    g = np.asarray(grid)
    return geometry.closed_grid_mesh(g, g[0].mean(axis=0), g[-1].mean(axis=0))


def _pitch_about_left(
    basis: _Basis, spheres: Sequence[Array], radii: Sequence[float]
) -> _Basis:
    """Basis pitched about ``left`` so the least-squares nadir line is level."""
    a = np.array([c @ basis.fwd for c in spheres])
    nadir = np.array([c @ basis.up - r for c, r in zip(spheres, radii, strict=True)])
    slope = 0.0
    if len(spheres) > 1 and np.ptp(a) > 1e-9:
        slope = float(np.polyfit(a, nadir, 1)[0])
    phi = math.atan(slope)
    if abs(math.degrees(phi)) > MAX_SOLE_PITCH_DEG:
        raise ValueError(
            f"contact spheres pitch the sole {math.degrees(phi):.1f} degrees from "
            f"the foot frame (limit {MAX_SOLE_PITCH_DEG} degrees)"
        )
    c, s = math.cos(phi), math.sin(phi)
    return _Basis(
        c * basis.fwd + s * basis.up,
        -s * basis.fwd + c * basis.up,
        basis.left,
        basis.medial_sign,
    )


def _sole_basis(
    spheres: Sequence[Mapping[str, Any]], section: BodySection, side: str
) -> _Basis:
    up, left = section.depth_axis, section.ml_axis
    fwd = np.asarray(np.cross(left, up), dtype=np.float64)
    flat = _Basis(fwd, up, left, 1.0 if side == "right" else -1.0)
    return _pitch_about_left(
        flat,
        [np.asarray(sp["position_m"], float) for sp in spheres],
        [float(sp["radius_m"]) for sp in spheres],
    )


def sole_frame(
    spec: Mapping[str, Any], section: BodySection, calcn_body: str
) -> tuple[Array, Array, Array]:
    """(forward, up, left) unit vectors of the sole frame of ``calcn_body``.

    ``up`` is the sole normal: the section depth axis pitched about the left
    axis until the contact-sphere nadir line is level. Precondition: the body
    has contact spheres. Postcondition: an orthonormal right-handed triple in
    the calcn spec frame.
    """
    spheres = [s for s in spec["contact"]["spheres"] if s["body"] == calcn_body]
    if not spheres:
        raise ValueError(f"{calcn_body} has no contact spheres to orient its sole")
    b = _sole_basis(spheres, section, "right")
    return b.fwd, b.up, b.left


def _sole_height(nadirs: Sequence[float]) -> float:
    """Flat sole height: ``SOLE_CLEARANCE_M`` above the lowest sphere nadir.

    Raises ``ValueError`` when any nadir misses it by more than
    ``SOLE_FIT_TOLERANCE_M``.
    """
    y = min(nadirs) + SOLE_CLEARANCE_M
    miss = max(abs(y - n) for n in nadirs)
    if miss > SOLE_FIT_TOLERANCE_M:
        raise ValueError(
            f"contact-sphere nadirs miss the flat sole plane by {1e3 * miss:.2f} mm "
            f"(limit {1e3 * SOLE_FIT_TOLERANCE_M:.1f} mm); the shoe is not moved to fit"
        )
    return float(y)


def _width_norm(length: float, breadth: float) -> float:
    """Scale so max lateral + max medial half width equals ``breadth``."""
    probe = _Last(length, breadth, 0.0, 0.0, 1.0)
    pts = [probe.section(length * f) for f in np.linspace(0.02, 0.98, 400)]
    return breadth / (max(p[0] for p in pts) + max(p[1] for p in pts))


def _side_of(leaf: str) -> str:
    return "right" if leaf.startswith("r") else "left"


def _foot_bodies(spec: Mapping[str, Any]) -> dict[str, dict[str, str]]:
    """Side -> {role: body name} for talus/calcn/toes bodies of the spec."""
    out: dict[str, dict[str, str]] = {}
    for body in spec["bodies"]:
        name = body["name"]
        match = _LEAF.match(name.lower().rsplit("/", 1)[-1])
        if match is None:
            continue
        role = "toes" if match.group(1).startswith("toe") else match.group(1)
        out.setdefault(_side_of(match.group(2)), {})[role] = name
    return out


def _to_body(poses: Mapping[str, Array], frm: str, to: str, pts: Array) -> Array:
    rel = np.linalg.inv(poses[to]) @ poses[frm]
    return pts @ rel[:3, :3].T + rel[:3, 3]


def _check_length(length_m: float) -> float:
    if not isinstance(length_m, (int, float)) or not (
        math.isfinite(length_m) and length_m > 0
    ):
        raise ValueError(f"foot length must be a positive finite metre, got {length_m}")
    return float(length_m)


def _spec_length(spec: Mapping[str, Any]) -> float | None:
    segs = spec.get("anthropometry", {}).get("segments", {})
    return segs.get("foot", {}).get("length_m")


def _ankle_stub(
    last: _Last, basis: _Basis, poses: Mapping[str, Array], roles: Mapping[str, str]
) -> geometry.Mesh:
    ankle = _to_body(poses, roles["talus"], roles["calcn"], np.zeros((1, 3)))[0]
    bottom = ankle - basis.up * (
        float(ankle @ basis.up) - (last.sole_y + ANKLE_DEPTH_M)
    )
    pts = _to_body(poses, roles["calcn"], roles["talus"], np.stack([ankle, bottom]))
    return geometry.lofted_segment(
        pts[0],
        pts[1],
        ANKLE_RADIUS_M,
        geometry.LoftOptions(rings=10, sides=24),
    )


def _build_side(
    spec: Mapping[str, Any],
    section: BodySection,
    roles: Mapping[str, str],
    poses: Mapping[str, Array],
    length: float,
    side: str,
) -> dict[str, tuple[FootPart, ...]]:
    spheres = [s for s in spec["contact"]["spheres"] if s["body"] == roles["calcn"]]
    basis = _sole_basis(spheres, section, side)
    centres = [np.asarray(sp["position_m"], float) for sp in spheres]
    radii = [float(sp["radius_m"]) for sp in spheres]
    a_back = min(float(c @ basis.fwd) for c in centres) - HEEL_CENTRE_TO_BACK_M
    sole_y = _sole_height(
        [float(c @ basis.up) - r for c, r in zip(centres, radii, strict=True)]
    )
    breadth = BREADTH_OVER_LENGTH * length
    last = _Last(length, breadth, a_back, sole_y, _width_norm(length, breadth))
    tip = a_back + length

    def build(
        a_lo: float,
        a_hi: float,
        ramp: Callable[[float], float],
        base: float,
    ) -> geometry.Mesh:
        return _shell(last, basis, a_lo, a_hi, lambda a: base + ramp(a))

    def frame_out(body: str, mesh: geometry.Mesh) -> geometry.Mesh:
        return geometry.Mesh(
            _to_body(poses, roles["calcn"], body, mesh.vertices), mesh.faces
        )

    out: dict[str, tuple[FootPart, ...]] = {}
    if "toes" in roles:
        mtp = _to_body(poses, roles["toes"], roles["calcn"], np.zeros((1, 3)))[0]
        cut = float(mtp @ basis.fwd)
        d = HINGE_OVERLAP_M
        fade_in = lambda a: HINGE_INSET_M * _smooth((a - cut) / d)  # noqa: E731
        fade_out = lambda a: HINGE_INSET_M * (1 - _smooth((a - (cut - d)) / d))  # noqa: E731
        lo = a_back + _TIP_EPS_M
        hi = tip - _TIP_EPS_M
        pieces = {
            roles["calcn"]: (lo, cut + d, fade_in),
            roles["toes"]: (cut - d, hi, fade_out),
        }
    else:
        pieces = {
            roles["calcn"]: (a_back + _TIP_EPS_M, tip - _TIP_EPS_M, lambda a: 0.0)
        }
    # The bare foot stops at the MTP plane (no overlap), so a hinge gap shows the
    # shoe's own hidden overlap shell and never skin.
    skin_range = {
        roles["calcn"]: (
            a_back + SKIN_INSET_M,
            cut if "toes" in roles else tip - SKIN_INSET_M,
        ),
    }
    if "toes" in roles:
        skin_range[roles["toes"]] = (cut, tip - SKIN_INSET_M)
    for body, (a_lo, a_hi, ramp) in pieces.items():
        s_lo, s_hi = skin_range[body]
        out[body] = (
            FootPart(
                "skin", frame_out(body, build(s_lo, s_hi, lambda a: 0.0, SKIN_INSET_M))
            ),
            FootPart("garment", frame_out(body, build(a_lo, a_hi, ramp, 0.0))),
        )
    if "talus" in roles:
        out[roles["talus"]] = (
            FootPart("ankle", _ankle_stub(last, basis, poses, roles)),
        )
    return out


def foot_parts(
    spec: Mapping[str, Any],
    sections: Mapping[str, BodySection],
    *,
    length_m: float | None = None,
    model: str = "procedural",
) -> dict[str, tuple[FootPart, ...]]:
    """Shoe and bare-foot shells of every foot body that can be sized.

    Preconditions: ``model`` in :data:`FOOT_MODELS` (``cc0_mesh`` is reserved
    and raises :class:`FootModelUnavailableError`); ``length_m``, when given,
    is a positive finite measurement overriding the spec foot length.
    Postconditions: bodies absent from the result keep the legacy loft (a side
    with no contact spheres, no cross-section or no foot length is never
    guessed); every returned shell is closed with outward winding in its own
    body's spec frame; right and left are mirror images; the spec is not
    modified.
    """
    if model not in FOOT_MODELS:
        raise ValueError(f"foot.model must be one of {FOOT_MODELS}, got {model!r}")
    if model == "cc0_mesh":
        raise FootModelUnavailableError(
            "foot.model 'cc0_mesh' is reserved: no licensed CC0 foot mesh is "
            "shipped yet, use 'procedural'"
        )
    if length_m is not None:
        length = _check_length(length_m)
    else:
        declared = _spec_length(spec)
        if declared is None:
            return {}
        length = _check_length(declared)
    contact = spec.get("contact", {}).get("spheres", [])
    poses = spec_body_poses(spec)
    out: dict[str, tuple[FootPart, ...]] = {}
    for side, roles in sorted(_foot_bodies(spec).items()):
        calcn = roles.get("calcn")
        if calcn is None or calcn not in sections:
            continue
        if not any(s["body"] == calcn for s in contact):
            continue
        out.update(_build_side(spec, sections[calcn], roles, poses, length, side))
    return out

"""Shared ground-reaction analysis core (GCV-1, #11707; epic #11706).

Turns contact wrenches into the complete ground-reaction breakdown every engine
and display surface shows: per-foot force, centre of pressure (CoP), free
moment, the net (both-feet) wrench with its own CoP and free moment, and the
moments of each foot and of the net wrench about the whole-body centre of mass.

Definitions (binding; world frame Z-up, ground plane ``z = z_g``, SI units,
force exerted **by the ground on the foot**).  Contact ``i`` of foot ``f``
applies force ``F_i`` at ``p_i`` with optional contact torque ``tau_i``:

1. ``F_f = sum F_i``;  ``M_O,f = sum (p_i x F_i + tau_i)`` about world origin O.
2. CoP of a wrench ``(F, M_O)`` on the plane, valid only when
   ``F_z >= cop_min_fz_n`` (:data:`COP_MIN_FZ_N`)::

       x_cop = (z_g F_x - M_O,y) / F_z
       y_cop = (M_O,x + z_g F_y) / F_z

   Below the threshold the CoP and free moment are ``None`` (never zero); the
   force is still reported.
3. Free moment ``T_z = M_O,z - (x_cop F_y - y_cop F_x)``, reported as
   ``T_z * z_hat``.
4. Net: ``F_net = F_L + F_R`` and ``M_O,net = M_O,L + M_O,R``; the net CoP and
   net free moment apply 2-3 to the **net** wrench.  The net free moment is in
   general *not* ``T_z,L + T_z,R``: each foot's ``T_z`` is taken about its own
   CoP, while the net one is about the net CoP, so the shear forces acting at
   the offset CoPs contribute a vertical moment to the net.  The two agree when
   the foot CoPs coincide.
5. About the whole-body centre of mass ``c``: ``M_c,f = M_O,f - c x F_f``
   (full wrench; equal to ``(cop_f - c) x F_f + T_z,f z_hat`` when the CoP
   exists).  The force-only term ``(cop_f - c) x F_f`` is exposed separately.
   ``M_c,net = M_c,L + M_c,R`` exactly.
6. A foot with no active contact reports zero force, ``cop=None`` and
   ``in_contact=False``.

Pure numpy; no engine imports.  Do not extend
``physics/ground_reaction_forces.py`` (GCV-6 consolidates it).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
import math
import re
from types import MappingProxyType
from typing import TYPE_CHECKING, TypeAlias

import numpy as np
from numpy.typing import ArrayLike, NDArray

from src.shared.python.force_overlay.contracts import OverlayWrench, WrenchKind
from src.shared.python.motion_matching.force_torque import (
    ContactReaction,
    validate_vec3,
)

if TYPE_CHECKING:  # pragma: no cover
    import pandas as pd

__all__ = [
    "COP_MIN_FZ_N",
    "NET_LABEL",
    "ContactSet",
    "FootReaction",
    "GroundReactionBreakdown",
    "GroundReactionSeries",
    "analyze_ground_reaction",
    "center_of_pressure",
    "foot_reaction",
    "grf_overlay_wrench",
    "overlay_label_part",
    "to_contact_reaction",
    "to_overlay_wrenches",
]

Array: TypeAlias = NDArray[np.float64]

#: Minimum vertical force [N] for a CoP / free moment to be reported.
COP_MIN_FZ_N: float = 10.0
#: Key of the both-feet result in the per-quantity mappings.
NET_LABEL = "net"
_LABEL_SAFE = re.compile(r"[^A-Za-z0-9_.-]+")
_AXES = ("x", "y", "z")
_ZHAT = np.array([0.0, 0.0, 1.0])
_DEFAULT_SOURCE = "shared:ground_reaction"


def overlay_label_part(text: str) -> str:
    """Sanitise ``text`` for use after the ``contact:<quantity>_`` label prefix."""
    return _LABEL_SAFE.sub("_", text).strip("_") or "x"


def _finite_scalar(value: float, name: str) -> float:
    if isinstance(value, bool) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number, got {value!r}")
    return float(value)


def _vec3(value: ArrayLike, name: str) -> Array:
    """A finite world 3-vector; the finiteness check is the shared one."""
    arr = np.asarray(value, dtype=float)
    if arr.shape != (3,):
        raise ValueError(f"{name} must have shape (3,), got {arr.shape}")
    return np.array(validate_vec3(arr.tolist(), name))


def _rows(value: ArrayLike, name: str) -> Array:
    arr = np.asarray(value, dtype=float)
    if arr.size == 0:
        return arr.reshape(0, 3)
    if arr.ndim != 2 or arr.shape[1] != 3:
        raise ValueError(f"{name} must have shape (n, 3), got {arr.shape}")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must be finite")
    return arr


@dataclass(frozen=True, eq=False)
class ContactSet:
    """Contacts of one foot: forces (n,3) [N] at points (n,3) [m], torques (n,3)."""

    forces_n: Array
    points_m: Array
    torques_nm: Array | None = None

    def __post_init__(self) -> None:
        forces = _rows(self.forces_n, "forces")
        points = _rows(self.points_m, "points")
        if forces.shape != points.shape:
            raise ValueError(
                f"forces and points must have the same shape, got "
                f"{forces.shape} and {points.shape}"
            )
        torques = None
        if self.torques_nm is not None:
            torques = _rows(self.torques_nm, "torques")
            if torques.shape != forces.shape:
                raise ValueError(
                    f"torques must have shape {forces.shape}, got {torques.shape}"
                )
        object.__setattr__(self, "forces_n", forces)
        object.__setattr__(self, "points_m", points)
        object.__setattr__(self, "torques_nm", torques)

    @classmethod
    def empty(cls) -> ContactSet:
        """A foot with no contacts."""
        return cls(np.zeros((0, 3)), np.zeros((0, 3)))


@dataclass(frozen=True, eq=False)
class FootReaction:
    """Resultant ground wrench of one foot (or the net of all feet).

    ``cop_m`` and ``free_moment_nm`` are ``None`` when ``F_z`` is below the CoP
    threshold or there is no contact (unavailable, never zero).
    ``contact_centroid_m`` is the mean of the loaded contact points, used only
    to anchor the force arrow when no CoP exists.
    """

    label: str
    force_n: Array
    moment_about_origin_nm: Array
    cop_m: Array | None
    free_moment_nm: Array | None
    in_contact: bool
    contact_centroid_m: Array | None = None


@dataclass(frozen=True, eq=False)
class GroundReactionBreakdown:
    """Per-foot and net ground reactions with moments about the CoM."""

    per_foot: Mapping[str, FootReaction]
    net: FootReaction
    com_m: Array
    moment_about_com_nm: Mapping[str, Array]
    force_moment_about_com_nm: Mapping[str, Array | None]
    ground_height_m: float


def center_of_pressure(
    force: ArrayLike,
    moment_about_origin: ArrayLike,
    ground_height_m: float = 0.0,
    min_fz: float = COP_MIN_FZ_N,
) -> Array | None:
    """Canonical centre of pressure of a wrench on the plane ``z = ground_height_m``.

    ``force`` is the ground force on the body and ``moment_about_origin`` its
    moment about the world origin (definition 2 in the module docstring).

    Preconditions: both are finite 3-vectors; ``ground_height_m`` is finite;
    ``min_fz`` is finite and non-negative.
    Postconditions: returns ``None`` (never zero) when ``F_z < min_fz``;
    otherwise a world point with ``z == ground_height_m``.
    """
    f = _vec3(force, "force")
    m = _vec3(moment_about_origin, "moment_about_origin")
    zg = _finite_scalar(ground_height_m, "ground_height_m")
    thr = _finite_scalar(min_fz, "min_fz")
    if thr < 0.0:
        raise ValueError(f"min_fz must be non-negative, got {thr}")
    return _cop(f, m, zg, thr)


def _cop(force: Array, moment: Array, zg: float, thr: float) -> Array | None:
    fz = float(force[2])
    if fz < thr:
        return None
    return np.array(
        [(zg * force[0] - moment[1]) / fz, (moment[0] + zg * force[1]) / fz, zg]
    )


def _resultant(
    label: str,
    force: Array,
    moment: Array,
    centroid: Array | None,
    in_contact: bool,
    ground_height_m: float,
    cop_min_fz_n: float,
) -> FootReaction:
    """Apply definitions 2-3 to a resultant wrench about the world origin."""
    cop = free = None
    if in_contact:
        cop = _cop(force, moment, ground_height_m, cop_min_fz_n)
    if cop is not None:
        free = (moment[2] - (cop[0] * force[1] - cop[1] * force[0])) * _ZHAT
    return FootReaction(label, force, moment, cop, free, in_contact, centroid)


def foot_reaction(
    label: str,
    forces: ArrayLike,
    points: ArrayLike,
    torques: ArrayLike | None = None,
    *,
    ground_height_m: float = 0.0,
    cop_min_fz_n: float = COP_MIN_FZ_N,
) -> FootReaction:
    """Resultant, CoP and free moment of one foot.

    Preconditions: ``forces``/``points``/``torques`` are finite ``(n, 3)``
    arrays in the world frame (``n`` may be 0); ``ground_height_m`` is finite;
    ``cop_min_fz_n`` is finite and non-negative.
    Postconditions: ``force_n`` and ``moment_about_origin_nm`` are always
    finite; ``cop_m``/``free_moment_nm`` are ``None`` unless ``F_z`` is at
    least ``cop_min_fz_n``; ``cop_m[2] == ground_height_m``.
    """
    if not isinstance(label, str) or not label.strip():
        raise ValueError("label must be a non-empty string")
    zg = _finite_scalar(ground_height_m, "ground_height_m")
    thr = _finite_scalar(cop_min_fz_n, "cop_min_fz_n")
    if thr < 0.0:
        raise ValueError(f"cop_min_fz_n must be non-negative, got {thr}")
    cs = ContactSet(np.asarray(forces), np.asarray(points), torques)  # type: ignore[arg-type]
    loaded = np.linalg.norm(cs.forces_n, axis=1) > 0.0
    in_contact = bool(loaded.any())
    force = cs.forces_n.sum(axis=0)
    moment = np.cross(cs.points_m, cs.forces_n).sum(axis=0)
    if cs.torques_nm is not None:
        moment = moment + cs.torques_nm.sum(axis=0)
    centroid = cs.points_m[loaded].mean(axis=0) if in_contact else None
    return _resultant(label, force, moment, centroid, in_contact, zg, thr)


def _net_reaction(
    feet: Sequence[FootReaction], ground_height_m: float, cop_min_fz_n: float
) -> FootReaction:
    force = np.zeros(3) + sum((r.force_n for r in feet), np.zeros(3))
    moment = sum((r.moment_about_origin_nm for r in feet), np.zeros(3))
    active = [
        r.contact_centroid_m
        for r in feet
        if r.in_contact and r.contact_centroid_m is not None
    ]
    centroid = np.mean(active, axis=0) if active else None
    return _resultant(
        NET_LABEL, force, moment, centroid, bool(active), ground_height_m, cop_min_fz_n
    )


def _readonly(mapping: dict) -> Mapping:
    return MappingProxyType(mapping)


def analyze_ground_reaction(
    contacts_by_foot: Mapping[str, ContactSet],
    com_m: ArrayLike,
    *,
    ground_height_m: float = 0.0,
    cop_min_fz_n: float = COP_MIN_FZ_N,
) -> GroundReactionBreakdown:
    """Full ground-reaction breakdown (definitions 1-6 in the module docstring).

    Preconditions: values are :class:`ContactSet`; keys are non-empty strings
    other than ``"net"``; ``com_m`` is a finite 3-vector (world frame).
    Postconditions: ``moment_about_com_nm["net"]`` equals the sum of the
    per-foot entries; ``force_moment_about_com_nm[k]`` is ``None`` exactly when
    foot/net ``k`` has no CoP.
    """
    com = _vec3(com_m, "com_m")
    zg = _finite_scalar(ground_height_m, "ground_height_m")
    per_foot: dict[str, FootReaction] = {}
    for name, cs in contacts_by_foot.items():
        if not isinstance(name, str) or not name.strip():
            raise ValueError("foot labels must be non-empty strings")
        if name == NET_LABEL:
            raise ValueError(f"foot label {NET_LABEL!r} is reserved for the net wrench")
        if not isinstance(cs, ContactSet):
            raise TypeError(f"contacts for {name!r} must be a ContactSet")
        per_foot[name] = foot_reaction(
            name,
            cs.forces_n,
            cs.points_m,
            cs.torques_nm,
            ground_height_m=zg,
            cop_min_fz_n=cop_min_fz_n,
        )
    net = _net_reaction(list(per_foot.values()), zg, cop_min_fz_n)
    everything = {**per_foot, NET_LABEL: net}
    about_com = {
        k: r.moment_about_origin_nm - np.cross(com, r.force_n)
        for k, r in everything.items()
    }
    # M_c,net is the exact sum of the foot moments (not recomputed from F_net).
    about_com[NET_LABEL] = (
        np.sum([about_com[k] for k in per_foot], axis=0) if per_foot else np.zeros(3)
    )
    force_only = {
        k: (None if r.cop_m is None else np.cross(r.cop_m - com, r.force_n))
        for k, r in everything.items()
    }
    return GroundReactionBreakdown(
        _readonly(per_foot), net, com, _readonly(about_com), _readonly(force_only), zg
    )


def _foot_part(reaction: FootReaction) -> str:
    return (
        NET_LABEL if reaction.label == NET_LABEL else overlay_label_part(reaction.label)
    )


def grf_overlay_wrench(
    reaction: FootReaction,
    *,
    source: str = _DEFAULT_SOURCE,
    body: str | None = None,
    label_part: str | None = None,
) -> OverlayWrench | None:
    """``contact:grf_<foot>`` force at the CoP (contact centroid below threshold).

    ``label_part`` overrides the sanitised label suffix.  Returns ``None`` when
    the foot is not in contact or has no anchor point.  Torque is left
    unavailable (``None``); the free moment is a separate wrench.
    """
    if not isinstance(reaction, FootReaction):
        raise TypeError("reaction must be a FootReaction")
    if not reaction.in_contact:
        return None
    anchor = (
        reaction.cop_m if reaction.cop_m is not None else reaction.contact_centroid_m
    )
    if anchor is None:
        return None
    default_body = "system" if reaction.label == NET_LABEL else reaction.label
    return OverlayWrench(
        WrenchKind.CONTACT,
        f"contact:grf_{label_part or _foot_part(reaction)}",
        body or default_body,
        _tuple(anchor),
        force_n=_tuple(reaction.force_n),
        source=source,
    )


def _tuple(v: Array) -> tuple[float, float, float]:
    return (float(v[0]), float(v[1]), float(v[2]))


def to_overlay_wrenches(
    breakdown: GroundReactionBreakdown, *, source: str = _DEFAULT_SOURCE
) -> list[OverlayWrench]:
    """Overlay wrenches (``WrenchKind.CONTACT``) for every available quantity.

    Labels: ``contact:grf_<foot>`` / ``contact:grf_net`` (force at the CoP),
    ``contact:free_moment_<foot>`` / ``..._net`` (pure torque at the CoP) and
    ``contact:moment_com_<foot>`` / ``..._net`` (pure torque at the CoM).
    Feet not in contact and unavailable CoP quantities are omitted.
    """
    if not isinstance(breakdown, GroundReactionBreakdown):
        raise TypeError("breakdown must be a GroundReactionBreakdown")
    if not isinstance(source, str) or not source.strip():
        raise ValueError("source must be a non-empty string")
    out: list[OverlayWrench] = []
    com = _tuple(breakdown.com_m)
    for reaction in (*breakdown.per_foot.values(), breakdown.net):
        if not reaction.in_contact:
            continue
        part = _foot_part(reaction)
        body = "system" if reaction.label == NET_LABEL else reaction.label
        grf = grf_overlay_wrench(reaction, source=source)
        if grf is not None:
            out.append(grf)
        if reaction.cop_m is not None and reaction.free_moment_nm is not None:
            out.append(
                OverlayWrench(
                    WrenchKind.CONTACT,
                    f"contact:free_moment_{part}",
                    body,
                    _tuple(reaction.cop_m),
                    torque_nm=_tuple(reaction.free_moment_nm),
                    source=source,
                )
            )
        out.append(
            OverlayWrench(
                WrenchKind.CONTACT,
                f"contact:moment_com_{part}",
                "system",
                com,
                torque_nm=_tuple(breakdown.moment_about_com_nm[reaction.label]),
                source=source,
            )
        )
    return out


def to_contact_reaction(
    breakdown: GroundReactionBreakdown,
    time_s: float,
    *,
    left_label: str = "left",
    right_label: str = "right",
) -> ContactReaction:
    """Populate the motion-matching :class:`ContactReaction` from a breakdown.

    A foot absent from the breakdown stays ``None`` (unavailable, not zero);
    CoPs are the ``(x, y)`` plane coordinates, ``None`` when unavailable.
    """
    feet = breakdown.per_foot

    def force(key: str):
        return _tuple(feet[key].force_n) if key in feet else None

    def cop(r: FootReaction | None):
        return (
            None
            if r is None or r.cop_m is None
            else (float(r.cop_m[0]), float(r.cop_m[1]))
        )

    return ContactReaction(
        time_s=time_s,
        net_grf_n=_tuple(breakdown.net.force_n),
        left_foot_grf_n=force(left_label),
        right_foot_grf_n=force(right_label),
        net_cop_m=cop(breakdown.net),
        left_foot_cop_m=cop(feet.get(left_label)),
        right_foot_cop_m=cop(feet.get(right_label)),
        contact_status={k: r.in_contact for k, r in feet.items()},
    )


@dataclass(frozen=True, eq=False)
class GroundReactionSeries:
    """Stacked breakdowns over time; NaN marks an unavailable value.

    Every mapping is keyed by foot label plus ``"net"``; vector quantities
    have shape ``(T, 3)``, ``in_contact`` has shape ``(T,)``.
    """

    times_s: Array
    force_n: Mapping[str, Array]
    cop_m: Mapping[str, Array]
    free_moment_nm: Mapping[str, Array]
    moment_about_com_nm: Mapping[str, Array]
    force_moment_about_com_nm: Mapping[str, Array]
    in_contact: Mapping[str, NDArray[np.bool_]]
    com_m: Array
    ground_height_m: Array = field(default_factory=lambda: np.zeros(0))

    @classmethod
    def from_breakdowns(
        cls, times_s: ArrayLike, breakdowns: Sequence[GroundReactionBreakdown]
    ) -> GroundReactionSeries:
        """Stack ``breakdowns`` sampled at strictly increasing ``times_s``."""
        times = np.asarray(times_s, dtype=float)
        if len(breakdowns) == 0:
            raise ValueError("breakdowns must not be empty")
        if times.shape != (len(breakdowns),):
            raise ValueError(
                f"times_s length {times.size} must equal breakdowns length "
                f"{len(breakdowns)}"
            )
        if not np.all(np.isfinite(times)) or np.any(np.diff(times) <= 0.0):
            raise ValueError("times_s must be finite and strictly increasing")
        feet = tuple(breakdowns[0].per_foot)
        if any(tuple(b.per_foot) != feet for b in breakdowns):
            raise ValueError("all breakdowns must have the same feet in the same order")
        keys = (*feet, NET_LABEL)

        def stack(getter, keys=keys):
            return _readonly(
                {k: np.array([getter(b, k) for b in breakdowns]) for k in keys}
            )

        def reaction(b: GroundReactionBreakdown, k: str) -> FootReaction:
            return b.net if k == NET_LABEL else b.per_foot[k]

        def opt(v: Array | None) -> Array:
            return np.full(3, np.nan) if v is None else v

        return cls(
            times_s=times,
            force_n=stack(lambda b, k: reaction(b, k).force_n),
            cop_m=stack(lambda b, k: opt(reaction(b, k).cop_m)),
            free_moment_nm=stack(lambda b, k: opt(reaction(b, k).free_moment_nm)),
            moment_about_com_nm=stack(lambda b, k: b.moment_about_com_nm[k]),
            force_moment_about_com_nm=stack(
                lambda b, k: opt(b.force_moment_about_com_nm[k])
            ),
            in_contact=stack(lambda b, k: reaction(b, k).in_contact),
            com_m=np.array([b.com_m for b in breakdowns]),
            ground_height_m=np.array([b.ground_height_m for b in breakdowns]),
        )

    def to_dataframe(self) -> pd.DataFrame:
        """Long-form columns ``<foot>_<quantity>_<axis>_<unit>`` plus ``time_s``."""
        import pandas as pd

        quantities = (
            ("force", "n", self.force_n),
            ("cop", "m", self.cop_m),
            ("free_moment", "nm", self.free_moment_nm),
            ("moment_com", "nm", self.moment_about_com_nm),
            ("force_moment_com", "nm", self.force_moment_about_com_nm),
        )
        cols: dict[str, NDArray] = {"time_s": self.times_s}
        for key in self.force_n:
            for name, unit, table in quantities:
                for i, axis in enumerate(_AXES):
                    cols[f"{key}_{name}_{axis}_{unit}"] = table[key][:, i]
            cols[f"{key}_in_contact"] = self.in_contact[key]
        return pd.DataFrame(cols)

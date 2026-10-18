"""Allocated hand-to-hand contact forces (HRG-H1, #13189; HRG-GS-G3, #13301). Reporting only.

In the full-body model the lead hand is the club body and the trail hand is
closed onto it by one rigid loop constraint, so only the total closure wrench
``lambda`` is observable: any hand-to-hand force acts between the same two bodies
as the weld. The quantity reported here is therefore an **allocated equivalent
force**, ``method == "allocation"`` everywhere it is shown. It is not a measured
biological force, not a new contact, and it never enters the equations of
motion.

Definition (binding). ``force_on_lead_n`` and ``torque_on_lead_nm`` are the wrench
of the closing (trail) hand on the lead side, i.e. ``-lambda``, at the closure
site ``O``. For each hand-hand pair in the topology, the allocated force is carried
to the patch point ``P`` as one force plus the residual couple:
``M_P = M_O - (P - O) x F`` (the couple a point contact cannot carry is kept,
not dropped). With ``n`` the unit normal from the lead site toward the trail site,
a closing force pushes on the lead hand along ``-n``:

* normal force ``F_n = max(0, -F . n)``; ``F . n >= 0`` is tension and is
  flagged ``unloaded`` rather than reported;
* tangential force ``F_t = |F - (F . n) n|`` (the part in the patch plane);
* friction ratio ``F_t / (mu F_n)`` against the spec ``static_friction``;
* share of the closure force ``F_n / |F_closure|``.

In physical mode on a welded topology, requesting physical hand-hand contact
raises :class:`SchemaFeatureUnavailableError` (tracked: #13309).
"""

from __future__ import annotations

import math
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

from src.shared.python.grip_contact.contact_topology import (
    ALLOCATED_HAND_HAND_MODES,
    ContactPair,
    ContactTopology,
    contact_topology,
)
from src.shared.python.grip_contact.pair_contact import (
    ContactCapsule,
    ContactSphere,
    capsule_capsule_distance,
    sphere_capsule_distance,
)
from src.shared.python.motion_matching.full_body_schema import (
    SchemaFeatureUnavailableError,
)
from src.shared.python.motion_matching.grip_contract import GripSpec

METHOD = "allocation"
#: Legacy overlay wrench label of the drawn palm-thumb normal force.
PALM_THUMB_LABEL = "external:palm_thumb_allocated"
#: Tolerance on the unit length of the patch normal.
NORMAL_TOLERANCE = 1e-9
#: Closure forces below this (N) have no meaningful share or direction.
MIN_CLOSURE_FORCE_N = 1e-12

Vec3 = tuple[float, float, float]


@dataclass(frozen=True)
class AllocatedPairForce:
    """Allocated contact force at a hand-hand patch, or why it is unavailable.

    An unavailable result carries ``available == False``, a non-empty ``reason``
    and ``None`` for every quantity (never a zero force). ``unloaded`` is
    ``True`` when the closure force pulls the patch apart.
    """

    available: bool
    reason: str = ""
    method: str = METHOD
    pair_name: str = "trail_palm_lead_thumb"
    label: str = PALM_THUMB_LABEL
    patch_point_m: Vec3 | None = None
    patch_normal: Vec3 | None = None
    force_on_lead_n: Vec3 | None = None
    couple_nm: Vec3 | None = None
    force_on_trail_n: Vec3 | None = None
    couple_on_trail_nm: Vec3 | None = None
    normal_force_n: float | None = None
    tangential_force_n: float | None = None
    unloaded: bool | None = None
    friction_ratio: float | None = None
    friction_reason: str = ""
    share_of_closure_force: float | None = None
    closure_force_n: float | None = None

    def to_dict(self) -> dict[str, Any]:
        """JSON-ready record; unavailable quantities stay ``None``, tuples become lists."""
        return {
            k: list(v) if isinstance(v, tuple) else v for k, v in asdict(self).items()
        }


#: Alias for backward compatibility with existing HRG-H1 callers.
PalmThumbForce = AllocatedPairForce


@dataclass(frozen=True)
class AllocatedHandContacts:
    """Allocated contact forces for all hand-hand pairs in a topology."""

    available: bool
    reason: str = ""
    method: str = METHOD
    pairs: dict[str, AllocatedPairForce] = field(default_factory=dict)
    total_force_on_lead_n: Vec3 | None = None
    total_couple_nm: Vec3 | None = None
    hand_club_force_n: Vec3 = (0.0, 0.0, 0.0)
    hand_club_torque_nm: Vec3 = (0.0, 0.0, 0.0)
    closure_force_n: Vec3 | None = None
    closure_torque_nm: Vec3 | None = None
    reproduction_error_n: float = 0.0

    def __getitem__(self, key: str) -> AllocatedPairForce:
        return self.pairs[key]

    def __iter__(self) -> Iterator[AllocatedPairForce]:
        return iter(self.pairs.values())

    def __len__(self) -> int:
        return len(self.pairs)

    def get(self, key: str, default: Any = None) -> Any:
        return self.pairs.get(key, default)

    @property
    def palm_thumb(self) -> AllocatedPairForce | None:
        return self.pairs.get("trail_palm_lead_thumb")

    def to_dict(self) -> dict[str, Any]:
        return {
            "available": self.available,
            "reason": self.reason,
            "method": self.method,
            "pairs": {k: v.to_dict() for k, v in self.pairs.items()},
            "total_force_on_lead_n": (
                list(self.total_force_on_lead_n) if self.total_force_on_lead_n else None
            ),
            "total_couple_nm": (
                list(self.total_couple_nm) if self.total_couple_nm else None
            ),
            "hand_club_force_n": list(self.hand_club_force_n),
            "hand_club_torque_nm": list(self.hand_club_torque_nm),
            "closure_force_n": (
                list(self.closure_force_n) if self.closure_force_n else None
            ),
            "closure_torque_nm": (
                list(self.closure_torque_nm) if self.closure_torque_nm else None
            ),
            "reproduction_error_n": self.reproduction_error_n,
        }


def unavailable_pair_force(
    pair_name: str, label: str, reason: str
) -> AllocatedPairForce:
    """An unavailable pair force result; ``reason`` must be non-empty."""
    if not isinstance(reason, str) or not reason.strip():
        raise ValueError("an unavailable force needs a non-empty reason")
    return AllocatedPairForce(
        available=False, reason=reason, pair_name=pair_name, label=label
    )


def unavailable_palm_thumb(reason: str) -> PalmThumbForce:
    """A result that could not be resolved; ``reason`` must be non-empty."""
    return unavailable_pair_force("trail_palm_lead_thumb", PALM_THUMB_LABEL, reason)


def _vec(name: str, value: Sequence[float] | np.ndarray) -> np.ndarray:
    arr = np.asarray(value, dtype=float)
    if arr.shape != (3,):
        raise ValueError(f"{name} must have shape (3,), got {arr.shape}")
    if not np.isfinite(arr).all():
        raise ValueError(f"{name} must be finite")
    return arr


def _tuple(x: np.ndarray) -> Vec3:
    return (float(x[0]), float(x[1]), float(x[2]))


def _friction(
    tangential: float, normal: float, mu: float, unloaded: bool
) -> tuple[float | None, str]:
    if unloaded:
        return None, "unloaded: no normal force, so no friction-cone ratio"
    if mu == 0.0:
        return None, "static_friction is zero: the cone has no width"
    return tangential / (mu * normal), ""


def pair_contact_patch(
    topology: ContactTopology,
    pair: ContactPair | str,
) -> tuple[np.ndarray, np.ndarray]:
    """Find contact point and unit normal (lead -> trail) for a hand-hand pair.

    Preconditions:
        topology is a ContactTopology.
        pair is a ContactPair or valid pair name within topology, and must be
        a hand-hand pair (site_b != GRIP).
    Postconditions:
        Returns (contact_point_m, normal_lead_to_trail) in the grip frame.
        normal_lead_to_trail has unit length to NORMAL_TOLERANCE.
    """
    if not isinstance(topology, ContactTopology):
        raise TypeError(
            f"topology must be ContactTopology, got {type(topology).__name__}"
        )
    if isinstance(pair, str):
        pair = topology.pair(pair)
    elif not isinstance(pair, ContactPair):
        raise TypeError(f"pair must be ContactPair or str, got {type(pair).__name__}")
    if not pair.is_hand_hand:
        raise ValueError(f"pair {pair.name!r} is not a hand-hand pair")

    site_a = topology.site(pair.site_a)
    site_b = topology.site(pair.site_b)

    prim_a: ContactSphere | ContactCapsule
    if site_a.primitive == "sphere":
        prim_a = ContactSphere(
            center_m=np.asarray(site_a.centre_m, dtype=float), radius_m=site_a.radius_m
        )
    else:
        p0, p1 = site_a.endpoints_m
        prim_a = ContactCapsule(
            p0_m=np.asarray(p0, dtype=float),
            p1_m=np.asarray(p1, dtype=float),
            radius_m=site_a.radius_m,
        )

    prim_b: ContactSphere | ContactCapsule
    if site_b.primitive == "sphere":
        prim_b = ContactSphere(
            center_m=np.asarray(site_b.centre_m, dtype=float), radius_m=site_b.radius_m
        )
    else:
        p0, p1 = site_b.endpoints_m
        prim_b = ContactCapsule(
            p0_m=np.asarray(p0, dtype=float),
            p1_m=np.asarray(p1, dtype=float),
            radius_m=site_b.radius_m,
        )

    if isinstance(prim_a, ContactSphere) and isinstance(prim_b, ContactCapsule):
        _, pt, normal_a_to_b, _ = sphere_capsule_distance(prim_a, prim_b)
    elif isinstance(prim_a, ContactCapsule) and isinstance(prim_b, ContactSphere):
        _, pt, normal_b_to_a, _ = sphere_capsule_distance(prim_b, prim_a)
        normal_a_to_b = -normal_b_to_a
    elif isinstance(prim_a, ContactCapsule) and isinstance(prim_b, ContactCapsule):
        _, pt, normal_a_to_b, _, _ = capsule_capsule_distance(prim_a, prim_b)
    elif isinstance(prim_a, ContactSphere) and isinstance(prim_b, ContactSphere):
        diff = prim_b.center_m - prim_a.center_m
        dist = float(np.linalg.norm(diff))
        normal_a_to_b = diff / dist if dist > 1e-12 else np.array([1.0, 0.0, 0.0])
        pt = (
            prim_a.center_m
            + (prim_a.radius_m + 0.5 * (dist - prim_a.radius_m - prim_b.radius_m))
            * normal_a_to_b
        )
    else:
        raise ValueError(f"unsupported primitive pair {pair.primitive_pair!r}")

    normal = normal_a_to_b if site_a.hand == "lead" else -normal_a_to_b
    norm = float(np.linalg.norm(normal))
    if abs(norm - 1.0) > NORMAL_TOLERANCE and norm > 1e-12:
        normal = normal / norm
    return np.asarray(pt, dtype=float), np.asarray(normal, dtype=float)


def allocate_pair_force(
    *,
    force_on_lead_n: Sequence[float] | np.ndarray,
    torque_on_lead_nm: Sequence[float] | np.ndarray,
    reference_point_m: Sequence[float] | np.ndarray,
    patch_point_m: Sequence[float] | np.ndarray,
    patch_normal: Sequence[float] | np.ndarray,
    static_friction: float,
    pair_name: str = "trail_palm_lead_thumb",
    label: str = PALM_THUMB_LABEL,
    closure_force_n: float | None = None,
) -> AllocatedPairForce:
    """Transport the closing-hand wrench to the patch and split it.

    All vectors share one frame. Preconditions: finite 3-vectors, a unit
    ``patch_normal`` (to 1e-9) and a finite ``static_friction >= 0``
    (``ValueError`` otherwise). Postconditions: ``force_on_lead_n`` and
    ``couple_nm`` plus ``(patch - reference) x force`` reproduce the input wrench
    exactly, and the trail-side reaction is the exact negative of both
    (Newton's third law).
    """
    force = _vec("force_on_lead_n", force_on_lead_n)
    torque = _vec("torque_on_lead_nm", torque_on_lead_nm)
    site = _vec("reference_point_m", reference_point_m)
    point = _vec("patch_point_m", patch_point_m)
    normal = _vec("patch_normal", patch_normal)
    if abs(float(np.linalg.norm(normal)) - 1.0) > NORMAL_TOLERANCE:
        raise ValueError("patch_normal must be a unit vector")
    if not (math.isfinite(static_friction) and static_friction >= 0.0):
        raise ValueError("static_friction must be finite and non-negative")
    couple = torque - np.cross(point - site, force)
    along = float(force @ normal)
    unloaded = along >= 0.0
    normal_force = 0.0 if unloaded else -along
    tangential = float(np.linalg.norm(force - along * normal))
    magnitude = (
        float(np.linalg.norm(force))
        if closure_force_n is None
        else float(closure_force_n)
    )
    ratio, why = _friction(tangential, normal_force, static_friction, unloaded)
    share = None if magnitude < MIN_CLOSURE_FORCE_N else normal_force / magnitude
    return AllocatedPairForce(
        available=True,
        pair_name=pair_name,
        label=label,
        patch_point_m=_tuple(point),
        patch_normal=_tuple(normal),
        force_on_lead_n=_tuple(force),
        couple_nm=_tuple(couple),
        force_on_trail_n=_tuple(-force),
        couple_on_trail_nm=_tuple(-couple),
        normal_force_n=normal_force,
        tangential_force_n=tangential,
        unloaded=unloaded,
        friction_ratio=ratio,
        friction_reason=why,
        share_of_closure_force=share,
        closure_force_n=magnitude,
    )


def allocate_palm_thumb(
    *,
    force_on_lead_n: Sequence[float] | np.ndarray,
    torque_on_lead_nm: Sequence[float] | np.ndarray,
    reference_point_m: Sequence[float] | np.ndarray,
    patch_point_m: Sequence[float] | np.ndarray,
    patch_normal: Sequence[float] | np.ndarray,
    static_friction: float,
    pair_name: str = "trail_palm_lead_thumb",
    label: str = PALM_THUMB_LABEL,
) -> PalmThumbForce:
    """Thin wrapper around :func:`allocate_pair_force` for backward compatibility."""
    return allocate_pair_force(
        force_on_lead_n=force_on_lead_n,
        torque_on_lead_nm=torque_on_lead_nm,
        reference_point_m=reference_point_m,
        patch_point_m=patch_point_m,
        patch_normal=patch_normal,
        static_friction=static_friction,
        pair_name=pair_name,
        label=label,
    )


def allocate_hand_hand_from_closure(
    closure: Any,
    *,
    club_rotation: Sequence[Sequence[float]] | np.ndarray,
    club_origin: Sequence[float] | np.ndarray,
    topology: ContactTopology | GripSpec | None = None,
    static_friction: float,
    hand_club_force_n: Sequence[float] | np.ndarray | None = None,
    hand_club_torque_nm: Sequence[float] | np.ndarray | None = None,
    weights: Mapping[str, float] | Sequence[float] | None = None,
    mode: str = "closure",
    schema_id: str = "full-body-v1",
) -> AllocatedHandContacts:
    """Allocate the closure wrench across all hand-hand pairs of a topology.

    Preconditions:
        club_rotation is a proper 3x3 rotation.
        club_origin is a finite 3-vector.
        static_friction >= 0 is finite.
        mode in ALLOCATED_HAND_HAND_MODES; requesting physical mode on a welded
        topology (schema_id == "full-body-v1") raises SchemaFeatureUnavailableError.
    Postconditions:
        The sum of allocated hand-hand forces plus hand-club forces reproduces
        the closure lambda to <= 1e-9 N.
    """
    if mode in ("physical", "separated_v2"):
        raise SchemaFeatureUnavailableError(
            schema_id,
            consumer="grip_contact",
            feature="physical hand-hand contact",
            child="HRG-GS-G11",
            issue=13309,
        )
    if mode not in ALLOCATED_HAND_HAND_MODES and mode != "allocation":
        raise ValueError(
            f"unsupported mode {mode!r}, expected one of {ALLOCATED_HAND_HAND_MODES}"
        )

    rot = np.asarray(club_rotation, dtype=float)
    if rot.shape != (3, 3) or not np.isfinite(rot).all():
        raise ValueError("club_rotation must be a finite 3x3 matrix")
    if not (
        np.allclose(rot.T @ rot, np.eye(3), atol=NORMAL_TOLERANCE)
        and math.isclose(float(np.linalg.det(rot)), 1.0, abs_tol=NORMAL_TOLERANCE)
    ):
        raise ValueError("club_rotation must be a proper rotation")
    origin = _vec("club_origin", club_origin)

    if topology is None:
        topo = contact_topology()
    elif isinstance(topology, GripSpec):
        topo = contact_topology(topology)
    elif isinstance(topology, ContactTopology):
        topo = topology
    else:
        raise TypeError(
            f"topology must be ContactTopology or GripSpec, got {type(topology).__name__}"
        )

    pairs = topo.hand_hand_pairs()
    n_pairs = len(pairs)
    if n_pairs == 0:
        raise ValueError("topology has no hand-hand pairs to allocate")

    f_hc = (
        np.zeros(3)
        if hand_club_force_n is None
        else _vec("hand_club_force_n", hand_club_force_n)
    )
    t_hc = (
        np.zeros(3)
        if hand_club_torque_nm is None
        else _vec("hand_club_torque_nm", hand_club_torque_nm)
    )

    w_map: dict[str, float] = {}
    if weights is None:
        w_map = {p.name: 1.0 / n_pairs for p in pairs}
    elif isinstance(weights, Mapping):
        for p in pairs:
            if p.name not in weights:
                raise ValueError(f"weights missing entry for pair {p.name!r}")
            w_val = float(weights[p.name])
            if not (math.isfinite(w_val) and w_val >= 0.0):
                raise ValueError(f"weight for {p.name!r} must be finite and >= 0")
            w_map[p.name] = w_val
        w_sum = sum(w_map.values())
        if abs(w_sum - 1.0) > 1e-9:
            raise ValueError(f"weights must sum to 1.0, got {w_sum}")
    elif isinstance(weights, Sequence):
        if len(weights) != n_pairs:
            raise ValueError(f"weights length {len(weights)} != pair count {n_pairs}")
        for p, w_raw in zip(pairs, weights, strict=True):
            w_val = float(w_raw)
            if not (math.isfinite(w_val) and w_val >= 0.0):
                raise ValueError(f"weight for {p.name!r} must be finite and >= 0")
            w_map[p.name] = w_val
        w_sum = sum(w_map.values())
        if abs(w_sum - 1.0) > 1e-9:
            raise ValueError(f"weights must sum to 1.0, got {w_sum}")
    else:
        raise TypeError("weights must be a Mapping, Sequence, or None")

    if not closure.available:
        reason = closure.reason or "closure unavailable"
        unavail = {
            p.name: unavailable_pair_force(p.name, p.allocated_label, reason)
            for p in pairs
        }
        return AllocatedHandContacts(
            available=False,
            reason=reason,
            pairs=unavail,
            hand_club_force_n=_tuple(f_hc),
            hand_club_torque_nm=_tuple(t_hc),
        )

    on_club = closure.wrenches[-1]
    if on_club.force_n is None or on_club.torque_nm is None:
        reason = "closure wrench has no force or torque half"
        unavail = {
            p.name: unavailable_pair_force(p.name, p.allocated_label, reason)
            for p in pairs
        }
        return AllocatedHandContacts(
            available=False,
            reason=reason,
            pairs=unavail,
            hand_club_force_n=_tuple(f_hc),
            hand_club_torque_nm=_tuple(t_hc),
        )

    f_lead = np.asarray(on_club.force_n, dtype=float)
    t_lead = np.asarray(on_club.torque_nm, dtype=float)
    site = _vec("reference_point_m", on_club.point_m)

    f_rem = f_lead - f_hc
    t_rem = t_lead - t_hc
    closure_mag = float(np.linalg.norm(f_lead))

    pair_results: dict[str, AllocatedPairForce] = {}
    for p in pairs:
        pt_grip, n_grip = pair_contact_patch(topo, p)
        pt_world = rot @ pt_grip + origin
        n_world = rot @ n_grip
        w_i = w_map[p.name]
        f_i = w_i * f_rem
        t_i = w_i * t_rem
        pair_results[p.name] = allocate_pair_force(
            force_on_lead_n=f_i,
            torque_on_lead_nm=t_i,
            reference_point_m=site,
            patch_point_m=pt_world,
            patch_normal=n_world,
            static_friction=static_friction,
            pair_name=p.name,
            label=p.allocated_label,
            closure_force_n=closure_mag,
        )

    sum_f_lead = sum(
        (np.asarray(pf.force_on_lead_n) for pf in pair_results.values()),
        start=np.zeros(3),
    )
    total_f_lead = sum_f_lead + f_hc
    sum_couple = sum(
        (np.asarray(pf.couple_nm) for pf in pair_results.values()),
        start=np.zeros(3),
    )
    error_n = float(np.linalg.norm(total_f_lead - f_lead))

    return AllocatedHandContacts(
        available=True,
        pairs=pair_results,
        total_force_on_lead_n=_tuple(total_f_lead),
        total_couple_nm=_tuple(sum_couple),
        hand_club_force_n=_tuple(f_hc),
        hand_club_torque_nm=_tuple(t_hc),
        closure_force_n=_tuple(f_lead),
        closure_torque_nm=_tuple(t_lead),
        reproduction_error_n=error_n,
    )


def allocate_from_closure(
    closure: Any,
    *,
    club_rotation: Sequence[Sequence[float]] | np.ndarray,
    club_origin: Sequence[float] | np.ndarray,
    patch: Any = None,
    topology: ContactTopology | GripSpec | None = None,
    static_friction: float,
    hand_club_force_n: Sequence[float] | np.ndarray | None = None,
    hand_club_torque_nm: Sequence[float] | np.ndarray | None = None,
    weights: Mapping[str, float] | Sequence[float] | None = None,
    mode: str = "closure",
    schema_id: str = "full-body-v1",
) -> PalmThumbForce | AllocatedHandContacts:
    """Allocate the palm-thumb or all hand-hand forces from a world-frame closure.

    When ``patch`` is provided and ``topology`` is ``None``, returns the legacy
    :class:`PalmThumbForce` for the palm-thumb pair. When ``topology`` is
    provided, returns :class:`AllocatedHandContacts` for all style pairs.
    """
    if patch is not None and topology is None:
        rot = np.asarray(club_rotation, dtype=float)
        if rot.shape != (3, 3) or not np.isfinite(rot).all():
            raise ValueError("club_rotation must be a finite 3x3 matrix")
        if not (
            np.allclose(rot.T @ rot, np.eye(3), atol=NORMAL_TOLERANCE)
            and math.isclose(float(np.linalg.det(rot)), 1.0, abs_tol=NORMAL_TOLERANCE)
        ):
            raise ValueError("club_rotation must be a proper rotation")
        origin = _vec("club_origin", club_origin)
        if not closure.available:
            return unavailable_palm_thumb(closure.reason or "closure unavailable")
        on_club = closure.wrenches[-1]
        if on_club.force_n is None or on_club.torque_nm is None:
            return unavailable_palm_thumb("closure wrench has no force or torque half")
        return allocate_palm_thumb(
            force_on_lead_n=on_club.force_n,
            torque_on_lead_nm=on_club.torque_nm,
            reference_point_m=on_club.point_m,
            patch_point_m=rot @ _vec("patch.point_club_m", patch.point_club_m) + origin,
            patch_normal=rot @ _vec("patch.normal_club", patch.normal_club),
            static_friction=static_friction,
        )

    return allocate_hand_hand_from_closure(
        closure,
        club_rotation=club_rotation,
        club_origin=club_origin,
        topology=topology,
        static_friction=static_friction,
        hand_club_force_n=hand_club_force_n,
        hand_club_torque_nm=hand_club_torque_nm,
        weights=weights,
        mode=mode,
        schema_id=schema_id,
    )

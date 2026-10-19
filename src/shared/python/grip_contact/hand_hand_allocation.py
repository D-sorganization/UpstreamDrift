"""Allocated hand-to-hand contact forces (HRG-H1, #13189; HRG-GS-G3, #13301). Reporting only.

In the full-body model the lead hand is the club body and the trail hand is
closed onto it by one rigid loop constraint, so only the total closure wrench
``lambda`` is observable: any hand-to-hand force acts between the same two bodies
as the weld. The quantity reported here is therefore an **allocated equivalent
force**, ``method == "allocation"`` everywhere it is shown. It is not a measured
biological force, not a new contact, and it never enters the equations of
motion.

The partitioning of the residual closure wrench across active hand-hand pairs
is an explicit, labelled modelling prior (:class:`AllocationPrior`), not an
identified quantity. When ``hand_club_force_n`` and ``hand_club_torque_nm`` are
not supplied, they default to zero, attributing the entire observable closure
to hand-hand pairs, exactly as HRG-H1 did. :class:`AllocatedHandContacts` records
``prior_name`` and ``hand_club_source`` ("supplied" vs "not_supplied") so the
modelling choices remain transparent.

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
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

from src.shared.python.grip_contact.contact_topology import (
    ALLOCATED_HAND_HAND_MODES,
    PHYSICAL_HAND_HAND_MODES,
    ContactPair,
    ContactSite,
    ContactTopology,
    coerce_contact_topology,
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


@dataclass(frozen=True)
class AllocationPrior:
    """Explicit, labelled modelling prior for partitioning closure wrenches.

    Preconditions (DbC):
        ``name`` must be a non-empty string.
        ``weights`` must map hand-hand pair names to finite, non-negative floats.
        Weights must sum to 1.0 within 1e-9.
    """

    name: str
    weights: Mapping[str, float]

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("prior name must be a non-empty string")
        if not isinstance(self.weights, Mapping) or len(self.weights) == 0:
            raise ValueError("prior weights must be a non-empty mapping")
        for k, v in self.weights.items():
            if not isinstance(k, str):
                raise TypeError(f"weight key {k!r} must be a str")
            val = float(v)
            if not (math.isfinite(val) and val >= 0.0):
                raise ValueError(f"weight for {k!r} must be finite and >= 0, got {v}")
        total = sum(float(v) for v in self.weights.values())
        if abs(total - 1.0) > 1e-9:
            raise ValueError(f"weights must sum to 1.0 within 1e-9, got {total}")

    def validate_for_topology(self, topology: ContactTopology) -> None:
        """Verify that weights keys exactly equal the topology's hand-hand pair names."""
        expected = {p.name for p in topology.hand_hand_pairs()}
        actual = set(self.weights.keys())
        if actual != expected:
            raise ValueError(
                f"prior weights keys {actual} do not match topology hand-hand pairs {expected}"
            )


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
    prior_name: str = "equal_split"
    hand_club_source: str = "not_supplied"

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
            "prior_name": self.prior_name,
            "hand_club_source": self.hand_club_source,
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


def _vec(name: str, value: Sequence[float] | np.ndarray) -> np.ndarray:
    arr = np.asarray(value, dtype=float)
    if arr.shape != (3,):
        raise ValueError(f"{name} must have shape (3,), got {arr.shape}")
    if not np.isfinite(arr).all():
        raise ValueError(f"{name} must be finite")
    return arr


def _tuple(x: np.ndarray) -> Vec3:
    return (float(x[0]), float(x[1]), float(x[2]))


def _check_rotation(
    club_rotation: Sequence[Sequence[float]] | np.ndarray,
) -> np.ndarray:
    rot = np.asarray(club_rotation, dtype=float)
    if rot.shape != (3, 3) or not np.isfinite(rot).all():
        raise ValueError("club_rotation must be a finite 3x3 matrix")
    if not (
        np.allclose(rot.T @ rot, np.eye(3), atol=NORMAL_TOLERANCE)
        and math.isclose(float(np.linalg.det(rot)), 1.0, abs_tol=NORMAL_TOLERANCE)
    ):
        raise ValueError("club_rotation must be a proper rotation")
    return rot


def _friction(
    tangential: float, normal: float, mu: float, unloaded: bool
) -> tuple[float | None, str]:
    if unloaded:
        return None, "unloaded: no normal force, so no friction-cone ratio"
    if mu == 0.0:
        return None, "static_friction is zero: the cone has no width"
    return tangential / (mu * normal), ""


def _primitive(site: ContactSite) -> ContactSphere | ContactCapsule:
    if site.primitive == "sphere":
        return ContactSphere(
            center_m=np.asarray(site.centre_m, dtype=float),
            radius_m=site.radius_m,
        )
    p0, p1 = site.endpoints_m
    return ContactCapsule(
        p0_m=np.asarray(p0, dtype=float),
        p1_m=np.asarray(p1, dtype=float),
        radius_m=site.radius_m,
    )


def _unavailable_contacts(
    pairs: Sequence[ContactPair],
    reason: str,
    f_hc: np.ndarray,
    t_hc: np.ndarray,
    prior_name: str,
    hand_club_source: str,
) -> AllocatedHandContacts:
    unavail = {
        p.name: unavailable_pair_force(p.name, p.allocated_label, reason) for p in pairs
    }
    return AllocatedHandContacts(
        available=False,
        reason=reason,
        pairs=unavail,
        hand_club_force_n=_tuple(f_hc),
        hand_club_torque_nm=_tuple(t_hc),
        prior_name=prior_name,
        hand_club_source=hand_club_source,
    )


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

    prim_a = _primitive(site_a)
    prim_b = _primitive(site_b)

    if isinstance(prim_a, ContactSphere) and isinstance(prim_b, ContactCapsule):
        _, pt, normal_a_to_b, _ = sphere_capsule_distance(prim_a, prim_b)
    elif isinstance(prim_a, ContactCapsule) and isinstance(prim_b, ContactSphere):
        _, pt, normal_b_to_a, _ = sphere_capsule_distance(prim_b, prim_a)
        normal_a_to_b = -normal_b_to_a
    elif isinstance(prim_a, ContactCapsule) and isinstance(prim_b, ContactCapsule):
        _, pt, normal_a_to_b, _, _ = capsule_capsule_distance(prim_a, prim_b)
    else:
        raise ValueError(
            f"unsupported primitive pair {prim_a.__class__.__name__} - {prim_b.__class__.__name__}"
        )

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
    patch: tuple[Sequence[float] | np.ndarray, Sequence[float] | np.ndarray],
    static_friction: float,
    pair_name: str = "trail_palm_lead_thumb",
    label: str = PALM_THUMB_LABEL,
    closure_force_n: float | None = None,
) -> AllocatedPairForce:
    """Transport the closing-hand wrench to the patch and split it.

    ``patch`` is ``(point, unit normal)``. All vectors share one frame.
    Preconditions: finite 3-vectors, a unit patch normal (to 1e-9) and a finite ``static_friction >= 0``
    (``ValueError`` otherwise). Postconditions: ``force_on_lead_n`` and
    ``couple_nm`` plus ``(patch - reference) x force`` reproduce the input wrench
    exactly, and the trail-side reaction is the exact negative of both
    (Newton's third law).
    """
    force = _vec("force_on_lead_n", force_on_lead_n)
    torque = _vec("torque_on_lead_nm", torque_on_lead_nm)
    site = _vec("reference_point_m", reference_point_m)
    point = _vec("patch_point_m", patch[0])
    normal = _vec("patch_normal", patch[1])
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


def equal_split_prior(
    topology: ContactTopology | GripSpec | None = None,
) -> AllocationPrior:
    """Construct an equal-split allocation prior for a topology's hand-hand pairs."""
    topo = coerce_contact_topology(topology)
    pairs = topo.hand_hand_pairs()
    n_pairs = len(pairs)
    if n_pairs == 0:
        raise ValueError("topology has no hand-hand pairs to allocate")
    return AllocationPrior(
        name="equal_split",
        weights={p.name: 1.0 / n_pairs for p in pairs},
    )


def require_allocated_mode(mode: str, schema_id: str) -> None:
    """Validate that mode is an accepted allocated hand-hand mode.

    Raises SchemaFeatureUnavailableError for physical hand-hand modes on a welded
    topology, and ValueError for unknown modes.
    """
    if mode in PHYSICAL_HAND_HAND_MODES or mode == "physical":
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


def _resolve_topology(
    topology: ContactTopology | GripSpec | None,
) -> ContactTopology:
    """The topology to allocate over (the default spec's when ``None``)."""
    return coerce_contact_topology(topology)


def _resolve_prior(
    prior: AllocationPrior | None, topology: ContactTopology
) -> AllocationPrior:
    """``prior`` validated against ``topology``, or the equal split when ``None``."""
    if prior is None:
        return equal_split_prior(topology)
    if not isinstance(prior, AllocationPrior):
        raise TypeError(
            f"prior must be an AllocationPrior or None, got {type(prior).__name__}"
        )
    prior.validate_for_topology(topology)
    return prior


def _hand_club_wrench(
    force_n: Sequence[float] | np.ndarray | None,
    torque_nm: Sequence[float] | np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray, str]:
    """``(force, torque, source)``; an omitted half is zero, labelled "not_supplied"."""
    supplied = force_n is not None or torque_nm is not None
    f = np.zeros(3) if force_n is None else _vec("hand_club_force_n", force_n)
    t = np.zeros(3) if torque_nm is None else _vec("hand_club_torque_nm", torque_nm)
    return f, t, "supplied" if supplied else "not_supplied"


def _sum_vectors(vectors: Iterable[Sequence[float] | None]) -> np.ndarray:
    """Component-wise sum of 3-vectors; NaN if any is unavailable (``None``)."""
    total = np.zeros(3)
    for v in vectors:
        if v is None:
            return np.full(3, np.nan)
        total = total + np.asarray(v, dtype=float)
    return total


def _closure_unavailable_reason(closure: Any) -> str | None:
    """Why ``closure`` cannot be allocated, or ``None`` when it can."""
    if not closure.available:
        return str(closure.reason or "closure unavailable")
    on_club = closure.wrenches[-1]
    if on_club.force_n is None or on_club.torque_nm is None:
        return "closure wrench has no force or torque half"
    return None


def _allocate_pairs(
    topology: ContactTopology,
    prior: AllocationPrior,
    *,
    wrench: tuple[np.ndarray, np.ndarray, np.ndarray],
    club_pose: tuple[np.ndarray, np.ndarray],
    static_friction: float,
    closure_force_n: float,
) -> dict[str, AllocatedPairForce]:
    """Split ``wrench`` = (force, torque, reference point) over the pairs by ``prior``.

    Each pair's patch is carried from the grip frame to the world by
    ``club_pose`` = (rotation, origin).
    """
    force, torque, site = wrench
    rot, origin = club_pose
    results: dict[str, AllocatedPairForce] = {}
    for pair in topology.hand_hand_pairs():
        point, normal = pair_contact_patch(topology, pair)
        weight = prior.weights[pair.name]
        results[pair.name] = allocate_pair_force(
            force_on_lead_n=weight * force,
            torque_on_lead_nm=weight * torque,
            reference_point_m=site,
            patch=(rot @ point + origin, rot @ normal),
            static_friction=static_friction,
            pair_name=pair.name,
            label=pair.allocated_label,
            closure_force_n=closure_force_n,
        )
    return results


def allocate_hand_hand_from_closure(
    closure: Any,
    *,
    club_rotation: Sequence[Sequence[float]] | np.ndarray,
    club_origin: Sequence[float] | np.ndarray,
    topology: ContactTopology | GripSpec | None = None,
    static_friction: float,
    hand_club_force_n: Sequence[float] | np.ndarray | None = None,
    hand_club_torque_nm: Sequence[float] | np.ndarray | None = None,
    prior: AllocationPrior | None = None,
) -> AllocatedHandContacts:
    """Allocate the closure wrench across all hand-hand pairs of a topology.

    The partitioning of the residual closure wrench is an explicit modelling
    prior, not an identified physical quantity. ``hand_club_force_n`` and
    ``hand_club_torque_nm`` default to zero when not supplied.

    Preconditions:
        club_rotation is a proper 3x3 rotation.
        club_origin is a finite 3-vector.
        static_friction >= 0 is finite.
        prior is an AllocationPrior matching topology pairs, or None (equal split).
    Postconditions:
        The sum of allocated hand-hand forces plus hand-club forces reproduces
        the closure lambda to <= 1e-9 N.
    """
    rot = _check_rotation(club_rotation)
    origin = _vec("club_origin", club_origin)

    topo = _resolve_topology(topology)
    pairs = topo.hand_hand_pairs()
    if len(pairs) == 0:
        raise ValueError("topology has no hand-hand pairs to allocate")
    alloc_prior = _resolve_prior(prior, topo)
    f_hc, t_hc, hc_source = _hand_club_wrench(hand_club_force_n, hand_club_torque_nm)

    reason = _closure_unavailable_reason(closure)
    if reason is not None:
        return _unavailable_contacts(
            pairs, reason, f_hc, t_hc, alloc_prior.name, hc_source
        )
    on_club = closure.wrenches[-1]

    f_lead = np.asarray(on_club.force_n, dtype=float)
    t_lead = np.asarray(on_club.torque_nm, dtype=float)
    site = _vec("reference_point_m", on_club.point_m)

    pair_results = _allocate_pairs(
        topo,
        alloc_prior,
        wrench=(f_lead - f_hc, t_lead - t_hc, site),
        club_pose=(rot, origin),
        static_friction=static_friction,
        closure_force_n=float(np.linalg.norm(f_lead)),
    )
    total_f_lead = _sum_vectors(pf.force_on_lead_n for pf in pair_results.values())
    total_f_lead = total_f_lead + f_hc
    sum_couple = _sum_vectors(pf.couple_nm for pf in pair_results.values())
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
        prior_name=alloc_prior.name,
        hand_club_source=hc_source,
    )

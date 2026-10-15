"""Bare-club vs welded-hand spatial inertia partition for grip-wrench dynamics.

Issue #12550 (NSX-01), parent epic #12593.

In the full-body specification (e.g., ``full_body_spec_anthro_driver.json``),
the club body (``solid_reference:GolfSwing3D_Kinetic/Club/Clubface Vector``)
carries both the physical club solids (``Clubhead``, ``Rigid Shaft``, ``Grip``)
and the welded hand solids (``LHand``, ``LHandStandoff``, ``RHand``).  Passing
the composite body inertia (``1.4212 kg`` for the driver) into the Newton-Euler
grip wrench attributes the hands' own inertial load (``1.1082 kg``) to the
hand-on-club interface.

This module partitions the club body's solids by specification provenance name
into ``bare_club_inertia`` (checked against ``spec["club"]["total_mass_kg"]``
to within ``1e-9``) and ``welded_hand_inertias``, and transforms the bare-club
spatial inertia and kinematics from the specification club body frame into the
world frame for ``holding_hand_wrench``.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import ArrayLike

Vec3 = tuple[float, float, float]

CLUB_BODY_SUFFIX = "Clubface Vector"
BARE_CLUB_SOLID_NAMES: frozenset[str] = frozenset({"Clubhead", "Rigid Shaft", "Grip"})
WELDED_HAND_SOLID_NAMES: frozenset[str] = frozenset({"LHand", "LHandStandoff", "RHand"})
_MASS_ATOL_KG = 1e-9
_REQUIRED_KINEMATICS_KEYS = (
    "body_angular_velocity_world_rad_s",
    "body_linear_acceleration_world_m_s2",
    "body_angular_acceleration_world_rad_s2",
    "gravity_m_s2",
)

__all__ = [
    "BARE_CLUB_SOLID_NAMES",
    "CLUB_BODY_SUFFIX",
    "ClubSolidsPartition",
    "SolidSpatialInertia",
    "WELDED_HAND_SOLID_NAMES",
    "bare_club_kinematics_from_body_frame",
    "club_solids_partition",
]


def _vec3_array(value: ArrayLike, name: str) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float64)
    if arr.shape != (3,) or not np.isfinite(arr).all():
        raise ValueError(f"{name} must be a finite 3-vector, got shape {arr.shape}")
    return arr


def _vec3_tuple(value: ArrayLike, name: str) -> Vec3:
    arr = _vec3_array(value, name)
    return (float(arr[0]), float(arr[1]), float(arr[2]))


def _rot3_matrix(value: ArrayLike, name: str) -> np.ndarray:
    rot = np.asarray(value, dtype=np.float64)
    if rot.shape != (3, 3) or not np.isfinite(rot).all():
        raise ValueError(f"{name} must be a finite 3x3 matrix, got shape {rot.shape}")
    if not np.allclose(rot.T @ rot, np.eye(3), atol=1e-6, rtol=0) or not np.isclose(
        float(np.linalg.det(rot)), 1.0, atol=1e-6, rtol=0
    ):
        raise ValueError(f"{name} must be a proper orthonormal rotation matrix")
    return rot


def _rigid_placement(value: ArrayLike, name: str) -> tuple[np.ndarray, np.ndarray]:
    mat = np.asarray(value, dtype=np.float64)
    if (
        mat.shape != (4, 4)
        or not np.isfinite(mat).all()
        or not np.allclose(mat[3], (0.0, 0.0, 0.0, 1.0), atol=1e-12, rtol=0)
    ):
        raise ValueError(f"{name} must be a finite 4x4 homogeneous transform")
    rot = _rot3_matrix(mat[:3, :3], f"{name}[:3, :3]")
    return rot, mat[:3, 3].copy()


@dataclass(frozen=True)
class SolidSpatialInertia:
    """Spatial inertia of a solid or aggregated solid group in the club body frame.

    Preconditions:
        - ``name`` is a non-empty string.
        - ``mass_kg`` is finite and strictly positive.
        - ``com_body_m`` is a finite 3-vector in the club body frame.
        - ``inertia_com_body_kg_m2`` is a finite symmetric positive-semidefinite
          3x3 rotational inertia tensor about ``com_body_m`` in club body axes.
    """

    name: str
    mass_kg: float
    com_body_m: Vec3
    inertia_com_body_kg_m2: np.ndarray

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("name must be a non-empty string")
        mass = float(self.mass_kg)
        if not np.isfinite(mass) or mass <= 0.0:
            raise ValueError(f"mass_kg must be finite and positive, got {self.mass_kg}")
        object.__setattr__(self, "mass_kg", mass)
        object.__setattr__(
            self, "com_body_m", _vec3_tuple(self.com_body_m, "com_body_m")
        )
        inertia = np.array(self.inertia_com_body_kg_m2, dtype=np.float64, copy=True)
        if inertia.shape != (3, 3) or not np.isfinite(inertia).all():
            raise ValueError("inertia_com_body_kg_m2 must be a finite 3x3 matrix")
        if not np.allclose(inertia, inertia.T, atol=1e-12, rtol=0):
            raise ValueError("inertia_com_body_kg_m2 must be symmetric")
        sym = 0.5 * (inertia + inertia.T)
        if float(np.linalg.eigvalsh(sym).min()) < -1e-12:
            raise ValueError("inertia_com_body_kg_m2 must be positive semidefinite")
        sym.setflags(write=False)
        object.__setattr__(self, "inertia_com_body_kg_m2", sym)


@dataclass(frozen=True)
class ClubSolidsPartition:
    """Partition of the club body into bare-club inertia and welded hand solids.

    Supports both attribute access (``bare_club_inertia``,
    ``welded_hand_inertias``, ``composite_inertia``) and two-tuple unpacking
    ``bare_club_inertia, welded_hand_inertias = club_solids_partition(spec)``.
    """

    bare_club_inertia: SolidSpatialInertia
    welded_hand_inertias: tuple[SolidSpatialInertia, ...]
    composite_inertia: SolidSpatialInertia

    def __post_init__(self) -> None:
        if not isinstance(self.bare_club_inertia, SolidSpatialInertia):
            raise TypeError("bare_club_inertia must be a SolidSpatialInertia")
        if not isinstance(self.composite_inertia, SolidSpatialInertia):
            raise TypeError("composite_inertia must be a SolidSpatialInertia")
        hands = tuple(self.welded_hand_inertias)
        if any(not isinstance(h, SolidSpatialInertia) for h in hands):
            raise TypeError(
                "welded_hand_inertias must contain SolidSpatialInertia items"
            )
        object.__setattr__(self, "welded_hand_inertias", hands)

    @property
    def hand_mass_kg(self) -> float:
        """Total mass [kg] of the hand solids welded onto the club body."""
        return float(sum(h.mass_kg for h in self.welded_hand_inertias))

    def __iter__(self) -> Iterator[Any]:
        """Yield ``(bare_club_inertia, welded_hand_inertias)`` for tuple unpacking."""
        yield self.bare_club_inertia
        yield self.welded_hand_inertias


def _solid_in_club_frame(solid: Mapping[str, Any]) -> SolidSpatialInertia:
    """Transform one specification solid entry into the club body frame."""
    name = str(solid.get("name", ""))
    rot, pos = _rigid_placement(solid.get("placement", ()), f"{name}.placement")
    com_local = _vec3_array(solid.get("com_m", ()), f"{name}.com_m")
    inertia_local = np.asarray(solid.get("inertia_com_kg_m2", ()), dtype=np.float64)
    if inertia_local.shape != (3, 3) or not np.isfinite(inertia_local).all():
        raise ValueError(f"{name}.inertia_com_kg_m2 must be a finite 3x3 matrix")
    com_body = rot @ com_local + pos
    inertia_body = rot @ inertia_local @ rot.T
    return SolidSpatialInertia(
        name=name,
        mass_kg=float(solid["mass_kg"]),
        com_body_m=_vec3_tuple(com_body, f"{name}.com_body_m"),
        inertia_com_body_kg_m2=inertia_body,
    )


def _aggregate_solids(
    name: str, solids: Sequence[SolidSpatialInertia]
) -> SolidSpatialInertia:
    """Aggregate solids in the club body frame via the parallel-axis theorem."""
    if not solids:
        raise ValueError(f"Cannot aggregate empty solid list for {name!r}")
    total_mass = float(sum(s.mass_kg for s in solids))
    weighted_com = sum(
        s.mass_kg * np.asarray(s.com_body_m, dtype=np.float64) for s in solids
    )
    com_total = weighted_com / total_mass
    inertia_total = np.zeros((3, 3), dtype=np.float64)
    eye3 = np.eye(3, dtype=np.float64)
    for solid in solids:
        d = np.asarray(solid.com_body_m, dtype=np.float64) - com_total
        shift = float(d @ d) * eye3 - np.outer(d, d)
        inertia_total += solid.inertia_com_body_kg_m2 + solid.mass_kg * shift
    return SolidSpatialInertia(
        name=name,
        mass_kg=total_mass,
        com_body_m=_vec3_tuple(com_total, f"{name}.com_body_m"),
        inertia_com_body_kg_m2=inertia_total,
    )


def _find_club_body(spec: Mapping[str, Any]) -> Mapping[str, Any]:
    bodies = spec.get("bodies")
    if not isinstance(bodies, Sequence):
        raise ValueError("spec must contain a 'bodies' sequence")
    closure = spec.get("closure")
    target_name = closure.get("body_b") if isinstance(closure, Mapping) else None
    for body in bodies:
        if not isinstance(body, Mapping):
            continue
        b_name = str(body.get("name", ""))
        if (target_name and b_name == target_name) or b_name.endswith(CLUB_BODY_SUFFIX):
            return body
    raise ValueError(f"spec has no club body matching {CLUB_BODY_SUFFIX!r}")


def club_solids_partition(spec: Mapping[str, Any]) -> ClubSolidsPartition:
    """Partition the club body's solids into bare-club and welded-hand inertias.

    Preconditions:
        - ``spec`` is a full-body specification mapping containing the club
          body (``Clubface Vector``) and ``spec["club"]["total_mass_kg"]``.
        - Every solid on the club body has a provenance leaf name in
          :data:`BARE_CLUB_SOLID_NAMES` or :data:`WELDED_HAND_SOLID_NAMES`.

    Postconditions:
        - ``abs(bare_club_inertia.mass_kg - spec["club"]["total_mass_kg"]) <= 1e-9``.
        - ``composite_inertia`` equals the parallel-axis aggregate of all club
          and welded hand solids in the club body frame.
    """
    if not isinstance(spec, Mapping):
        raise TypeError(f"spec must be a Mapping, got {type(spec).__name__}")
    club_meta = spec.get("club")
    if not isinstance(club_meta, Mapping) or "total_mass_kg" not in club_meta:
        raise ValueError("spec must define club.total_mass_kg")
    expected_bare_mass = float(club_meta["total_mass_kg"])
    if not np.isfinite(expected_bare_mass) or expected_bare_mass <= 0.0:
        raise ValueError(
            f"club.total_mass_kg must be finite and positive, got {expected_bare_mass}"
        )

    club_body = _find_club_body(spec)
    raw_solids = club_body.get("solids")
    if not isinstance(raw_solids, Sequence) or not raw_solids:
        raise ValueError("club body has no solids")

    bare_solids: list[SolidSpatialInertia] = []
    hand_solids: list[SolidSpatialInertia] = []
    for raw in raw_solids:
        if not isinstance(raw, Mapping):
            raise TypeError("solid entry must be a Mapping")
        leaf = str(raw.get("name", "")).rsplit("/", 1)[-1]
        transformed = _solid_in_club_frame(raw)
        if leaf in BARE_CLUB_SOLID_NAMES:
            bare_solids.append(transformed)
        elif leaf in WELDED_HAND_SOLID_NAMES:
            hand_solids.append(transformed)
        else:
            raise ValueError(
                f"Unknown club-body solid provenance {raw.get('name')!r} "
                f"(expected one of {sorted(BARE_CLUB_SOLID_NAMES | WELDED_HAND_SOLID_NAMES)})"
            )

    if not bare_solids:
        raise ValueError("club body has no bare-club solids")

    bare_inertia = _aggregate_solids("bare_club", bare_solids)
    if abs(bare_inertia.mass_kg - expected_bare_mass) > _MASS_ATOL_KG:
        raise ValueError(
            f"Bare-club mass {bare_inertia.mass_kg:.12g} kg does not match "
            f"club.total_mass_kg {expected_bare_mass:.12g} kg within {_MASS_ATOL_KG}"
        )
    composite_inertia = _aggregate_solids(
        "composite_club_with_hands", [*bare_solids, *hand_solids]
    )
    return ClubSolidsPartition(
        bare_club_inertia=bare_inertia,
        welded_hand_inertias=tuple(hand_solids),
        composite_inertia=composite_inertia,
    )


def bare_club_kinematics_from_body_frame(
    partition: ClubSolidsPartition | SolidSpatialInertia,
    *,
    body_position_world_m: ArrayLike,
    body_rotation_world: ArrayLike,
    body_linear_velocity_world_m_s: ArrayLike | None = None,
    **rates_and_gravity: ArrayLike,
) -> dict[str, Any]:
    """Compute world-frame Newton-Euler club keywords from club-body-frame kinematics.

    Given the world pose ``(p_B, R_WB)``, velocity ``(v_B, omega)``, and
    classical acceleration ``(a_B, alpha)`` of the specification's club body
    frame ``B``, transforms the spatial inertia (``partition.bare_club_inertia``
    or a supplied :class:`SolidSpatialInertia`) into the world frame:

    .. math::

        c_{\\text{bare}} &= p_B + R_{WB}\\, c_{\\text{bare},B} \\\\
        a_{c,\\text{bare}} &= a_B + \\alpha \\times r + \\omega \\times (\\omega \\times r),
            \\quad r = R_{WB}\\, c_{\\text{bare},B} \\\\
        I_{\\text{bare},W} &= R_{WB}\\, I_{\\text{bare},B}\\, R_{WB}^T

    Returns the dictionary expected by :func:`holding_hand_wrench`.
    """
    if isinstance(partition, ClubSolidsPartition):
        inertia_body = partition.bare_club_inertia
    elif isinstance(partition, SolidSpatialInertia):
        inertia_body = partition
    else:
        raise TypeError(
            "partition must be a ClubSolidsPartition or SolidSpatialInertia"
        )

    missing = [k for k in _REQUIRED_KINEMATICS_KEYS if k not in rates_and_gravity]
    unknown = sorted(set(rates_and_gravity) - set(_REQUIRED_KINEMATICS_KEYS))
    if missing or unknown:
        raise ValueError(
            f"kinematics keywords missing {missing} / unknown {unknown}; "
            f"expected {list(_REQUIRED_KINEMATICS_KEYS)}"
        )

    p_b = _vec3_array(body_position_world_m, "body_position_world_m")
    rot_wb = _rot3_matrix(body_rotation_world, "body_rotation_world")
    if body_linear_velocity_world_m_s is not None:
        _vec3_array(body_linear_velocity_world_m_s, "body_linear_velocity_world_m_s")
    omega = _vec3_array(
        rates_and_gravity["body_angular_velocity_world_rad_s"],
        "body_angular_velocity_world_rad_s",
    )
    a_b = _vec3_array(
        rates_and_gravity["body_linear_acceleration_world_m_s2"],
        "body_linear_acceleration_world_m_s2",
    )
    alpha = _vec3_array(
        rates_and_gravity["body_angular_acceleration_world_rad_s2"],
        "body_angular_acceleration_world_rad_s2",
    )
    gravity = _vec3_array(rates_and_gravity["gravity_m_s2"], "gravity_m_s2")

    c_bare_b = np.asarray(inertia_body.com_body_m, dtype=np.float64)
    r_w = rot_wb @ c_bare_b
    c_bare_w = p_b + r_w
    a_c_bare = a_b + np.cross(alpha, r_w) + np.cross(omega, np.cross(omega, r_w))
    inertia_w = rot_wb @ inertia_body.inertia_com_body_kg_m2 @ rot_wb.T

    return {
        "mass_kg": float(inertia_body.mass_kg),
        "gravity_m_s2": gravity,
        "com_m": c_bare_w,
        "com_acceleration_m_s2": a_c_bare,
        "inertia_world_kg_m2": inertia_w,
        "angular_velocity_rad_s": omega,
        "angular_acceleration_rad_s2": alpha,
    }

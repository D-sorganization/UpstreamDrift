"""Club wrench and external-load assembly for full-body driving (SPN-9c, #12758).

Why hands and not a club body.  The full-body target models have no club.  The
matched swing carries one (``msk_club``: a rigid ``Club`` body held by both
hands).  Two options were weighed: add a club body to the licensed model (it
changes the asset, its inertia and its constraint topology, and needs a
grip-closure model in a file we may not modify) or apply the club's net reaction
as ``ExternalLoads`` on the hand bodies.  The hand loads leave the target model
untouched, keep the receipt hash of the model file valid, and carry exactly the
quantity spinal loading needs: the wrench the club exchanges with the hands.
The cost is that grip compliance and the exact point of application are not
modelled (see the limitations in the SPN-9c section).

Newton-Euler of the rigid club (ground frame, X anterior, Y up, Z right; SI)::

    F_h  = m (a_c - g)                       force the HANDS exert ON the club
    M_c  = I_w alpha + omega x (I_w omega)   moment about the club COM,
    I_w  = R I_local R^T

with ``g = (0, -9.80665, 0)``.  ``F_h`` and ``M_c`` are the sum of both hands.
Split.  The indeterminate split between the hands uses fixed weights ``w_i``
(``sum w_i = 1``, assumption): ``F_i = w_i F_h`` at the grip point ``p_i`` and a
couple ``tau_i = w_i (M_c - sum_j (p_j - c) x F_j)``, so that
``sum_i (tau_i + (p_i - c) x F_i) = M_c`` exactly.
Reaction on the hand.  The hand receives the opposite wrench; transported to
the hand body origin ``o_i`` it is the force ``-F_i`` and the torque
``-tau_i - (p_i - o_i) x F_i``, applied by ``ExternalForce`` at the origin with
force and torque expressed in ground.

The ground reaction is not computed here: it reuses
``musculoskeletal_grf.estimate_ground_reactions`` (force-free ID, friction
limited NNLS over sole points) and is combined with the hand loads in one
``ExternalLoads`` file by :func:`write_load_set`.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy.interpolate import CubicSpline

from src.engines.physics_engines.opensim.python.musculoskeletal_grf import (
    ContactPoint,
    write_external_loads,
)
from src.shared.python.contracts import ensure
from src.shared.python.core.constants import GRAVITY_FLOAT
from src.shared.python.motion_matching.kinematic_smoothing import (
    smooth_kinematic_trajectory,
)

__all__ = [
    "KINDS",
    "POLICIES",
    "ClubInertia",
    "ClubWrench",
    "HandWrenches",
    "LoadSet",
    "club_net_wrench",
    "combine_load_sets",
    "describe_load_set",
    "hand_reaction_loads",
    "loads_record",
    "split_between_hands",
    "write_load_set",
]

POLICIES = ("none", "supplied", "grf_only", "club_and_grf")
KINDS = ("club_hand_wrench", "ground_reaction")
_MIN_SAMPLES = 4
_DATAFILE = re.compile(r"<datafile>[^<]*</datafile>")
_ROT_TOL = 1e-6
_SYM_TOL = 1e-9
_WEIGHT_TOL = 1e-9


def _finite_array(value: object, shape: tuple[int, ...], label: str) -> np.ndarray:
    arr = np.array(value, dtype=float, copy=True)
    if arr.shape != shape:
        raise ValueError(f"{label} must have shape {shape}, got {arr.shape}")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{label} must be finite")
    arr.setflags(write=False)
    return arr


@dataclass(frozen=True)
class ClubInertia:
    """Rigid club mass properties in the club body frame.

    ``com_local`` is the centre of mass [m] and ``inertia_local`` the inertia
    tensor about the COM [kg m^2], both in the body frame.

    Preconditions (``ValueError``): ``mass_kg`` finite and positive,
    ``com_local`` three finite numbers, ``inertia_local`` a symmetric
    positive-definite (3, 3) matrix.
    """

    mass_kg: float
    com_local: Any
    inertia_local: Any

    def __post_init__(self) -> None:
        mass = float(self.mass_kg)
        if not np.isfinite(mass) or mass <= 0.0:
            raise ValueError(f"mass_kg must be finite and positive, got {mass}")
        com = _finite_array(self.com_local, (3,), "com_local")
        inertia = _finite_array(self.inertia_local, (3, 3), "inertia_local")
        if not np.allclose(inertia, inertia.T, atol=_SYM_TOL, rtol=0):
            raise ValueError("inertia_local must be symmetric")
        if np.linalg.eigvalsh(inertia).min() <= 0.0:
            raise ValueError("inertia_local must be positive definite")
        object.__setattr__(self, "mass_kg", mass)
        object.__setattr__(self, "com_local", com)
        object.__setattr__(self, "inertia_local", inertia)


@dataclass(frozen=True)
class ClubWrench:
    """Net wrench the hands exert ON the club: force and moment about the COM.

    Arrays are ``(n, 3)`` in the ground frame; force in N, moment in N m.
    """

    force_n: np.ndarray
    moment_com_nm: np.ndarray

    def __post_init__(self) -> None:
        force = np.array(self.force_n, dtype=float, copy=True)
        if force.ndim != 2 or force.shape[1] != 3:
            raise ValueError(f"force_n must have shape (n, 3), got {force.shape}")
        object.__setattr__(self, "force_n", force)
        moment = _finite_array(self.moment_com_nm, force.shape, "moment_com_nm")
        object.__setattr__(self, "moment_com_nm", moment)
        force.setflags(write=False)


@dataclass(frozen=True)
class HandWrenches:
    """Per-hand wrench ON the club: force at the grip point plus a couple.

    ``force_n`` and ``torque_nm`` are ``(n, H, 3)`` (ground frame).
    """

    force_n: np.ndarray
    torque_nm: np.ndarray


def _validated_series(
    time_s: object, com: object, rot: object
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    t = np.asarray(time_s, dtype=float)
    if t.ndim != 1:
        raise ValueError(f"time_s must be 1-D, got shape {t.shape}")
    n = t.size
    com_w = np.asarray(com, dtype=float)
    rot_w = np.asarray(rot, dtype=float)
    if com_w.shape != (n, 3) or rot_w.shape != (n, 3, 3):
        raise ValueError(
            f"com_world must have shape ({n}, 3) and rot_world ({n}, 3, 3); "
            f"got {com_w.shape} and {rot_w.shape}"
        )
    if n < _MIN_SAMPLES:
        raise ValueError(f"need at least {_MIN_SAMPLES} samples, got {n}")
    if not (np.all(np.isfinite(t)) and np.all(np.isfinite(com_w))):
        raise ValueError("time_s and com_world must be finite")
    if np.any(np.diff(t) <= 0.0):
        raise ValueError("time_s must be strictly increasing")
    gram = np.einsum("nij,nkj->nik", rot_w, rot_w)
    if not np.all(np.isfinite(rot_w)) or not np.allclose(
        gram, np.eye(3), atol=_ROT_TOL, rtol=0
    ):
        raise ValueError("rot_world must contain finite proper rotation matrices")
    return t, com_w, rot_w


def _smoothed(time_s: np.ndarray, series: np.ndarray, filter_hz: float | None):
    if filter_hz is None:
        return series
    result = smooth_kinematic_trajectory(time_s, series, cutoff_hz=filter_hz)
    if not result.smoothing_applied:
        raise ValueError(f"low-pass filter at {filter_hz} Hz could not be applied")
    return result.q


def _vee(skew: np.ndarray) -> np.ndarray:
    """Axial vectors of ``(n, 3, 3)`` skew parts."""
    return 0.5 * np.stack(
        [
            skew[:, 2, 1] - skew[:, 1, 2],
            skew[:, 0, 2] - skew[:, 2, 0],
            skew[:, 1, 0] - skew[:, 0, 1],
        ],
        axis=1,
    )


def club_net_wrench(
    time_s: np.ndarray,
    com_world: np.ndarray,
    rot_world: np.ndarray,
    inertia: ClubInertia,
    *,
    filter_hz: float | None = None,
) -> ClubWrench:
    """Net force and COM moment the hands exert on the club (Newton-Euler).

    Derivatives come from not-a-knot cubic splines of the COM position and of
    the nine rotation entries (``omega x = dR/dt R^T``,
    ``alpha x = d2R/dt2 R^T - (omega x)^2``), after an optional zero-phase
    Butterworth low-pass at ``filter_hz`` (the same filter as the inverse
    dynamics).  End samples are less reliable than the interior.

    Preconditions (``ValueError``): ``time_s`` ``(n,)`` strictly increasing with
    ``n >= 4``; ``com_world`` ``(n, 3)`` finite; ``rot_world`` ``(n, 3, 3)``
    proper rotations (club frame to ground); ``filter_hz`` positive if given.
    Postconditions: finite ``(n, 3)`` force [N] and moment [N m] in ground.
    """
    t, com, rot = _validated_series(time_s, com_world, rot_world)
    if filter_hz is not None and not (np.isfinite(filter_hz) and filter_hz > 0.0):
        raise ValueError(f"filter_hz must be positive and finite, got {filter_hz!r}")
    n = t.size
    com_s = _smoothed(t, com, filter_hz)
    rot_flat = _smoothed(t, rot.reshape(n, 9), filter_hz).reshape(n, 3, 3)
    accel = CubicSpline(t, com_s, axis=0)(t, 2)
    spline = CubicSpline(t, rot_flat.reshape(n, 9), axis=0)
    rdot = spline(t, 1).reshape(n, 3, 3)
    rddot = spline(t, 2).reshape(n, 3, 3)
    omega_hat = rdot @ np.transpose(rot_flat, (0, 2, 1))
    omega = _vee(omega_hat)
    alpha = _vee(rddot @ np.transpose(rot_flat, (0, 2, 1)) - omega_hat @ omega_hat)
    world_inertia = rot_flat @ inertia.inertia_local @ np.transpose(rot_flat, (0, 2, 1))
    moment = np.einsum("nij,nj->ni", world_inertia, alpha) + np.cross(
        omega, np.einsum("nij,nj->ni", world_inertia, omega)
    )
    gravity = np.array([0.0, -GRAVITY_FLOAT, 0.0])
    force = inertia.mass_kg * (accel - gravity)
    ensure(bool(np.all(np.isfinite(force))), "club force must be finite")
    return ClubWrench(force, moment)


def split_between_hands(
    net: ClubWrench,
    com_world: np.ndarray,
    grips_world: np.ndarray,
    weights: Sequence[float],
) -> HandWrenches:
    """Distribute the net club wrench over the hands (see the module docstring).

    Postcondition: ``sum_i F_i = F`` and
    ``sum_i (tau_i + (p_i - c) x F_i) = M_c`` for every frame.

    Preconditions (``ValueError``): ``com_world`` ``(n, 3)``, ``grips_world``
    ``(n, H, 3)``; one non-negative weight per hand, summing to 1.
    """
    n = net.force_n.shape[0]
    com = np.asarray(com_world, dtype=float)
    grips = np.asarray(grips_world, dtype=float)
    if com.shape != (n, 3) or grips.ndim != 3 or grips.shape[::2] != (n, 3):
        raise ValueError(
            f"com_world must have shape ({n}, 3) and grips_world ({n}, H, 3); "
            f"got {com.shape} and {grips.shape}"
        )
    w = np.asarray(weights, dtype=float)
    if w.shape != (grips.shape[1],):
        raise ValueError("one weight per hand is required")
    if np.any(w < 0.0) or not np.all(np.isfinite(w)):
        raise ValueError("hand weights must be finite and non-negative")
    if abs(float(w.sum()) - 1.0) > _WEIGHT_TOL:
        raise ValueError(f"hand weights must sum to 1, got {float(w.sum())}")
    force = w[None, :, None] * net.force_n[:, None, :]
    lever = grips - com[:, None, :]
    leftover = net.moment_com_nm - np.cross(lever, force).sum(axis=1)
    torque = w[None, :, None] * leftover[:, None, :]
    return HandWrenches(force, torque)


def hand_reaction_loads(
    hands: HandWrenches, grips_world: np.ndarray, origins_world: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Wrench the club exerts ON each hand, at the hand body origin.

    Returns ``(force, torque)`` of shape ``(n, H, 3)`` in ground: the force is
    the opposite of the hand's force on the club and the torque is
    ``-tau - (p - o) x F`` (see the module docstring).
    """
    grips = np.asarray(grips_world, dtype=float)
    origins = np.asarray(origins_world, dtype=float)
    if grips.shape != hands.force_n.shape or origins.shape != grips.shape:
        raise ValueError("grips_world and origins_world must match the hand wrenches")
    force = -hands.force_n
    torque = -hands.torque_nm - np.cross(grips - origins, hands.force_n)
    return force, torque


@dataclass(frozen=True)
class LoadSet:
    """External point loads over a frame series (ground-expressed).

    ``forces``/``torques`` are ``(n, P, 3)``; each point is a
    :class:`ContactPoint` (name, body, local application point); ``kinds``
    labels every point with one of :data:`KINDS`.
    """

    points: tuple[ContactPoint, ...]
    forces: np.ndarray
    torques: np.ndarray
    kinds: tuple[str, ...]

    def __post_init__(self) -> None:
        points = tuple(self.points)
        forces = np.array(self.forces, dtype=float, copy=True)
        if forces.ndim != 3 or forces.shape[1:] != (len(points), 3):
            raise ValueError(
                f"forces must have shape (n, {len(points)}, 3), got {forces.shape}"
            )
        torques = _finite_array(self.torques, forces.shape, "torques")
        if not np.all(np.isfinite(forces)):
            raise ValueError("forces must be finite")
        if len(self.kinds) != len(points) or any(k not in KINDS for k in self.kinds):
            raise ValueError(f"one kind per point from {KINDS} is required")
        names = [p.name for p in points]
        if len(set(names)) != len(names):
            raise ValueError(f"duplicate point names: {names}")
        forces.setflags(write=False)
        object.__setattr__(self, "points", points)
        object.__setattr__(self, "forces", forces)
        object.__setattr__(self, "torques", torques)
        object.__setattr__(self, "kinds", tuple(self.kinds))


def combine_load_sets(*sets: LoadSet) -> LoadSet:
    """Concatenate load sets over the same frames (point order preserved).

    Raises ``ValueError`` for no sets, a frame-count mismatch or a duplicate
    point name.
    """
    if not sets:
        raise ValueError("combine_load_sets needs at least one load set")
    n_frames = {s.forces.shape[0] for s in sets}
    if len(n_frames) != 1:
        raise ValueError(f"load sets differ in frame count: {sorted(n_frames)}")
    return LoadSet(
        tuple(p for s in sets for p in s.points),
        np.concatenate([s.forces for s in sets], axis=1),
        np.concatenate([s.torques for s in sets], axis=1),
        tuple(k for s in sets for k in s.kinds),
    )


def write_load_set(
    directory: str | Path, time_s: np.ndarray, loads: LoadSet, stem: str
) -> Path:
    """Write ``<stem>.sto`` and ``<stem>.xml`` (OpenSim ``ExternalLoads``).

    Reuses :func:`musculoskeletal_grf.write_external_loads`; force, point and
    torque identifiers per point, force and torque expressed in ground, the
    point in the applied body.  The ``<datafile>`` entry is rewritten to the
    basename so the XML bytes do not depend on the output directory (the
    ``.sto`` sits next to it).  Returns the XML path.
    """
    xml = write_external_loads(
        directory,
        np.asarray(time_s, dtype=float),
        loads.forces,
        loads.points,
        stem,
        torques=loads.torques,
    )
    text = xml.read_text(encoding="utf-8")
    patched, count = _DATAFILE.subn(f"<datafile>{stem}.sto</datafile>", text)
    ensure(count == 1, "ExternalLoads XML must name exactly one datafile", count)
    xml.write_text(patched, encoding="utf-8", newline="\n")
    return xml


def describe_load_set(loads: LoadSet) -> list[dict[str, str]]:
    """Name, body, kind and frame of every load (receipt listing)."""
    return [
        {"name": p.name, "body": p.body, "kind": k, "frame": "ground"}
        for p, k in zip(loads.points, loads.kinds, strict=True)
    ]


_EXCLUSION_REASONS: Mapping[str, Mapping[str, str]] = {
    "none": {
        "club_hand_wrench": "policy none applies no external loads",
        "ground_reaction": "policy none applies no external loads",
    },
    "grf_only": {
        "club_hand_wrench": "policy grf_only excludes the club hand wrench",
    },
    "club_and_grf": {
        "club_hand_wrench": "not produced by this run",
        "ground_reaction": "not produced by this run",
    },
}


def _statement(policy: str, kinds: set[str], supplied: str | None) -> str:
    if policy == "supplied":
        return (
            f"external loads supplied by file {supplied}: contents are not "
            "classified by this run"
        )
    if not kinds:
        return (
            "no external loads: club excluded, ground reaction excluded, gravity only"
        )
    if kinds == set(KINDS):
        return "club hand wrench and ground reaction included"
    if kinds == {"ground_reaction"}:
        return "ground reaction included, club excluded"
    return "club hand wrench included, ground reaction excluded"


def loads_record(
    policy: str,
    loads: LoadSet | None,
    *,
    map_id: str | None,
    supplied_name: str | None = None,
) -> dict[str, Any]:
    """Receipt record of which external loads a run includes and excludes.

    Raises ``ValueError`` for a policy outside :data:`POLICIES`.  Postcondition:
    ``included`` lists every load applied and ``excluded`` every load class
    (``club_hand_wrench``, ``ground_reaction``) the policy leaves out, with the
    reason, so an excluded club is always stated.
    """
    if policy not in POLICIES:
        raise ValueError(f"policy must be one of {POLICIES}, got {policy!r}")
    included = [] if loads is None else describe_load_set(loads)
    kinds = {item["kind"] for item in included}
    reasons = _EXCLUSION_REASONS.get(policy, {})
    excluded = [
        {"item": kind, "reason": reasons.get(kind, "not produced by this run")}
        for kind in KINDS
        if kind not in kinds and policy != "supplied"
    ]
    record: dict[str, Any] = {
        "policy": policy,
        "map_id": map_id,
        "included": included,
        "excluded": excluded,
        "statement": _statement(policy, kinds, supplied_name),
    }
    if supplied_name is not None:
        record["supplied_file"] = supplied_name
    return record

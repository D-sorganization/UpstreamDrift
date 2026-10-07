"""Ground-reaction estimation for the musculoskeletal swing (issue #11617).

The matched swing has no measured ground-reaction forces and the golf humanoid has
no foot-ground contact model.  Without ground forces the leg joint torques that
inverse dynamics returns are meaningless (the stance leg would have to be
weightless), so the missing external wrench is *estimated from the kinematics*:

1. Inverse dynamics with every force element off gives the six pelvis (root)
   generalised forces that the ground would have to supply.
2. Each foot carries two sole points (heel, ball).  Per frame, non-negative
   friction-pyramid weights are found by regularised NNLS so that the pelvis
   generalised forces of the point forces equal the required wrench.
3. The resulting forces are written as an OpenSim ``ExternalLoads`` file.

The distribution of force between two feet and between heel and ball is
statically indeterminate; the minimum-norm choice is an assumption, not a
measurement.  Leg-muscle forces inherit that uncertainty.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import nnls

from src.engines.physics_engines.opensim.python.musculoskeletal_swing import (
    ROOT_COORDINATES,
    write_sto,
)
from src.shared.python.contracts import ensure, require

DEFAULT_FRICTION = 1.0
DEFAULT_SOLE_OFFSET_M = 0.025
DEFAULT_RIDGE = 5e-2
DEFAULT_CONTACT_HEIGHT_TOL_M = 0.12
_FD_STEP = 1e-5


@dataclass(frozen=True)
class ContactPoint:
    """A sole point rigidly attached to a body."""

    name: str
    body: str
    local: tuple[float, float, float]
    foot: str = "r"


def default_contact_points(
    sole_offset_m: float = DEFAULT_SOLE_OFFSET_M,
) -> tuple[ContactPoint, ...]:
    """Medial/lateral heel and ball sole points of both feet (8 points).

    Heel points sit on the calcaneus body, ball points on the toes (MTP) body;
    two lateral positions per row give the contact patch a finite width so a
    centre of pressure can move across the foot.
    """
    require(sole_offset_m >= 0.0, "sole_offset_m must be non-negative")
    pts: list[ContactPoint] = []
    for side in ("r", "l"):
        for row, body, x, z in (
            ("heel", "calcn", -0.02, 0.03),
            ("ball", "toes", 0.02, 0.045),
        ):
            for tag, sign in (("a", 1.0), ("b", -1.0)):
                pts.append(
                    ContactPoint(
                        f"{row}_{tag}_{side}",
                        f"{body}_{side}",
                        (x, -sole_offset_m, sign * z),
                        side,
                    )
                )
    return tuple(pts)


def friction_generators(mu: float) -> np.ndarray:
    """Four friction-pyramid edge directions ``(±mu, 1, ±mu)``, shape ``(4, 3)``."""
    require(mu > 0.0, "friction coefficient must be positive")
    return np.array(
        [[mu, 1.0, mu], [mu, 1.0, -mu], [-mu, 1.0, mu], [-mu, 1.0, -mu]], dtype=float
    )


def solve_contact_forces(
    required: np.ndarray,
    jacobians: np.ndarray,
    active: np.ndarray,
    *,
    feet: tuple[str, ...] | None = None,
    spin_generalised: np.ndarray | None = None,
    mu: float = DEFAULT_FRICTION,
    ridge: float = 1e-4,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Find non-negative-vertical, friction-limited point forces for one frame.

    Args:
        required: ``(6,)`` root generalised force the ground must supply.
        jacobians: ``(n_points, 3, 6)`` point-position Jacobians w.r.t. the root.
        active: ``(n_points,)`` bool; inactive points carry zero force.
        feet: foot label per point; with ``spin_generalised`` each foot that has
            an active point may also apply a free vertical (spin) moment.
        spin_generalised: ``(6,)`` root generalised force of a unit vertical moment.
        mu: friction coefficient (pyramid half-width per unit normal force).
        ridge: Tikhonov weight pulling the pyramid weights towards an equal
            share of the required vertical force (tie-break prior).

    Returns:
        ``(forces (n, 3), spin (n,), residual_norm)``.  The spin moment of a foot
        is reported on that foot's first active point; all in ground axes.

    Raises:
        ValueError: on inconsistent shapes or non-positive ridge.
    """
    require(required.shape == (6,), "required must have shape (6,)")
    require(
        jacobians.ndim == 3 and jacobians.shape[1:] == (3, 6),
        "jacobians must have shape (n, 3, 6)",
    )
    n = jacobians.shape[0]
    require(active.shape == (n,), "active must have shape (n,)")
    require(ridge > 0.0, "ridge must be positive")
    gens = friction_generators(mu)
    forces = np.zeros((n, 3))
    spin = np.zeros(n)
    idx = [int(i) for i in np.flatnonzero(active)]
    if not idx:
        return forces, spin, float(np.linalg.norm(required))
    blocks = [(jacobians[i].T @ gens.T) for i in idx]
    spin_owner: list[int] = []
    if spin_generalised is not None and feet is not None:
        for foot in dict.fromkeys(feet[i] for i in idx):
            owner = next(i for i in idx if feet[i] == foot)
            spin_owner.append(owner)
            blocks.append(np.column_stack([spin_generalised, -spin_generalised]))
    cols = np.concatenate(blocks, axis=1)
    # Prior: share the required vertical force equally over the active pyramid
    # edges (mid-foot centre of pressure, equal loading) instead of min-norm,
    # which would concentrate the load on a few edge points.
    prior = np.zeros(cols.shape[1])
    prior[: 4 * len(idx)] = max(float(required[4]), 0.0) / (4.0 * len(idx))
    a = np.vstack([cols, np.sqrt(ridge) * np.eye(cols.shape[1])])
    b = np.concatenate([required, np.sqrt(ridge) * prior])
    lam, _ = nnls(a, b, maxiter=4000)
    for slot, i in enumerate(idx):
        forces[i] = gens.T @ lam[4 * slot : 4 * slot + 4]
    base = 4 * len(idx)
    for slot, owner in enumerate(spin_owner):
        spin[owner] = lam[base + 2 * slot] - lam[base + 2 * slot + 1]
    residual = float(np.linalg.norm(cols @ lam - required))
    ensure(bool(np.all(forces[:, 1] >= -1e-9)), "vertical force must be >= 0", forces)
    return forces, spin, residual


def _mobility_indices(model: Any, state: Any, names: list[str]) -> dict[str, int]:
    out: dict[str, int] = {}
    for name in names:
        coord = model.getCoordinateSet().get(name)
        state.updU().setToZero()
        coord.setSpeedValue(state, 1.0)
        u = state.getU().to_numpy()
        out[name] = int(np.argmax(np.abs(u)))
    state.updU().setToZero()
    return out


def _pelvis_rotation(model: Any, state: Any) -> np.ndarray:
    model.realizePosition(state)
    mat = model.getBodySet().get("pelvis").getTransformInGround(state).R().asMat33()
    return np.array([[mat.get(i, j) for j in range(3)] for i in range(3)])


def _force_free_copy(model: Any) -> Any:
    osim = __import__("opensim")
    free = osim.Model(model)
    for i in range(free.getForceSet().getSize()):
        free.updForceSet().get(i).set_appliesForce(False)
    free.initSystem()
    return free


def _root_jacobian(
    root_coords: list[Any], state: Any, positions: Any, n_points: int
) -> np.ndarray:
    """Central-difference Jacobian ``(P, 3, 6)`` of sole points w.r.t. root coordinates."""
    jac = np.zeros((n_points, 3, 6))
    for j, rc in enumerate(root_coords):
        v0 = rc.getValue(state)
        rc.setValue(state, v0 + _FD_STEP, False)
        plus = positions(state)
        rc.setValue(state, v0 - _FD_STEP, False)
        minus = positions(state)
        rc.setValue(state, v0, False)
        jac[:, :, j] = (plus - minus) / (2 * _FD_STEP)
    return jac


def _spin_generalised(model: Any, root_coords: list[Any], state: Any) -> np.ndarray:
    """Generalised force of a unit ground-vertical moment: ``J_omega^T e_y``."""
    rot0 = _pelvis_rotation(model, state)
    spin_gen = np.zeros(6)
    for j in range(3):
        v0 = root_coords[j].getValue(state)
        root_coords[j].setValue(state, v0 + _FD_STEP, False)
        rp = _pelvis_rotation(model, state)
        root_coords[j].setValue(state, v0 - _FD_STEP, False)
        rm = _pelvis_rotation(model, state)
        root_coords[j].setValue(state, v0, False)
        skew = ((rp - rm) / (2 * _FD_STEP)) @ rot0.T
        spin_gen[j] = (skew[0, 2] - skew[2, 0]) / 2.0  # omega_y
    return spin_gen


def estimate_ground_reactions(
    model: Any,
    times: np.ndarray,
    coords: dict[str, np.ndarray],
    points: tuple[ContactPoint, ...] | None = None,
    *,
    mu: float = DEFAULT_FRICTION,
    height_tol_m: float = DEFAULT_CONTACT_HEIGHT_TOL_M,
    ridge: float = DEFAULT_RIDGE,
) -> dict[str, Any]:
    """Estimate point ground-reaction forces from prescribed kinematics.

    Args:
        model: musculoskeletal model (initialised).
        times: ``(N,)`` uniformly sampled, smooth kinematics.
        coords: ``{coordinate: (N,)}`` values (rad / m) for every coordinate.
        points: sole points (default heel/ball of both feet).
        mu: friction coefficient.
        height_tol_m: a point is in contact when within this of the floor.
        ridge: min-norm weight spreading force over the available sole points.

    Returns:
        dict with ``forces (N, P, 3)``, ``spins (N, P)``, ``points_world (N, P, 3)``, ``floor_y``,
        ``residual_norm (N,)``, ``required (N, 6)``, ``active (N, P)``.
    """
    osim = __import__("opensim")
    pts = points or default_contact_points()
    n = len(times)
    require(n > 12, "need at least 13 frames")
    free = _force_free_copy(model)
    state = free.initSystem()
    names = [c.getName() for c in free.getCoordinateSet()]
    require(all(k in coords for k in names), "coords must cover every model coordinate")
    mob = _mobility_indices(free, state, names)
    dt = float(np.mean(np.diff(times)))
    vel = {k: np.gradient(v, dt) for k, v in coords.items()}
    acc = {k: np.gradient(v, dt) for k, v in vel.items()}
    root_idx = [mob[c] for c in ROOT_COORDINATES]
    solver = osim.InverseDynamicsSolver(free)
    bodies = [free.getBodySet().get(p.body) for p in pts]
    locals_ = [osim.Vec3(*p.local) for p in pts]
    root_coords = [free.getCoordinateSet().get(c) for c in ROOT_COORDINATES]

    def point_positions(st: Any) -> np.ndarray:
        free.realizePosition(st)
        return np.array(
            [
                b.findStationLocationInGround(st, loc).to_numpy()
                for b, loc in zip(bodies, locals_, strict=True)
            ]
        )

    forces = np.zeros((n, len(pts), 3))
    spins = np.zeros((n, len(pts)))
    feet = tuple(p.foot for p in pts)
    world = np.zeros((n, len(pts), 3))
    required = np.zeros((n, 6))
    resid = np.zeros(n)
    for k in range(n):
        for name in names:
            c = free.getCoordinateSet().get(name)
            c.setValue(state, float(coords[name][k]), False)
            c.setSpeedValue(state, float(vel[name][k]))
        udot = osim.Vector(state.getNU(), 0.0)
        for name in names:
            udot.set(mob[name], float(acc[name][k]))
        free.realizeVelocity(state)
        tau = solver.solve(state, udot).to_numpy()
        # ID output is the root force needed to produce the motion: ground supplies it.
        required[k] = tau[root_idx]
        base = point_positions(state)
        world[k] = base
        jac = _root_jacobian(root_coords, state, point_positions, len(pts))
        spin_gen = _spin_generalised(free, root_coords, state)
        # sign: tau_root(F) = J^T F supplies the ground wrench.
        world_k = base
        active = world_k[:, 1] <= world_k[:, 1].min() + height_tol_m
        forces[k], spins[k], resid[k] = solve_contact_forces(
            required[k],
            jac,
            active,
            feet=feet,
            spin_generalised=spin_gen,
            mu=mu,
            ridge=ridge,
        )
    floor_y = float(world[:, :, 1].min())
    act = world[:, :, 1] <= floor_y + height_tol_m
    return {
        "forces": forces,
        "spins": spins,
        "points_world": world,
        "floor_y": floor_y,
        "residual_norm": resid,
        "required": required,
        "active": act,
        "points": pts,
    }


def write_external_loads(
    directory: str | Path,
    times: np.ndarray,
    forces: np.ndarray,
    points: tuple[ContactPoint, ...],
    stem: str = "grf_estimated",
    spins: np.ndarray | None = None,
) -> Path:
    """Write ``<stem>.sto`` and ``<stem>.xml`` (ExternalLoads); return the XML path."""
    osim = __import__("opensim")
    require(forces.shape[:2] == (len(times), len(points)), "forces/points mismatch")
    out = Path(directory)
    out.mkdir(parents=True, exist_ok=True)
    cols: dict[str, np.ndarray] = {}
    loads = osim.ExternalLoads()
    sto = out / f"{stem}.sto"
    for j, pt in enumerate(points):
        tag = f"{pt.name}"
        for ax, comp in zip("xyz", range(3), strict=True):
            cols[f"{tag}_force_v{ax}"] = forces[:, j, comp]
        for ax, val in zip("xyz", pt.local, strict=True):
            cols[f"{tag}_point_p{ax}"] = np.full(len(times), float(val))
        spin = np.zeros(len(times)) if spins is None else spins[:, j]
        cols[f"{tag}_torque_x"] = np.zeros(len(times))
        cols[f"{tag}_torque_y"] = spin
        cols[f"{tag}_torque_z"] = np.zeros(len(times))
        ef = osim.ExternalForce()
        ef.set_torque_identifier(f"{tag}_torque_")
        ef.setName(tag)
        ef.set_applied_to_body(pt.body)
        ef.set_force_expressed_in_body("ground")
        ef.set_point_expressed_in_body(pt.body)
        ef.set_force_identifier(f"{tag}_force_v")
        ef.set_point_identifier(f"{tag}_point_p")
        loads.cloneAndAppend(ef)
    write_sto(sto, times, cols)
    loads.setDataFileName(str(sto.resolve()))
    xml = out / f"{stem}.xml"
    loads.printToXML(str(xml))
    return xml

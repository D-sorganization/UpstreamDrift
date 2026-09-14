"""Ground-supported forward dynamics of the MuJoCo full-body model.

The root six coordinates (pelvis translation and rotation) are unactuated:
the golfer is carried by the shared contact law at the foot spheres, never by
root forces. Joint torques on the remaining coordinates come from a controller
callable. Integration is fixed-step RK4 over the adapter's constrained
accelerations, so the physics (weld closure, contact law) is exactly the
FB-3-M adapter's and stays comparable with the Pinocchio and Drake lanes.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from src.engines.physics_engines.mujoco.python.full_body_model import (
    NativeMujocoFullBodyModel,
)
from src.engines.physics_engines.mujoco.python.native_model import (
    _evaluate_weld_closure,
)
from src.shared.python.motion_matching.contact_law import GroundPlane
from src.shared.python.motion_matching.ground_support import (
    SupportReport,
    convex_hull_contains,
    support_report,
)

Array = NDArray[np.float64]
Controller = Callable[[float, Array, Array], Array]

MIN_SINGULAR_VALUE = 1e-2  # 1 / (kg m^2): directions below this are closure-locked
ROOT_COORDINATES = (
    "TranslationInputX",
    "TranslationInputY",
    "TranslationInputZ",
    "HipInputX",
    "HipInputY",
    "HipInputZ",
)


@dataclass(frozen=True)
class SimulationRecord:
    """Sampled state, torque and support history of one run."""

    time_s: Array
    q: Array
    v: Array
    tau: Array
    normal_force_n: Array
    weight_fraction: Array
    centre_of_pressure_m: Array
    inside_support_polygon: Array
    lowest_sphere_height_m: Array


class FullBodySimulator:
    """RK4 forward dynamics with unactuated root and shared ground contact."""

    def __init__(self, adapter: NativeMujocoFullBodyModel) -> None:
        names = tuple(adapter.coordinate_order)
        if names[:6] != ROOT_COORDINATES:
            raise ValueError("The first six coordinates must be the pelvis root")
        self.adapter = adapter
        self.names = names
        self.nv = len(names)
        self.root = np.arange(6)
        self.actuated = np.arange(6, self.nv)
        self.lower_limb = np.arange(adapter.upper_body_coordinates, self.nv)
        self.mass_kg = float(np.sum(adapter.model.body_mass))
        self.gravity = np.array(adapter.model.opt.gravity, dtype=float)
        # Adapter force vectors follow MuJoCo DOF order; states follow spec order.
        self._dof = np.array([adapter.model.joint(name).dofadr[0] for name in names])

    def _map(self, values: Array) -> dict[str, float]:
        return dict(zip(self.names, values.tolist(), strict=True))

    def root_translation_axes(self, q: Array) -> Array:
        """World directions (columns) of the three root slide coordinates at ``q``.

        The Simscape root translation primitives are not world-aligned, so a
        world displacement ``d`` corresponds to ``solve(axes, d)`` in q.
        """
        adapter = self.adapter
        adapter.frame_poses(self._map(q))
        model = adapter.model
        return np.column_stack(
            [adapter.data.xaxis[model.joint(name).id] for name in self.names[:3]]
        )

    def acceleration(self, q: Array, v: Array, tau: Array) -> Array:
        """Constrained acceleration with contact; root torques are forced to zero."""
        effort = np.asarray(tau, dtype=float).copy()
        if effort.shape != (self.nv,):
            raise ValueError("Torque vector must match the coordinate count")
        effort[self.root] = 0.0
        result = self.adapter.accelerations(
            self._map(q), self._map(v), self._map(effort)
        )
        return np.array([result[name] for name in self.names])

    def feedforward(
        self, q: Array, v: Array, *, compensate_contact: bool = True
    ) -> Array:
        """Torque cancelling the bias (and, optionally, contact) generalized forces.

        Only actuated joints receive torque. With ``compensate_contact`` the
        instantaneous contact wrench mapped to the joints is cancelled too, which
        is exact at equilibrium but follows every contact transient; without it
        the PD loop carries the ground reaction.
        """
        bias, contact, _ = self.adapter.generalized_forces(self._map(q), self._map(v))
        tau = np.zeros(self.nv)
        force = bias - contact if compensate_contact else bias
        tau[self.actuated] = force[self._dof][self.actuated]
        return tau

    def static_penetration_m(self) -> float:
        """Penetration at which equally loaded spheres carry the weight at rest."""
        stiffness = float(self.adapter.contact_parameters.stiffness_n_m)
        return (
            self.mass_kg
            * float(np.linalg.norm(self.gravity))
            / (stiffness * len(self.adapter._spheres))
        )

    def affine_dynamics(self, q: Array, v: Array) -> tuple[Array, Array]:
        """Return ``(A, b)`` with ``a = A @ tau_actuated + b`` at the state.

        One KKT factorisation of the same closure-constrained system the
        adapter integrates (mass matrix, dual-grip weld, bias and contact
        forces), so joint torques can be chosen for a desired acceleration.
        Postcondition: ``A`` is (nv, actuated), ``b`` equals the zero-torque
        acceleration.
        """
        adapter = self.adapter
        bias, contact, _ = adapter.generalized_forces(self._map(q), self._map(v))
        mj, model, data = adapter._mj, adapter.model, adapter.data
        mass = np.zeros((model.nv, model.nv))
        mj.mj_fullM(model, mass, data.qM)
        jac, drift = _evaluate_weld_closure(mj, model, data, adapter._closure)
        m = jac.shape[0]
        kkt = np.block([[mass, -jac.T], [jac, np.zeros((m, m))]])
        rhs = np.zeros((model.nv + m, 1 + self.actuated.size))
        rhs[: model.nv, 0] = contact - bias
        rhs[model.nv :, 0] = -drift
        rhs[self._dof[self.actuated], 1:] = np.eye(self.actuated.size)
        solution = np.linalg.solve(kkt, rhs)[: model.nv]
        ordered = solution[self._dof]  # spec coordinate order
        return ordered[:, 1:], ordered[:, 0]

    def inverse_dynamics(self, q: Array, v: Array, joint_acceleration: Array) -> Array:
        """Torques giving the actuated joints exactly ``joint_acceleration``.

        The root is unactuated, so its acceleration follows from the contact
        and gravity; ``joint_acceleration`` has one entry per actuated joint.
        Directions removed by the grip closure are matched in the least-squares
        sense with the least-norm torque.
        """
        target = np.asarray(joint_acceleration, dtype=float)
        if target.shape != (self.actuated.size,) or not np.isfinite(target).all():
            raise ValueError("Joint acceleration must be finite, one per joint")
        affine, offset = self.affine_dynamics(q, v)
        rows = self.actuated
        # The grip closure removes six controllable directions, so the square
        # joint block is rank deficient: invert only the singular directions a
        # real joint can move (apparent inertia below 1 / MIN_SINGULAR_VALUE)
        # and take the least-norm torque in the rest.
        affine, offset = self.affine_dynamics(q, v)
        rows = self.actuated
        u, sv, vt = np.linalg.svd(affine[rows], full_matrices=False)
        keep = sv > MIN_SINGULAR_VALUE
        inverse = (vt[keep].T / sv[keep]) @ u[:, keep].T
        tau = np.zeros(self.nv)
        tau[rows] = inverse @ (target - offset[rows])
        return tau

    def centre_of_mass(self, q: Array) -> tuple[Array, Array]:
        """Whole-body centre of mass and its Jacobian (spec coordinate order)."""
        adapter = self.adapter
        adapter.frame_poses(self._map(q))
        mj, model, data = adapter._mj, adapter.model, adapter.data
        mj.mj_comPos(model, data)
        jac = np.zeros((3, model.nv))
        mj.mj_jacSubtreeCom(model, data, jac, 1)  # body 1 roots the whole tree
        return data.subtree_com[1].copy(), jac[:, self._dof]

    def support(self, q: Array, v: Array) -> tuple[SupportReport, float]:
        """Support report and the lowest sphere height at a state."""
        samples = self.adapter.evaluate_contact_samples(self._map(q), self._map(v))
        plane = self.adapter.ground_plane
        n = np.asarray(plane.normal, dtype=float)
        n = n / np.linalg.norm(n)
        points: dict[str, Array] = {}
        lowest = np.inf
        for name, info in self.adapter._spheres.items():
            centre = self.adapter.data.site_xpos[info["site_id"]].copy()
            height = float(centre @ n - plane.height_m)
            lowest = min(lowest, height - info["radius"])
            points[name] = centre - (height) * n
        report = support_report(
            samples, points, plane, self.mass_kg, self.gravity.tolist()
        )
        return report, float(lowest)

    def step(
        self, t: float, q: Array, v: Array, controller: Controller, dt: float
    ) -> tuple[Array, Array, Array]:
        """One RK4 step; returns the new state and the torque used at the start."""

        def rate(t_k: float, q_k: Array, v_k: Array) -> tuple[Array, Array, Array]:
            tau_k = np.asarray(controller(t_k, q_k, v_k), dtype=float)
            return v_k, self.acceleration(q_k, v_k, tau_k), tau_k

        k1q, k1v, tau = rate(t, q, v)
        k2q, k2v, _ = rate(t + dt / 2, q + dt / 2 * k1q, v + dt / 2 * k1v)
        k3q, k3v, _ = rate(t + dt / 2, q + dt / 2 * k2q, v + dt / 2 * k2v)
        k4q, k4v, _ = rate(t + dt, q + dt * k3q, v + dt * k3v)
        q_next = q + dt / 6 * (k1q + 2 * k2q + 2 * k3q + k4q)
        v_next = v + dt / 6 * (k1v + 2 * k2v + 2 * k3v + k4v)
        if not (np.isfinite(q_next).all() and np.isfinite(v_next).all()):
            raise FloatingPointError(f"Nonfinite state at t={t:.4f}s")
        return q_next, v_next, tau

    def run(
        self,
        q0: Array,
        v0: Array,
        controller: Controller,
        *,
        duration_s: float,
        dt_s: float,
        record_every: int = 1,
    ) -> SimulationRecord:
        """Integrate from ``(q0, v0)`` and sample every ``record_every`` steps.

        Precondition: finite state of model size, positive duration and step.
        Postcondition: the record starts at t=0 with the initial state and the
        support columns are evaluated at every sampled state.
        """
        q = np.asarray(q0, dtype=float).copy()
        v = np.asarray(v0, dtype=float).copy()
        if q.shape != (self.nv,) or v.shape != (self.nv,):
            raise ValueError("Initial state must match the coordinate count")
        if not (np.isfinite(q).all() and np.isfinite(v).all()):
            raise ValueError("Initial state must be finite")
        if duration_s <= 0 or dt_s <= 0 or record_every < 1:
            raise ValueError("Duration, step and record interval must be positive")
        steps = int(round(duration_s / dt_s))
        times, qs, vs, taus = [0.0], [q.copy()], [v.copy()], []
        supports: list[tuple[SupportReport, float]] = [self.support(q, v)]
        tau_prev = np.zeros(self.nv)
        for k in range(1, steps + 1):
            q, v, tau_prev = self.step((k - 1) * dt_s, q, v, controller, dt_s)
            if k % record_every == 0 or k == steps:
                times.append(k * dt_s)
                qs.append(q.copy())
                vs.append(v.copy())
                taus.append(tau_prev.copy())
                supports.append(self.support(q, v))
        taus.insert(0, taus[0] if taus else tau_prev)
        cops = np.array(
            [
                r.centre_of_pressure_m
                if r.centre_of_pressure_m is not None
                else (np.nan,) * 3
                for r, _ in supports
            ]
        )
        return SimulationRecord(
            time_s=np.array(times),
            q=np.array(qs),
            v=np.array(vs),
            tau=np.array(taus),
            normal_force_n=np.array([r.total_normal_force_n for r, _ in supports]),
            weight_fraction=np.array([r.weight_fraction for r, _ in supports]),
            centre_of_pressure_m=cops,
            inside_support_polygon=np.array(
                [r.inside_support_polygon for r, _ in supports]
            ),
            lowest_sphere_height_m=np.array([h for _, h in supports]),
        )


def preload_feet(
    simulator: FullBodySimulator, q: Array, *, preload: bool = True
) -> Array:
    """Translate the root along the ground normal so the feet rest on the plane.

    Only the root translation changes, so the posture (and the CoM offset from
    the support polygon, which the pose solver's ``balance_weight`` controls)
    is untouched. With ``preload`` the lowest sphere sits at the static
    penetration so the contact already carries the weight; otherwise it just
    touches. Postcondition: the lowest sphere bottom is at ``-depth``.
    """
    q = np.asarray(q, dtype=float).copy()
    depth = simulator.static_penetration_m() if preload else 0.0
    adapter = simulator.adapter
    plane = adapter.ground_plane
    n = np.asarray(plane.normal, dtype=float)
    n = n / np.linalg.norm(n)
    adapter.frame_poses(simulator._map(q))
    centres = np.array(
        [adapter.data.site_xpos[i["site_id"]] for i in adapter._spheres.values()]
    )
    radii = np.array([i["radius"] for i in adapter._spheres.values()])
    lowest = float(np.min(centres @ n - radii)) - plane.height_m + depth
    q[:3] += np.linalg.solve(simulator.root_translation_axes(q), -lowest * n)
    return q


def _check_gains(
    omega_rad_s: float | Array, zeta: float, balance: tuple[float, float] | None
) -> None:
    omega = np.asarray(omega_rad_s, dtype=float)
    if not np.isfinite(omega).all() or np.any(omega <= 0) or zeta <= 0:
        raise ValueError("Natural frequency and damping ratio must be positive")
    if balance is not None and (len(balance) != 2 or min(balance) < 0):
        raise ValueError("Balance gains must be two nonnegative numbers")


def joint_natural_frequencies(
    simulator: FullBodySimulator, *, upper_body: float, lower_limb: float
) -> Array:
    """Per-coordinate natural frequency vector: stiff upper body, compliant legs.

    Compliant legs let the ground constraint, not the joint servo, decide
    where the feet rest, so a small root deviation does not turn into foot
    penetration or lift-off. Root entries are unused.
    """
    if upper_body <= 0 or lower_limb <= 0:
        raise ValueError("Natural frequencies must be positive")
    omega = np.full(simulator.nv, float(upper_body))
    omega[simulator.lower_limb] = float(lower_limb)
    return omega


def _planted_coupling(simulator: FullBodySimulator) -> Array:
    """``S`` with ``v_root = S v_legs`` when every contact sphere is held still.

    From ``J_feet,root v_root + J_feet,legs v_legs = 0`` at the current state
    (the adapter's kinematics must already be evaluated).
    """
    adapter = simulator.adapter
    mj, model, data = adapter._mj, adapter.model, adapter.data
    rows = []
    for info in adapter._spheres.values():
        buffer = np.zeros((3, model.nv))
        mj.mj_jacSite(model, data, buffer, None, info["site_id"])
        rows.append(buffer[:, simulator._dof])
    jac_feet = np.concatenate(rows)
    legs = simulator.lower_limb
    return -np.linalg.pinv(jac_feet[:, simulator.root]) @ jac_feet[:, legs]


def _root_regulation_acceleration(
    simulator: FullBodySimulator,
    q: Array,
    v: Array,
    q_ref: Array,
    v_ref: Array,
    gains: tuple[float, float],
) -> Array:
    """Lower-limb acceleration steering the root (pelvis) pose to its reference.

    With the feet planted the root pose is a function of the leg joints; a PD
    law on the six root coordinates gives a desired root acceleration that the
    least-norm inverse of the planted coupling maps onto the legs. Returns a
    full-size vector with only lower-limb entries nonzero.
    """
    simulator.adapter.frame_poses(simulator._map(q))
    simulator.adapter._mj.mj_comPos(simulator.adapter.model, simulator.adapter.data)
    coupling = _planted_coupling(simulator)
    root = simulator.root
    wanted = gains[0] * (q_ref[root] - q[root]) + gains[1] * (v_ref[root] - v[root])
    out = np.zeros(simulator.nv)
    out[simulator.lower_limb] = np.linalg.pinv(coupling) @ wanted
    return out


def _planted_com_jacobian(
    simulator: FullBodySimulator, q: Array
) -> tuple[Array, Array]:
    """CoM position and its Jacobian over the lower-limb joints with the feet planted.

    With every contact sphere held still the root motion is a function of the
    joint motion (``J_feet,root v_root + J_feet,joints v_joints = 0``), so the
    centre of mass responds to the legs through ``J_com,legs + J_com,root S``
    with ``S = -pinv(J_feet,root) J_feet,legs``. That is the ankle and hip
    strategy of a standing body.
    """
    com, jac_com = simulator.centre_of_mass(q)
    legs = simulator.lower_limb
    coupling = _planted_coupling(simulator)
    return com, jac_com[:, legs] + jac_com[:, simulator.root] @ coupling


def _balance_acceleration(
    simulator: FullBodySimulator,
    q: Array,
    v: Array,
    com_ref: Array,
    gains: tuple[float, float],
) -> Array:
    """Actuated-joint acceleration steering the CoM over ``com_ref`` (legs only).

    A PD law on the horizontal centre-of-mass error gives a desired CoM
    acceleration, mapped to the lower-limb joints through the least-norm
    inverse of the feet-planted CoM Jacobian.
    """
    com, jac = _planted_com_jacobian(simulator, q)
    n = np.asarray(simulator.adapter.ground_plane.normal, dtype=float)
    n = n / np.linalg.norm(n)
    error = com_ref - com
    error -= (error @ n) * n
    velocity = jac @ v[simulator.lower_limb]
    velocity -= (velocity @ n) * n
    wanted = gains[0] * error - gains[1] * velocity
    out = np.zeros(simulator.nv)
    out[simulator.lower_limb] = np.linalg.pinv(jac) @ wanted
    return out[simulator.actuated]


def _computed_torque(
    simulator: FullBodySimulator,
    q: Array,
    v: Array,
    q_ref: Array,
    v_ref: Array,
    a_ref: Array,
    omega_rad_s: float | Array,
    zeta: float,
    balance: tuple[float, float] | None,
    com_ref: Array | None,
    root_regulation: tuple[float, float] | None = None,
) -> Array:
    act = simulator.actuated
    omega = np.broadcast_to(np.asarray(omega_rad_s, dtype=float), (simulator.nv,))[act]
    wanted = (
        a_ref[act]
        + 2.0 * zeta * omega * (v_ref[act] - v[act])
        + omega**2 * (q_ref[act] - q[act])
    )
    if balance is not None and com_ref is not None:
        wanted = wanted + _balance_acceleration(simulator, q, v, com_ref, balance)
    if root_regulation is not None:
        wanted = (
            wanted
            + _root_regulation_acceleration(
                simulator, q, v, q_ref, v_ref, root_regulation
            )[act]
        )
    return simulator.inverse_dynamics(q, v, wanted)


def hold_pose_controller(
    simulator: FullBodySimulator,
    q_ref: Array,
    *,
    omega_rad_s: float | Array,
    zeta: float = 1.0,
    balance: tuple[float, float] | None = None,
    root_regulation: tuple[float, float] | None = None,
) -> Controller:
    """Computed-torque hold of a posture with optional centre-of-mass balance.

    Every actuated joint is driven to ``q_ref`` as a second-order system with
    natural frequency ``omega_rad_s`` (scalar or per-coordinate vector, see
    :func:`joint_natural_frequencies`) and damping ratio ``zeta`` through the
    exact contact-aware inverse dynamics; the root stays unactuated. ``balance``
    gives the (stiffness 1/s^2, damping 1/s) of the CoM law that keeps the
    centre of mass over its reference ground point.
    """
    reference = np.asarray(q_ref, dtype=float).copy()
    if reference.shape != (simulator.nv,) or not np.isfinite(reference).all():
        raise ValueError("Reference posture must be finite with model size")
    _check_gains(omega_rad_s, zeta, balance)
    _check_gains(1.0, 1.0, root_regulation)
    com_ref = simulator.centre_of_mass(reference)[0] if balance is not None else None
    zero = np.zeros(simulator.nv)

    def controller(t: float, q: Array, v: Array) -> Array:
        return _computed_torque(
            simulator,
            q,
            v,
            reference,
            zero,
            zero,
            omega_rad_s,
            zeta,
            balance,
            com_ref,
            root_regulation,
        )

    return controller


def tracking_controller(
    simulator: FullBodySimulator,
    time_ref: Sequence[float],
    q_ref: Array,
    *,
    omega_rad_s: float | Array,
    zeta: float = 1.0,
    balance: tuple[float, float] | None = None,
    root_regulation: tuple[float, float] | None = None,
    acceleration_feedforward: float = 1.0,
) -> Controller:
    """Computed-torque tracking of a reference trajectory (linear interpolation).

    ``root_regulation`` gives the (stiffness 1/s^2, damping 1/s) of the PD law
    that steers the pelvis root coordinates to the reference through the
    planted legs; combine it with compliant lower-limb frequencies.
    ``acceleration_feedforward`` scales the reference acceleration term in
    [0, 1]: 1 is exact computed torque, 0 leaves only the PD law on position
    and velocity, which a noisy (not dynamically consistent) reference needs.

    Reference velocity and acceleration come from finite differences of the
    rows. Precondition: strictly increasing times and one q row per time.
    """
    if not 0.0 <= acceleration_feedforward <= 1.0:
        raise ValueError("acceleration_feedforward must lie in [0, 1]")
    times = np.asarray(time_ref, dtype=float)
    reference = np.asarray(q_ref, dtype=float)
    if (
        times.ndim != 1
        or np.any(np.diff(times) <= 0)
        or reference.shape != (times.size, simulator.nv)
        or not np.isfinite(reference).all()
    ):
        raise ValueError("Reference times must increase with one finite q row each")
    _check_gains(omega_rad_s, zeta, balance)
    _check_gains(1.0, 1.0, root_regulation)
    if times.size > 1:
        velocity = np.gradient(reference, times, axis=0)
        acceleration = acceleration_feedforward * np.gradient(velocity, times, axis=0)
    else:
        velocity = np.zeros_like(reference)
        acceleration = np.zeros_like(reference)

    def sample(table: Array, t: float) -> Array:
        return np.array([np.interp(t, times, table[:, k]) for k in range(simulator.nv)])

    def controller(t: float, q: Array, v: Array) -> Array:
        q_t, v_t, a_t = (
            sample(reference, t),
            sample(velocity, t),
            sample(acceleration, t),
        )
        com_ref = simulator.centre_of_mass(q_t)[0] if balance is not None else None
        return _computed_torque(
            simulator,
            q,
            v,
            q_t,
            v_t,
            a_t,
            omega_rad_s,
            zeta,
            balance,
            com_ref,
            root_regulation,
        )

    return controller


def _distance_outside(point_xy: Array, hull_xy: Array) -> float:
    """Distance from a point to a convex polygon, zero inside."""
    if convex_hull_contains(point_xy, hull_xy):
        return 0.0
    best = np.inf
    n = len(hull_xy)
    for i in range(n):
        a, b = hull_xy[i], hull_xy[(i + 1) % n]
        ab = b - a
        s = float(np.clip((point_xy - a) @ ab / max(float(ab @ ab), 1e-12), 0.0, 1.0))
        best = min(best, float(np.linalg.norm(point_xy - (a + s * ab))))
    return best


def reference_zmp(
    simulator: FullBodySimulator,
    time_ref: Sequence[float],
    q_ref: Array,
    ground: GroundPlane,
    *,
    contact_tolerance_m: float = 0.005,
    min_load_fraction: float = 0.1,
) -> dict[str, Array]:
    """Zero-moment point a reference trajectory demands of this model.

    From the whole-body linear and angular momentum rates (MuJoCo subtree
    momentum, finite differences of the reference) the total ground reaction
    ``R = m (a_com + g)`` and the moment about the centre of mass follow; the
    zero-moment point is where that reaction must act on the plane for the
    moment to vanish. It is compared with the convex hull of the contact
    spheres within ``contact_tolerance_m`` of the plane at that frame (all
    spheres when fewer than three touch). Returns arrays over frames:
    ``zmp_xy``, ``com``, ``grf_over_weight`` (3), ``outside_m`` (distance
    outside the hull, zero inside), ``hull_xy`` (an object array of the
    per-frame support vertices) and ``unloaded`` (vertical reaction below
    ``min_load_fraction`` of the weight: the reference is near free fall and
    its zero-moment point is meaningless; ``outside_m`` is zero there). A
    loaded frame outside cannot be realised by any unilateral foot contact,
    whatever the controller. Preconditions: strictly increasing times with
    one finite row each, at least three frames.
    """
    times = np.asarray(time_ref, dtype=float)
    ref = np.asarray(q_ref, dtype=float)
    if (
        times.ndim != 1
        or times.size < 3
        or np.any(np.diff(times) <= 0)
        or ref.shape != (times.size, simulator.nv)
        or not np.isfinite(ref).all()
    ):
        raise ValueError(
            "Reference needs at least three increasing times and finite rows"
        )
    if contact_tolerance_m < 0 or not 0.0 < min_load_fraction < 1.0:
        raise ValueError(
            "contact tolerance must be nonnegative, load fraction in (0, 1)"
        )
    adapter = simulator.adapter
    mj, model, data = adapter._mj, adapter.model, adapter.data
    n = np.asarray(ground.normal, dtype=float)
    n = n / np.linalg.norm(n)
    g = float(np.linalg.norm(simulator.gravity))
    mass = simulator.mass_kg
    velocity = np.gradient(ref, times, axis=0)
    acceleration = np.gradient(velocity, times, axis=0)

    def momentum(q: Array, v: Array) -> tuple[Array, Array, Array]:
        adapter.frame_poses(simulator._map(q))
        data.qvel[simulator._dof] = v
        mj.mj_forward(model, data)
        mj.mj_subtreeVel(model, data)
        linear = data.subtree_linvel[1] * model.body_subtreemass[1]
        return data.subtree_com[1].copy(), linear.copy(), data.subtree_angmom[1].copy()

    dt = 1e-4
    frames = times.size
    zmp = np.empty((frames, 2))
    com = np.empty((frames, 3))
    grf = np.empty((frames, 3))
    outside = np.empty(frames)
    unloaded = np.zeros(frames, dtype=bool)
    hulls = np.empty(frames, dtype=object)
    sphere_names = list(adapter._spheres)
    for k in range(frames):
        c0, p0, l0 = momentum(ref[k], velocity[k])
        c1, p1, l1 = momentum(
            ref[k] + velocity[k] * dt, velocity[k] + acceleration[k] * dt
        )
        reaction = (p1 - p0) / dt - simulator.gravity * mass
        moment = (l1 - l0) / dt  # about the centre of mass
        normal_load = max(float(reaction @ n), 1e-6)
        height = float(c0 @ n) - ground.height_m
        # Point on the plane where the reaction gives zero moment about the CoM:
        # r = c - height n + (n x (moment + height n x R_t)) / R_n, worked per axis
        r_t = reaction - n * float(reaction @ n)
        point = c0 - n * height + (np.cross(n, moment) - height * r_t) / normal_load
        adapter.frame_poses(simulator._map(ref[k]))  # momentum left the +dt state
        centres = np.array(
            [data.site_xpos[adapter._spheres[s]["site_id"]] for s in sphere_names]
        )
        radii = np.array([adapter._spheres[s]["radius"] for s in sphere_names])
        sphere_height = centres @ n - radii - ground.height_m
        touching = sphere_height <= contact_tolerance_m
        feet = centres[touching] if touching.sum() >= 3 else centres
        hull = feet[:, :2]
        zmp[k] = point[:2]
        com[k] = c0
        grf[k] = reaction / (mass * g)
        unloaded[k] = float(reaction @ n) < min_load_fraction * mass * g
        outside[k] = 0.0 if unloaded[k] else _distance_outside(point[:2], hull)
        hulls[k] = hull.copy()
    return {
        "zmp_xy": zmp,
        "com": com,
        "grf_over_weight": grf,
        "outside_m": outside,
        "unloaded": unloaded,
        "hull_xy": hulls,
    }

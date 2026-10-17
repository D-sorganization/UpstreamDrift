"""Barbell-on-bushings runs: quasi-static hold, bundle, cross-engine parity.

LIFT-4 (#11744) on the LIFT-5 back-squat same-input bundle (#11745).  Uses
:class:`~src.shared.python.lifting.barbell_bushing.BarbellBushingPlant`.

* :func:`quasi_static_hold` holds one pose in MuJoCo with the pipeline
  computed-torque controller and the bar on both hands, and records the
  shared-law ground reaction, the per-hand bushing forces and deflections.
  It is the physical check: once settled, the vertical ground reaction must
  equal the weight of body plus bar (the bar has no other support).
* :func:`build_barbell_bundle` re-tracks a back-squat bundle's reference with
  the bar attached and returns a same-input bundle whose body arrays are the
  loaded run; the bar's states and parameters travel in the provenance.
* :func:`barbell_plant` rebuilds the coupled plant of such a bundle in any
  same-input engine and :func:`instantaneous_parity` compares, at chosen
  reference states, the per-hand bushing wrenches and the coupled
  accelerations of each engine against MuJoCo.

The controller is the pipeline computed-torque law built on
:func:`rigid_bar_spec`: the same body with the bar welded to the hand body, so
the law's inverse dynamics, centre-of-mass balance and foot preload include the
bar's weight and inertia on the whole floating-base system.  The bushings'
compliance is left to feedback.  Until #13072 the law was built on the unloaded
body with the bar's static weight subtracted as a joint-space feedforward
``J^T m g`` on the actuated rows only; the root rows of that load (about 980 N
down and 635 N m of pelvis pitch on the squat bundle) were left out of the
controller's model, so the commanded accelerations were wrong by up to about
4000 rad/s^2 from the first step and the re-track diverged.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any

import numpy as np
from numpy.typing import NDArray

from src.shared.python.lifting.barbell_bushing import (
    BAR_COORDINATES,
    SIDES,
    Barbell,
    BarbellBushingPlant,
    barbell_dynamics,
    barbell_interface,
)
from src.shared.python.motion_matching.same_input import (
    InputBundle,
    Rollout,
    StepPolicy,
    VectorPlant,
    integrate,
    project_to_closure,
)

Array = NDArray[np.float64]
Controller = Callable[[float, Array, Array], Array]
PROVENANCE_KEY = "barbell"
#: Solid name of the bar in :func:`rigid_bar_spec`.
RIGID_BAR_SOLID = "barbell_rigid_controller_model"
#: Key of the club record under which :func:`rigid_bar_spec` keeps the
#: unloaded club mass, so :func:`bushing_body_spec` restores it exactly.
UNLOADED_CLUB_MASS_KEY = "unloaded_total_mass_kg"
#: Provenance key of :func:`retrack_landmark_deviation` (#13072).
RETRACK_DEVIATION_KEY = "retrack_landmark_deviation_m"
SAMPLE_EVERY = 10
#: Step policy of every bar-loaded tracking run (OSL-16, #13104): 64 RK4
#: substeps (15.6 us) per 1 ms step and the controller held 0.25 ms.  With
#: the bar on the hands the foot contact's stiffest open-loop mode grows with
#: the ground force to 1.3e5 1/s in the squat window (the Coulomb
#: regularisation slope mu f_n / v_t on the light foot segments), so the
#: default eight 0.125 ms substeps give |lambda h| above RK4's real stability
#: limit 2.785 and the toe, subtalar and ankle run away; 32 substeps reach
#: 2.80 and leave the converged trajectory by 2e-3 rad.  64 substeps give at
#: most 2.03 and agree with 128 to 1.7e-8 rad
#: (:func:`~src.shared.python.motion_matching.same_input.stiffness.substeps_for`).
#: The 0.25 ms hold bounds the first-order error of holding the computed
#: torque while the ~35 us contact re-equilibrates (measurements in
#: ``scripts/measure_loaded_foot_tracking.py``).
LOADED_TRACKING_POLICY = StepPolicy(substeps=64, effort_holds=4)

__all__ = [
    "HoldResult",
    "HoldStart",
    "barbell_plant",
    "build_barbell_bundle",
    "bushing_body_spec",
    "contact_equilibrium_pose",
    "contact_wrench_about_com",
    "hold_start",
    "instantaneous_parity",
    "LOADED_TRACKING_POLICY",
    "loaded_effort_source",
    "quasi_static_hold",
    "retrack_landmark_deviation",
    "rigid_bar_spec",
]


def loaded_effort_source(
    plant: BarbellBushingPlant, controller: Controller
) -> Callable[[int, float, Array, Array], Array]:
    """Effort source: the rigid-bar body controller; the bar entries are zero.

    ``controller`` must be built on :func:`rigid_bar_spec` (see
    :func:`_tracking`), so its computed torque already carries the bar's
    weight and inertia, including their load on the unactuated root.
    """
    nv = plant.nv_body

    def source(_k: int, t: float, q: Array, v: Array) -> Array:
        tau = np.zeros(nv + len(BAR_COORDINATES))
        tau[:nv] = np.asarray(controller(t, q[:nv], v[:nv]), dtype=float)
        return tau

    return source


def rigid_bar_spec(spec_bytes: bytes, barbell: Barbell) -> bytes:
    """``spec_bytes`` with the bar welded rigidly to the hand (club) body.

    The bar's mass, centre of mass and inertia
    (:func:`~src.shared.python.lifting.barbell_bushing.barbell_dynamics`, in
    the club body frame) are appended as one solid of the closure's club
    body.  This is the tracking controller's model of the loaded lifter
    (#13072): its computed torque and centre-of-mass balance see the bar on
    the floating base as well as on the arms, and its foot preload carries
    body plus bar.  The bushing compliance is the only bar dynamics it omits.

    Since #13093 the back-squat source bundle carries this spec: its IK and
    its tracking run both see the bar load, and :func:`bushing_body_spec`
    recovers the body the LIFT-4 bushings attach to.

    Postcondition: the spec's total mass rises by ``barbell.load_kg``.  A
    ``club`` record's ``total_mass_kg`` rises by the same load (the bar is
    part of the club body the hands hold, so the grip partition sees club plus
    bar, OSL-16, #13104) and its unloaded value is kept under
    :data:`UNLOADED_CLUB_MASS_KEY`; every other field is unchanged;
    ``bushing_body_spec`` of the result returns a compact (``","``/``":"``)
    spec equal to ``spec_bytes`` when that was compact.

    Raises:
        ValueError: If the spec has no closure club body or already carries
            the rigid bar (the load would count twice).
    """
    if not isinstance(spec_bytes, bytes | bytearray):
        raise TypeError("spec_bytes must be the raw spec document bytes")
    if not isinstance(barbell, Barbell):
        raise TypeError("barbell must be a Barbell")
    spec = json.loads(spec_bytes)
    if _rigid_bar_body(spec) is not None:
        raise ValueError("spec already carries the rigid bar; it would count twice")
    dynamics = barbell_dynamics(barbell, barbell_interface(spec, barbell))
    club = spec["closure"]["body_b"]
    body = next((b for b in spec["bodies"] if b["name"] == club), None)
    if body is None:
        raise ValueError(f"spec has no closure club body {club!r}")
    body["solids"].append(
        {
            "name": RIGID_BAR_SOLID,
            "mass_kg": float(dynamics.mass_kg),
            "com_m": [float(x) for x in dynamics.com_m],
            "inertia_com_kg_m2": np.asarray(dynamics.inertia_com_kg_m2).tolist(),
            "placement": np.eye(4).tolist(),
        }
    )
    club_record = spec.get("club")
    if isinstance(club_record, dict) and "total_mass_kg" in club_record:
        unloaded = club_record["total_mass_kg"]
        club_record["total_mass_kg"] = float(unloaded) + float(dynamics.mass_kg)
        club_record[UNLOADED_CLUB_MASS_KEY] = unloaded
    return json.dumps(spec, separators=(",", ":")).encode()


def _rigid_bar_body(spec: Mapping[str, Any]) -> dict[str, Any] | None:
    """The body carrying the :data:`RIGID_BAR_SOLID`, if any."""
    for body in spec["bodies"]:
        if any(s["name"] == RIGID_BAR_SOLID for s in body["solids"]):
            found: dict[str, Any] = body
            return found
    return None


def bushing_body_spec(spec_bytes: bytes) -> bytes:
    """The lifter body the bushings attach to: ``spec_bytes`` without the rigid bar.

    A bar-loaded source bundle (#13093) carries :func:`rigid_bar_spec`; the
    LIFT-4 plant needs the body alone.  A spec without the rigid bar is
    returned unchanged (same bytes).
    """
    if not isinstance(spec_bytes, bytes | bytearray):
        raise TypeError("spec_bytes must be the raw spec document bytes")
    spec = json.loads(spec_bytes)
    body = _rigid_bar_body(spec)
    if body is None:
        return bytes(spec_bytes)
    body["solids"] = [s for s in body["solids"] if s["name"] != RIGID_BAR_SOLID]
    club_record = spec.get("club")
    if isinstance(club_record, dict) and UNLOADED_CLUB_MASS_KEY in club_record:
        club_record["total_mass_kg"] = club_record.pop(UNLOADED_CLUB_MASS_KEY)
    return json.dumps(spec, separators=(",", ":")).encode()


def _tracking(
    spec_bytes: bytes,
    barbell: Barbell,
    times: Array,
    track: Array,
    duration_s: float,
) -> tuple[Any, Array, Array, int, Any]:
    """Rigid-bar controller and start state; simulator of the body alone.

    The controller and the start state (feet preloaded for body plus bar)
    come from :func:`rigid_bar_spec`; the returned simulator is the
    unloaded body's, for its contact law, mass and centre of mass.
    """
    from src.engines.physics_engines.mujoco.python.full_body_model import (
        NativeMujocoFullBodyModel,
    )
    from src.shared.python.motion_matching import full_body_forward_dynamics as fs
    from src.shared.python.motion_matching.same_input.reference import (
        _tracking_setup,
    )

    controller, q0, v0, steps = _tracking_setup(
        rigid_bar_spec(spec_bytes, barbell), times, track, duration_s
    )
    simulator = fs.FullBodySimulator(NativeMujocoFullBodyModel(spec_bytes))
    return controller, q0, v0, steps, simulator


def _attached_start(
    spec_bytes: bytes, q: Array, v: Array, barbell: Barbell, engine: str = "mujoco"
) -> tuple[BarbellBushingPlant, Array, Array]:
    plant = VectorPlant(engine, spec_bytes)
    start = project_to_closure(plant, q, v)
    loaded = BarbellBushingPlant(plant, spec_bytes, barbell, start.q)
    q_aug, v_aug = loaded.attached_state(start.q, start.v)
    return loaded, q_aug, v_aug


#: Convergence of :func:`contact_equilibrium_pose`: vertical force within this
#: fraction of the weight and tangential moment within this many weight-metres.
EQUILIBRIUM_TOLERANCE = 1e-12
_EQUILIBRIUM_FD_STEP = 1e-7


def contact_wrench_about_com(simulator: Any, q: Array) -> tuple[Array, Array]:
    """Total contact force and its moment about the centre of mass, at rest at ``q``.

    ``simulator`` is a :class:`FullBodySimulator`; the force is the shared
    contact law summed over every sphere with zero rates (world frame, N and
    N m).
    """
    names = simulator.names
    samples = simulator.adapter.evaluate_contact_samples(
        dict(zip(names, map(float, q), strict=True)), dict.fromkeys(names, 0.0)
    )
    com = np.asarray(simulator.centre_of_mass(q)[0], dtype=float)
    force, moment = np.zeros(3), np.zeros(3)
    for sample in samples.values():
        f = np.asarray(sample.normal_force_n) + np.asarray(sample.friction_force_n)
        force += f
        moment += np.cross(np.asarray(sample.contact_point_m) - com, f)
    return force, moment


def _equilibrium_residual(simulator: Any, q: Array, basis: Array) -> Array:
    """``[F.n / W - 1, M.t1 / W, M.t2 / W]`` (``basis`` rows: n, t1, t2)."""
    force, moment = contact_wrench_about_com(simulator, q)
    weight = float(simulator.mass_kg) * float(np.linalg.norm(simulator.gravity))
    return np.array([force @ basis[0] / weight - 1.0, *(basis[1:] @ moment / weight)])


def _root_step(simulator: Any, q: Array, step: Array, normal: Array) -> Array:
    """``q`` moved ``step[0]`` along the ground normal and ``step[1:]`` in root angles."""
    out = np.array(q, dtype=float)
    axes = simulator.root_translation_axes(out)
    out[:3] += np.linalg.solve(axes, step[0] * normal)
    out[3:6] += step[1:]
    return out


def contact_equilibrium_pose(
    simulator: Any, q: Array, *, max_iterations: int = 30
) -> Array:
    """``q`` with its root placed in static equilibrium on the foot contact.

    Only the unactuated root moves: along the ground normal and in its three
    angles (minimum-norm Gauss-Newton from :func:`preload_feet`), until the
    contact law at rest carries the simulator's weight with zero tangential
    moment about its centre of mass, i.e. the centre of pressure lies under
    the centre of mass.  ``simulator`` is the model whose weight the feet must
    carry (for a loaded hold, the :func:`rigid_bar_spec` body plus bar).

    Postconditions: the actuated coordinates and the horizontal root
    translations equal ``q``'s; the residuals are within
    :data:`EQUILIBRIUM_TOLERANCE`.

    Raises:
        ValueError: If ``q`` is not finite with one value per coordinate, or
            the contact equilibrium is not reached in ``max_iterations``.
    """
    from src.shared.python.motion_matching import full_body_forward_dynamics as fs

    pose = np.asarray(q, dtype=float)
    if pose.shape != (simulator.nv,) or not np.isfinite(pose).all():
        raise ValueError(f"q must be {simulator.nv} finite values in spec order")
    normal = -np.asarray(simulator.gravity, dtype=float)
    normal /= np.linalg.norm(normal)
    tangent = np.cross(normal, [1.0, 0.0, 0.0])
    if np.linalg.norm(tangent) < 0.5:
        tangent = np.cross(normal, [0.0, 1.0, 0.0])
    tangent /= np.linalg.norm(tangent)
    basis = np.vstack([normal, np.cross(tangent, normal), tangent])
    current = fs.preload_feet(simulator, pose)
    for _ in range(max_iterations):
        residual = _equilibrium_residual(simulator, current, basis)
        if np.abs(residual).max() <= EQUILIBRIUM_TOLERANCE:
            current[simulator.actuated] = pose[simulator.actuated]
            return current
        jacobian = np.zeros((3, 4))
        for k in range(4):
            step = np.zeros(4)
            step[k] = _EQUILIBRIUM_FD_STEP
            jacobian[:, k] = (
                _equilibrium_residual(
                    simulator, _root_step(simulator, current, step, normal), basis
                )
                - _equilibrium_residual(
                    simulator, _root_step(simulator, current, -step, normal), basis
                )
            ) / (2.0 * _EQUILIBRIUM_FD_STEP)
        delta = np.linalg.lstsq(jacobian, -residual, rcond=1e-8)[0]
        current = _root_step(simulator, current, delta, normal)
    raise ValueError(
        f"contact equilibrium not reached in {max_iterations} iterations "
        "(are the feet in contact?)"
    )


@dataclass(frozen=True)
class HoldStart:
    """Start of the quasi-static hold: the loaded plant and its rest state."""

    plant: BarbellBushingPlant
    q: Array
    v: Array

    @property
    def q_body(self) -> Array:
        """Body coordinates of the start state (spec order)."""
        return self.q[: self.plant.nv_body]

    @property
    def track(self) -> Array:
        """Constant two-row posture reference holding :attr:`q_body`."""
        return np.vstack([self.q_body, self.q_body])


def hold_start(spec_bytes: bytes, q_hold: Array, barbell: Barbell) -> HoldStart:
    """Hold start at the contact equilibrium of ``q_hold``'s joint angles (#11744).

    ``q_hold`` is projected onto the hand closure, then its root is placed by
    :func:`contact_equilibrium_pose` on the :func:`rigid_bar_spec` body, so
    the feet carry body plus bar with the centre of pressure under the
    combined centre of mass.  The bar hangs at its static bushing deflection
    and every rate is zero.  A bar-loaded spec is reduced to its body first.
    """
    from src.engines.physics_engines.mujoco.python.full_body_model import (
        NativeMujocoFullBodyModel,
    )
    from src.shared.python.motion_matching import full_body_forward_dynamics as fs

    spec_bytes = bushing_body_spec(spec_bytes)
    plant = VectorPlant("mujoco", spec_bytes)
    pose = np.asarray(q_hold, dtype=float)
    closed = project_to_closure(plant, pose, np.zeros_like(pose)).q
    rigid = fs.FullBodySimulator(
        NativeMujocoFullBodyModel(rigid_bar_spec(spec_bytes, barbell))
    )
    q_eq = contact_equilibrium_pose(rigid, closed)
    loaded = BarbellBushingPlant(plant, spec_bytes, barbell, q_eq)
    q_aug, v_aug = loaded.attached_state(q_eq, np.zeros_like(q_eq))
    return HoldStart(loaded, q_aug, v_aug)


@dataclass(frozen=True)
class HoldResult:
    """Sampled quasi-static hold (every :data:`SAMPLE_EVERY` steps)."""

    time_s: Array
    grf_n: Array
    hand_force_n: dict[str, Array]
    deflection_m: Array
    body_weight_n: float
    bar_weight_n: float
    rollout: Rollout

    def summary(self, settle_window_s: float) -> dict[str, Any]:
        """Means over the last ``settle_window_s`` and whole-run peaks."""
        if not (np.isfinite(settle_window_s) and settle_window_s > 0.0):
            raise ValueError("settle_window_s must be positive")
        late = self.time_s >= self.time_s[-1] - settle_window_s + 1e-12
        total = self.body_weight_n + self.bar_weight_n
        hands = {s: self.hand_force_n[s][late].mean(axis=0) for s in SIDES}
        hand_sum = hands["L"] + hands["R"]
        grf = self.grf_n[late].mean(axis=0)
        return {
            "duration_s": float(self.time_s[-1]),
            "settle_window_s": float(settle_window_s),
            "body_weight_n": self.body_weight_n,
            "bar_weight_n": self.bar_weight_n,
            "mean_grf_n": grf.tolist(),
            "grf_vertical_over_total_weight": float(grf[2] / total),
            "mean_hand_force_on_bar_n": {s: hands[s].tolist() for s in SIDES},
            "hand_vertical_sum_over_bar_weight": float(hand_sum[2] / self.bar_weight_n),
            "hand_vertical_split_left": float(hands["L"][2] / hand_sum[2]),
            "peak_deflection_m": float(self.deflection_m.max()),
            "failure": self.rollout.failure,
        }


def _contact_force(adapter: Any, names: Sequence[str], q: Array, v: Array) -> Array:
    samples = adapter.evaluate_contact_samples(
        dict(zip(names, map(float, q), strict=True)),
        dict(zip(names, map(float, v), strict=True)),
    )
    return np.sum([s.normal_force_n + s.friction_force_n for s in samples.values()], 0)


def quasi_static_hold(
    spec_bytes: bytes, q_hold: Array, barbell: Barbell, *, duration_s: float
) -> HoldResult:
    """Hold ``q_hold`` in MuJoCo with the bar on both hands for ``duration_s``.

    The run starts at rest at :func:`hold_start`: ``q_hold``'s joint angles
    with the root in contact equilibrium for body plus bar, and the bar
    hanging at its static bushing deflection.  The controller holds that
    posture and its centre of mass.  Until #11744 the start was the
    ``preload_feet`` pose, whose feet carried 0.595 of the load with the centre
    of pressure 6.7 cm ahead of the centre of mass; the fall onto the heels
    rang a lightly damped fore-aft sway that sagged the last 0.5 s.  A
    bar-loaded spec is reduced to its body first (:func:`bushing_body_spec`).
    The run uses :data:`LOADED_TRACKING_POLICY` and stops on failure.
    Postcondition: samples every :data:`SAMPLE_EVERY` steps.
    """
    spec_bytes = bushing_body_spec(spec_bytes)
    start = hold_start(spec_bytes, q_hold, barbell)
    times = np.array([0.0, float(duration_s)])
    controller, _, _, steps, simulator = _tracking(
        spec_bytes, barbell, times, start.track, duration_s
    )
    loaded = start.plant
    rollout = integrate(
        loaded,
        start.q,
        start.v,
        loaded_effort_source(loaded, controller),
        steps=steps,
        dt_s=float(duration_s) / steps,
        policy=replace(LOADED_TRACKING_POLICY, stop_on_failure=True),
    )
    rows = range(0, rollout.q.shape[0], SAMPLE_EVERY)
    nv, names = loaded.nv_body, loaded.body_order
    grf: list[Array] = []
    forces: dict[str, list[Array]] = {s: [] for s in SIDES}
    deflection: list[float] = []
    for k in rows:
        q, v = rollout.q[k], rollout.v[k]
        grf.append(_contact_force(simulator.adapter, names, q[:nv], v[:nv]))
        sample = loaded.sample(q, v)
        for side in SIDES:
            forces[side].append(sample.wrench[side].force_n)
        deflection.append(sample.max_deflection_m())
    gravity = float(np.linalg.norm(simulator.gravity))
    return HoldResult(
        time_s=rollout.time_s[list(rows)],
        grf_n=np.array(grf),
        hand_force_n={s: np.array(forces[s]) for s in SIDES},
        deflection_m=np.array(deflection),
        body_weight_n=float(simulator.mass_kg) * gravity,
        bar_weight_n=barbell.load_kg * gravity,
        rollout=rollout,
    )


def hold_balance_geometry(
    spec_bytes: bytes, q_hold: Array, barbell: Barbell
) -> dict[str, Any]:
    """Centres of mass, centre of pressure and foot points at the hold start.

    Returns, at :func:`hold_start`, the body and the body-plus-bar centres of
    mass, the contact law's centre of pressure at rest (on the ground plane),
    the forefoot and heel contact-sphere centres (by sphere name) and the
    horizontal unit ``forward_axis`` from the mean heel to the mean forefoot,
    all in metres in the world frame, so a caller can see on which side of the
    feet each centre of mass falls.  A bar-loaded spec is reduced to its body
    first.
    """
    from src.engines.physics_engines.mujoco.python.full_body_model import (
        NativeMujocoFullBodyModel,
    )
    from src.shared.python.motion_matching import full_body_forward_dynamics as fs

    spec_bytes = bushing_body_spec(spec_bytes)
    start = hold_start(spec_bytes, q_hold, barbell)
    loaded, q_aug = start.plant, start.q
    q0, v0 = start.q_body, np.zeros_like(start.q_body)
    simulator = fs.FullBodySimulator(NativeMujocoFullBodyModel(spec_bytes))
    adapter = simulator.adapter
    body_com = np.asarray(simulator.centre_of_mass(q0)[0], dtype=float)
    bar_com = np.asarray(q_aug[-6:-3], dtype=float)
    mass = float(simulator.mass_kg)
    samples = adapter.evaluate_contact_samples(  # leaves the sites placed at q0
        dict(zip(loaded.body_order, map(float, q0), strict=True)),
        dict(zip(loaded.body_order, map(float, v0), strict=True)),
    )
    weights = np.array(
        [float(np.linalg.norm(c.normal_force_n)) for c in samples.values()]
    )
    points = np.array([np.asarray(c.contact_point_m) for c in samples.values()])
    spheres = adapter._spheres  # noqa: SLF001 - contact-sphere site ids
    sites = {k: adapter.data.site_xpos[i["site_id"]].copy() for k, i in spheres.items()}
    forefoot = {k: p for k, p in sites.items() if k.startswith("forefoot")}
    heel = {k: p for k, p in sites.items() if k.startswith("heel")}
    normal = np.asarray(adapter.ground_plane.normal, dtype=float)
    normal /= np.linalg.norm(normal)
    axis = np.mean(list(forefoot.values()), 0) - np.mean(list(heel.values()), 0)
    axis -= (axis @ normal) * normal
    return {
        "body_com_m": body_com,
        "combined_com_m": (mass * body_com + barbell.load_kg * bar_com)
        / (mass + barbell.load_kg),
        "centre_of_pressure_m": weights @ points / weights.sum(),
        "forefoot_m": forefoot,
        "heel_m": heel,
        "forward_axis": axis / np.linalg.norm(axis),
    }


def _landmark_kinematics(source: InputBundle) -> tuple[Any, tuple[str, ...]]:
    """MuJoCo marker kinematics of the LIFT-5 landmarks of ``source``'s spec.

    The bar landmark is the hand-held bar-centre frame (a body frame), so it
    measures where the hands carry the bar, not the bushing-borne bar body.
    """
    from src.engines.physics_engines.mujoco.python.full_body_ik import (
        FullBodyMarkerKinematics,
    )
    from src.engines.physics_engines.mujoco.python.full_body_model import (
        NativeMujocoFullBodyModel,
    )
    from src.shared.python.lifting.back_bar import BAR_CENTRE_FRAME
    from src.shared.python.lifting.bundles import LANDMARK_ATTACHMENTS

    attachments = dict(LANDMARK_ATTACHMENTS)
    if "back_bar" in source.provenance.get("ik", {}):
        attachments["bar"] = (BAR_CENTRE_FRAME, (0.0, 0.0, 0.0))
    model = NativeMujocoFullBodyModel(bytes(source.spec_bytes))
    return FullBodyMarkerKinematics(model, attachments), tuple(attachments)


def retrack_landmark_deviation(
    source: InputBundle, reference_q: Array
) -> dict[str, float]:
    """Largest landmark distance (m) of a re-tracked reference from ``source``'s.

    Each LIFT-5 landmark (:data:`~src.shared.python.lifting.bundles.\
LANDMARK_ATTACHMENTS`) is placed by forward kinematics of ``reference_q`` and
    of ``source.reference_q`` row by row; the maximum distance over all rows is
    returned per landmark.  This is the quantity bounded by the LIFT-5 IK
    residual thresholds (#13072).

    Raises:
        ValueError: if ``reference_q`` does not match the source's shape or is
            not finite.
    """
    track = np.asarray(reference_q, dtype=float)
    expected = np.asarray(source.reference_q, dtype=float)
    if track.shape != expected.shape or not np.isfinite(track).all():
        raise ValueError(
            f"reference_q must be finite with the source shape {expected.shape}"
        )
    kin, labels = _landmark_kinematics(source)
    peaks = np.zeros(len(labels))
    for row, ref in zip(track, expected, strict=True):
        here = np.array(kin.marker_positions(row), dtype=float)
        gap = here - np.array(kin.marker_positions(ref), dtype=float)
        peaks = np.maximum(peaks, np.linalg.norm(gap, axis=1))
    return {label: float(p) for label, p in zip(labels, peaks, strict=True)}


def build_barbell_bundle(source: InputBundle, barbell: Barbell) -> InputBundle:
    """Re-track ``source``'s reference in MuJoCo with the bar on both hands.

    The returned bundle keeps the body coordinates (so it validates as an
    ordinary same-input bundle); the bar's reference states, the bar and
    bushing parameters and the attachment pose are under
    ``provenance["barbell"]``.  Replay it with :func:`barbell_plant`.

    A bar-loaded source (#13093) is re-tracked on its body
    (:func:`bushing_body_spec`): the bushing bar replaces the rigid one, and
    the returned bundle carries the body spec.  The re-track integrates
    with ``source``'s step policy (substeps and effort holds) and records it.
    """
    body = bushing_body_spec(source.spec_bytes)
    times = np.arange(source.reference_q.shape[0], dtype=float) * source.dt_s
    duration = source.steps * source.dt_s
    controller, q0, v0, steps, _ = _tracking(
        body,
        barbell,
        times,
        np.asarray(source.reference_q, dtype=float),
        duration,
    )
    loaded, q_aug, v_aug = _attached_start(body, q0, v0, barbell)
    rollout = integrate(
        loaded,
        q_aug,
        v_aug,
        loaded_effort_source(loaded, controller),
        steps=steps,
        dt_s=source.dt_s,
        policy=source.step_policy(),
    )
    nv, holds = loaded.nv_body, source.effort_holds
    record = {
        **loaded.record(),
        "attach_q": q_aug[:nv].tolist(),
        "reference_bar_q": rollout.q[:, nv:].tolist(),
        "reference_bar_v": rollout.v[:, nv:].tolist(),
        "controller": "pipeline computed-torque (kkt) on the rigid-bar spec "
        "(rigid_bar_spec, #13072)",
    }
    provenance = {
        **source.provenance,
        "max_pose_drift_m": float(rollout.pose_drift.max()),
        RETRACK_DEVIATION_KEY: retrack_landmark_deviation(source, rollout.q[:, :nv]),
        PROVENANCE_KEY: record,
        "source_bundle_spec_sha256": source.spec_sha256,
    }
    return InputBundle(
        spec_bytes=body,
        coordinate_order=source.coordinate_order,
        dt_s=source.dt_s,
        q0=rollout.q[0, :nv],
        v0=rollout.v[0, :nv],
        efforts=rollout.efforts[::holds, :nv],
        reference_q=rollout.q[:, :nv],
        reference_v=rollout.v[:, :nv],
        reference_engine="mujoco",
        provenance=provenance,
        effort_holds=holds,
        hold_efforts=None if holds == 1 else rollout.efforts[:, :nv],
        substeps=source.substeps,
    )


def barbell_plant(engine: str, bundle: InputBundle) -> BarbellBushingPlant:
    """The coupled plant of a barbell bundle in ``engine``.

    Raises:
        ValueError: if the bundle carries no barbell record.
    """
    record = bundle.provenance.get(PROVENANCE_KEY)
    if not isinstance(record, Mapping):
        raise ValueError("bundle has no barbell record; build it with the bar")
    barbell = Barbell(**record["barbell"])
    plant = VectorPlant(engine, bundle.spec_bytes)
    return BarbellBushingPlant(
        plant, bundle.spec_bytes, barbell, np.asarray(record["attach_q"], float)
    )


def augmented_reference(bundle: InputBundle) -> tuple[Array, Array]:
    """Reference states ``(q, v)`` with the bar coordinates appended."""
    record = bundle.provenance[PROVENANCE_KEY]
    return (
        np.hstack([bundle.reference_q, np.asarray(record["reference_bar_q"])]),
        np.hstack([bundle.reference_v, np.asarray(record["reference_bar_v"])]),
    )


def _engine_values(
    plant: BarbellBushingPlant, q: Array, v: Array, tau: Array
) -> dict[str, Array]:
    sample = plant.sample(q, v)
    out = {f"hand_force_{s}": sample.wrench[s].force_n for s in SIDES}
    out |= {f"hand_moment_{s}": sample.wrench[s].moment_nm for s in SIDES}
    acceleration = plant.acceleration(q, v, tau)
    out["body_acceleration"] = acceleration[: plant.nv_body]
    out["bar_acceleration"] = acceleration[plant.nv_body :]
    return out


def instantaneous_parity(
    bundle: InputBundle, engines: Sequence[str], indices: Sequence[int]
) -> dict[str, Any]:
    """Largest difference from MuJoCo per quantity, at reference states.

    State ``k`` uses the reference state and the effort applied from it (the
    last state reuses the last effort).  An engine that fails to build is
    reported unavailable with the reason, never as zero difference.
    """
    q_ref, v_ref = augmented_reference(bundle)
    efforts = np.hstack([bundle.efforts, np.zeros((bundle.steps, 6))])

    def values(plant: BarbellBushingPlant) -> list[dict[str, Array]]:
        return [
            _engine_values(plant, q_ref[k], v_ref[k], efforts[min(k, bundle.steps - 1)])
            for k in indices
        ]

    reference = values(barbell_plant("mujoco", bundle))
    report: dict[str, Any] = {
        "indices": list(indices),
        "mujoco_scale": {
            key: float(max(np.abs(row[key]).max() for row in reference))
            for key in reference[0]
        },
        "engines": {},
    }
    for engine in engines:
        try:
            rows = values(barbell_plant(engine, bundle))
        except (ImportError, RuntimeError, ValueError, AttributeError) as exc:
            report["engines"][engine] = {"unavailable": f"{type(exc).__name__}: {exc}"}
            continue
        report["engines"][engine] = {
            key: float(
                max(
                    np.abs(row[key] - ref[key]).max()
                    for row, ref in zip(rows, reference, strict=True)
                )
            )
            for key in reference[0]
        }
    return report

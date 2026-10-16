"""Per-level joint reactions with muscle forces (SPN-10c, #12762).

Observable.  The per-level output is :attr:`LoadObservable.JOINT_REACTION`
(ADR-0055 item 5): the force and moment exerted ON the superior vertebra BY
the inferior vertebra through the intervertebral joint, expressed in the
INFERIOR vertebra ISB frame (X anterior, Y cranial, Z right) about the
mid-disc joint centre.  Compression is ``+F_y``.  The joint centre is the
origin of the joint's child frame, which for the segmental models is also the
origin of the parent offset frame (the translations of a level are constant
zero), so no reference-point transport is needed (the offset passed to
:func:`spine.loads.from_wrench_in_frame` is zero).  Couplers are disabled
(ADR-0055 item 6).

Reaction definition.  Muscles are external forces on the bodies and are NOT
part of the reaction.  The reaction is what the joint transmits: the
translational constraint force plus the torque of the reserve (residual)
actuation that closes the dynamics at the joint, because a reserve is a
generalized force acting at the joint and not a muscle.  Where the muscles
fully explain the inverse-dynamics effort the reserve is zero and the
reaction moment vanishes (the joint rotations are free).

Method.  For every frame selected by the static optimisation (SO):

1. Coordinates ``q``, speeds ``u`` and accelerations ``udot`` come from the
   same motion, derivatives and mobility layout as the inverse dynamics
   (:func:`inverse_dynamics._load_kinematics`); the optional low-pass cutoff
   must equal the one used for the SO input, and the sample times must match
   the SO frame times.
2. Each muscle is set to the SO tendon force ``F_m`` with
   ``overrideActuation`` (rigid tendon, so no fibre equilibrium is needed).
3. ``InverseDynamicsSolver`` then returns the residual generalized force
   ``r`` on EVERY coordinate that, together with gravity, external loads and
   the muscles, produces the measured ``udot``.  On the level coordinates this
   is the reserve; on the six ground-pelvis coordinates it is the pelvis
   residual wrench.  The level residual must equal the SO reserve within
   ``reserve_tolerance_nm`` (a mismatch means the SO result does not belong to
   this model, motion or filter and is refused).
4. ``r`` is applied through one ``CoordinateActuator`` per coordinate
   (override actuation, optimal force 1), so forward dynamics at the state
   gives accelerations equal to the measured ones.  This is the semantics of
   OpenSim's ``JointReaction`` analysis driven by CMC/SO actuators plus
   residuals; the consistency ``max |udot - udot_measured|`` is verified and
   recorded.
   A massless body (the coupler driver body of the fixtures) makes the mass
   matrix singular and the accelerations NaN, so a body with zero mass is
   given ``1e-9 kg`` and an isotropic inertia of ``1e-12 kg m^2`` in the run
   copy (weight below ``1e-8 N``, negligible against the ``1e-6 N`` accuracy;
   recorded in the provenance).
5. ``Joint.calcReactionOnChildExpressedInGround`` (the call behind
   ``JointReaction`` with ``apply_on_bodies=child``) gives the wrench on the
   child at the child frame origin in ground.  It is rotated into the inferior
   ISB frame with ``R_isb = R_i0 R_gP^T`` (``R_gP`` the ground orientation of
   the parent offset frame, ``R_i0`` the inventory neutral rotation, the same
   convention as the inverse-dynamics moments).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from src.engines.physics_engines.opensim.python.spine._osim_helpers import (
    transform_to_matrix,
)
from src.engines.physics_engines.opensim.python.spine.drive import PelvisCoordinates
from src.engines.physics_engines.opensim.python.spine.inventory import (
    ModelLevelMap,
    SpineInventory,
)
from src.engines.physics_engines.opensim.python.spine.inverse_dynamics import (
    PreparedModel,
    SpineIDOptions,
    _Kinematics,
    _frame_speeds,
    _load_kinematics,
    _sha256,
    _solve_frame,
    prepare_model_for_id,
)
from src.engines.physics_engines.opensim.python.spine.static_optimization import (
    SpineStaticOptimizationResult,
)
from src.shared.python.contracts import ensure
from src.shared.python.engine_core.opensim_runtime import import_opensim
from src.shared.python.spine.loads import (
    LoadObservable,
    SpinalLoadSeries,
    from_wrench_in_frame,
)

__all__ = ["JointReactionOptions", "compute_joint_reactions"]

_TIME_MATCH_TOL_S = 1e-9
_RESERVE_PREFIX = "jr_residual_"
_REGULARISED_MASS_KG = 1e-9
_REGULARISED_INERTIA = 1e-12


@dataclass(frozen=True)
class JointReactionOptions:
    """Optional settings of :func:`compute_joint_reactions`.

    ``pelvis`` / ``level_map``: as for :class:`SpineIDOptions`;
    ``external_loads``: ExternalLoads XML (must be the one the SO input used);
    ``filter_hz``: low-pass cutoff [Hz] (must equal the SO input's);
    ``reserve_tolerance_nm``: largest accepted ``|measured level residual - SO
    reserve|`` [N*m]; ``acceleration_tolerance``: largest accepted
    ``|udot - udot_measured|`` [rad/s^2 or m/s^2].
    """

    pelvis: PelvisCoordinates | None = None
    level_map: ModelLevelMap | None = None
    external_loads: str | Path | None = None
    filter_hz: float | None = None
    reserve_tolerance_nm: float = 1e-6
    acceleration_tolerance: float = 1e-6

    def __post_init__(self) -> None:
        if self.pelvis is not None and not isinstance(self.pelvis, PelvisCoordinates):
            raise TypeError("pelvis must be a PelvisCoordinates or None")
        for name in ("reserve_tolerance_nm", "acceleration_tolerance"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive, got {value!r}")
        hz = self.filter_hz
        if hz is not None and not (math.isfinite(hz) and hz > 0):
            raise ValueError(f"filter_hz must be positive and finite, got {hz!r}")


# ------------------------------------------------------------ SWIG adapters


def _muscle_objects(model: object) -> list:
    muscles = model.getMuscles()  # type: ignore[attr-defined]
    return [muscles.get(i) for i in range(muscles.getSize())]


def _add_residual_actuators(model: object, names: list[str]) -> None:
    """Add one unit-optimal-force ``CoordinateActuator`` per coordinate."""
    osim = import_opensim()
    for name in names:
        actuator = osim.CoordinateActuator(name)
        actuator.setName(_RESERVE_PREFIX + name)
        actuator.setOptimalForce(1.0)
        model.addForce(actuator)  # type: ignore[attr-defined]


def _residual_actuators(model: object, names: list[str]) -> list:
    osim = import_opensim()
    forces = model.getForceSet()  # type: ignore[attr-defined]
    return [
        osim.CoordinateActuator.safeDownCast(forces.get(_RESERVE_PREFIX + name))
        for name in names
    ]


def _override(actuator: object, state: object, value: float) -> None:
    actuator.overrideActuation(state, True)  # type: ignore[attr-defined]
    actuator.setOverrideActuation(state, float(value))  # type: ignore[attr-defined]


def _regularise_massless_bodies(model: object) -> tuple[str, ...]:
    """Give zero-mass bodies a negligible mass and inertia; return their names."""
    osim = import_opensim()
    bodies = model.getBodySet()  # type: ignore[attr-defined]
    changed = []
    for i in range(bodies.getSize()):
        body = bodies.get(i)
        if body.getMass() == 0.0:
            body.setMass(_REGULARISED_MASS_KG)
            body.setInertia(
                osim.Inertia(
                    _REGULARISED_INERTIA, _REGULARISED_INERTIA, _REGULARISED_INERTIA
                )
            )
            changed.append(str(body.getName()))
    return tuple(changed)


def _joint(model: object, name: str) -> object:
    return model.getJointSet().get(name)  # type: ignore[attr-defined]


def _child_reaction(joint: object, state: object) -> tuple[np.ndarray, np.ndarray]:
    """``(force, moment)`` on the child at the child frame origin, in ground."""
    wrench = joint.calcReactionOnChildExpressedInGround(state)  # type: ignore[attr-defined]
    moment, force = wrench.get(0), wrench.get(1)
    return (
        np.array([force.get(i) for i in range(3)]),
        np.array([moment.get(i) for i in range(3)]),
    )


def _parent_frame_rotation(joint: object, state: object) -> np.ndarray:
    frame = joint.getParentFrame()  # type: ignore[attr-defined]
    return transform_to_matrix(frame.getTransformInGround(state))[:3, :3]


def _acceleration_error(state: object, expected: np.ndarray) -> float:
    udot = state.getUDot()  # type: ignore[attr-defined]
    return float(max(abs(udot.get(i) - expected[i]) for i in range(expected.size)))


# ------------------------------------------------------------------ driver


def _check_inputs(so: object, options: object) -> JointReactionOptions:
    if not isinstance(so, SpineStaticOptimizationResult):
        raise TypeError(
            "so_result must be a SpineStaticOptimizationResult, got "
            f"{type(so).__name__}"
        )
    if options is None:
        return JointReactionOptions()
    if not isinstance(options, JointReactionOptions):
        raise TypeError("options must be a JointReactionOptions or None")
    return options


def _prepare(
    model_path: str | Path,
    so: SpineStaticOptimizationResult,
    opts: JointReactionOptions,
) -> tuple[PreparedModel, tuple[str, ...]]:
    """Load the model: couplers off, rigid tendons, a residual actuator per DOF."""
    prepared = prepare_model_for_id(
        model_path, disable_couplers=True, external_loads=opts.external_loads
    )
    loads = so.provenance.get("external_loads", "n/a")
    if prepared.external_loads != loads:
        raise ValueError(
            "external loads differ from the static-optimisation input: "
            f"{prepared.external_loads!r} vs {loads!r}"
        )
    muscles = _muscle_objects(prepared.model)
    if tuple(str(m.getName()) for m in muscles) != tuple(so.muscle_names):
        raise ValueError(
            "muscle names of the model differ from the static-optimisation "
            f"result: {[str(m.getName()) for m in muscles]} vs {list(so.muscle_names)}"
        )
    for muscle in muscles:
        muscle.set_ignore_tendon_compliance(True)
    names = [
        str(prepared.model.getCoordinateSet().get(i).getName())  # type: ignore[attr-defined]
        for i in range(prepared.model.getCoordinateSet().getSize())  # type: ignore[attr-defined]
    ]
    _add_residual_actuators(prepared.model, names)
    regularised = _regularise_massless_bodies(prepared.model)
    prepared.state = prepared.model.initSystem()  # type: ignore[attr-defined]
    return prepared, regularised


def _check_frames(kin: _Kinematics, so: SpineStaticOptimizationResult) -> None:
    index = so.frame_index
    if index.size and (index.min() < 0 or index.max() >= kin.times.size):
        raise ValueError("SO frame indices fall outside the motion samples")
    if not np.allclose(kin.times[index], so.time_s, rtol=0.0, atol=_TIME_MATCH_TOL_S):
        raise ValueError("SO frame times do not match the motion sample times")
    names = kin.layout.names
    cols = [names.index(c) for c in so.coordinate_names]
    expected = [
        c for lv in kin.inventory.levels for c in kin.inventory.coordinate_names[lv]
    ]
    if [kin.layout.names[i] for i in cols] != expected:
        raise ValueError("SO level coordinates differ from the model inventory")


@dataclass(frozen=True)
class _Frame:
    """One solved frame: ground wrenches, rotations, residual and accuracy."""

    force: np.ndarray
    moment: np.ndarray
    rotation: np.ndarray
    level_residual: np.ndarray
    pelvis_residual: float
    acceleration_error: float


def _solve_one(
    prepared: PreparedModel,
    solver: object,
    kin: _Kinematics,
    so: SpineStaticOptimizationResult,
    row: int,
) -> _Frame:
    model, state, layout = prepared.model, prepared.state, kin.layout
    frame = int(so.frame_index[row])
    for actuator in _residual_actuators(model, layout.names):
        _override(actuator, state, 0.0)
    for muscle, force in zip(_muscle_objects(model), so.force_n[row], strict=True):
        _override(muscle, state, float(force))
    sample = (kin.q[frame], kin.qd[frame], kin.qdd[frame])
    tau = _solve_frame(prepared, solver, layout, float(kin.times[frame]), sample)
    for actuator, name in zip(
        _residual_actuators(model, layout.names), layout.names, strict=True
    ):
        _override(actuator, state, tau[layout.u_index[name]])
    model.realizeAcceleration(state)  # type: ignore[attr-defined]
    expected = np.zeros(len(layout.names))
    _u, udot = _frame_speeds(layout, kin.q[frame], kin.qd[frame], kin.qdd[frame])
    for j, name in enumerate(layout.names):
        expected[layout.u_index[name]] = udot[j]
    inventory = kin.inventory
    n_lv = len(inventory.levels)
    force, moment, rot = (
        np.zeros((n_lv, 3)),
        np.zeros((n_lv, 3)),
        np.zeros((n_lv, 3, 3)),
    )
    for k, level in enumerate(inventory.levels):
        joint = _joint(model, inventory.joint_names[level])
        force[k], moment[k] = _child_reaction(joint, state)
        rot[k] = (
            inventory.neutral[level][:3, :3] @ _parent_frame_rotation(joint, state).T
        )
    coords = [c for lv in inventory.levels for c in inventory.coordinate_names[lv]]
    level_res = np.array([tau[layout.u_index[c]] for c in coords])
    pelvis = [*layout.pelvis.rotation, *layout.pelvis.translation]
    pelvis_res = max(abs(tau[layout.u_index[c]]) for c in pelvis)
    return _Frame(
        force,
        moment,
        rot,
        level_res,
        float(pelvis_res),
        _acceleration_error(state, expected),
    )


def _provenance(
    model_path: Path,
    model_id: str,
    so: SpineStaticOptimizationResult,
    prepared: PreparedModel,
    frames: list[_Frame],
    checks: tuple[float, tuple[str, ...]],
    opts: JointReactionOptions,
) -> dict[str, str]:
    reserve_error, regularised = checks
    couplers = (
        f"disabled ({len(prepared.disabled_constraints)} of "
        f"{prepared.n_constraints} constraints)"
        if prepared.disabled_constraints
        else f"none present ({prepared.n_constraints} constraints)"
    )
    return {
        "model_id": model_id,
        "model_sha256": _sha256(model_path),
        "solver": (
            "OpenSim JointReaction semantics: Joint.calcReactionOnChild"
            "ExpressedInGround after forward dynamics with SO muscle forces "
            "(override actuation) and residual coordinate actuators"
        ),
        "cost_exponent": str(so.cost_exponent),
        "distribution_source": so.provenance["distribution_source"],
        "external_loads": so.provenance["external_loads"],
        "couplers": couplers,
        "muscles": (
            f"{len(so.muscle_names)} muscles prescribed to the SO tendon force "
            "(rigid tendon); muscles are external forces, not in the reaction"
        ),
        "reserves": (
            "reserve (residual) torque is part of the reaction; max |level "
            f"residual - SO reserve| = {reserve_error:.3e} N*m"
        ),
        "acceleration_consistency": (
            f"max |udot - measured| = {max(f.acceleration_error for f in frames):.3e}"
        ),
        "pelvis_residual": (
            "max |ground-pelvis residual generalized force| = "
            f"{max(f.pelvis_residual for f in frames):.3e}"
        ),
        "massless_bodies": (
            ", ".join(regularised) + " (given 1e-9 kg for forward dynamics)"
            if regularised
            else "none"
        ),
        "filter_hz": "none" if opts.filter_hz is None else str(float(opts.filter_hz)),
        "units": "N, N*m",
        "frame": "inferior vertebra ISB frame, about the mid-disc joint centre",
        "sign": "on the superior body by the inferior body; compression = +F_y",
    }


def compute_joint_reactions(
    model_path: str | Path,
    motion: object,
    so_result: SpineStaticOptimizationResult,
    model_id: str,
    *,
    options: JointReactionOptions | None = None,
) -> SpinalLoadSeries:
    """Per-level joint reactions with the SO muscle forces applied.

    Args:
        model_path: muscle-bearing ``.osim`` used for the static optimisation.
        motion: a ``DriveResult`` or ``.mot`` path (as for the inverse
            dynamics) that produced ``so_result``.
        so_result: :func:`run_spine_static_optimization` output; its frames are
            the output samples.
        model_id: level-map key (or ``options.level_map``).
        options: :class:`JointReactionOptions`.

    Returns:
        A :attr:`LoadObservable.JOINT_REACTION` series (N, N*m) on the SO
        frames, in the inferior vertebra ISB frame about the joint centre.
        Postconditions: provenance carries every required key, the
        accelerations equal the measured ones within the tolerance and the
        level residual equals the SO reserve within the tolerance.

    Raises:
        TypeError: ``so_result`` or ``options`` of the wrong type.
        ValueError: muscle, coordinate, external-load or time mismatch with
            the SO result, a reserve or acceleration inconsistency, a missing
            file, or an inverse-dynamics failure.
    """
    opts = _check_inputs(so_result, options)
    so = so_result
    path = Path(model_path)
    prepared, regularised = _prepare(path, so, opts)
    id_opts = SpineIDOptions(pelvis=opts.pelvis, level_map=opts.level_map)
    kin = _load_kinematics(prepared, motion, model_id, id_opts, opts.filter_hz)  # type: ignore[arg-type]
    _check_frames(kin, so)
    solver = import_opensim().InverseDynamicsSolver(prepared.model)
    try:
        frames = [
            _solve_one(prepared, solver, kin, so, row)
            for row in range(so.frame_index.size)
        ]
    except RuntimeError as exc:
        raise ValueError(f"joint reaction failed: {exc}") from exc
    reserve_error = max(
        float(np.max(np.abs(f.level_residual - so.reserve_nm[row])))
        for row, f in enumerate(frames)
    )
    if not reserve_error <= opts.reserve_tolerance_nm:
        raise ValueError(
            f"measured level reserve differs from the SO reserve by "
            f"{reserve_error:.3e} N*m (> {opts.reserve_tolerance_nm:g}); the "
            "SO result does not belong to this model, motion or filter"
        )
    worst = max(f.acceleration_error for f in frames)
    if not worst <= opts.acceleration_tolerance:
        raise ValueError(
            f"forward-dynamics accelerations differ from the measured ones by "
            f"{worst:.3e} (> {opts.acceleration_tolerance:g})"
        )
    inventory: SpineInventory = kin.inventory
    n_lv = len(inventory.levels)
    series = from_wrench_in_frame(
        time_s=so.time_s,
        levels=inventory.levels,
        observable=LoadObservable.JOINT_REACTION,
        force=np.stack([f.force for f in frames]),
        moment=np.stack([f.moment for f in frames]),
        rotation_frame_to_isb=np.stack([f.rotation for f in frames]),
        point_offset_m=np.zeros((n_lv, 3)),
        provenance=_provenance(
            path, model_id, so, prepared, frames, (reserve_error, regularised), opts
        ),
    )
    ensure(
        series.observable is LoadObservable.JOINT_REACTION,
        "observable must be JOINT_REACTION",
    )
    return series

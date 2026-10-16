"""Spinal inverse dynamics with external loads (SPN-10a, #12760).

Observable.  The per-level output is :attr:`LoadObservable.GENERALIZED_EFFORT`
(ADR-0055 item 5): the moment the inferior body exerts on the superior body
through the three rotational coordinates of the level, expressed in the
inferior vertebra ISB frame about the mid-disc joint centre, in N*m.  It is
*not* a joint reaction: the translational force of the joint is not an
unknown of the ID problem for a joint with constant (zero) translations, so
``force_n`` is exactly zero.  Couplers prescribe kinematics only and are
disabled for ID (ADR-0055 item 6), which frees every spinal coordinate.

Method.  For each frame the coordinates are set, ``InverseDynamicsSolver``
returns the generalized forces ``tau`` that produce the prescribed
accelerations, and ``tau`` is mapped to moments:

* Level ``i`` (ZXY ISB angles ``theta = s * q``, ``s`` the inventory axis
  sign).  With ``J(theta) = [e_z, R_z e_x, R_z R_x e_y]`` the angular-velocity
  columns in the joint's parent offset frame ``P``, power balance
  ``tau_theta = J^T M_P`` gives ``M_P = J^{-T} (s * tau)``, and the inferior
  ISB frame is ``R_i0 M_P`` with ``R_i0`` the inventory neutral offset
  rotation.  ``J`` is singular at ``|lateral bending| = 90 deg``.
* Pelvis.  The six generalized forces of the ground-pelvis joint form the
  pelvis residual wrench in the ground frame about the pelvis origin.  For a
  free joint (quaternion state, ``nq != nu``) OpenSim speeds are the
  ground-frame angular velocity and the origin velocity, so ``tau`` is the
  wrench; the convention is verified by a runtime probe.  For a custom joint
  with Euler coordinates ``M = J^{-T} tau`` with ``J`` of the pelvis sequence.

Derivatives.  Coordinates are optionally low-passed (zero-phase Butterworth,
:func:`smooth_kinematic_trajectory`) and then differentiated with a
not-a-knot ``scipy`` ``CubicSpline`` evaluated at the sample times.  Pelvis
Euler rates are converted to ground-frame angular velocity and acceleration
analytically (:func:`euler_rate_columns`).  Spline end effects make the first
and last samples less reliable than the interior.

Residual budget.  :class:`PelvisResidualBudget` reports peak and RMS of the
pelvis residual force and moment and their ratios to body weight and to body
weight times body height, judged against a declared :class:`ResidualBudget`.
With no external loads the residual is the whole weight and the budget fails
by construction; that is the correct statement, not a defect.  The body
height of the moment ratio is the extent of the body origins along the model's
gravity direction at the FIRST frame of the motion (``body_height_source =
model_extent_first_frame``) unless the caller gives ``body_height_m``; it does
not depend on the pose the run ends in or on the model's up axis.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
import tempfile
from types import MappingProxyType
from typing import Any

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.spatial.transform import Rotation

from src.engines.physics_engines.opensim.python.spine.drive import (
    FIXTURE_PELVIS,
    GOLF_PELVIS,
    DriveResult,
    PelvisCoordinates,
)
from src.engines.physics_engines.opensim.python.spine.muscle_model import (
    millard_equivalent_model,
)
from src.engines.physics_engines.opensim.python.spine.inventory import (
    ModelLevelMap,
    SpineInventory,
    inventory_spine,
    load_level_maps,
)
from src.shared.python.contracts import ensure, invariant
from src.shared.python.engine_core.opensim_runtime import import_opensim
from src.shared.python.motion_matching.kinematic_smoothing import (
    smooth_kinematic_trajectory,
)
from src.shared.python.spine.loads import LoadObservable, SpinalLoadSeries

__all__ = [
    "HEIGHT_SOURCES",
    "HEIGHT_SOURCE_GIVEN",
    "HEIGHT_SOURCE_MODEL_EXTENT",
    "PelvisResidualBudget",
    "PreparedModel",
    "ResidualBudget",
    "ResidualReport",
    "SpinalGeneralizedForces",
    "SpineIDOptions",
    "euler_jacobian",
    "euler_rate_columns",
    "prepare_model_for_id",
    "run_spinal_inverse_dynamics",
    "spinal_generalized_forces",
]

_MIN_SAMPLES = 4
_SINGULAR_TOL = 1e-6
_UNIT_TOL = 1e-9
_LEVEL_SEQUENCE = "ZXY"
_NO_LOADS = "none (excluded): gravity only"


@dataclass(frozen=True)
class ResidualBudget:
    """Declared acceptance limits of the pelvis residual wrench.

    ``force_fraction_bw``: peak residual force over body weight;
    ``moment_fraction_bw_height``: peak residual moment over body weight times
    body height.  Both must be finite and positive.
    """

    force_fraction_bw: float = 0.05
    moment_fraction_bw_height: float = 0.01

    def __post_init__(self) -> None:
        for name in ("force_fraction_bw", "moment_fraction_bw_height"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive, got {value!r}")


@dataclass(frozen=True)
class SpineIDOptions:
    """Optional settings of :func:`run_spinal_inverse_dynamics`.

    ``pelvis``: ground-pelvis coordinate names (``None`` selects the fixture
    names when the model has ``pelvis_rx`` and the golf humanoid otherwise);
    ``level_map``: explicit level map (else the registry entry of ``model_id``);
    ``budget``: declared residual limits; ``body_height_m``: height used in the
    moment ratio (``None`` uses the extent of the body origins along the model
    gravity direction in the first frame, which is a model extent and not a
    stature).
    """

    pelvis: PelvisCoordinates | None = None
    level_map: ModelLevelMap | None = None
    budget: ResidualBudget = field(default_factory=ResidualBudget)
    body_height_m: float | None = None

    def __post_init__(self) -> None:
        if self.pelvis is not None and not isinstance(self.pelvis, PelvisCoordinates):
            raise TypeError("pelvis must be a PelvisCoordinates or None")
        if not isinstance(self.budget, ResidualBudget):
            raise TypeError("budget must be a ResidualBudget")
        height = self.body_height_m
        if height is not None and (not math.isfinite(height) or height <= 0):
            raise ValueError(f"body_height_m must be finite and positive: {height!r}")


def _ro(array: np.ndarray) -> np.ndarray:
    out = np.array(array, dtype=float, copy=True)
    out.setflags(write=False)
    return out


HEIGHT_SOURCE_GIVEN = "body_height_m"
HEIGHT_SOURCE_MODEL_EXTENT = "model_extent_first_frame"
HEIGHT_SOURCES = (HEIGHT_SOURCE_GIVEN, HEIGHT_SOURCE_MODEL_EXTENT)


@dataclass(frozen=True)
class PelvisResidualBudget:
    """Pelvis residual wrench per frame and its budget verdict (read-only).

    ``generalized`` (n, 6) holds the six ground-pelvis generalized forces in
    joint order (rotations then translations); ``moment_nm`` and ``force_n``
    (n, 3) are the residual wrench in the ground frame about the pelvis origin
    (N*m, N).  ``force_ratio`` is the peak force norm over ``body_weight_n``;
    ``moment_ratio`` the peak moment norm over ``body_weight_n *
    body_height_m``.  ``passed`` is true when both are within ``budget``.
    """

    time_s: np.ndarray
    generalized: np.ndarray
    moment_nm: np.ndarray
    force_n: np.ndarray
    body_weight_n: float
    body_height_m: float
    budget: ResidualBudget
    body_height_source: str = HEIGHT_SOURCE_MODEL_EXTENT

    def __post_init__(self) -> None:
        if self.body_height_source not in HEIGHT_SOURCES:
            raise ValueError(
                f"body_height_source must be one of {HEIGHT_SOURCES}, "
                f"got {self.body_height_source!r}"
            )
        for name in ("time_s", "generalized", "moment_nm", "force_n"):
            object.__setattr__(self, name, _ro(getattr(self, name)))

    @property
    def peak_force_n(self) -> float:
        """Largest residual force norm [N]."""
        return float(np.max(np.linalg.norm(self.force_n, axis=1)))

    @property
    def rms_force_n(self) -> float:
        """RMS of the residual force norm [N]."""
        return float(np.sqrt(np.mean(np.sum(self.force_n**2, axis=1))))

    @property
    def peak_moment_nm(self) -> float:
        """Largest residual moment norm [N*m]."""
        return float(np.max(np.linalg.norm(self.moment_nm, axis=1)))

    @property
    def rms_moment_nm(self) -> float:
        """RMS of the residual moment norm [N*m]."""
        return float(np.sqrt(np.mean(np.sum(self.moment_nm**2, axis=1))))

    @property
    def force_ratio(self) -> float:
        """Peak residual force over body weight."""
        return self.peak_force_n / self.body_weight_n

    @property
    def moment_ratio(self) -> float:
        """Peak residual moment over body weight times body height."""
        return self.peak_moment_nm / (self.body_weight_n * self.body_height_m)

    @property
    def passed(self) -> bool:
        """True when both ratios are within the declared budget."""
        return bool(
            self.force_ratio <= self.budget.force_fraction_bw
            and self.moment_ratio <= self.budget.moment_fraction_bw_height
        )


ResidualReport = PelvisResidualBudget


# ------------------------------------------------------------ rotation maths


def _axis(letter: str) -> np.ndarray:
    return np.eye(3)["XYZ".index(letter)]


def _rot(letter: str, angle: float) -> np.ndarray:
    return Rotation.from_euler(letter, angle).as_matrix()


def euler_jacobian(q: np.ndarray, sequence: str) -> np.ndarray:
    """Columns of the angular-velocity Jacobian of intrinsic Euler angles.

    For ``R = R_a(q1) R_b(q2) R_c(q3)`` (``sequence = "abc"``) the angular
    velocity in the parent frame is ``omega = J @ qdot`` with
    ``J = [e_a, R_a e_b, R_a R_b e_c]``.  Returns a (3, 3) array.
    """
    a, b, c = list(sequence)
    ra = _rot(a, float(q[0]))
    rab = ra @ _rot(b, float(q[1]))
    return np.column_stack([_axis(a), ra @ _axis(b), rab @ _axis(c)])


def euler_rate_columns(
    q: np.ndarray, qd: np.ndarray, qdd: np.ndarray, sequence: str
) -> tuple[np.ndarray, np.ndarray]:
    """Parent-frame angular velocity and acceleration of intrinsic Euler angles.

    ``omega = J qd`` and ``alpha = J qdd + qd2 * dc2 + qd3 * dc3`` with
    ``dc2 = qd1 (c1 x c2)`` and ``dc3 = qd1 (c1 x c3) + qd2 (c2 x c3)`` the
    time derivatives of the Jacobian columns ``c_k``.
    """
    jac = euler_jacobian(q, sequence)
    c1, c2, c3 = jac[:, 0], jac[:, 1], jac[:, 2]
    dc2 = qd[0] * np.cross(c1, c2)
    dc3 = qd[0] * np.cross(c1, c3) + qd[1] * np.cross(c2, c3)
    return jac @ qd, jac @ qdd + qd[1] * dc2 + qd[2] * dc3


def _check_regular(jac: np.ndarray, what: str) -> None:
    if abs(float(np.linalg.det(jac))) < _SINGULAR_TOL:
        raise ValueError(f"{what}: Euler angles are at a gimbal singularity")


# ------------------------------------------------------------- model adapters


def _coordinate_names(model: object) -> list[str]:
    coords = model.getCoordinateSet()  # type: ignore[attr-defined]
    return [str(coords.get(i).getName()) for i in range(coords.getSize())]


def _coordinate(model: object, name: str) -> object:
    return model.getCoordinateSet().get(name)  # type: ignore[attr-defined]


def _is_rotational(model: object, name: str) -> bool:
    osim = import_opensim()
    return _coordinate(model, name).getMotionType() == osim.Coordinate.Rotational  # type: ignore[attr-defined]


def _u_indices(model: object, state: object) -> dict[str, int]:
    """Map each coordinate to its mobility (speed) index by unit-speed probing."""
    n_u = state.getNU()  # type: ignore[attr-defined]
    out: dict[str, int] = {}
    for name in _coordinate_names(model):
        for i in range(n_u):
            state.updU().set(i, 0.0)  # type: ignore[attr-defined]
        _coordinate(model, name).setSpeedValue(state, 1.0)  # type: ignore[attr-defined]
        hits = [i for i in range(n_u) if state.getU().get(i) == 1.0]  # type: ignore[attr-defined]
        invariant(len(hits) == 1, f"coordinate {name!r} maps to {hits} speeds")
        out[name] = hits[0]
    for i in range(n_u):
        state.updU().set(i, 0.0)  # type: ignore[attr-defined]
    return out


def _body_angular_velocity(model: object, state: object, body: str) -> np.ndarray:
    model.realizeVelocity(state)  # type: ignore[attr-defined]
    vel = model.getBodySet().get(body).getVelocityInGround(state)  # type: ignore[attr-defined]
    omega = vel.get(0)
    return np.array([omega.get(i) for i in range(3)])


def _check_free_pelvis_speeds(
    model: object, state: object, pelvis: PelvisCoordinates
) -> None:
    """Verify that free-joint rotation speeds are ground-frame angular velocity."""
    for k, name in enumerate(pelvis.rotation):
        for i in range(state.getNU()):  # type: ignore[attr-defined]
            state.updU().set(i, 0.0)  # type: ignore[attr-defined]
        _coordinate(model, name).setSpeedValue(state, 1.0)  # type: ignore[attr-defined]
        omega = _body_angular_velocity(model, state, "pelvis")
        if not np.allclose(omega, np.eye(3)[k], atol=_UNIT_TOL):
            raise ValueError(
                f"free-joint speed convention not supported: unit speed of "
                f"{name!r} gives angular velocity {omega.tolist()} in ground"
            )
    for i in range(state.getNU()):  # type: ignore[attr-defined]
        state.updU().set(i, 0.0)  # type: ignore[attr-defined]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


# ------------------------------------------------------------ model preparing


@dataclass
class PreparedModel:
    """A model prepared for ID (the model copy used by the run).

    ``disabled_constraints`` lists the constraints whose enforcement was
    switched off; ``n_constraints`` is the number of constraints in the model;
    ``external_loads`` describes the loads applied (``_NO_LOADS`` if none).
    """

    model: Any
    state: Any
    disabled_constraints: tuple[str, ...]
    n_constraints: int
    external_loads: str

    def enforced_flags(self) -> list[bool]:
        """``isEnforced`` of every constraint of the prepared model."""
        constraints = self.model.getConstraintSet()  # type: ignore[attr-defined]
        return [
            bool(constraints.get(i).get_isEnforced())
            for i in range(constraints.getSize())
        ]


def _describe_external_loads(xml: Path) -> str:
    osim = import_opensim()
    loads = osim.ExternalLoads(str(xml), True)
    items = [
        f"{loads.get(i).getName()}->{loads.get(i).getAppliedToBodyName()}"
        for i in range(loads.getSize())
    ]
    return (
        f"included: {len(items)} ExternalForce [{', '.join(items)}] from "
        f"{xml.name}; data {Path(loads.getDataFileName()).name}"
    )


def _load_model(model_path: Path, external_loads: Path | None) -> Any:
    osim = import_opensim()
    if external_loads is None:
        return osim.Model(str(model_path))
    processor = osim.ModelProcessor(str(model_path))
    processor.append(osim.ModOpAddExternalLoads(str(external_loads.resolve())))
    return processor.process()


def _clear_coordinate_locks(model: Any) -> None:
    """Unlock every coordinate: the drive moves them, locks only corrupt ID/SO/JR.

    A locked coordinate has zero acceleration in forward dynamics and no
    moment arm in OpenSim, while the inverse-dynamics solver ignores the lock.
    """
    coords = model.getCoordinateSet()
    for i in range(coords.getSize()):
        coords.get(i).set_locked(False)


def prepare_model_for_id(
    model_path: str | Path,
    *,
    disable_couplers: bool = True,
    external_loads: str | Path | None = None,
    millard_muscles: bool = False,
) -> PreparedModel:
    """Load ``model_path`` for ID: disable constraints, add external loads.

    ``millard_muscles`` swaps ``Thelen2003Muscle`` elements for Millard muscles
    before loading (rigid-tendon solves; see ``muscle_model``).

    Preconditions: the model file exists; ``external_loads`` is ``None`` or an
    existing ExternalLoads XML.  Postconditions: with ``disable_couplers`` every
    constraint of the model has ``isEnforced == False`` (couplers prescribe
    kinematics only, ADR-0055 item 6) and no coordinate is locked; the returned
    state is initialised.

    Raises:
        TypeError: ``disable_couplers`` is not a bool.
        ValueError: a file is missing.
    """
    if not isinstance(disable_couplers, bool):
        raise TypeError("disable_couplers must be a bool")
    path = Path(model_path)
    if not path.is_file():
        raise ValueError(f"model file not found: {path}")
    xml = None if external_loads is None else Path(external_loads)
    if xml is not None and not xml.is_file():
        raise ValueError(f"external loads file not found: {xml}")
    if millard_muscles:
        with tempfile.TemporaryDirectory() as tmp:
            swapped = millard_equivalent_model(path, Path(tmp) / path.name)
            model = _load_model(swapped or path, xml)
    else:
        model = _load_model(path, xml)
    if disable_couplers:
        _clear_coordinate_locks(model)
    constraints = model.getConstraintSet()
    disabled = []
    for i in range(constraints.getSize()):
        constraint = constraints.get(i)
        if disable_couplers:
            constraint.set_isEnforced(False)
            disabled.append(str(constraint.getName()))
    state = model.initSystem()
    note = _NO_LOADS if xml is None else _describe_external_loads(xml)
    prepared = PreparedModel(
        model, state, tuple(disabled), int(constraints.getSize()), note
    )
    ensure(
        not disable_couplers or not any(prepared.enforced_flags()),
        "every constraint must be disabled for ID",
    )
    return prepared


# ---------------------------------------------------------------- motion I/O


def _read_mot(model: object, path: Path) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    osim = import_opensim()
    table = osim.TimeSeriesTable(str(path))
    degrees = (
        table.hasTableMetaDataKey("inDegrees")
        and table.getTableMetaDataAsString("inDegrees").lower() == "yes"
    )
    n_rows = int(table.getNumRows())
    time_col = table.getIndependentColumn()
    times = np.array([time_col[i] for i in range(n_rows)], dtype=float)
    known = set(_coordinate_names(model))
    cols: dict[str, np.ndarray] = {}
    labels = table.getColumnLabels()
    for k in range(len(labels)):
        name = str(labels[k])
        if name not in known:
            continue
        column = table.getDependentColumn(name)
        values = np.array([column[i] for i in range(n_rows)], dtype=float)
        if degrees and _is_rotational(model, name):
            values = np.radians(values)
        cols[name] = values
    return times, cols


def _motion_table(
    model: object, motion: DriveResult | str | Path
) -> tuple[np.ndarray, dict[str, np.ndarray], str, str]:
    """Return ``(time_s, coordinates, distribution_source, motion_label)``."""
    if isinstance(motion, DriveResult):
        known = set(_coordinate_names(model))
        cols = {k: np.asarray(v) for k, v in motion.coordinates.items() if k in known}
        source = motion.kinematics.distribution_source
        return np.asarray(motion.time_s, dtype=float), cols, source, "DriveResult"
    if isinstance(motion, str | Path):
        path = Path(motion)
        if path.suffix not in (".mot", ".sto") or not path.is_file():
            raise ValueError(f"motion must be an existing .mot/.sto file: {path}")
        times, cols = _read_mot(model, path)
        return times, cols, f"motion file: {path.name}", path.name
    raise TypeError(
        f"motion must be a DriveResult or a .mot path, got {type(motion).__name__}"
    )


def _derivatives(
    times: np.ndarray, q: np.ndarray, filter_hz: float | None
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if times.size < _MIN_SAMPLES or np.any(np.diff(times) <= 0):
        raise ValueError(
            f"need at least {_MIN_SAMPLES} strictly increasing time samples"
        )
    if not np.isfinite(q).all():
        raise ValueError("coordinate values must be finite")
    smooth = q
    if filter_hz is not None:
        result = smooth_kinematic_trajectory(times, q, cutoff_hz=filter_hz)
        if not result.smoothing_applied:
            raise ValueError(f"low-pass filter at {filter_hz} Hz could not be applied")
        smooth = result.q
    spline = CubicSpline(times, smooth, axis=0)
    return smooth, spline(times, 1), spline(times, 2)


# --------------------------------------------------------------- per-frame ID


@dataclass(frozen=True)
class _Layout:
    """Coordinate bookkeeping shared by all frames."""

    names: list[str]
    u_index: dict[str, int]
    pelvis: PelvisCoordinates
    free_pelvis: bool


def _frame_speeds(
    layout: _Layout, q: np.ndarray, qd: np.ndarray, qdd: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Speeds ``u`` and ``udot`` per coordinate name order of the layout."""
    u, udot = qd.copy(), qdd.copy()
    if layout.free_pelvis:
        pos = [layout.names.index(n) for n in layout.pelvis.rotation]
        omega, alpha = euler_rate_columns(
            q[pos], qd[pos], qdd[pos], layout.pelvis.sequence
        )
        u[pos], udot[pos] = omega, alpha
    return u, udot


def _solve_frame(
    prepared: PreparedModel,
    solver: object,
    layout: _Layout,
    time_s: float,
    row: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> np.ndarray:
    model, state = prepared.model, prepared.state
    q, qd, qdd = row
    u, udot = _frame_speeds(layout, q, qd, qdd)
    state.setTime(time_s)  # type: ignore[attr-defined]
    _set_positions(model, state, layout, q)
    for j, name in enumerate(layout.names):
        _coordinate(model, name).setSpeedValue(state, float(u[j]))  # type: ignore[attr-defined]
    model.realizeDynamics(state)  # type: ignore[attr-defined]
    osim = import_opensim()
    acc = osim.Vector(len(layout.names), 0.0)
    for j, name in enumerate(layout.names):
        acc.set(layout.u_index[name], float(udot[j]))
    tau = solver.solve(state, acc)  # type: ignore[attr-defined]
    return np.array([tau.get(i) for i in range(len(layout.names))])


def _level_moments(
    tau: np.ndarray,
    q: np.ndarray,
    layout: _Layout,
    inventory: SpineInventory,
    signs: Mapping,
) -> np.ndarray:
    """(L, 3) moments on the superior body in the inferior ISB frame [N*m]."""
    out = np.zeros((len(inventory.levels), 3))
    for k, level in enumerate(inventory.levels):
        names = inventory.coordinate_names[level]
        sign = np.array(signs[level], dtype=float)
        cols = [layout.names.index(n) for n in names]
        theta = sign * q[cols]
        jac = euler_jacobian(theta, _LEVEL_SEQUENCE)
        _check_regular(jac, f"level {level.name}")
        tau_theta = sign * np.array([tau[layout.u_index[n]] for n in names])
        moment_p = np.linalg.solve(jac.T, tau_theta)
        out[k] = inventory.neutral[level][:3, :3] @ moment_p
    return out


def _pelvis_wrench(
    tau: np.ndarray, q: np.ndarray, layout: _Layout
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generalized forces, ground moment (about the origin) and ground force."""
    pel = layout.pelvis
    rot = np.array([tau[layout.u_index[n]] for n in pel.rotation])
    trans = np.array([tau[layout.u_index[n]] for n in pel.translation])
    moment = rot
    if not layout.free_pelvis:
        jac = euler_jacobian(
            q[[layout.names.index(n) for n in pel.rotation]], pel.sequence
        )
        _check_regular(jac, "pelvis")
        moment = np.linalg.solve(jac.T, rot)
    return np.concatenate([rot, trans]), moment, trans


def _gravity_direction(model: object) -> np.ndarray:
    """Unit vector of the model gravity in ground (the "down" direction)."""
    gravity = model.getGravity()  # type: ignore[attr-defined]
    vec = np.array([gravity.get(i) for i in range(3)], dtype=float)
    norm = float(np.linalg.norm(vec))
    if not math.isfinite(norm) or norm <= 0.0:
        raise ValueError("model gravity must be a finite non-zero vector")
    return vec / norm


def _body_origin(bodies: object, index: int, state: object) -> np.ndarray:
    pos = bodies.get(index).getTransformInGround(state).p()  # type: ignore[attr-defined]
    return np.array([pos.get(i) for i in range(3)], dtype=float)


def _set_positions(
    model: object, state: object, layout: _Layout, q: np.ndarray
) -> None:
    """Set every coordinate value of ``state`` to ``q`` (layout order)."""
    for j, name in enumerate(layout.names):
        _coordinate(model, name).setValue(state, float(q[j]), False)  # type: ignore[attr-defined]


def _body_extent(
    model: object, state: object, layout: _Layout, q0: np.ndarray
) -> float:
    """Extent of the body origins along the gravity direction at ``q0`` [m].

    Evaluated at the first frame of the motion (``q0``), not at the state the
    run leaves behind.  The projection is on the model's own gravity direction,
    so the model's up axis is not assumed.  Mutates ``state`` (positions only).
    """
    _set_positions(model, state, layout, q0)
    model.realizePosition(state)  # type: ignore[attr-defined]
    bodies = model.getBodySet()  # type: ignore[attr-defined]
    down = _gravity_direction(model)
    heights = [
        float(down @ _body_origin(bodies, i, state)) for i in range(bodies.getSize())
    ]
    return float(max(heights) - min(heights))


def _resolve_pelvis(model: object, options: SpineIDOptions) -> PelvisCoordinates:
    if options.pelvis is not None:
        return options.pelvis
    names = set(_coordinate_names(model))
    return FIXTURE_PELVIS if "pelvis_rx" in names else GOLF_PELVIS


def _coordinate_matrix(
    names: list[str], cols: Mapping[str, np.ndarray], required: list[str]
) -> np.ndarray:
    for name in required:
        if name not in cols:
            raise ValueError(f"motion has no column for required coordinate {name!r}")
    n = len(next(iter(cols.values())))
    return np.column_stack([cols.get(name, np.zeros(n)) for name in names])


def _provenance(
    prepared: PreparedModel,
    inputs: tuple[Path, str, str, float | None],
    defaulted: list[str],
) -> dict[str, str]:
    model_path, model_id, source, filter_hz = inputs
    couplers = (
        f"disabled ({len(prepared.disabled_constraints)} of "
        f"{prepared.n_constraints} constraints)"
        if prepared.disabled_constraints
        else f"not disabled ({prepared.n_constraints} constraints present)"
    )
    return {
        "model_id": model_id,
        "model_sha256": _sha256(model_path),
        "solver": "OpenSim InverseDynamicsSolver (generalized effort)",
        "cost_exponent": "n/a",
        "distribution_source": source,
        "external_loads": prepared.external_loads,
        "couplers": couplers,
        "filter_hz": "none" if filter_hz is None else str(float(filter_hz)),
        "derivatives": "scipy CubicSpline (not-a-knot) at the sample times",
        "units": "N*m",
        "frame": "inferior vertebra ISB frame, about the mid-disc joint centre",
        "coordinates_defaulted_to_zero": ", ".join(defaulted) or "none",
    }


@dataclass(frozen=True)
class _Trajectory:
    """Everything one ID pass produces (shared by the public entry points)."""

    prepared: PreparedModel
    layout: _Layout
    inventory: SpineInventory
    times: np.ndarray
    q: np.ndarray
    qd: np.ndarray
    tau: np.ndarray  # (n, nu) by mobility index
    provenance: dict[str, str]


def _disable_muscles(prepared: PreparedModel) -> int:
    """Stop every muscle applying force in the prepared state; return the count."""
    muscles = prepared.model.getMuscles()  # type: ignore[attr-defined]
    for i in range(muscles.getSize()):
        muscles.get(i).setAppliesForce(prepared.state, False)
    return int(muscles.getSize())


@dataclass(frozen=True)
class _Kinematics:
    """Motion samples, derivatives and coordinate bookkeeping of one run."""

    times: np.ndarray
    q: np.ndarray
    qd: np.ndarray
    qdd: np.ndarray
    layout: _Layout
    inventory: SpineInventory
    source: str
    defaulted: list[str]


def _load_kinematics(
    prepared: PreparedModel,
    motion: DriveResult | str | Path,
    model_id: str,
    opts: SpineIDOptions,
    filter_hz: float | None,
) -> _Kinematics:
    """Read ``motion`` and derive ``q, qd, qdd`` for the prepared model.

    Shared by inverse dynamics and the joint-reaction run (SPN-10c) so both
    use identical coordinates, derivatives and mobility layout.
    """
    model, state = prepared.model, prepared.state
    times, cols, source, _label = _motion_table(model, motion)
    inventory = inventory_spine(model, model_id, level_map=opts.level_map)
    pelvis = _resolve_pelvis(model, opts)
    names = _coordinate_names(model)
    required = [c for lv in inventory.levels for c in inventory.coordinate_names[lv]]
    required += [*pelvis.rotation, *pelvis.translation]
    q, qd, qdd = _derivatives(
        times, _coordinate_matrix(names, cols, required), filter_hz
    )
    free = state.getNQ() != state.getNU()
    if free:
        _check_free_pelvis_speeds(model, state, pelvis)
    layout = _Layout(names, _u_indices(model, state), pelvis, free)
    defaulted = [x for x in names if x not in cols]
    return _Kinematics(times, q, qd, qdd, layout, inventory, source, defaulted)


def _solve_trajectory(
    model_path: str | Path,
    motion: DriveResult | str | Path,
    model_id: str,
    io: tuple[str | Path | None, float | None, bool],
    opts: SpineIDOptions,
    exclude_muscles: bool,
) -> _Trajectory:
    """Run the ID solve over every sample (``io`` = loads, filter, couplers)."""
    external_loads, filter_hz, disable_couplers = io
    if filter_hz is not None and not (math.isfinite(filter_hz) and filter_hz > 0):
        raise ValueError(f"filter_hz must be positive and finite, got {filter_hz!r}")
    prepared = prepare_model_for_id(
        model_path, disable_couplers=disable_couplers, external_loads=external_loads
    )
    n_muscles = _disable_muscles(prepared) if exclude_muscles else 0
    model = prepared.model
    kin = _load_kinematics(prepared, motion, model_id, opts, filter_hz)
    times, q, qd, qdd = kin.times, kin.q, kin.qd, kin.qdd
    layout, inventory = kin.layout, kin.inventory
    solver = import_opensim().InverseDynamicsSolver(model)
    tau = np.zeros((times.size, len(layout.names)))
    try:
        for i in range(times.size):
            tau[i] = _solve_frame(
                prepared, solver, layout, float(times[i]), (q[i], qd[i], qdd[i])
            )
    except RuntimeError as exc:
        raise ValueError(f"inverse dynamics failed: {exc}") from exc
    prov = _provenance(
        prepared,
        (Path(model_path), model_id, kin.source, filter_hz),
        kin.defaulted,
    )
    prov["muscles"] = (
        f"excluded ({n_muscles} muscles not applying force)"
        if exclude_muscles
        else "as modelled (muscle forces at the default state are included)"
    )
    return _Trajectory(prepared, layout, inventory, times, q, qd, tau, prov)


def run_spinal_inverse_dynamics(
    model_path: str | Path,
    motion: DriveResult | str | Path,
    model_id: str,
    *,
    external_loads: str | Path | None = None,
    filter_hz: float | None = None,
    disable_couplers: bool = True,
    options: SpineIDOptions | None = None,
) -> tuple[SpinalLoadSeries, PelvisResidualBudget]:
    """Spinal inverse dynamics: generalized effort per level + pelvis residual.

    Args:
        model_path: ``.osim`` file of the segmental model.
        motion: a :class:`DriveResult` or a ``.mot`` path whose columns include
            every level coordinate and the six pelvis coordinates.
        model_id: level-map key (or use ``options.level_map``).
        external_loads: ExternalLoads XML, or ``None`` (recorded as ``none
            (excluded): gravity only`` in the provenance).
        filter_hz: optional zero-phase low-pass cutoff [Hz] for the coordinates.
        disable_couplers: disable every constraint (couplers) for ID.
        options: :class:`SpineIDOptions`.

    Returns:
        ``(series, residual)``: a ``GENERALIZED_EFFORT`` series (moments in
        N*m, forces exactly zero) and the pelvis residual budget.

    Raises:
        TypeError: wrong ``motion`` or ``options`` type.
        ValueError: ``filter_hz`` not positive, a missing file or coordinate,
            too few samples, or a gimbal singularity.
    """
    opts = SpineIDOptions() if options is None else options
    if not isinstance(opts, SpineIDOptions):
        raise TypeError("options must be a SpineIDOptions or None")
    run = _solve_trajectory(
        model_path,
        motion,
        model_id,
        (external_loads, filter_hz, disable_couplers),
        opts,
        True,  # GENERALIZED_EFFORT is the net joint moment: muscles excluded
    )
    inventory, layout = run.inventory, run.layout
    signs = _axis_signs(inventory, opts, model_id)
    n = run.times.size
    moments = np.zeros((n, len(inventory.levels), 3))
    generalized, res_m, res_f = np.zeros((n, 6)), np.zeros((n, 3)), np.zeros((n, 3))
    for i in range(n):
        moments[i] = _level_moments(run.tau[i], run.q[i], layout, inventory, signs)
        generalized[i], res_m[i], res_f[i] = _pelvis_wrench(
            run.tau[i], run.q[i], layout
        )
    prov = dict(run.provenance)
    series = SpinalLoadSeries(
        time_s=run.times,
        levels=inventory.levels,
        observable=LoadObservable.GENERALIZED_EFFORT,
        force_n=np.zeros_like(moments),
        moment_nm=moments,
        provenance=prov,
    )
    model, state = run.prepared.model, run.prepared.state
    weight = _body_weight(model, state)
    if opts.body_height_m is None:
        height = _body_extent(model, state, layout, run.q[0])
        source = HEIGHT_SOURCE_MODEL_EXTENT
    else:
        height, source = opts.body_height_m, HEIGHT_SOURCE_GIVEN
    residual = PelvisResidualBudget(
        run.times, generalized, res_m, res_f, weight, height, opts.budget, source
    )
    ensure(
        series.observable is LoadObservable.GENERALIZED_EFFORT,
        "observable must be GENERALIZED_EFFORT",
    )
    return series, residual


@dataclass(frozen=True)
class SpinalGeneralizedForces:
    """Per-coordinate ID generalized forces for static optimisation (SPN-10b).

    ``model_coordinates`` are all model coordinate names; ``q`` / ``qd`` /
    ``tau`` are ``(n, len(model_coordinates))`` in that order (rad, rad/s,
    m, m/s; ``tau`` in N*m for rotational and N for translational
    coordinates, OpenSim's sign of each coordinate).  ``q``/``qd`` are the
    (optionally filtered) values the ID used.  ``levels`` and
    ``level_coordinates`` (``3 * len(levels)``, caudal first, (flex_ext,
    lat_bend, axial_rot) per level) come from the inventory.  Arrays are
    read-only.
    """

    time_s: np.ndarray
    model_coordinates: tuple[str, ...]
    q: np.ndarray
    qd: np.ndarray
    tau: np.ndarray
    levels: tuple
    level_coordinates: tuple[str, ...]
    model_id: str
    provenance: Mapping[str, str]

    def __post_init__(self) -> None:
        for name in ("time_s", "q", "qd", "tau"):
            object.__setattr__(self, name, _ro(getattr(self, name)))
        object.__setattr__(self, "provenance", MappingProxyType(dict(self.provenance)))

    def level_tau(self) -> np.ndarray:
        """``(n, 3 L)`` generalized forces of the level coordinates [N*m]."""
        cols = [self.model_coordinates.index(c) for c in self.level_coordinates]
        return self.tau[:, cols]


def spinal_generalized_forces(
    model_path: str | Path,
    motion: DriveResult | str | Path,
    model_id: str,
    *,
    external_loads: str | Path | None = None,
    filter_hz: float | None = None,
    disable_couplers: bool = True,
    options: SpineIDOptions | None = None,
    exclude_muscles: bool = True,
) -> SpinalGeneralizedForces:
    """The same ID solve as :func:`run_spinal_inverse_dynamics`, per coordinate.

    Static optimisation needs ``tau`` per coordinate, not the per-level ISB
    moments.  With ``exclude_muscles`` (default) every muscle is switched off
    in the ID state, so ``tau`` is the net generalized force the muscles (and
    reserves) must supply; a muscle-bearing model then gives the same ``tau``
    as its muscle-free skeleton.  Arguments and errors are those of
    :func:`run_spinal_inverse_dynamics`; ``exclude_muscles`` must be a bool.
    """
    opts = SpineIDOptions() if options is None else options
    if not isinstance(opts, SpineIDOptions):
        raise TypeError("options must be a SpineIDOptions or None")
    if not isinstance(exclude_muscles, bool):
        raise TypeError("exclude_muscles must be a bool")
    run = _solve_trajectory(
        model_path,
        motion,
        model_id,
        (external_loads, filter_hz, disable_couplers),
        opts,
        exclude_muscles,
    )
    order = [run.layout.u_index[name] for name in run.layout.names]
    inventory = run.inventory
    level_coords = tuple(
        c for lv in inventory.levels for c in inventory.coordinate_names[lv]
    )
    return SpinalGeneralizedForces(
        time_s=run.times,
        model_coordinates=tuple(run.layout.names),
        q=run.q,
        qd=run.qd,
        tau=run.tau[:, order],
        levels=inventory.levels,
        level_coordinates=level_coords,
        model_id=model_id,
        provenance=run.provenance,
    )


def _body_weight(model: object, state: object) -> float:
    gravity = model.getGravity()  # type: ignore[attr-defined]
    magnitude = math.sqrt(sum(gravity.get(i) ** 2 for i in range(3)))
    return float(model.getTotalMass(state) * magnitude)  # type: ignore[attr-defined]


def _axis_signs(inventory: SpineInventory, opts: SpineIDOptions, model_id: str) -> dict:
    """Per-level axis signs from the level map used for the inventory."""
    entry = opts.level_map or load_level_maps()[model_id]
    return {lv: entry.levels[lv].axis_sign for lv in inventory.levels}

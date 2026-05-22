"""OpenSim forward simulator with polynomial torque controller (issue #4120).

This module implements ``simulate_with_coefficients`` -- the canonical
engine-agnostic forward-sim wrapper specified in
``OPENSIM_PARITY_SPEC.md`` (sec 2.2) and ``CROSS_ENGINE_PARITY_SPEC.md``
(sec 2.2).

The entry point is :func:`simulate_with_coefficients`, which:

1. Loads the canonical golf-humanoid ``.osim`` (lazy + cached).
2. Reshapes the flat coefficient vector ``theta`` into one degree-6
   polynomial of joint torque per actuated coordinate.
3. Wires a :class:`PolynomialTorqueController` onto every
   ``CoordinateActuator`` so each integrator step samples
   ``tau_j(t) = sum_{k=0..6} a_{jk} t^k``.
4. Drives ``osim.Manager.integrate(t_final)`` over the canonical time
   grid; samples the SimTK state at every grid point.
5. Re-walks the trajectory once for grip / clubhead world poses via
   :func:`extract_full_pose`.
6. Packs everything into the canonical :class:`SimOut`.

Pure-numpy helpers (``evaluate_polynomial_torque``, ``SimOptions``,
``SimOut`` schema validation) are importable without ``opensim`` so unit
tests run on every CI even when the OpenSim wheel is not installed.
"""

from __future__ import annotations

import collections
import threading
import time as _time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np
import numpy.typing as npt

from src.shared.python.motion_matching.validate_theta import validate_theta

# Canonical polynomial form: tau_j(t) = sum_{k=0}^{6} a_{j,k} * t^k.
POLY_DEGREE: int = 6
COEFFS_PER_JOINT: int = POLY_DEGREE + 1

# Default model path lives next to the engine; resolved lazily so tests
# can override via SimOptions.osim_path.
_DEFAULT_OSIM = Path(__file__).resolve().parents[2] / "models" / "golf_humanoid.osim"

# Canonical frame / body names exposed in SimOut. Sourced from the
# committed golf_humanoid.osim (issue #4110).
GRIP_FRAME_NAME: str = "hand_r_grip_offset"
CLUBHEAD_FRAME_NAME: str = "club_head_offset"


# --------------------------------------------------------------------------- #
# Public dataclasses
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class SimOptions:
    """Forward-simulation options for the OpenSim engine.

    Mirrors the cross-engine ``SimOptions`` contract; field names match
    the Pinocchio/Drake/MuJoCo siblings so callers can swap engines
    without changing options assembly.

    Attributes:
        t_final: Simulation horizon (seconds). Must be > 0.
        dt: Output / sample timestep (seconds). The internal integrator
            adapts; ``dt`` is the rate at which we *report* state.
            Must satisfy ``0 < dt <= t_final``.
        integrator: Integrator scheme passed to ``osim.Manager``.
            ``"rk_merson"`` (default) is OpenSim's stock 4th-order RK
            with adaptive step. ``"semi_explicit_euler"`` is faster and
            accepted as an option; both names accepted defensively.
        accuracy: Integrator absolute / relative tolerance. Default
            ``1e-5`` matches OpenSim Manager defaults.
        gravity: Optional ``(3,)`` gravity override (m/s^2). ``None``
            keeps the model's default. Pass ``np.zeros(3)`` for the
            energy-conservation regression.
        osim_path: Override the cached ``.osim`` location. ``None`` uses
            the packaged ``golf_humanoid.osim``.
    """

    t_final: float = 1.0
    dt: float = 1e-3
    integrator: Literal["rk_merson", "semi_explicit_euler"] = "rk_merson"
    accuracy: float = 1e-5
    gravity: npt.NDArray[np.float64] | None = None
    osim_path: str | Path | None = None

    def __post_init__(self) -> None:  # DbC preconditions
        if not (self.t_final > 0):
            msg = f"t_final must be positive, got {self.t_final!r}"
            raise ValueError(msg)
        if not (0 < self.dt <= self.t_final):
            msg = (
                f"dt must satisfy 0 < dt <= t_final; "
                f"got dt={self.dt!r}, t_final={self.t_final!r}"
            )
            raise ValueError(msg)
        if self.integrator not in ("rk_merson", "semi_explicit_euler"):
            msg = (
                f"integrator={self.integrator!r} not supported; "
                "use 'rk_merson' or 'semi_explicit_euler'"
            )
            raise ValueError(msg)
        if not (self.accuracy > 0):
            msg = f"accuracy must be positive, got {self.accuracy!r}"
            raise ValueError(msg)
        if self.gravity is not None:
            g = np.asarray(self.gravity, dtype=np.float64)
            if g.shape != (3,):
                msg = f"gravity must have shape (3,), got {g.shape!r}"
                raise ValueError(msg)


@dataclass(frozen=True)
class SimOut:
    """Canonical forward-simulation output.

    Schema mirrors the cross-engine spec §2.2; fields with the same name
    on every engine carry identical semantics so the cost function can
    treat them uniformly.

    All time series are sampled at ``options.dt`` from ``t=0`` to
    ``t=options.t_final`` inclusive (length ``n_steps + 1``).

    Attributes:
        time: ``(N,)`` sample times (s).
        q: ``(N, n_joints)`` joint configuration trajectory (rad).
        qd: ``(N, n_joints)`` joint velocity trajectory (rad/s).
        qdd: ``(N, n_joints)`` joint acceleration trajectory (rad/s^2).
        tau: ``(N, n_joints)`` applied joint torques (N*m).
        grip: ``(N, 3)`` grip world position (m). Origin of the
            ``hand_r_grip_offset`` frame.
        grip_quat: ``(N, 4)`` grip world orientation, ``[w, x, y, z]``.
        clubhead: ``(N, 3)`` clubhead world position (m). Origin of
            the ``club_head_offset`` frame.
        club_quat: ``(N, 4)`` clubhead world orientation, ``[w, x, y, z]``.
        solver_status: ``"success" | "warning" | "failed"``.
        duration_s: Wall-clock time (s) for the simulate call.
        meta: Free-form diagnostics (frame ids, integrator name, etc.).
    """

    time: npt.NDArray[np.float64]
    q: npt.NDArray[np.float64]
    qd: npt.NDArray[np.float64]
    qdd: npt.NDArray[np.float64]
    tau: npt.NDArray[np.float64]
    grip: npt.NDArray[np.float64]
    grip_quat: npt.NDArray[np.float64]
    clubhead: npt.NDArray[np.float64]
    club_quat: npt.NDArray[np.float64]
    solver_status: str
    duration_s: float
    meta: dict[str, Any] = field(default_factory=dict)


# --------------------------------------------------------------------------- #
# Pure-numpy polynomial torque (testable without opensim)
# --------------------------------------------------------------------------- #


def evaluate_polynomial_torque(
    coeffs: npt.NDArray[np.float64],
    t: float,
) -> npt.NDArray[np.float64]:
    """Evaluate the per-joint degree-6 polynomial torque at time ``t``.

    Implements the canonical contract::

        tau_j(t) = sum_{k=0}^{6} a_{j,k} * t^k

    Args:
        coeffs: ``(n_joints, 7)`` matrix where ``coeffs[j, k]`` is the
            coefficient on ``t^k``. Element ``k=0`` is the constant
            term, ``k=6`` is the highest degree.
        t: Evaluation time (scalar, seconds).

    Returns:
        ``(n_joints,)`` joint-torque vector.

    Raises:
        ValueError: If ``coeffs`` is not 2D with 7 columns, or ``t`` is
            non-finite.
    """
    coeffs_arr = np.asarray(coeffs, dtype=np.float64)
    if coeffs_arr.ndim != 2:
        msg = f"coeffs must be 2D (n_joints, 7); got ndim={coeffs_arr.ndim}"
        raise ValueError(msg)
    if coeffs_arr.shape[1] != COEFFS_PER_JOINT:
        msg = (
            f"coeffs must have {COEFFS_PER_JOINT} columns "
            f"(degree {POLY_DEGREE}); got shape {coeffs_arr.shape}"
        )
        raise ValueError(msg)
    if not np.isfinite(t):
        raise ValueError(f"t must be finite, got {t!r}")

    # Horner's method for numerical stability and ~7x fewer multiplies.
    out = coeffs_arr[:, POLY_DEGREE].copy()
    for k in range(POLY_DEGREE - 1, -1, -1):
        out = out * t + coeffs_arr[:, k]
    return out


# --------------------------------------------------------------------------- #
# Model cache (process-local; SimTK::State is per-call so we cache only
# the immutable Model after initSystem()).
# --------------------------------------------------------------------------- #


_MODEL_CACHE: dict[str, Any] = {}
# Per-model locks: keyed by resolved model path string.
# Using defaultdict avoids a single global lock for all models,
# which would serialize concurrent loads of different model files.
_MODEL_CACHE_LOCKS: dict[str, threading.Lock] = collections.defaultdict(threading.Lock)


def _resolve_osim_path(osim_path: str | Path | None) -> Path:
    if osim_path is None:
        return _DEFAULT_OSIM
    return Path(osim_path)


def _load_model(osim_path: Path) -> Any:
    """Load and cache the OpenSim Model for the given ``.osim`` path.

    Returns a *fresh copy* of the cached model so callers can safely
    edit gravity / actuators without leaking state across calls.
    """
    import opensim as osim  # noqa: PLC0415  -- optional engine dep

    key = str(osim_path.resolve())
    with _MODEL_CACHE_LOCKS[key]:
        cached = _MODEL_CACHE.get(key)
        if cached is None:
            if not osim_path.exists():
                raise FileNotFoundError(f"OpenSim model not found at {osim_path}")
            # Load once; tests may reuse this path across many calls.
            cached = osim.Model(str(osim_path))
            _MODEL_CACHE[key] = cached
    # Always return a clone -- the controller will be added per call.
    return osim.Model(cached)


# --------------------------------------------------------------------------- #
# Polynomial torque controller (osim.Controller subclass).
#
# OpenSim's Python bindings dispatch the C++ ``computeControls`` callback
# back into Python at every integrator stage. We override it to evaluate
# the polynomial torque law for every ``CoordinateActuator``.
# --------------------------------------------------------------------------- #


def _make_controller_class() -> type:  # noqa: C901, PLR0915 -- factory holds class body
    """Build the ``PolynomialTorqueController`` subclass lazily.

    OpenSim's ``Controller`` base class is only available once
    ``import opensim`` succeeds; we defer construction so the module
    imports cleanly without the optional dep.
    """
    import opensim as osim  # noqa: PLC0415  -- optional engine dep

    class PolynomialTorqueController(osim.Controller):
        """Evaluates ``tau_j(t) = sum_k a_jk * t^k`` per coordinate.

        The controller stores the coefficient matrix and the actuator
        indices; ``computeControls`` is hot-called by the integrator
        and writes one scalar per actuator into the provided
        ``controls`` SimTK::Vector.
        """

        def __init__(
            self,
            coeffs: npt.NDArray[np.float64],
            actuator_names: list[str],
        ) -> None:
            super().__init__()
            self._coeffs = np.asarray(coeffs, dtype=np.float64).copy()
            self._actuator_names = list(actuator_names)
            self.setName("PolynomialTorqueController")
            self._validate_shape()

        def _validate_shape(self) -> None:
            if self._coeffs.ndim != 2:
                msg = f"coeffs must be 2D; got ndim={self._coeffs.ndim}"
                raise ValueError(msg)
            if self._coeffs.shape[1] != COEFFS_PER_JOINT:
                msg = (
                    f"coeffs must have {COEFFS_PER_JOINT} columns; "
                    f"got shape {self._coeffs.shape}"
                )
                raise ValueError(msg)
            if self._coeffs.shape[0] != len(self._actuator_names):
                msg = (
                    f"coeffs has {self._coeffs.shape[0]} rows but "
                    f"{len(self._actuator_names)} actuators were provided"
                )
                raise ValueError(msg)

        # --- Public accessors (used by tests + fit driver) --------------- #

        def set_theta(self, theta: npt.NDArray[np.float64]) -> None:
            """Update the coefficient matrix in-place.

            Args:
                theta: Either a flat ``(n_joints * 7,)`` vector or a
                    ``(n_joints, 7)`` matrix matching the controller's
                    actuator count.
            """
            arr = np.asarray(theta, dtype=np.float64)
            if arr.ndim == 1:
                arr = arr.reshape(self._coeffs.shape)
            if arr.shape != self._coeffs.shape:
                msg = f"theta has shape {arr.shape}; expected {self._coeffs.shape}"
                raise ValueError(msg)
            self._coeffs[...] = arr

        def get_theta(self) -> npt.NDArray[np.float64]:
            """Return a copy of the current coefficient matrix."""
            return self._coeffs.copy()

        def tau_at(self, t: float, joint_idx: int) -> float:
            """Evaluate one joint's torque at time ``t`` without SimTK."""
            if not (0 <= joint_idx < self._coeffs.shape[0]):
                msg = f"joint_idx={joint_idx} out of range [0, {self._coeffs.shape[0]})"
                raise IndexError(msg)
            return float(evaluate_polynomial_torque(self._coeffs, float(t))[joint_idx])

        # --- OpenSim callback -------------------------------------------- #

        def computeControls(  # noqa: N802 -- OpenSim API name
            self,
            state: Any,
            controls: Any,
        ) -> None:
            """Write polynomial-torque controls into ``controls``.

            OpenSim invokes this callback once per integrator stage.
            The contract is to add (not overwrite) into ``controls``
            so that multiple controllers can compose; for our use case
            we are the only controller and the additions are equivalent
            to writes.
            """
            t = float(state.getTime())
            tau = evaluate_polynomial_torque(self._coeffs, t)

            # Build a SimTK::Vector of length 1 for each actuator and add.
            # The controls vector indexing is [actuator_idx_in_model];
            # we wired the controller to actuators in order of insertion.
            for j in range(len(self._actuator_names)):
                # ``getModel().getActuators().get(name).getControlIndex(0)``
                # would be one-indexed safe; instead we use the controller's
                # getActuatorSet(), which OpenSim populates from
                # ``addActuator``. This stays LOD-2.
                act_idx = self._actuator_index_in_controls[j]
                # SimTK::Vector add via ``set`` (no operator+= in Python).
                controls.set(act_idx, controls.get(act_idx) + float(tau[j]))

        # Internal mapping populated by the simulate harness after the
        # controller has been added to the model and the actuators have
        # been bound. Allows ``computeControls`` to skip a name lookup
        # on every callback.
        _actuator_index_in_controls: list[int] = []

        def _bind_indices(self, indices: list[int]) -> None:
            """Cache the per-actuator control-vector indices."""
            if len(indices) != len(self._actuator_names):
                msg = (
                    f"indices length {len(indices)} != actuator count "
                    f"{len(self._actuator_names)}"
                )
                raise ValueError(msg)
            self._actuator_index_in_controls = list(indices)

    return PolynomialTorqueController


# --------------------------------------------------------------------------- #
# Forward-kinematics helper -- minimal wrapper over osim API that works
# with the shipped golf_humanoid.osim (uses hand_r_grip_offset and
# club_head_offset PhysicalOffsetFrames).
# --------------------------------------------------------------------------- #


def extract_full_pose(state: Any, model: Any) -> dict[str, npt.NDArray[np.float64]]:
    """Extract grip + clubhead world poses from a single SimTK state.

    Mirrors the FK API in ``opensim_golf/fk.py`` but uses the canonical
    ``hand_r_grip_offset`` / ``club_head_offset`` frames from the
    committed golf_humanoid.osim.

    Args:
        state: SimTK::State (must already be realized to Position or higher).
        model: Initialized ``osim.Model``.

    Returns:
        Dict with ``grip``, ``grip_quat``, ``clubhead``, ``club_quat`` keys
        (each a numpy array; quats in canonical ``[w, x, y, z]`` order).
    """
    model.realizePosition(state)

    grip_frame = model.getComponent(f"/jointset/hand_r_to_club/{GRIP_FRAME_NAME}")
    clubhead_frame = model.getComponent(
        f"/jointset/hand_r_to_club/{CLUBHEAD_FRAME_NAME}"
    )

    grip_t = grip_frame.getTransformInGround(state)
    clubhead_t = clubhead_frame.getTransformInGround(state)

    grip_pos = _vec3_to_array(grip_t.p())
    clubhead_pos = _vec3_to_array(clubhead_t.p())

    grip_quat = _rotation_to_quat(grip_t.R())
    club_quat = _rotation_to_quat(clubhead_t.R())

    return {
        "grip": grip_pos,
        "grip_quat": grip_quat,
        "clubhead": clubhead_pos,
        "club_quat": club_quat,
    }


def _vec3_to_array(v: Any) -> npt.NDArray[np.float64]:
    """Convert a SimTK::Vec3 to a (3,) numpy array."""
    return np.array([v.get(0), v.get(1), v.get(2)], dtype=np.float64)


def _rotation_to_quat(rot: Any) -> npt.NDArray[np.float64]:
    """Convert a SimTK::Rotation to a (4,) [w, x, y, z] quaternion.

    Uses Shepperd's method for numerical stability across all rotations.
    """
    m = np.array(
        [
            [rot.get(0, 0), rot.get(0, 1), rot.get(0, 2)],
            [rot.get(1, 0), rot.get(1, 1), rot.get(1, 2)],
            [rot.get(2, 0), rot.get(2, 1), rot.get(2, 2)],
        ],
        dtype=np.float64,
    )
    trace = np.trace(m)
    if trace > 0.0:
        s = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (m[2, 1] - m[1, 2]) * s
        y = (m[0, 2] - m[2, 0]) * s
        z = (m[1, 0] - m[0, 1]) * s
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = 2.0 * np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2])
        w = (m[2, 1] - m[1, 2]) / s
        x = 0.25 * s
        y = (m[0, 1] + m[1, 0]) / s
        z = (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = 2.0 * np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2])
        w = (m[0, 2] - m[2, 0]) / s
        x = (m[0, 1] + m[1, 0]) / s
        y = 0.25 * s
        z = (m[1, 2] + m[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1])
        w = (m[1, 0] - m[0, 1]) / s
        x = (m[0, 2] + m[2, 0]) / s
        y = (m[1, 2] + m[2, 1]) / s
        z = 0.25 * s
    return np.array([w, x, y, z], dtype=np.float64)


# --------------------------------------------------------------------------- #
# Public entry point
# --------------------------------------------------------------------------- #


def simulate_with_coefficients(
    theta: npt.NDArray[np.float64],
    options: SimOptions | None = None,
    initial_pose: dict[str, Any] | None = None,
) -> SimOut:
    """Forward-simulate the golf humanoid with polynomial joint torques.

    Implements the canonical ``simulate_with_coefficients`` contract from
    the cross-engine and OpenSim parity specs.

    Args:
        theta: ``(n_joints * 7,)`` flat coefficient vector. Each chunk
            of 7 entries is the polynomial coefficients for one
            ``CoordinateActuator`` in the order they appear in the
            model's ForceSet.
        options: Integrator settings. ``None`` uses defaults (1 s window,
            1 ms reporting timestep, RK Merson).
        initial_pose: Optional initial state. Recognized keys:

            * ``"q"``: joint configuration vector ``(n_coords,)`` —
              defaults to the model's default coordinates.
            * ``"qd"``: joint velocity vector ``(n_coords,)`` —
              defaults to zeros.

    Returns:
        Canonical :class:`SimOut`. See class docstring for shapes.

    Raises:
        ValueError: If ``theta`` shape is incompatible with the model,
            or ``initial_pose`` entries are mis-shaped.
        FileNotFoundError: If the ``.osim`` cannot be located.
        ImportError: If ``opensim`` is not installed.
    """
    import opensim as osim  # noqa: PLC0415  -- optional engine dep

    opts = options if options is not None else SimOptions()
    osim_path = _resolve_osim_path(opts.osim_path)
    model = _load_model(osim_path)

    # Optional gravity override.
    if opts.gravity is not None:
        g = np.asarray(opts.gravity, dtype=np.float64)
        model.setGravity(osim.Vec3(float(g[0]), float(g[1]), float(g[2])))

    # ----- Validate theta shape (CROSS_ENGINE_PARITY_SPEC.md §2.2) ----- #
    n_coords = int(model.getNumCoordinates())
    actuator_names = _coordinate_actuator_names(model)
    n_act = len(actuator_names)
    theta_arr = validate_theta(theta, n_joints=n_act)
    coeffs = theta_arr.reshape(n_act, COEFFS_PER_JOINT)

    # ----- Wire the polynomial torque controller ----------------------- #
    Controller = _make_controller_class()
    controller = Controller(coeffs=coeffs, actuator_names=actuator_names)
    actuator_set = model.getActuators()
    control_indices = []
    for name in actuator_names:
        actuator = actuator_set.get(name)
        controller.addActuator(actuator)
        # Each CoordinateActuator has exactly one control input.
        # We rely on insertion order matching getActuators() order;
        # fall back to a global control-vector index probe below.
        control_indices.append(_actuator_global_control_index(model, name))
    controller._bind_indices(control_indices)
    model.addController(controller)

    # ----- Build the system + state ------------------------------------ #
    state = model.initSystem()

    # Apply initial_pose overrides AFTER initSystem (mutating Coordinate
    # values on the live SimTK::State is the supported path; setting the
    # default values must happen before initSystem and would invalidate
    # the cache).
    _apply_initial_pose(model, state, initial_pose)

    # ----- Allocate output buffers ------------------------------------- #
    n_steps = int(round(opts.t_final / opts.dt))
    if not np.isclose(n_steps * opts.dt, opts.t_final, rtol=1e-9, atol=1e-12):
        n_steps = int(np.ceil(opts.t_final / opts.dt))
    n_samples = n_steps + 1

    time_grid = np.arange(n_samples, dtype=np.float64) * opts.dt
    q_traj = np.empty((n_samples, n_coords), dtype=np.float64)
    qd_traj = np.empty((n_samples, n_coords), dtype=np.float64)
    qdd_traj = np.empty((n_samples, n_coords), dtype=np.float64)
    tau_traj = np.empty((n_samples, n_act), dtype=np.float64)
    grip_pos = np.empty((n_samples, 3), dtype=np.float64)
    grip_quat = np.empty((n_samples, 4), dtype=np.float64)
    club_pos = np.empty((n_samples, 3), dtype=np.float64)
    club_quat = np.empty((n_samples, 4), dtype=np.float64)

    # ----- Integrate sample-by-sample using osim.Manager --------------- #
    manager = osim.Manager(model)
    manager.setIntegratorAccuracy(opts.accuracy)
    state.setTime(0.0)
    manager.initialize(state)

    solver_status = "success"
    t_start = _time.perf_counter()

    # Record sample 0 (initial conditions).
    _record_sample(
        model,
        state,
        controller,
        0,
        time_grid,
        q_traj,
        qd_traj,
        qdd_traj,
        tau_traj,
        grip_pos,
        grip_quat,
        club_pos,
        club_quat,
    )

    try:
        for i in range(1, n_samples):
            target_time = float(time_grid[i])
            state = manager.integrate(target_time)
            _record_sample(
                model,
                state,
                controller,
                i,
                time_grid,
                q_traj,
                qd_traj,
                qdd_traj,
                tau_traj,
                grip_pos,
                grip_quat,
                club_pos,
                club_quat,
            )
    except RuntimeError as err:  # pragma: no cover -- exercised by stress tests
        solver_status = "failed"
        # Pad un-recorded samples with NaN so postcondition shape holds.
        for buf in (
            q_traj,
            qd_traj,
            qdd_traj,
            tau_traj,
            grip_pos,
            grip_quat,
            club_pos,
            club_quat,
        ):
            buf[i:] = np.nan  # noqa: PLW2901 -- intentional fill
        meta_err: dict[str, Any] = {"integrator_error": str(err)}
    else:
        meta_err = {}

    duration_s = _time.perf_counter() - t_start

    meta: dict[str, Any] = {
        "n_actuators": n_act,
        "n_coords": n_coords,
        "actuator_names": actuator_names,
        "grip_frame": GRIP_FRAME_NAME,
        "clubhead_frame": CLUBHEAD_FRAME_NAME,
        "osim_path": str(osim_path),
        "integrator": opts.integrator,
        "accuracy": float(opts.accuracy),
        "dt": float(opts.dt),
        "t_final": float(opts.t_final),
        "n_steps": int(n_steps),
        **meta_err,
    }

    return SimOut(
        time=time_grid,
        q=q_traj,
        qd=qd_traj,
        qdd=qdd_traj,
        tau=tau_traj,
        grip=grip_pos,
        grip_quat=grip_quat,
        clubhead=club_pos,
        club_quat=club_quat,
        solver_status=solver_status,
        duration_s=float(duration_s),
        meta=meta,
    )


# --------------------------------------------------------------------------- #
# Internal helpers
# --------------------------------------------------------------------------- #


def _coordinate_actuator_names(model: Any) -> list[str]:
    """Return the names of every ``CoordinateActuator`` in insertion order.

    The controller's torque coefficients are paired with actuators in
    the order returned here, so the result is also the canonical
    ``theta`` chunk ordering.
    """
    names: list[str] = []
    actuator_set = model.getActuators()
    for i in range(actuator_set.getSize()):
        a = actuator_set.get(i)
        # Filter to CoordinateActuators only (skip muscles, contact forces, etc.).
        # We use the type-name comparison rather than ``isinstance`` because
        # OpenSim's SWIG bindings sometimes downcast.
        cls_name = type(a).__name__
        if cls_name == "CoordinateActuator":
            names.append(a.getName())
        else:
            # Fallback: include any actuator whose name starts with "tau_"
            # which is the convention used by the canonical golf_humanoid.osim.
            try:
                a_name = a.getName()
            except Exception:  # noqa: BLE001 -- defensive against SWIG quirks
                continue
            if a_name.startswith("tau_"):
                names.append(a_name)
    return names


def _actuator_global_control_index(model: Any, name: str) -> int:
    """Return the model-global control-vector index for an actuator.

    For ``CoordinateActuator`` the actuator has exactly one scalar
    control. We probe the model's controls cache by walking actuators
    in the order OpenSim assigns control indices (insertion order in
    ForceSet).
    """
    actuator_set = model.getActuators()
    offset = 0
    for i in range(actuator_set.getSize()):
        a = actuator_set.get(i)
        a_name = a.getName()
        n_ctrls = a.numControls()
        if a_name == name:
            return offset
        offset += n_ctrls
    raise KeyError(f"actuator {name!r} not found in model")


def _apply_initial_pose(
    model: Any,
    state: Any,
    initial_pose: dict[str, Any] | None,
) -> None:
    """Apply optional ``q`` / ``qd`` overrides onto a live SimTK state."""
    if initial_pose is None:
        return
    coord_set = model.getCoordinateSet()
    n = coord_set.getSize()

    q_in = initial_pose.get("q")
    if q_in is not None:
        q_arr = np.asarray(q_in, dtype=np.float64)
        if q_arr.shape != (n,):
            msg = f"initial_pose['q'] has shape {q_arr.shape}; expected ({n},)"
            raise ValueError(msg)
        for i in range(n):
            coord_set.get(i).setValue(state, float(q_arr[i]), False)

    qd_in = initial_pose.get("qd")
    if qd_in is not None:
        qd_arr = np.asarray(qd_in, dtype=np.float64)
        if qd_arr.shape != (n,):
            msg = f"initial_pose['qd'] has shape {qd_arr.shape}; expected ({n},)"
            raise ValueError(msg)
        for i in range(n):
            coord_set.get(i).setSpeedValue(state, float(qd_arr[i]))

    # Re-assemble the multibody system after coordinate edits.
    model.assemble(state)


def _record_sample(  # noqa: PLR0913 -- buffers required for vectorised fill
    model: Any,
    state: Any,
    controller: Any,
    i: int,
    time_grid: npt.NDArray[np.float64],
    q_traj: npt.NDArray[np.float64],
    qd_traj: npt.NDArray[np.float64],
    qdd_traj: npt.NDArray[np.float64],
    tau_traj: npt.NDArray[np.float64],
    grip_pos: npt.NDArray[np.float64],
    grip_quat: npt.NDArray[np.float64],
    club_pos: npt.NDArray[np.float64],
    club_quat: npt.NDArray[np.float64],
) -> None:
    """Snapshot ``(q, qd, qdd, tau, grip, clubhead)`` into output buffers."""
    # Realize to acceleration so qdd is valid.
    model.realizeAcceleration(state)

    coord_set = model.getCoordinateSet()
    n = coord_set.getSize()
    for j in range(n):
        c = coord_set.get(j)
        q_traj[i, j] = c.getValue(state)
        qd_traj[i, j] = c.getSpeedValue(state)
        qdd_traj[i, j] = c.getAccelerationValue(state)

    # Sample torque from the controller's polynomial law (cheaper than
    # walking the actuator force outputs).
    t_now = float(state.getTime())
    tau_traj[i, :] = evaluate_polynomial_torque(controller.get_theta(), t_now)

    # Mirror the chronology metadata.
    time_grid[i]  # noqa: B018 -- index check
    pose = extract_full_pose(state, model)
    grip_pos[i] = pose["grip"]
    grip_quat[i] = pose["grip_quat"]
    club_pos[i] = pose["clubhead"]
    club_quat[i] = pose["club_quat"]


__all__ = [
    "POLY_DEGREE",
    "COEFFS_PER_JOINT",
    "GRIP_FRAME_NAME",
    "CLUBHEAD_FRAME_NAME",
    "SimOptions",
    "SimOut",
    "evaluate_polynomial_torque",
    "extract_full_pose",
    "simulate_with_coefficients",
]

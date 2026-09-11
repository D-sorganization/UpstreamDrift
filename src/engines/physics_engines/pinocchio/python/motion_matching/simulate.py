"""RK4 + ABA forward simulator for the Pinocchio golfer model.

This module implements ``simulate_with_coefficients`` -- the canonical
engine-agnostic forward-sim wrapper specified in
``PINOCCHIO_PARITY_SPEC.md`` (sec 2.2) and the cross-engine spec (sec 2.2).

The entry point is :func:`simulate_with_coefficients`, which:

1. Lazily builds and caches a ``pin.Model`` from ``golfer.urdf``.
2. Reshapes the flat coefficient vector ``theta`` into one degree-6
   polynomial of joint torque per actuated joint (power or Bernstein basis).
3. Integrates the equations of motion with classical RK4 over a window
   ``[0, t_final]`` at fixed timestep ``dt``, using
   ``pin.aba(model, data, q, qd, tau)`` for forward dynamics at each
   stage.
4. Re-walks the trajectory once with ``pin.computeForwardKinematics`` /
   ``updateFramePlacements`` to extract the canonical ``grip``
   (``mid_hands``) and ``clubhead`` (``club_head``) Cartesian poses and
   quaternions.
5. Packs everything into the canonical :class:`SimOut` matching Simscape.

Pinocchio gotchas (CLAUDE.md, parity spec sec 2.2):

* **Never call** ``pin.computeTotalEnergy``. Always use
  ``pin.computeKineticEnergy`` + ``pin.computePotentialEnergy``
  separately and sum.
* ``pin.Data`` is **not** thread-safe. We hold a single ``pin.Data``
  per call frame; the module-level cache stores only the immutable
  ``pin.Model``.
* ``pin.aba`` mutates ``data`` in place. RK4 copies ``q, qd`` between
  stages so each stage starts from a consistent state.

Performance: target < 100 ms for a 1.0 s swing at 1 kHz on a single
modern CPU core, post-warmup. The hot loop avoids Python-level allocs
beyond the per-step state copies and keeps all heavy lifting in
Pinocchio's C++ core.
"""

from __future__ import annotations

import collections
import threading
from dataclasses import dataclass, field
from math import comb
from pathlib import Path
from typing import Any, Literal

import numpy as np
import numpy.typing as npt

from src.shared.python.math_utils.quaternion import rotmat_to_quat
from src.shared.python.motion_matching.piecewise_polynomial import (
    bernstein_to_power_matrix,
)
from src.shared.python.motion_matching.polynomial_torque import (
    COEFFS_PER_JOINT,
    evaluate_polynomial_torque,
)
from src.shared.python.motion_matching.polynomial_torque import (
    POLY_DEGREE as _POLY_DEGREE,
)
from src.shared.python.motion_matching.validate_theta import validate_theta

POLY_DEGREE: int = _POLY_DEGREE

# Default URDF lives next to the engine; resolved lazily so tests can override.
_DEFAULT_URDF = (
    Path(__file__).resolve().parents[2] / "models" / "generated" / "golfer.urdf"
)

# Canonical frame names exposed in SimOut. Defined here, not in the URDF
# loader, so tests can introspect the contract without importing pinocchio.
GRIP_FRAME_NAME: str = "mid_hands"
CLUBHEAD_FRAME_NAME: str = "club_head"


# --------------------------------------------------------------------------- #
# Runtime availability probe
# --------------------------------------------------------------------------- #


def is_pinocchio_available() -> bool:
    """Return whether a functional Pinocchio C++ runtime is available."""
    try:
        import pinocchio as pin  # noqa: PLC0415
    except ImportError:
        return False
    if type(pin).__module__ == "unittest.mock":
        return False
    builder = getattr(pin, "buildModelFromUrdf", None)
    return callable(builder) and type(builder).__module__ != "unittest.mock"


# --------------------------------------------------------------------------- #
# Polynomial & Quaternion helpers
# --------------------------------------------------------------------------- #


def evaluate_bernstein_torque(
    coeffs: npt.NDArray[np.float64],
    t: float,
    T_s: float = 1.0,
    t0: float = 0.0,
) -> npt.NDArray[np.float64]:
    """Evaluate per-joint degree-6 torques in Bernstein polynomial basis.

    tau_j(t) = sum_{k=0}^6 c_{j,k} B_{k,6}((t - t0) / T_s)
    where B_{k,6}(s) = comb(6, k) * s^k * (1 - s)^(6 - k).

    Args:
        coeffs: 2D array of shape ``(n_joints, 7)`` containing control points.
        t: Time in seconds.
        T_s: Horizon duration in seconds.
        t0: Time offset in seconds.

    Returns:
        1D array of shape ``(n_joints,)`` with applied torques at time ``t``.
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
    if T_s <= 0.0:
        raise ValueError(f"T_s must be positive, got {T_s!r}")

    s = (t - t0) / T_s
    basis = np.array(
        [
            comb(POLY_DEGREE, k) * (s**k) * ((1.0 - s) ** (POLY_DEGREE - k))
            for k in range(COEFFS_PER_JOINT)
        ],
        dtype=np.float64,
    )
    return coeffs_arr @ basis


def _quat_to_rotmat_series(quats: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Convert (N, 4) unit quaternions [w, x, y, z] to (N, 3, 3) rotation matrices."""
    q = np.asarray(quats, dtype=np.float64)
    if q.ndim != 2 or q.shape[1] != 4:
        raise ValueError(f"quats must have shape (N, 4); got {q.shape}")
    n = q.shape[0]
    rots = np.empty((n, 3, 3), dtype=np.float64)
    w = q[:, 0]
    x = q[:, 1]
    y = q[:, 2]
    z = q[:, 3]

    rots[:, 0, 0] = 1.0 - 2.0 * (y * y + z * z)
    rots[:, 0, 1] = 2.0 * (x * y - z * w)
    rots[:, 0, 2] = 2.0 * (x * z + y * w)

    rots[:, 1, 0] = 2.0 * (x * y + z * w)
    rots[:, 1, 1] = 1.0 - 2.0 * (x * x + z * z)
    rots[:, 1, 2] = 2.0 * (y * z - x * w)

    rots[:, 2, 0] = 2.0 * (x * z - y * w)
    rots[:, 2, 1] = 2.0 * (y * z + x * w)
    rots[:, 2, 2] = 1.0 - 2.0 * (x * x + y * y)
    return rots


# --------------------------------------------------------------------------- #
# Public dataclasses
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class SimOptions:
    """Forward-simulation options.

    Attributes:
        t_final: Simulation horizon in seconds. Must be > 0.
        dt: Fixed integrator timestep in seconds. Must satisfy
            ``0 < dt <= t_final``.
        integrator: Integrator scheme. Only ``"rk4"`` is implemented.
        gravity: World gravity vector (m/s^2). ``None`` -> Pinocchio
            default (``[0, 0, -9.80665]``). Pass ``np.zeros(3)`` for the
            energy-conservation test.
        urdf_path: Override the cached URDF location. ``None`` uses the
            packaged ``golfer.urdf``.
        compute_energy: Whether to populate ``SimOut.kinetic_energy`` and
            ``SimOut.potential_energy``. Costs ~2 extra Pinocchio calls
            per saved sample; default ``True``.
        basis: Basis used for torque parametrization: ``"power"`` or
            ``"bernstein"``. Default is ``"power"``.
        T_s: Horizon duration in seconds for Bernstein basis normalization.
            Defaults to ``t_final`` if unspecified.
        t0: Reference start time in seconds. Default 0.0.
        output_rate_hz: Optional output grid sample rate in Hz.
        compute_qdd: Whether to compute generalized accelerations. Default True.
    """

    t_final: float = 1.0
    dt: float = 1e-3
    integrator: Literal["rk4"] = "rk4"
    gravity: npt.NDArray[np.float64] | None = None
    urdf_path: str | Path | None = None
    compute_energy: bool = True
    basis: Literal["power", "bernstein"] = "power"
    T_s: float | None = None
    t0: float = 0.0
    output_rate_hz: float | None = None
    compute_qdd: bool = True

    def __post_init__(self) -> None:  # DbC preconditions, parity spec sec 5.2
        if self.T_s is not None:
            if not (self.T_s > 0):
                msg = f"T_s must be positive, got {self.T_s!r}"
                raise ValueError(msg)
            if self.t_final == 1.0 and self.T_s != 1.0:
                object.__setattr__(self, "t_final", float(self.T_s))
        else:
            object.__setattr__(self, "T_s", float(self.t_final))

        if self.output_rate_hz is not None:
            if not (self.output_rate_hz > 0):
                msg = f"output_rate_hz must be positive, got {self.output_rate_hz!r}"
                raise ValueError(msg)
            if self.dt == 1e-3:
                object.__setattr__(self, "dt", 1.0 / self.output_rate_hz)

        if not (self.t_final > 0):
            msg = f"t_final must be positive, got {self.t_final!r}"
            raise ValueError(msg)
        if not (0 < self.dt <= self.t_final):
            msg = (
                f"dt must satisfy 0 < dt <= t_final; "
                f"got dt={self.dt!r}, t_final={self.t_final!r}"
            )
            raise ValueError(msg)
        if self.integrator != "rk4":
            msg = (
                f"integrator={self.integrator!r} not yet supported; "
                "only 'rk4' is implemented (issue #4118)"
            )
            raise ValueError(msg)
        if self.basis not in ("power", "bernstein"):
            msg = f"basis must be 'power' or 'bernstein', got {self.basis!r}"
            raise ValueError(msg)
        if self.gravity is not None:
            g = np.asarray(self.gravity, dtype=np.float64)
            if g.shape != (3,):
                msg = f"gravity must have shape (3,), got {g.shape!r}"
                raise ValueError(msg)


@dataclass(frozen=True)
class SimOut:
    """Canonical forward-simulation output matching Simscape contract.

    All time series are sampled at ``options.dt`` from ``t=0`` to
    ``t=options.t_final`` inclusive (so length ``n_steps + 1``).

    Attributes:
        time: Sample times, shape ``(n_steps + 1,)``.
        q: Joint configuration trajectory, shape ``(n_steps + 1, nq)``.
        qd: Joint velocity trajectory, shape ``(n_steps + 1, nv)``.
        qdd: Joint acceleration trajectory, shape ``(n_steps + 1, nv)``.
        tau: Applied joint-torque trajectory, shape
            ``(n_steps + 1, nv)``.
        grip: Mid-hands Cartesian position, shape
            ``(n_steps + 1, 3)``, world frame.
        grip_quat: Mid-hands unit quaternions [w, x, y, z], shape
            ``(n_steps + 1, 4)``, world frame.
        clubhead: Club-head Cartesian position, shape
            ``(n_steps + 1, 3)``, world frame.
        club_quat: Club-head unit quaternions [w, x, y, z], shape
            ``(n_steps + 1, 4)``, world frame.
        solver_status: Solver execution status ("success", "warning", or "failed").
        duration_s: Wallclock duration in seconds.
        kinetic_energy: Per-sample kinetic energy (J), shape
            ``(n_steps + 1,)``. Populated when
            ``options.compute_energy=True``.
        potential_energy: Per-sample potential energy (J), shape
            ``(n_steps + 1,)``. Populated when
            ``options.compute_energy=True``.
        meta: Free-form diagnostics (joint count, frame ids, wallclock).
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
    solver_status: str = "success"
    duration_s: float = 0.0
    kinetic_energy: npt.NDArray[np.float64] = field(
        default_factory=lambda: np.zeros(0, dtype=np.float64)
    )
    potential_energy: npt.NDArray[np.float64] = field(
        default_factory=lambda: np.zeros(0, dtype=np.float64)
    )
    meta: dict[str, Any] = field(default_factory=dict)

    def __init__(
        self,
        time: npt.NDArray[np.float64] | None = None,
        q: npt.NDArray[np.float64] | None = None,
        qd: npt.NDArray[np.float64] | None = None,
        qdd: npt.NDArray[np.float64] | None = None,
        tau: npt.NDArray[np.float64] | None = None,
        grip: npt.NDArray[np.float64] | None = None,
        grip_quat: npt.NDArray[np.float64] | None = None,
        clubhead: npt.NDArray[np.float64] | None = None,
        club_quat: npt.NDArray[np.float64] | None = None,
        solver_status: str = "success",
        duration_s: float = 0.0,
        kinetic_energy: npt.NDArray[np.float64] | None = None,
        potential_energy: npt.NDArray[np.float64] | None = None,
        meta: dict[str, Any] | None = None,
        *,
        t: npt.NDArray[np.float64] | None = None,
        grip_position: npt.NDArray[np.float64] | None = None,
        grip_rotation: npt.NDArray[np.float64] | None = None,
        clubhead_position: npt.NDArray[np.float64] | None = None,
        clubhead_rotation: npt.NDArray[np.float64] | None = None,
    ) -> None:
        t_arr = time if time is not None else t
        if t_arr is None:
            raise ValueError("Either 'time' or 't' must be provided")
        t_vec = np.asarray(t_arr, dtype=np.float64)
        n = t_vec.shape[0]

        q_mat = (
            np.asarray(q, dtype=np.float64)
            if q is not None
            else np.zeros((n, 0), dtype=np.float64)
        )
        qd_mat = (
            np.asarray(qd, dtype=np.float64)
            if qd is not None
            else np.zeros((n, 0), dtype=np.float64)
        )
        nv = qd_mat.shape[1] if qd_mat.ndim == 2 else 0

        qdd_mat = (
            np.asarray(qdd, dtype=np.float64)
            if qdd is not None
            else np.zeros((n, nv), dtype=np.float64)
        )
        tau_mat = (
            np.asarray(tau, dtype=np.float64)
            if tau is not None
            else np.zeros((n, nv), dtype=np.float64)
        )

        grip_mat = grip if grip is not None else grip_position
        if grip_mat is None:
            grip_mat = np.zeros((n, 3), dtype=np.float64)
        else:
            grip_mat = np.asarray(grip_mat, dtype=np.float64)

        if grip_quat is not None:
            g_quat = np.asarray(grip_quat, dtype=np.float64)
        elif grip_rotation is not None:
            g_quat = rotmat_to_quat(grip_rotation)
        else:
            g_quat = np.zeros((n, 4), dtype=np.float64)

        head_mat = clubhead if clubhead is not None else clubhead_position
        if head_mat is None:
            head_mat = np.zeros((n, 3), dtype=np.float64)
        else:
            head_mat = np.asarray(head_mat, dtype=np.float64)

        if club_quat is not None:
            c_quat = np.asarray(club_quat, dtype=np.float64)
        elif clubhead_rotation is not None:
            c_quat = rotmat_to_quat(clubhead_rotation)
        else:
            c_quat = np.zeros((n, 4), dtype=np.float64)

        ke_vec = (
            np.asarray(kinetic_energy, dtype=np.float64)
            if kinetic_energy is not None
            else np.zeros(n, dtype=np.float64)
        )
        pe_vec = (
            np.asarray(potential_energy, dtype=np.float64)
            if potential_energy is not None
            else np.zeros(n, dtype=np.float64)
        )
        meta_dict = meta if meta is not None else {}

        object.__setattr__(self, "time", t_vec)
        object.__setattr__(self, "q", q_mat)
        object.__setattr__(self, "qd", qd_mat)
        object.__setattr__(self, "qdd", qdd_mat)
        object.__setattr__(self, "tau", tau_mat)
        object.__setattr__(self, "grip", grip_mat)
        object.__setattr__(self, "grip_quat", g_quat)
        object.__setattr__(self, "clubhead", head_mat)
        object.__setattr__(self, "club_quat", c_quat)
        object.__setattr__(self, "solver_status", solver_status)
        object.__setattr__(self, "duration_s", float(duration_s))
        object.__setattr__(self, "kinetic_energy", ke_vec)
        object.__setattr__(self, "potential_energy", pe_vec)
        object.__setattr__(self, "meta", meta_dict)

    @property
    def t(self) -> npt.NDArray[np.float64]:
        return self.time

    @property
    def grip_position(self) -> npt.NDArray[np.float64]:
        return self.grip

    @property
    def grip_rotation(self) -> npt.NDArray[np.float64]:
        """(N, 3, 3) rotation matrices computed from grip_quat."""
        return _quat_to_rotmat_series(self.grip_quat)

    @property
    def clubhead_position(self) -> npt.NDArray[np.float64]:
        return self.clubhead

    @property
    def clubhead_rotation(self) -> npt.NDArray[np.float64]:
        """(N, 3, 3) rotation matrices computed from club_quat."""
        return _quat_to_rotmat_series(self.club_quat)


# --------------------------------------------------------------------------- #
# Model cache (process-local; pinocchio.Data is not thread-safe so we
# only cache the immutable Model and rebuild Data per call).
# --------------------------------------------------------------------------- #


_MODEL_CACHE: dict[str, Any] = {}
# Per-model locks: keyed by resolved URDF path string.
# Using defaultdict avoids a single global lock for all models,
# which would serialize concurrent loads of different URDF files.
_MODEL_CACHE_LOCKS: dict[str, threading.Lock] = collections.defaultdict(threading.Lock)


def _resolve_urdf_path(urdf_path: str | Path | None) -> Path:
    if urdf_path is None:
        return _DEFAULT_URDF
    return Path(urdf_path)


def _get_cached_model(urdf_path: Path) -> Any:
    """Build and cache the Pinocchio model for the given URDF."""
    if not is_pinocchio_available():
        msg = (
            "Pinocchio C++ runtime (with buildModelFromUrdf) is required. "
            "Install pinocchio bindings to use this function."
        )
        raise ImportError(msg)

    import pinocchio as pin  # noqa: PLC0415

    key = str(urdf_path.resolve())
    with _MODEL_CACHE_LOCKS[key]:
        cached = _MODEL_CACHE.get(key)
        if cached is not None:
            return cached
        if not urdf_path.exists():
            msg = f"URDF not found at {urdf_path}"
            raise FileNotFoundError(msg)
        model = pin.buildModelFromUrdf(str(urdf_path))
        _MODEL_CACHE[key] = model
        return model


def _frame_id(model: Any, name: str) -> int:
    """Return the frame id; helpful error if the URDF is stale."""
    if not model.existFrame(name):
        msg = (
            f"URDF is missing canonical frame {name!r}. Cherry-pick "
            "PIN-MODEL-GRIP-FRAME (#4112) before running this simulator."
        )
        raise RuntimeError(msg)
    return int(model.getFrameId(name))


# --------------------------------------------------------------------------- #
# Pinocchio dynamics helpers
# --------------------------------------------------------------------------- #


def _aba_qdd(
    pin_mod: Any,
    model: Any,
    data: Any,
    q: npt.NDArray[np.float64],
    qd: npt.NDArray[np.float64],
    tau: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """Forward dynamics via ABA. Returns a *copy* of qdd (data is reused)."""
    pin_mod.aba(model, data, q, qd, tau)
    # Copy because pin_mod.aba mutates data.ddq on the next call.
    return np.asarray(data.ddq, dtype=np.float64).copy()


def _rk4_step(
    pin_mod: Any,
    model: Any,
    data: Any,
    coeffs: npt.NDArray[np.float64],
    q: npt.NDArray[np.float64],
    qd: npt.NDArray[np.float64],
    t: float,
    dt: float,
) -> tuple[
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
]:
    """Classical RK4 step. Returns (q_next, qd_next, tau_at_start, k1_qdd)."""
    # Stage 1 (at t)
    tau1 = evaluate_polynomial_torque(coeffs, t)
    k1_qd = qd
    k1_qdd = _aba_qdd(pin_mod, model, data, q, qd, tau1)

    # Stage 2 (at t + dt/2)
    half = 0.5 * dt
    q2 = q + half * k1_qd
    qd2 = qd + half * k1_qdd
    tau2 = evaluate_polynomial_torque(coeffs, t + half)
    k2_qd = qd2
    k2_qdd = _aba_qdd(pin_mod, model, data, q2, qd2, tau2)

    # Stage 3 (at t + dt/2)
    q3 = q + half * k2_qd
    qd3 = qd + half * k2_qdd
    k3_qd = qd3
    k3_qdd = _aba_qdd(pin_mod, model, data, q3, qd3, tau2)

    # Stage 4 (at t + dt)
    q4 = q + dt * k3_qd
    qd4 = qd + dt * k3_qdd
    tau4 = evaluate_polynomial_torque(coeffs, t + dt)
    k4_qd = qd4
    k4_qdd = _aba_qdd(pin_mod, model, data, q4, qd4, tau4)

    sixth = dt / 6.0
    q_next = q + sixth * (k1_qd + 2.0 * k2_qd + 2.0 * k3_qd + k4_qd)
    qd_next = qd + sixth * (k1_qdd + 2.0 * k2_qdd + 2.0 * k3_qdd + k4_qdd)
    return q_next, qd_next, tau1, k1_qdd


# --------------------------------------------------------------------------- #
# Public entry point
# --------------------------------------------------------------------------- #


def simulate_with_coefficients(  # noqa: C901
    theta: npt.NDArray[np.float64],
    options: SimOptions | None = None,
    initial_pose: dict[str, Any] | None = None,
) -> SimOut:
    """Forward-simulate the golfer + club system with polynomial torques.

    Implements the canonical engine-agnostic ``simulate_with_coefficients``
    contract from the cross-engine and Pinocchio parity specs.

    Args:
        theta: Flat coefficient vector of shape ``(n_joints * 7,)`` such
            that ``theta.reshape(n_joints, 7)[j, k]`` is the coefficient of
            joint ``j``'s torque polynomial (power or Bernstein basis).
            ``n_joints`` must equal ``model.nv`` for the loaded URDF.
        options: Integrator settings. ``None`` uses defaults (1 s window,
            1 ms timestep, RK4, default gravity, power basis).
        initial_pose: Optional initial state. Keys:

            * ``"q"``: configuration vector, shape ``(model.nq,)``.
              Defaults to ``pin.neutral(model)``.
            * ``"qd"``: velocity vector, shape ``(model.nv,)``. Defaults
              to ``np.zeros(model.nv)``.

    Returns:
        Canonical :class:`SimOut` matching Simscape contract.

    Raises:
        ValueError: If ``theta`` is the wrong shape, ``initial_pose``
            entries are mis-shaped, or any state becomes non-finite
            during integration.
        FileNotFoundError: If the URDF cannot be located.
        RuntimeError: If the URDF is missing the canonical ``mid_hands``
            or ``club_head`` frames.
        ImportError: If functional ``pinocchio`` C++ runtime is not installed.
    """
    import time as _time  # noqa: PLC0415

    if not is_pinocchio_available():
        msg = (
            "Pinocchio C++ runtime (with buildModelFromUrdf) is required for "
            "simulate_with_coefficients. Install via conda or python bindings."
        )
        raise ImportError(msg)

    import pinocchio as pin  # noqa: PLC0415

    opts = options if options is not None else SimOptions()
    urdf_path = _resolve_urdf_path(opts.urdf_path)
    model = _get_cached_model(urdf_path)

    if opts.gravity is not None:
        model = model.copy()
        model.gravity.linear = np.asarray(opts.gravity, dtype=np.float64)

    data = model.createData()

    # ----- Validate theta shape (CROSS_ENGINE_PARITY_SPEC.md §2.2) ----- #
    n_joints = int(model.nv)
    theta_arr = validate_theta(theta, n_joints=n_joints)
    coeffs = theta_arr.reshape(n_joints, COEFFS_PER_JOINT)

    # Convert Bernstein control points to equivalent power basis for fast Horner RK4 evaluation
    if opts.basis == "bernstein":
        m_bern = bernstein_to_power_matrix(POLY_DEGREE)
        p_coeffs = coeffs @ m_bern
        t_span = opts.T_s if opts.T_s is not None else opts.t_final
        scale_powers = t_span ** np.arange(COEFFS_PER_JOINT, dtype=np.float64)
        rk4_coeffs = p_coeffs / scale_powers[None, :]
    else:
        rk4_coeffs = coeffs

    # ----- Initial state ----------------------------------------------- #
    if initial_pose is None:
        q0 = pin.neutral(model)
        qd0 = np.zeros(n_joints, dtype=np.float64)
    else:
        q0_in = initial_pose.get("q")
        qd0_in = initial_pose.get("qd")
        q0 = (
            np.asarray(q0_in, dtype=np.float64).copy()
            if q0_in is not None
            else pin.neutral(model)
        )
        qd0 = (
            np.asarray(qd0_in, dtype=np.float64).copy()
            if qd0_in is not None
            else np.zeros(n_joints, dtype=np.float64)
        )
    if q0.shape != (model.nq,):
        msg = f"initial_pose['q'] has shape {q0.shape}; expected ({model.nq},)"
        raise ValueError(msg)
    if qd0.shape != (n_joints,):
        msg = f"initial_pose['qd'] has shape {qd0.shape}; expected ({n_joints},)"
        raise ValueError(msg)

    # ----- Allocate output buffers ------------------------------------- #
    n_steps = int(round(opts.t_final / opts.dt))
    if not np.isclose(n_steps * opts.dt, opts.t_final, rtol=1e-9, atol=1e-12):
        n_steps = int(np.ceil(opts.t_final / opts.dt))
    n_samples = n_steps + 1

    t_grid = np.arange(n_samples, dtype=np.float64) * opts.dt
    q_traj = np.empty((n_samples, model.nq), dtype=np.float64)
    qd_traj = np.empty((n_samples, n_joints), dtype=np.float64)
    qdd_traj = np.empty((n_samples, n_joints), dtype=np.float64)
    tau_traj = np.empty((n_samples, n_joints), dtype=np.float64)

    grip_pos = np.empty((n_samples, 3), dtype=np.float64)
    grip_rot = np.empty((n_samples, 3, 3), dtype=np.float64)
    clubhead_pos = np.empty((n_samples, 3), dtype=np.float64)
    clubhead_rot = np.empty((n_samples, 3, 3), dtype=np.float64)
    if opts.compute_energy:
        ke = np.empty((n_samples,), dtype=np.float64)
        pe = np.empty((n_samples,), dtype=np.float64)
    else:
        ke = np.zeros((n_samples,), dtype=np.float64)
        pe = np.zeros((n_samples,), dtype=np.float64)

    # ----- Integrate ---------------------------------------------------- #
    q_traj[0] = q0
    qd_traj[0] = qd0
    tau_traj[0] = evaluate_polynomial_torque(rk4_coeffs, 0.0)

    t_start = _time.perf_counter()

    q = q0.copy()
    qd = qd0.copy()
    for i in range(1, n_samples):
        t_i = t_grid[i - 1] - opts.t0
        q, qd, tau_i, k1_qdd = _rk4_step(
            pin, model, data, rk4_coeffs, q, qd, t_i, opts.dt
        )
        if not (np.all(np.isfinite(q)) and np.all(np.isfinite(qd))):
            msg = (
                f"Integration diverged at step {i} (t={t_grid[i]:.4f}); "
                "non-finite q/qd. Reduce dt, damp the polynomial, or "
                "tighten initial_pose."
            )
            raise ValueError(msg)
        q_traj[i] = q
        qd_traj[i] = qd
        tau_traj[i] = tau_i
        qdd_traj[i - 1] = k1_qdd

    qdd_traj[-1] = _aba_qdd(pin, model, data, q, qd, tau_traj[-1])

    integrate_secs = _time.perf_counter() - t_start

    # ----- Re-walk for FK + energy ------------------------------------- #
    grip_id = _frame_id(model, GRIP_FRAME_NAME)
    clubhead_id = _frame_id(model, CLUBHEAD_FRAME_NAME)

    fk_start = _time.perf_counter()
    for i in range(n_samples):
        q_i = q_traj[i]
        pin.forwardKinematics(model, data, q_i, qd_traj[i])
        pin.updateFramePlacements(model, data)
        grip_pos[i] = data.oMf[grip_id].translation
        grip_rot[i] = data.oMf[grip_id].rotation
        clubhead_pos[i] = data.oMf[clubhead_id].translation
        clubhead_rot[i] = data.oMf[clubhead_id].rotation
        if opts.compute_energy:
            # CLAUDE.md gotcha: NEVER use pin.computeTotalEnergy.
            ke[i] = float(pin.computeKineticEnergy(model, data, q_i, qd_traj[i]))
            pe[i] = float(pin.computePotentialEnergy(model, data, q_i))
    fk_secs = _time.perf_counter() - fk_start

    meta = {
        "n_joints": n_joints,
        "model_nq": int(model.nq),
        "model_nv": int(model.nv),
        "grip_frame": GRIP_FRAME_NAME,
        "clubhead_frame": CLUBHEAD_FRAME_NAME,
        "grip_frame_id": int(grip_id),
        "clubhead_frame_id": int(clubhead_id),
        "urdf_path": str(urdf_path),
        "integrator": opts.integrator,
        "dt": float(opts.dt),
        "t_final": float(opts.t_final),
        "n_steps": int(n_steps),
        "wallclock_integrate_s": float(integrate_secs),
        "wallclock_fk_s": float(fk_secs),
        "wallclock_total_s": float(integrate_secs + fk_secs),
        "compute_energy": bool(opts.compute_energy),
        "basis": str(opts.basis),
    }

    return SimOut(
        time=t_grid,
        q=q_traj,
        qd=qd_traj,
        qdd=qdd_traj,
        tau=tau_traj,
        grip=grip_pos,
        grip_quat=rotmat_to_quat(grip_rot),
        clubhead=clubhead_pos,
        club_quat=rotmat_to_quat(clubhead_rot),
        solver_status="success",
        duration_s=float(integrate_secs + fk_secs),
        kinetic_energy=ke,
        potential_energy=pe,
        meta=meta,
    )

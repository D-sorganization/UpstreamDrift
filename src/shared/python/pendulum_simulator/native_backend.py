"""Optional Rust-backed kernels for pendulum simulation.

This module adapts the compiled ``pendulum_core`` extension to the desktop
Python APIs. Native execution is intentionally opt-in and model-specific so the
pure-Python implementations remain the contract-preserving fallback path.
"""

from __future__ import annotations

import logging
import os
import warnings
from typing import TYPE_CHECKING, Any

import numpy as np

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from .physics import PendulumParams
    from .physics_golfer import GolferParams
    from .physics_triple import TriplePendulumParams

_DOUBLE_BACKEND_ENV = "PENDULUM_DOUBLE_BACKEND"
_TRIPLE_BACKEND_ENV = "PENDULUM_TRIPLE_BACKEND"
_GOLFER_BACKEND_ENV = "PENDULUM_GOLFER_BACKEND"
_WARNED_CALLS: set[str] = set()

try:
    import pendulum_core as _pendulum_core

    _NATIVE_IMPORT_ERROR: str | None = None
except ImportError as exc:
    _pendulum_core = None
    _NATIVE_IMPORT_ERROR = str(exc)


def _warn_once(call_name: str, exc: Exception) -> None:
    """Warn once per native call site, then fall back to Python."""
    if not (call_name is not None):
        raise ValueError("call_name must be provided")
    if call_name in _WARNED_CALLS:
        return
    _WARNED_CALLS.add(call_name)
    warnings.warn(
        (
            f"Rust backend call '{call_name}' failed with "
            f"{exc!r}; falling back to the Python implementation."
        ),
        RuntimeWarning,
        stacklevel=2,
    )


def _truncate_q(q: np.ndarray) -> np.ndarray:
    """Normalize q to the 8 generalized coordinates expected by the Rust core."""
    q_arr = np.asarray(q, dtype=float)
    if q_arr.shape[0] > 8:
        q_arr = q_arr[:8]
    if not (q_arr.shape == (8):
        raise ValueError(), f"q must have shape (8,), got {q_arr.shape}")
    return q_arr


def _vector8(values: np.ndarray, name: str) -> np.ndarray:
    """Normalize a vector argument to a finite length-8 array."""
    arr = np.asarray(values, dtype=float)
    if not (arr.shape == (8):
        raise ValueError(), f"{name} must have shape (8,), got {arr.shape}")
    if not (np.all(np.isfinite(arr))):
        raise ValueError(f"{name} must be finite")
    return arr


def _backend_mode(env_name: str) -> str:
    """Return the configured backend mode for a model.

    Default is ``"auto"``, which selects Rust when ``pendulum_core`` is
    importable and falls back to Python otherwise.  Set the environment
    variable to ``"python"`` or ``"rust"`` to force a specific backend.
    """
    mode = os.getenv(env_name, "auto").strip().lower()
    if mode == "auto":
        return "rust" if _pendulum_core is not None else "python"
    return mode if mode in {"python", "rust"} else "python"


def golfer_native_constraint_dynamics_supported(params: GolferParams) -> bool:
    """Whether native constrained dynamics matches the Python model assumptions."""
    return (
        params.b_hub == 0.0
        and params.b_rs == 0.0
        and params.b_re == 0.0
        and params.b_rh == 0.0
        and params.b_ls == 0.0
        and params.b_le == 0.0
        and params.b_lh == 0.0
    )


def golfer_backend_mode() -> str:
    """Return the configured golfer backend mode."""
    return _backend_mode(_GOLFER_BACKEND_ENV)


def double_backend_mode() -> str:
    """Return the configured double-pendulum backend mode."""
    return _backend_mode(_DOUBLE_BACKEND_ENV)


def triple_backend_mode() -> str:
    """Return the configured triple-pendulum backend mode."""
    return _backend_mode(_TRIPLE_BACKEND_ENV)


def golfer_native_available() -> bool:
    """Whether the compiled ``pendulum_core`` Python extension is importable."""
    return _pendulum_core is not None


def golfer_native_enabled() -> bool:
    """Whether golfer kernels should use the Rust extension."""
    return golfer_backend_mode() == "rust" and golfer_native_available()


def double_native_enabled() -> bool:
    """Whether double-pendulum kernels should use the Rust extension."""
    return double_backend_mode() == "rust" and golfer_native_available()


def triple_native_enabled() -> bool:
    """Whether triple-pendulum kernels should use the Rust extension."""
    return triple_backend_mode() == "rust" and golfer_native_available()


def get_native_backend_info() -> dict[str, object]:
    """Return backend configuration and availability details."""
    return {
        "configured_backend": {
            "double": double_backend_mode(),
            "triple": triple_backend_mode(),
            "golfer": golfer_backend_mode(),
        },
        "native_available": golfer_native_available(),
        "native_import_error": _NATIVE_IMPORT_ERROR,
        "supported_models": {"golfer": True, "double": True, "triple": True},
        "supports_constraint_dynamics": True,
    }


def _to_rust_double_params(params: PendulumParams) -> Any:
    """Convert the Python double-pendulum params dataclass to the PyO3 wrapper."""
    if _pendulum_core is None:
        raise RuntimeError("pendulum_core is not available")

    return _pendulum_core.PyDoublePendulumParams(
        params.m1,
        params.m2,
        params.L1,
        params.L2,
        params.g,
        params.b1,
        params.b2,
        params.mClub,
    )


def _to_rust_triple_params(params: TriplePendulumParams) -> Any:
    """Convert the Python triple-pendulum params dataclass to the PyO3 wrapper."""
    if _pendulum_core is None:
        raise RuntimeError("pendulum_core is not available")

    return _pendulum_core.PyTriplePendulumParams(
        params.m1,
        params.m2,
        params.m3,
        params.L1,
        params.L2,
        params.L3,
        params.g,
        params.b1,
        params.b2,
        params.b3,
    )


def _to_rust_golfer_params(params: GolferParams) -> Any:
    """Convert the Python golfer params dataclass to the PyO3 wrapper type."""
    if _pendulum_core is None:
        raise RuntimeError("pendulum_core is not available")

    return _pendulum_core.PyGolferParams(
        params.L_hub,
        params.m_hub,
        params.d_rs,
        params.d_ls,
        params.L_r_upper,
        params.m_r_upper,
        params.L_r_fore,
        params.m_r_fore,
        params.L_l_upper,
        params.m_l_upper,
        params.L_l_fore,
        params.m_l_fore,
        params.L_club,
        params.m_club,
        params.m_clubhead,
        params.grip_right,
        params.grip_left,
        params.g,
    )


def double_mass_matrix(phi: float, params: PendulumParams) -> np.ndarray | None:
    """Return the native double-pendulum mass matrix, or ``None`` if unavailable."""
    if not (phi is not None):
        raise ValueError("phi must be provided")
    if not double_native_enabled():
        return None

    try:
        q_arr = np.array([0.0, phi], dtype=float)
        result = _pendulum_core.py_double_mass_matrix(
            q_arr.tolist(), _to_rust_double_params(params)
        )
        return np.array(result, dtype=float)
    except (
        RuntimeError,
        AttributeError,
        TypeError,
    ) as exc:  # pragma: no cover - exercised when extension exists
        logger.debug(
            "double_mass_matrix: Rust call failed (%s), falling back to Python",
            type(exc).__name__,
        )
        _warn_once("double_mass_matrix", exc)
        return None


def double_gravity_vector(
    theta1: float, phi: float, params: PendulumParams
) -> np.ndarray | None:
    """Return the native double-pendulum gravity vector, or ``None`` if unavailable."""
    if not (theta1 is not None):
        raise ValueError("theta1 must be provided")
    if not double_native_enabled():
        return None

    try:
        q_arr = np.array([theta1, phi], dtype=float)
        result = _pendulum_core.py_double_gravity_vector(
            q_arr.tolist(), _to_rust_double_params(params)
        )
        return np.array(result, dtype=float)
    except (
        RuntimeError,
        AttributeError,
        TypeError,
    ) as exc:  # pragma: no cover - exercised when extension exists
        logger.debug(
            "double_gravity_vector: Rust call failed (%s), falling back to Python",
            type(exc).__name__,
        )
        _warn_once("double_gravity_vector", exc)
        return None


def double_coriolis_vector(
    phi: float,
    dtheta1: float,
    dphi: float,
    params: PendulumParams,
) -> np.ndarray | None:
    """Return the native double-pendulum Coriolis vector, or ``None`` if unavailable."""
    if not (phi is not None):
        raise ValueError("phi must be provided")
    if not double_native_enabled():
        return None

    try:
        q_arr = np.array([0.0, phi], dtype=float)
        qdot_arr = np.array([dtheta1, dphi], dtype=float)
        result = _pendulum_core.py_double_coriolis(
            q_arr.tolist(),
            qdot_arr.tolist(),
            _to_rust_double_params(params),
        )
        return np.array(result, dtype=float)
    except (
        RuntimeError,
        AttributeError,
        TypeError,
    ) as exc:  # pragma: no cover - exercised when extension exists
        logger.debug(
            "double_coriolis_vector: Rust call failed (%s), falling back to Python",
            type(exc).__name__,
        )
        _warn_once("double_coriolis_vector", exc)
        return None


def double_forward_kinematics(
    theta1: float, phi: float, params: PendulumParams
) -> dict[str, tuple[float, float]] | None:
    """Return native double-pendulum forward kinematics mapped to desktop keys."""
    if not (theta1 is not None):
        raise ValueError("theta1 must be provided")
    if not double_native_enabled():
        return None

    try:
        q_arr = np.array([theta1, phi], dtype=float)
        result = _pendulum_core.py_double_forward_kinematics(
            q_arr.tolist(), _to_rust_double_params(params)
        )
    except (
        RuntimeError,
        AttributeError,
        TypeError,
    ) as exc:  # pragma: no cover - exercised when extension exists
        logger.debug(
            "double_forward_kinematics: Rust call failed (%s), falling back to Python",
            type(exc).__name__,
        )
        _warn_once("double_forward_kinematics", exc)
        return None

    return {
        "shoulder": (0.0, 0.0),
        "wrist": (float(result["wrist_x"]), float(result["wrist_y"])),
        "tip": (float(result["club_tip_x"]), float(result["club_tip_y"])),
    }


def triple_mass_matrix(
    phi1: float, phi2: float, params: TriplePendulumParams
) -> np.ndarray | None:
    """Return the native triple-pendulum mass matrix, or ``None`` if unavailable."""
    if not (phi1 is not None):
        raise ValueError("phi1 must be provided")
    if not triple_native_enabled():
        return None

    try:
        q_arr = np.array([0.0, phi1, phi2], dtype=float)
        result = _pendulum_core.py_triple_mass_matrix(
            q_arr.tolist(), _to_rust_triple_params(params)
        )
        return np.array(result, dtype=float)
    except (
        RuntimeError,
        AttributeError,
        TypeError,
    ) as exc:  # pragma: no cover - exercised when extension exists
        logger.debug(
            "triple_mass_matrix: Rust call failed (%s), falling back to Python",
            type(exc).__name__,
        )
        _warn_once("triple_mass_matrix", exc)
        return None


def triple_gravity_vector(
    theta1: float, phi1: float, phi2: float, params: TriplePendulumParams
) -> np.ndarray | None:
    """Return the native triple-pendulum gravity vector, or ``None`` if unavailable."""
    if not (theta1 is not None):
        raise ValueError("theta1 must be provided")
    if not triple_native_enabled():
        return None

    try:
        q_arr = np.array([theta1, phi1, phi2], dtype=float)
        result = _pendulum_core.py_triple_gravity_vector(
            q_arr.tolist(), _to_rust_triple_params(params)
        )
        return np.array(result, dtype=float)
    except (
        RuntimeError,
        AttributeError,
        TypeError,
    ) as exc:  # pragma: no cover - exercised when extension exists
        logger.debug(
            "triple_gravity_vector: Rust call failed (%s), falling back to Python",
            type(exc).__name__,
        )
        _warn_once("triple_gravity_vector", exc)
        return None


def triple_coriolis_vector(
    phi1: float,
    phi2: float,
    dtheta1: float,
    dphi1: float,
    dphi2: float,
    params: TriplePendulumParams,
) -> np.ndarray | None:
    """Return the native triple-pendulum Coriolis vector, or ``None`` if unavailable."""
    if not (phi1 is not None):
        raise ValueError("phi1 must be provided")
    if not triple_native_enabled():
        return None

    try:
        q_arr = np.array([0.0, phi1, phi2], dtype=float)
        qdot_arr = np.array([dtheta1, dphi1, dphi2], dtype=float)
        result = _pendulum_core.py_triple_coriolis(
            q_arr.tolist(),
            qdot_arr.tolist(),
            _to_rust_triple_params(params),
        )
        return np.array(result, dtype=float)
    except (
        RuntimeError,
        AttributeError,
        TypeError,
    ) as exc:  # pragma: no cover - exercised when extension exists
        logger.debug(
            "triple_coriolis_vector: Rust call failed (%s), falling back to Python",
            type(exc).__name__,
        )
        _warn_once("triple_coriolis_vector", exc)
        return None


def triple_forward_kinematics(
    theta1: float,
    phi1: float,
    phi2: float,
    params: TriplePendulumParams,
) -> dict[str, tuple[float, float]] | None:
    """Return native triple-pendulum forward kinematics mapped to desktop keys."""
    if not (theta1 is not None):
        raise ValueError("theta1 must be provided")
    if not triple_native_enabled():
        return None

    try:
        q_arr = np.array([theta1, phi1, phi2], dtype=float)
        result = _pendulum_core.py_triple_forward_kinematics(
            q_arr.tolist(), _to_rust_triple_params(params)
        )
    except (
        RuntimeError,
        AttributeError,
        TypeError,
    ) as exc:  # pragma: no cover - exercised when extension exists
        logger.debug(
            "triple_forward_kinematics: Rust call failed (%s), falling back to Python",
            type(exc).__name__,
        )
        _warn_once("triple_forward_kinematics", exc)
        return None

    return {
        "shoulder": (0.0, 0.0),
        "wrist1": (float(result["joint1_x"]), float(result["joint1_y"])),
        "wrist2": (float(result["joint2_x"]), float(result["joint2_y"])),
        "tip": (float(result["joint3_x"]), float(result["joint3_y"])),
    }


def golfer_mass_matrix(q: np.ndarray, params: GolferParams) -> np.ndarray | None:
    """Return the native golfer mass matrix, or ``None`` if disabled/unavailable."""
    if not (q is not None):
        raise ValueError("q must be provided")
    if not golfer_native_enabled():
        return None

    try:
        q_arr = _truncate_q(q)
        result = _pendulum_core.py_golfer_mass_matrix(
            q_arr.tolist(), _to_rust_golfer_params(params)
        )
        return np.array(result, dtype=float)
    except (
        RuntimeError,
        AttributeError,
        TypeError,
    ) as exc:  # pragma: no cover - exercised when extension exists
        logger.debug(
            "golfer_mass_matrix: Rust call failed (%s), falling back to Python",
            type(exc).__name__,
        )
        _warn_once("golfer_mass_matrix", exc)
        return None


def golfer_gravity_vector(q: np.ndarray, params: GolferParams) -> np.ndarray | None:
    """Return the native golfer gravity vector, or ``None`` if disabled/unavailable."""
    if not (q is not None):
        raise ValueError("q must be provided")
    if not golfer_native_enabled():
        return None

    try:
        q_arr = _truncate_q(q)
        result = _pendulum_core.py_golfer_gravity_vector(
            q_arr.tolist(), _to_rust_golfer_params(params)
        )
        return np.array(result, dtype=float)
    except (
        RuntimeError,
        AttributeError,
        TypeError,
    ) as exc:  # pragma: no cover - exercised when extension exists
        logger.debug(
            "golfer_gravity_vector: Rust call failed (%s), falling back to Python",
            type(exc).__name__,
        )
        _warn_once("golfer_gravity_vector", exc)
        return None


def golfer_forward_kinematics(
    q: np.ndarray, params: GolferParams
) -> dict[str, tuple[float, float]] | None:
    """Return native golfer forward kinematics mapped to Python GUI keys."""
    if not (q is not None):
        raise ValueError("q must be provided")
    if not golfer_native_enabled():
        return None

    try:
        q_arr = _truncate_q(q)
        result = _pendulum_core.py_golfer_forward_kinematics(
            q_arr.tolist(), _to_rust_golfer_params(params)
        )
    except (
        RuntimeError,
        AttributeError,
        TypeError,
    ) as exc:  # pragma: no cover - exercised when extension exists
        logger.debug(
            "golfer_forward_kinematics: Rust call failed (%s), falling back to Python",
            type(exc).__name__,
        )
        _warn_once("golfer_forward_kinematics", exc)
        return None

    theta_club = float(q_arr[7])
    club_dx = float(np.sin(theta_club))
    club_dy = float(-np.cos(theta_club))
    club_base = tuple(result["club_base"])
    grip_left = (
        float(club_base[0] + params.grip_left * club_dx),
        float(club_base[1] - params.grip_left * club_dy),
    )

    return {
        "origin": (0.0, 0.0),
        "hub": tuple(result["hub"]),
        "rs": tuple(result["r_shoulder"]),
        "re": tuple(result["r_elbow"]),
        "rh": tuple(result["r_wrist"]),
        "ls": tuple(result["l_shoulder"]),
        "le": tuple(result["l_elbow"]),
        "lh": tuple(result["l_wrist"]),
        "club_base": tuple(result["club_base"]),
        "club_tip": tuple(result["club_tip"]),
        "grip_right": tuple(result["r_wrist"]),
        "grip_left": grip_left,
    }


def golfer_constrained_dynamics(
    q: np.ndarray,
    qdot: np.ndarray,
    tau: np.ndarray,
    params: GolferParams,
    alpha: float,
    beta: float,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Return native golfer accelerations and multipliers when supported."""
    if not (q is not None):
        raise ValueError("q must be provided")
    if not golfer_native_enabled() or not golfer_native_constraint_dynamics_supported(
        params
    ):
        return None

    try:
        q_arr = _truncate_q(q)
        qdot_arr = _vector8(qdot, "qdot")
        tau_arr = _vector8(tau, "tau")
        qddot, lambda_forces = _pendulum_core.py_golfer_constrained_dynamics(
            q_arr.tolist(),
            qdot_arr.tolist(),
            tau_arr.tolist(),
            _to_rust_golfer_params(params),
            alpha,
            beta,
        )
        return (
            np.array(qddot, dtype=float),
            np.array(lambda_forces, dtype=float),
        )
    except (
        RuntimeError,
        AttributeError,
        TypeError,
    ) as exc:  # pragma: no cover - exercised when extension exists
        logger.debug(
            "golfer_constrained_dynamics: Rust call failed (%s), falling back to Python",
            type(exc).__name__,
        )
        _warn_once("golfer_constrained_dynamics", exc)
        return None


def golfer_project_to_constraints(
    q: np.ndarray,
    params: GolferParams,
    max_iters: int,
    tol: float,
) -> np.ndarray | None:
    """Return native golfer position projection, or ``None`` if unavailable."""
    if not (q is not None):
        raise ValueError("q must be provided")
    if not golfer_native_enabled():
        return None

    try:
        q_arr = _truncate_q(q)
        result = _pendulum_core.py_golfer_project_to_constraints(
            q_arr.tolist(),
            _to_rust_golfer_params(params),
            max_iters,
            tol,
        )
        return np.array(result, dtype=float)
    except (
        RuntimeError,
        AttributeError,
        TypeError,
    ) as exc:  # pragma: no cover - exercised when extension exists
        logger.debug(
            "golfer_project_to_constraints: Rust call failed (%s), falling back to Python",
            type(exc).__name__,
        )
        _warn_once("golfer_project_to_constraints", exc)
        return None


def golfer_project_velocity(
    q: np.ndarray,
    qdot: np.ndarray,
    params: GolferParams,
) -> np.ndarray | None:
    """Return native golfer velocity projection, or ``None`` if unavailable."""
    if not (q is not None):
        raise ValueError("q must be provided")
    if not golfer_native_enabled():
        return None

    try:
        q_arr = _truncate_q(q)
        qdot_arr = _vector8(qdot, "qdot")
        result = _pendulum_core.py_golfer_project_velocity(
            q_arr.tolist(),
            qdot_arr.tolist(),
            _to_rust_golfer_params(params),
        )
        return np.array(result, dtype=float)
    except (
        RuntimeError,
        AttributeError,
        TypeError,
    ) as exc:  # pragma: no cover - exercised when extension exists
        logger.debug(
            "golfer_project_velocity: Rust call failed (%s), falling back to Python",
            type(exc).__name__,
        )
        _warn_once("golfer_project_velocity", exc)
        return None


def batch_evaluate_double(
    params: PendulumParams,
    coeffs_batch: list[list[float]],
    n_coeffs_per_joint: int,
    q0: list[float],
    qdot0: list[float],
    t_end: float,
) -> list[tuple[float, float, bool]] | None:
    """Batch-evaluate polynomial torque profiles via Rust rayon.

    Returns a list of ``(max_tip_speed, tip_speed_at_bottom, success)``
    tuples, or ``None`` if the native backend is unavailable.
    """
    if not (params is not None):
        raise ValueError("params must be provided")
    if _pendulum_core is None or not hasattr(
        _pendulum_core, "py_batch_evaluate_double"
    ):
        return None

    try:
        result: list[tuple[float, float, bool]] = (
            _pendulum_core.py_batch_evaluate_double(
                _to_rust_double_params(params),
                coeffs_batch,
                n_coeffs_per_joint,
                q0,
                qdot0,
                t_end,
            )
        )
        return result
    except (RuntimeError, AttributeError, TypeError) as exc:  # pragma: no cover
        logger.debug(
            "batch_evaluate_double: Rust call failed (%s), falling back to Python",
            type(exc).__name__,
        )
        _warn_once("batch_evaluate_double", exc)
        return None


def simulate_double(
    params: PendulumParams,
    q0: list[float],
    qdot0: list[float],
    coeffs: list[float],
    n_coeffs_per_joint: int,
    t_span: tuple[float, float],
    max_steps: int = 100000,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Run RK45 integration in Rust for the double pendulum.

    Returns ``(times, states)``, where ``times`` is a 1D array and ``states``
    is a 2D array of shape ``(N, 4)`` containing ``[q1, q2, qdot1, qdot2]``.
    Returns ``None`` if the native backend is unavailable.
    """
    if not (params is not None):
        raise ValueError("params must be provided")
    if _pendulum_core is None or not hasattr(_pendulum_core, "py_simulate_double"):
        return None

    try:
        times, states = _pendulum_core.py_simulate_double(
            _to_rust_double_params(params),
            q0,
            qdot0,
            coeffs,
            n_coeffs_per_joint,
            t_span,
            max_steps,
        )
        return np.array(times, dtype=float), np.array(states, dtype=float)
    except (RuntimeError, AttributeError, TypeError) as exc:  # pragma: no cover
        logger.debug(
            "simulate_double: Rust call failed (%s), falling back to Python",
            type(exc).__name__,
        )
        _warn_once("simulate_double", exc)
        return None

"""Rust-backed Hill muscle kernel interface for UpstreamDrift.

This module provides a clean Python facade over the ``upstream_muscle``
Rust binary (built via PyO3/Maturin) for the Hill-type muscle model used
by the biomechanics stack: scalar force-length / force-velocity / tendon
curves, activation dynamics, and the batched RL-inner-loop kernels that
back ``stable_baselines3`` training.

If the Rust wheel is not installed, a graceful fallback to the existing
pure-Python implementations in this directory is provided so callers
never break.

Principles:
- **DRY**: same kernels back the Rust crate and the WASM/RL paths.
- **DbC**: every function validates inputs before forwarding.
- **TDD**: see ``tests/unit/biomechanics/test_rust_muscle.py``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from src.shared.python.logging_pkg.logging_config import get_logger

if TYPE_CHECKING:
    from numpy.typing import NDArray

logger = get_logger(__name__)


# ── Try importing the Rust wheel ─────────────────────────────────────────────

_RUST_AVAILABLE = False

try:
    import upstream_muscle as _rust  # type: ignore[import-untyped]

    _RUST_AVAILABLE = True
    logger.info("upstream_muscle Rust kernel loaded successfully")
except ImportError:
    _rust = None  # type: ignore[assignment]
    logger.info(
        "upstream_muscle Rust wheel not installed — "
        "falling back to pure-Python Hill muscle. "
        "Build with: maturin develop --release -m rust_core/upstream-muscle/Cargo.toml"
    )


def is_rust_available() -> bool:
    """Return True if the Rust muscle kernel is importable."""
    return _RUST_AVAILABLE


# ── Scalar curves ────────────────────────────────────────────────────────────


def f_l(l_norm: float, width: float | None = None) -> float:
    """Active force-length Gaussian curve.

    Args:
        l_norm: Normalised fiber length ``l_CE / l_opt``.
        width: Optional Gaussian width (defaults to 0.56 / Thelen 2003).
    """
    if _RUST_AVAILABLE:
        return float(_rust.f_l(l_norm, width))
    from src.shared.python.biomechanics.hill_muscle import (
        HillMuscleModel,
        MuscleParameters,
    )

    model = HillMuscleModel(
        MuscleParameters(F_max=1.0, l_opt=1.0, l_slack=1.0),
        force_length_width=width
        if width is not None
        else HillMuscleModel.DEFAULT_FORCE_LENGTH_WIDTH,
    )
    return float(model.force_length_active(l_norm))


def f_p(l_norm: float) -> float:
    """Passive (PEE) force-length curve."""
    if _RUST_AVAILABLE:
        return float(_rust.f_p(l_norm))
    from src.shared.python.biomechanics.hill_muscle import (
        HillMuscleModel,
        MuscleParameters,
    )

    model = HillMuscleModel(MuscleParameters(F_max=1.0, l_opt=1.0, l_slack=1.0))
    return float(model.force_length_passive(l_norm))


def f_v(v_norm: float) -> float:
    """Force-velocity (Hill hyperbola + eccentric plateau)."""
    if _RUST_AVAILABLE:
        return float(_rust.f_v(v_norm))
    from src.shared.python.biomechanics.hill_muscle import (
        HillMuscleModel,
        MuscleParameters,
    )

    model = HillMuscleModel(MuscleParameters(F_max=1.0, l_opt=1.0, l_slack=1.0))
    return float(model.force_velocity(v_norm))


def f_t(l_tendon_norm: float) -> float:
    """Tendon (SEE) force-length curve."""
    if _RUST_AVAILABLE:
        return float(_rust.f_t(l_tendon_norm))
    from src.shared.python.biomechanics.hill_muscle import (
        HillMuscleModel,
        MuscleParameters,
    )

    model = HillMuscleModel(MuscleParameters(F_max=1.0, l_opt=1.0, l_slack=1.0))
    return float(model.tendon_force(l_tendon_norm))


# ── Activation dynamics ──────────────────────────────────────────────────────


def activation_step(
    u: float,
    a: float,
    dt: float,
    *,
    tau_act: float | None = None,
    tau_deact: float | None = None,
    min_activation: float | None = None,
) -> float:
    """Advance activation by one Euler step.

    Args:
        u: Neural excitation in ``[0, 1]``.
        a: Current activation in ``[0, 1]``.
        dt: Time step ``[s]``, must be positive.
    """
    if dt <= 0:
        raise ValueError(f"dt must be positive, got {dt}")
    if _RUST_AVAILABLE:
        dyn_rust = _rust.ActivationDynamics(
            tau_act if tau_act is not None else 0.010,
            tau_deact if tau_deact is not None else 0.040,
            min_activation if min_activation is not None else 0.001,
        )
        return float(dyn_rust.update(u, a, dt))
    from src.shared.python.biomechanics.activation_dynamics import ActivationDynamics

    dyn = ActivationDynamics(
        tau_act=tau_act if tau_act is not None else 0.010,
        tau_deact=tau_deact if tau_deact is not None else 0.040,
        min_activation=min_activation if min_activation is not None else 0.001,
    )
    return float(dyn.update(u, a, dt))


def activation_step_batch(
    u: NDArray[np.float64],
    a: NDArray[np.float64],
    dt: float,
    *,
    tau_act: float | None = None,
    tau_deact: float | None = None,
    min_activation: float | None = None,
) -> NDArray[np.float64]:
    """Batched single-Euler-step activation update for ``M`` muscles.

    Both arrays must be 1-D, contiguous, float64, with matching length.
    Falls back to a Python loop if the Rust wheel is absent.
    """
    u_arr = np.ascontiguousarray(u, dtype=np.float64)
    a_arr = np.ascontiguousarray(a, dtype=np.float64)
    if u_arr.shape != a_arr.shape or u_arr.ndim != 1:
        raise ValueError(
            f"u and a must be 1-D arrays of matching shape; got {u_arr.shape} vs {a_arr.shape}"
        )
    if dt <= 0:
        raise ValueError(f"dt must be positive, got {dt}")
    if _RUST_AVAILABLE:
        return np.asarray(
            _rust.activation_step_batch(
                u_arr,
                a_arr,
                dt,
                tau_act=tau_act,
                tau_deact=tau_deact,
                min_activation=min_activation,
            )
        )
    from src.shared.python.biomechanics.activation_dynamics import ActivationDynamics

    dyn = ActivationDynamics(
        tau_act=tau_act if tau_act is not None else 0.010,
        tau_deact=tau_deact if tau_deact is not None else 0.040,
        min_activation=min_activation if min_activation is not None else 0.001,
    )
    out = np.empty_like(u_arr)
    for i in range(u_arr.shape[0]):
        out[i] = dyn.update(float(u_arr[i]), float(a_arr[i]), dt)
    return out


# ── Muscle force ─────────────────────────────────────────────────────────────


def muscle_force_batch(
    activations: NDArray[np.float64],
    l_ce: NDArray[np.float64],
    v_ce: NDArray[np.float64],
    params: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Compute Hill total fiber-projected force for ``M`` muscles.

    Args:
        activations: ``(M,)`` activation in ``[0, 1]``.
        l_ce: ``(M,)`` fiber length [m].
        v_ce: ``(M,)`` fiber velocity [m/s].
        params: ``(M, 7)`` float64; columns
            ``[f_max, l_opt, l_slack, v_max, pennation_angle, damping,
            force_length_width]``.

    Returns:
        ``(M,)`` float64 of muscle forces in newtons (clamped non-negative).
    """
    a = np.ascontiguousarray(activations, dtype=np.float64)
    ll = np.ascontiguousarray(l_ce, dtype=np.float64)
    v = np.ascontiguousarray(v_ce, dtype=np.float64)
    p = np.ascontiguousarray(params, dtype=np.float64)
    if a.ndim != 1 or ll.shape != a.shape or v.shape != a.shape:
        raise ValueError("activations/l_ce/v_ce must be matching 1-D arrays")
    if p.ndim != 2 or p.shape != (a.shape[0], 7):
        raise ValueError(f"params must have shape (M=={a.shape[0]}, 7); got {p.shape}")
    if _RUST_AVAILABLE:
        return np.asarray(_rust.muscle_force_batch(a, ll, v, p))

    # Python fallback
    from src.shared.python.biomechanics.hill_muscle import (
        HillMuscleModel,
        MuscleParameters,
        MuscleState,
    )

    out = np.empty(a.shape[0], dtype=np.float64)
    for i in range(a.shape[0]):
        row = p[i]
        params_obj = MuscleParameters(
            F_max=float(row[0]),
            l_opt=float(row[1]),
            l_slack=float(row[2]),
            v_max=float(row[3]),
            pennation_angle=float(row[4]),
            damping=float(row[5]),
        )
        model = HillMuscleModel(params_obj, force_length_width=float(row[6]))
        state = MuscleState(
            activation=float(a[i]),
            l_CE=float(ll[i]),
            v_CE=float(v[i]),
            l_MT=0.0,
        )
        out[i] = model.compute_force(state)
    return out


def joint_torques_batch(
    moment_arms: NDArray[np.float64],
    forces: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Compute joint torques ``tau = R · F``.

    Args:
        moment_arms: ``(J, M)`` float64 moment-arm matrix.
        forces: ``(M,)`` float64 muscle forces.

    Returns:
        ``(J,)`` float64 joint torques.
    """
    r = np.ascontiguousarray(moment_arms, dtype=np.float64)
    f = np.ascontiguousarray(forces, dtype=np.float64)
    if r.ndim != 2 or f.ndim != 1 or r.shape[1] != f.shape[0]:
        raise ValueError(
            f"moment_arms must be (J, M) and forces (M,); got {r.shape} vs {f.shape}"
        )
    if _RUST_AVAILABLE:
        return np.asarray(_rust.joint_torques_batch(r, f))
    return r @ f


def step_full(
    excitations: NDArray[np.float64],
    activations: NDArray[np.float64],
    l_ce: NDArray[np.float64],
    v_ce: NDArray[np.float64],
    params: NDArray[np.float64],
    moment_arms: NDArray[np.float64],
    dt: float,
    *,
    tau_act: float | None = None,
    tau_deact: float | None = None,
    min_activation: float | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Combined RL inner-loop step: ``u -> a' -> F -> tau``.

    Returns ``(new_activations [M], joint_torques [J])``.
    """
    if dt <= 0:
        raise ValueError(f"dt must be positive, got {dt}")
    u = np.ascontiguousarray(excitations, dtype=np.float64)
    a = np.ascontiguousarray(activations, dtype=np.float64)
    ll = np.ascontiguousarray(l_ce, dtype=np.float64)
    v = np.ascontiguousarray(v_ce, dtype=np.float64)
    p = np.ascontiguousarray(params, dtype=np.float64)
    r = np.ascontiguousarray(moment_arms, dtype=np.float64)
    if not (u.shape == a.shape == ll.shape == v.shape) or u.ndim != 1:
        raise ValueError(
            "excitations, activations, l_ce, v_ce must be 1-D arrays of equal length"
        )
    if p.ndim != 2 or p.shape != (u.shape[0], 7):
        raise ValueError(f"params must have shape (M=={u.shape[0]}, 7); got {p.shape}")
    if r.ndim != 2 or r.shape[1] != u.shape[0]:
        raise ValueError(f"moment_arms must be (J, M={u.shape[0]}); got {r.shape}")

    if _RUST_AVAILABLE:
        a_out, tau_out = _rust.step_full(
            u,
            a,
            ll,
            v,
            p,
            r,
            dt,
            tau_act=tau_act,
            tau_deact=tau_deact,
            min_activation=min_activation,
        )
        return np.asarray(a_out), np.asarray(tau_out)

    # Python fallback: a' then F then tau.
    a_new = activation_step_batch(
        u,
        a,
        dt,
        tau_act=tau_act,
        tau_deact=tau_deact,
        min_activation=min_activation,
    )
    f_out = muscle_force_batch(a_new, ll, v, p)
    return a_new, joint_torques_batch(r, f_out)


__all__ = [
    "is_rust_available",
    "f_l",
    "f_p",
    "f_v",
    "f_t",
    "activation_step",
    "activation_step_batch",
    "muscle_force_batch",
    "joint_torques_batch",
    "step_full",
]

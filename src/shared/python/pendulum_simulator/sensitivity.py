"""Shared state transition matrix (STM) sensitivity harness and parity receipts (SAP-4, #13793).

Design by Contract & DRY
-------------------------
- Pre: Positive finite difference step sizes and positive state scales (fail closed).
- Pre: Valid tolerance and epsilon with tolerance >= epsilon for agreement horizon.
- Post: SensitivityReceipt records sigma_max, FTLE, agreement horizon,
  dominant sensitivity direction, FD step, and state scale conventions.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
import math
from typing import Any, TypeAlias

import numpy as np
import numpy.typing as npt

FloatArray: TypeAlias = npt.NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class StateScales:
    """Positive scales for state coordinates."""

    values: tuple[float, ...]

    def __post_init__(self) -> None:
        values = tuple(float(v) for v in self.values)
        if len(values) == 0 or any(not math.isfinite(v) or v <= 0.0 for v in values):
            raise ValueError("state scales must contain finite positive scale values")
        object.__setattr__(self, "values", values)

    @property
    def array(self) -> FloatArray:
        return np.asarray(self.values, dtype=np.float64)


@dataclass(frozen=True, slots=True)
class FiniteTimeSpectra:
    singular_values: FloatArray
    exponents_per_s: FloatArray


def normalized_transition(
    transition: npt.ArrayLike, state_scales: StateScales
) -> FloatArray:
    matrix = np.asarray(transition, dtype=np.float64)
    if (
        matrix.ndim != 2
        or matrix.shape[0] == 0
        or matrix.shape[0] != matrix.shape[1]
        or not np.all(np.isfinite(matrix))
    ):
        raise ValueError("transition must be a nonempty finite square matrix")
    scales = state_scales.array
    if matrix.shape != (scales.size, scales.size):
        raise ValueError("transition dimension must match state scales")
    return matrix * scales[np.newaxis, :] / scales[:, np.newaxis]


def finite_time_spectra(
    transitions: npt.ArrayLike, elapsed_time_s: npt.ArrayLike
) -> FiniteTimeSpectra:
    maps = np.asarray(transitions, dtype=np.float64)
    times = np.asarray(elapsed_time_s, dtype=np.float64)
    if (
        maps.ndim != 3
        or maps.shape[0] != times.size
        or maps.shape[1] == 0
        or maps.shape[1] != maps.shape[2]
        or not np.all(np.isfinite(maps))
    ):
        raise ValueError("transitions must be finite square maps on the time grid")
    if times.size < 2 or times[0] != 0.0 or np.any(np.diff(times) <= 0.0):
        raise ValueError("elapsed time must start at zero and increase strictly")
    singular = np.asarray(
        [np.linalg.svd(m, compute_uv=False) for m in maps], dtype=np.float64
    )
    exponents = np.full_like(singular, np.nan)
    exponents[1:] = np.log(singular[1:]) / times[1:, np.newaxis]
    return FiniteTimeSpectra(singular_values=singular, exponents_per_s=exponents)


def compute_agreement_horizon(tolerance: float, epsilon: float, ftle: float) -> float:
    if (
        not math.isfinite(tolerance)
        or not math.isfinite(epsilon)
        or tolerance <= 0.0
        or epsilon <= 0.0
    ):
        raise ValueError("tolerance and epsilon must be finite and positive")
    if tolerance < epsilon:
        raise ValueError(
            f"tolerance ({tolerance}) must be greater than or equal to epsilon ({epsilon})"
        )
    return float("inf") if ftle <= 0.0 else float(math.log(tolerance / epsilon) / ftle)


@dataclass(frozen=True)
class SensitivityReceipt:
    sigma_max: float
    ftle_per_s: float
    agreement_horizon_s: float
    dominant_direction: tuple[float, ...]
    fd_step: float
    state_scales: tuple[float, ...]
    sigma_max_phases: dict[str, float] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not math.isfinite(self.fd_step) or self.fd_step <= 0.0:
            raise ValueError(f"fd_step must be finite and positive, got {self.fd_step}")
        if not math.isfinite(self.sigma_max) or self.sigma_max < 0.0:
            raise ValueError(
                f"sigma_max must be finite and non-negative, got {self.sigma_max}"
            )
        if not math.isfinite(self.ftle_per_s):
            raise ValueError(f"ftle_per_s must be finite, got {self.ftle_per_s}")
        if len(self.state_scales) == 0 or any(
            not math.isfinite(s) or s <= 0.0 for s in self.state_scales
        ):
            raise ValueError("state_scales must contain finite positive values")

    def to_dict(self) -> dict[str, Any]:
        return {
            "sigma_max": self.sigma_max,
            "ftle_per_s": self.ftle_per_s,
            "agreement_horizon_s": self.agreement_horizon_s,
            "dominant_direction": list(self.dominant_direction),
            "fd_step": self.fd_step,
            "state_scales": list(self.state_scales),
            "sigma_max_phases": dict(self.sigma_max_phases),
            "metadata": dict(self.metadata),
        }


def compute_step_jacobian(
    step_fn: Callable[[FloatArray, float, float, Any], FloatArray],
    state: FloatArray,
    t: float,
    dt: float,
    control: Any,
    fd_step: float,
) -> FloatArray:
    n = state.size
    jacobian = np.empty((n, n), dtype=np.float64)
    for j in range(n):
        upper, lower = state.copy(), state.copy()
        upper[j] += fd_step
        lower[j] -= fd_step
        jacobian[:, j] = (
            step_fn(upper, t, dt, control) - step_fn(lower, t, dt, control)
        ) / (2.0 * fd_step)
    if not np.all(np.isfinite(jacobian)):
        raise ValueError("step Jacobian contains non-finite values")
    return jacobian


def propagate_state_transition_matrix(
    step_fn: Callable[[FloatArray, float, float, Any], FloatArray],
    initial_state: FloatArray,
    controls: Sequence[Any],
    time_grid: Sequence[float],
    dt: float,
    fd_step: float = 1e-6,
) -> tuple[FloatArray, FloatArray]:
    n, num_steps = initial_state.size, len(controls)
    states, transitions = (
        np.empty((num_steps + 1, n), dtype=np.float64),
        np.empty((num_steps + 1, n, n), dtype=np.float64),
    )
    states[0], transitions[0] = initial_state, np.eye(n, dtype=np.float64)
    curr_state = initial_state.copy()
    for k, ctrl in enumerate(controls):
        t_curr = time_grid[k]
        transitions[k + 1] = (
            compute_step_jacobian(step_fn, curr_state, t_curr, dt, ctrl, fd_step)
            @ transitions[k]
        )
        curr_state = step_fn(curr_state, t_curr, dt, ctrl)
        states[k + 1] = curr_state
    return states, transitions


def compute_linear_system_stm(
    A: npt.ArrayLike, t_end: float = 0.5, dt: float = 0.001, fd_step: float = 1e-5
) -> FloatArray:
    a_mat = np.asarray(A, dtype=np.float64)

    def _rk4_step(x: FloatArray, t: float, step_dt: float, ctrl: Any) -> FloatArray:
        k1 = a_mat @ x
        k2 = a_mat @ (x + 0.5 * step_dt * k1)
        k3 = a_mat @ (x + 0.5 * step_dt * k2)
        k4 = a_mat @ (x + step_dt * k3)
        return x + (step_dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

    num_steps = int(round(t_end / dt))
    _, transitions = propagate_state_transition_matrix(
        _rk4_step,
        np.ones(a_mat.shape[0], dtype=np.float64),
        [None] * num_steps,
        [k * dt for k in range(num_steps)],
        dt,
        fd_step=fd_step,
    )
    return transitions[-1]


def run_pendulum_sensitivity_study(
    engine: Any,
    torque_profile: Any = None,
    state_scales: StateScales | None = None,
    t_end: float = 0.2,
    dt: float = 0.001,
    fd_step: float = 1e-6,
    tolerance: float = 0.20,
    initial_state: Sequence[float] | None = None,
) -> SensitivityReceipt:
    if not math.isfinite(fd_step) or fd_step <= 0.0:
        raise ValueError(f"fd_step must be finite and positive, got {fd_step}")
    if not math.isfinite(t_end) or t_end <= 0.0 or t_end <= dt:
        raise ValueError("t_end must be positive and greater than dt")
    if not math.isfinite(dt) or dt <= 0.0:
        raise ValueError("dt must be positive")

    scales = state_scales or StateScales((1.0, 1.0, 1.0, 10.0, 10.0, 10.0))
    from src.shared.python.pendulum_simulator.step_convergence import (
        BreakpointTorqueProfile,
    )

    profile = torque_profile or BreakpointTorqueProfile(
        initial_torque=(80.0, 30.0, 10.0)
    )

    def _step_fn(
        x: FloatArray, t: float, step_dt: float, ctrl: tuple[float, float, float]
    ) -> FloatArray:
        engine.reset(tuple(x[:3]), tuple(x[3:]))
        engine.step(t, step_dt, ctrl)
        if hasattr(engine, "state"):
            s = engine.state
            return np.array(
                [s.theta1, s.theta2, s.theta3, s.omega1, s.omega2, s.omega3],
                dtype=np.float64,
            )
        if hasattr(engine, "data"):
            return np.concatenate([engine.data.qpos, engine.data.qvel]).astype(
                np.float64
            )
        raise TypeError(f"unsupported engine adapter type: {type(engine)}")

    num_steps = int(round(t_end / dt))
    init_x = (
        np.asarray(initial_state, dtype=np.float64)
        if initial_state is not None
        else np.array([-1.5, 0.5, 0.2, 0.0, 0.0, 0.0], dtype=np.float64)
    )
    if init_x.shape != (6,):
        raise ValueError(
            f"initial_state must have 6 elements, got shape {init_x.shape}"
        )
    _, transitions = propagate_state_transition_matrix(
        _step_fn,
        init_x,
        [profile.evaluate(k * dt) for k in range(num_steps)],
        [k * dt for k in range(num_steps)],
        dt,
        fd_step=fd_step,
    )
    m_final_norm = normalized_transition(transitions[-1], scales)
    _, s_vals, vt = np.linalg.svd(m_final_norm)
    sigma_max = float(s_vals[0])
    ftle = float(math.log(max(sigma_max, 1e-15)) / t_end)

    phases = {"release": 1.0}
    if num_steps >= 2:
        m_mid_norm = normalized_transition(transitions[num_steps // 2], scales)
        phases["mid_swing"] = float(np.linalg.svd(m_mid_norm, compute_uv=False)[0])
    phases["impact"] = sigma_max

    return SensitivityReceipt(
        sigma_max=sigma_max,
        ftle_per_s=ftle,
        agreement_horizon_s=compute_agreement_horizon(
            tolerance=tolerance, epsilon=min(fd_step, 1e-6), ftle=ftle
        ),
        dominant_direction=tuple(float(v) for v in vt[0]),
        fd_step=fd_step,
        state_scales=tuple(scales.values),
        sigma_max_phases=phases,
        metadata={
            "engine": getattr(engine, "engine_name", type(engine).__name__),
            "t_end_s": t_end,
            "dt_s": dt,
        },
    )


def _stage_rhs(dyn: Any, x: FloatArray, tau: tuple[float, float, float]) -> FloatArray:
    from double_pendulum_model.physics import triple_pendulum as tp

    acc = dyn.forward_dynamics(tp.TriplePendulumState(*x), tau)
    return np.array([x[3], x[4], x[5], acc[0], acc[1], acc[2]], dtype=np.float64)


def _stage_jacobian(
    dyn: Any, x_ref: FloatArray, tau_ff: Any, t_s: float, imp: Any = None
) -> FloatArray:
    from double_pendulum_model.physics import triple_pendulum as tp

    a_mat, h, eye = np.zeros((6, 6)), 1e-7, np.eye(6)
    a_mat[:3, 3:] = np.eye(3)
    tau = (float(tau_ff[0]), float(tau_ff[1]), float(tau_ff[2]))
    for j in range(6):
        up = dyn.forward_dynamics(tp.TriplePendulumState(*(x_ref + h * eye[j])), tau)
        lo = dyn.forward_dynamics(tp.TriplePendulumState(*(x_ref - h * eye[j])), tau)
        a_mat[3:, j] = (np.asarray(up) - np.asarray(lo)) / (2.0 * h)
    if imp is not None and not imp.is_zero:
        mass = dyn.mass_matrix(tp.TriplePendulumState(*x_ref))
        a_mat[3:, :3] -= np.linalg.solve(mass, imp.stiffness_at(t_s))
        a_mat[3:, 3:] -= np.linalg.solve(mass, imp.damping_at(t_s))
    return a_mat


def toy_1dof_sigma_max(
    damping: float, stiffness: float = 10.0, mass: float = 1.0, t_end: float = 0.5
) -> float:
    ok = math.isfinite(damping) and damping >= 0.0 and min(stiffness, mass, t_end) > 0.0
    if not ok:
        raise ValueError("invalid 1DOF oracle parameters")
    a_mat = np.array([[0.0, 1.0], [-stiffness / mass, -damping / mass]]) * t_end
    vals, vecs = np.linalg.eig(a_mat)
    phi = np.real(vecs @ np.diag(np.exp(vals)) @ np.linalg.inv(vecs))
    return float(np.linalg.svd(phi, compute_uv=False)[0])


def rollout_nominal_trajectory(
    x0: Sequence[float] = (-1.5, 0.5, 0.2, 0.0, 0.0, 0.0),
    prof: Any = None,
    t_end: float = 0.2,
    dt: float = 0.001,
) -> Any:
    from double_pendulum_model.physics import triple_pendulum as tp
    from src.shared.python.pendulum_simulator import drive_modes as dm
    from src.shared.python.pendulum_simulator import step_convergence as sc

    if not math.isfinite(t_end) or not math.isfinite(dt) or dt <= 0.0 or t_end <= dt:
        raise ValueError("t_end and dt must be positive with t_end > dt")
    prof = prof or sc.BreakpointTorqueProfile((80.0, 30.0, 10.0))
    n = int(round(t_end / dt))
    times = np.array([k * dt for k in range(n + 1)], dtype=np.float64)
    states, torques = np.empty((n + 1, 6)), np.empty((n, 3))
    st_grid, und_grid = np.empty((n, 4, 6)), np.empty((n, 4, 3))
    curr = np.asarray(x0, dtype=np.float64)
    states[0] = curr
    c_dyn, u_dyn = dm._CANONICAL_DYN, dm._UNDAMPED_DYN
    for k in range(n):
        tau_c = prof.evaluate(float(times[k]))
        torques[k] = tau_c
        k1 = _stage_rhs(c_dyn, curr, tau_c)
        s1 = curr + 0.5 * dt * k1
        k2 = _stage_rhs(c_dyn, s1, tau_c)
        s2 = curr + 0.5 * dt * k2
        k3 = _stage_rhs(c_dyn, s2, tau_c)
        s3 = curr + dt * k3
        k4 = _stage_rhs(c_dyn, s3, tau_c)
        for idx, (sv, rv) in enumerate(((curr, k1), (s1, k2), (s2, k3), (s3, k4))):
            st_grid[k, idx] = sv
            acc = (float(rv[3]), float(rv[4]), float(rv[5]))
            und_grid[k, idx] = u_dyn.inverse_dynamics(tp.TriplePendulumState(*sv), acc)
        curr = curr + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        states[k + 1] = curr
    q_nom, v_nom = states[:, :3], states[:, 3:]
    return dm.NominalTrajectory(times, q_nom, v_nom, torques, st_grid, und_grid)


def step_drive_mode_rk4(
    mode: Any, x: FloatArray, k: int, dt: float, nom: Any, imp: Any = None
) -> FloatArray:
    from src.shared.python.pendulum_simulator import drive_modes as dm

    m = dm._coerce_mode(mode)
    dyn = dm._UNDAMPED_DYN if m is dm.DriveMode.TORQUE_UNDAMPED else dm._CANONICAL_DYN
    t_k, tau_c = float(nom.times[k]), nom.torques[k]
    t_st = (t_k, t_k + 0.5 * dt, t_k + 0.5 * dt, t_k + dt)

    def _ev(i: int, st: FloatArray) -> FloatArray:
        ref = (nom.stage_states[k, i], tau_c, nom.stage_undamped_torques[k, i])
        tau = dm.drive_torque(m, t_st[i], st[:3], st[3:], nom, imp, ref_stage=ref)
        return _stage_rhs(dyn, st, tau)

    k1 = _ev(0, x)
    k2 = _ev(1, x + 0.5 * dt * k1)
    k3 = _ev(2, x + 0.5 * dt * k2)
    k4 = _ev(3, x + dt * k3)
    return x + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def integrate_drive_mode_stm(
    mode: Any, nom: Any, imp: Any = None, window_s: float = 0.15
) -> tuple[FloatArray, FloatArray, FloatArray, Any]:
    from src.shared.python.loop_nullspace.impedance_spec import HYPOTHESIS_ID
    from src.shared.python.pendulum_simulator import drive_modes as dm

    m = dm._coerce_mode(mode)
    t_end = float(nom.times[-1])
    if not math.isfinite(window_s) or window_s <= 0.0 or window_s > t_end + 1e-12:
        raise ValueError("window_s must lie inside (0, t_e]")
    if m is dm.DriveMode.JOINT_IMPEDANCE:
        if imp is None:
            raise ValueError("impedance is required for JOINT_IMPEDANCE")
        if not imp.is_zero and any(
            w.t_start_s < -1e-12 or w.t_end_s > t_end + 1e-12 for w in imp.windows
        ):
            raise ValueError("impedance window must lie inside [0, t_e]")
    scales = StateScales((1.0, 1.0, 1.0, 10.0, 10.0, 10.0))
    is_und = m is dm.DriveMode.TORQUE_UNDAMPED
    dyn = dm._UNDAMPED_DYN if is_und else dm._CANONICAL_DYN
    n, dt = len(nom.torques), float(nom.times[1] - nom.times[0])
    states, trans = np.empty((n + 1, 6)), np.empty((n + 1, 6, 6))
    states[0], trans[0] = np.concatenate([nom.q[0], nom.v[0]]), np.eye(6)
    imp_m = imp if m is dm.DriveMode.JOINT_IMPEDANCE else None
    for k in range(n):
        states[k + 1] = step_drive_mode_rk4(m, states[k], k, dt, nom, imp_m)
        t_k = float(nom.times[k])
        t_st = (t_k, t_k + 0.5 * dt, t_k + 0.5 * dt, t_k + dt)
        ts = nom.stage_undamped_torques[k] if is_und else [nom.torques[k]] * 4
        st_k = nom.stage_states[k]
        a_st = [_stage_jacobian(dyn, st_k[i], ts[i], t_st[i], imp_m) for i in range(4)]
        phi = trans[k]
        kp1 = a_st[0] @ phi
        kp2 = a_st[1] @ (phi + 0.5 * dt * kp1)
        kp3 = a_st[2] @ (phi + 0.5 * dt * kp2)
        kp4 = a_st[3] @ (phi + dt * kp3)
        trans[k + 1] = phi + (dt / 6.0) * (kp1 + 2.0 * kp2 + 2.0 * kp3 + kp4)
    w_idx = min(int(round(max(0.0, t_end - window_s) / dt)), n - 1)
    phi_win = np.linalg.solve(trans[w_idx].T, trans[-1].T).T
    phi_norm = normalized_transition(phi_win, scales)
    s_vals = tuple(float(v) for v in np.linalg.svd(phi_norm, compute_uv=False))
    lens = np.array([seg.length_m for seg in dm._CANONICAL_DYN.parameters.segments])
    xf, h, eye = states[-1], 1e-7, np.eye(6)

    def _vtip(s: FloatArray) -> float:
        a, w = np.cumsum(s[:3]), np.cumsum(s[3:])
        vx, vz = np.sum(-lens * np.cos(a) * w), np.sum(lens * np.sin(a) * w)
        return float(math.hypot(vx, vz))

    g_vt = np.array([_vtip(xf + h * e) - _vtip(xf - h * e) for e in eye])
    u_vt = g_vt / np.linalg.norm(g_vt)
    u_ln = np.array([1.0, 1.0, 1.0, 0.0, 0.0, 0.0]) / math.sqrt(3.0)
    gains = {
        "grad_v_tip": float(np.linalg.norm(phi_norm @ u_vt)),
        "in_plane_path_lean_proxy": float(np.linalg.norm(phi_norm @ u_ln)),
    }
    hyps = imp.hypotheses if imp is not None else (HYPOTHESIS_ID,)
    dt_w = float(nom.times[-1] - nom.times[w_idx])
    ftle = float(math.log(max(s_vals[0], 1e-15)) / dt_w)
    det_w = float(np.linalg.det(phi_win))
    rec = dm.DriveModeSensitivityReceipt(
        m.value, hyps, s_vals[0], s_vals, ftle, det_w, gains, phi_norm
    )
    return states, trans[-1], phi_win, rec

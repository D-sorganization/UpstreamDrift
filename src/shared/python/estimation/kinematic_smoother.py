"""White-jerk Rauch-Tung-Striebel (RTS) kinematic smoother with uncertainty.

Implements a continuous-time white-jerk kinematic state-space model:
    state x = [q, q_dot, q_ddot]^T
discretized exactly for sample interval dt = 1 / rate_hz:
    x_{k+1} = F x_k + w_k,  w_k ~ N(0, q_c * Q_unit)
    y_k = H x_k + v_k,      v_k ~ N(0, r)

NaN measurements are treated as missing observations (predict without update),
allowing the posterior uncertainty to widen across gaps while maintaining
continuous smoothed kinematics.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.linalg import solve
from scipy.optimize import minimize

from src.shared.python.core.contracts import require


@dataclass(frozen=True)
class NoiseParameters:
    """Process and measurement noise parameters for kinematic smoothing."""

    jerk_psd: np.ndarray
    measurement_var: np.ndarray
    success: tuple[bool, ...] | None = None

    def __post_init__(self) -> None:
        self.jerk_psd.setflags(write=False)
        self.measurement_var.setflags(write=False)


@dataclass(frozen=True)
class KinematicSmoothingResult:
    """Result of white-jerk RTS kinematic smoothing with posterior uncertainty."""

    position: np.ndarray
    velocity: np.ndarray
    acceleration: np.ndarray
    position_std: np.ndarray
    velocity_std: np.ndarray
    acceleration_std: np.ndarray
    log_marginal_likelihood: float
    noise_parameters: NoiseParameters

    def __post_init__(self) -> None:
        for arr in (
            self.position,
            self.velocity,
            self.acceleration,
            self.position_std,
            self.velocity_std,
            self.acceleration_std,
        ):
            arr.setflags(write=False)


def white_jerk_transition(dt: float) -> tuple[np.ndarray, np.ndarray]:
    """Compute state transition matrix F and unit process noise covariance Q_unit.

    Args:
        dt: Sample period in seconds (> 0).

    Returns:
        tuple[F, Q_unit] of shape ((3, 3), (3, 3)).
    """
    require(dt > 0.0, "dt must be positive", dt)
    dt2 = dt * dt
    dt3 = dt2 * dt
    dt4 = dt3 * dt
    dt5 = dt4 * dt
    F = np.array(
        [
            [1.0, dt, 0.5 * dt2],
            [0.0, 1.0, dt],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    Q_unit = np.array(
        [
            [dt5 / 20.0, dt4 / 8.0, dt3 / 6.0],
            [dt4 / 8.0, dt3 / 3.0, dt2 / 2.0],
            [dt3 / 6.0, dt2 / 2.0, dt],
        ],
        dtype=np.float64,
    )
    return F, Q_unit


def _default_prior(
    y: np.ndarray,
    dt: float,
    measurement_var: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Construct diffuse initial state prior scaled to coordinate units."""
    n_frames, nq = y.shape
    m0 = np.zeros((nq, 3), dtype=np.float64)
    P0 = np.zeros((nq, 3, 3), dtype=np.float64)
    dt2 = dt * dt
    dt4 = dt2 * dt2

    for j in range(nq):
        yj = y[:, j]
        valid_idx = np.where(~np.isnan(yj))[0]
        if len(valid_idx) == 0:
            y0, v0 = 0.0, 0.0
        elif len(valid_idx) == 1:
            y0, v0 = float(yj[valid_idx[0]]), 0.0
        else:
            k0 = valid_idx[0]
            k1 = valid_idx[1]
            y0 = float(yj[k0])
            dt_eff = float(k1 - k0) * dt
            v0 = float(yj[k1] - yj[k0]) / dt_eff

        m0[j] = [y0, v0, 0.0]
        rj = float(measurement_var[j])
        P0[j] = 1e6 * np.diag([rj, rj / dt2, rj / dt4])

    return m0, P0


def _forward_filter(
    y: np.ndarray,
    F: np.ndarray,
    Q_unit: np.ndarray,
    jerk_psd: np.ndarray,
    measurement_var: np.ndarray,
    m0: np.ndarray,
    P0: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Run forward Kalman filter across all coordinates."""
    n_frames, nq = y.shape
    m_pred = np.zeros((n_frames, nq, 3), dtype=np.float64)
    P_pred = np.zeros((n_frames, nq, 3, 3), dtype=np.float64)
    m_filt = np.zeros((n_frames, nq, 3), dtype=np.float64)
    P_filt = np.zeros((n_frames, nq, 3, 3), dtype=np.float64)
    log_lik = np.zeros(nq, dtype=np.float64)

    Q = jerk_psd[:, None, None] * Q_unit[None, :, :]
    r = measurement_var
    log_2pi = math.log(2.0 * math.pi)

    for k in range(n_frames):
        if k == 0:
            mp = m0.copy()
            Pp = P0.copy()
        else:
            prev_m = m_filt[k - 1]
            prev_P = P_filt[k - 1]
            mp = prev_m @ F.T
            Pp = F @ prev_P @ F.T + Q
        m_pred[k] = mp
        P_pred[k] = Pp

        yk = y[k]
        mf = mp.copy()
        Pf = Pp.copy()
        valid = ~np.isnan(yk)

        if np.any(valid):
            v = yk - mp[:, 0]
            S = Pp[:, 0, 0] + r
            v_val = v[valid]
            S_val = S[valid]
            log_lik[valid] += -0.5 * (log_2pi + np.log(S_val) + (v_val * v_val) / S_val)

            K = Pp[:, :, 0] / S[:, None]
            mf[valid] = mp[valid] + K[valid] * v[valid, None]

            col = Pp[valid, :, 0:1]
            upd = (col @ col.transpose(0, 2, 1)) / S[valid, None, None]
            P_upd = Pp[valid] - upd
            Pf[valid] = 0.5 * (P_upd + P_upd.transpose(0, 2, 1))

        m_filt[k] = mf
        P_filt[k] = Pf

    return m_pred, P_pred, m_filt, P_filt, log_lik


def _backward_smoother(
    m_pred: np.ndarray,
    P_pred: np.ndarray,
    m_filt: np.ndarray,
    P_filt: np.ndarray,
    F: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Run Rauch-Tung-Striebel (RTS) backward smoothing pass."""
    n_frames, nq, _ = m_filt.shape
    m_smooth = np.zeros_like(m_filt)
    P_smooth = np.zeros_like(P_filt)

    m_smooth[-1] = m_filt[-1]
    P_smooth[-1] = P_filt[-1]

    for k in range(n_frames - 2, -1, -1):
        target = F @ P_filt[k]
        C_T = solve(P_pred[k + 1], target)
        C = C_T.transpose(0, 2, 1)

        m_diff = m_smooth[k + 1] - m_pred[k + 1]
        step_m = (C @ m_diff[:, :, None])[:, :, 0]
        m_smooth[k] = m_filt[k] + step_m

        P_diff = P_smooth[k + 1] - P_pred[k + 1]
        step_P = C @ P_diff @ C.transpose(0, 2, 1)
        P_k = P_filt[k] + step_P
        P_smooth[k] = 0.5 * (P_k + P_k.transpose(0, 2, 1))

    return m_smooth, P_smooth


def _estimate_single_noise(
    yj: np.ndarray,
    rate_hz: float,
    dt: float,
    F: np.ndarray,
    Q_unit: np.ndarray,
) -> tuple[float, float, bool]:
    """Estimate jerk_psd and measurement_var for a single coordinate."""
    valid_idx = np.where(~np.isnan(yj))[0]
    require(
        len(valid_idx) >= 3, "coordinate has fewer than 3 valid frames", len(valid_idx)
    )
    y_valid = yj[valid_idx]

    diff2 = np.diff(y_valid, n=2)
    var2 = float(np.var(diff2))
    r_seed = max(var2 / 6.0, 1e-12)

    diff3 = np.diff(y_valid, n=3)
    var3 = float(np.var(diff3))
    qc_seed = max(var3 * rate_hz, 1e-6)

    def _neg_log_lik(params: np.ndarray) -> float:
        qc_val = math.exp(float(params[0]))
        r_val = math.exp(float(params[1]))
        m0, P0 = _default_prior(yj[:, None], dt, np.array([r_val]))
        _, _, _, _, lik = _forward_filter(
            yj[:, None],
            F,
            Q_unit,
            np.array([qc_val]),
            np.array([r_val]),
            m0,
            P0,
        )
        return -float(lik[0])

    init_params = np.array([math.log(qc_seed), math.log(r_seed)], dtype=np.float64)
    res = minimize(_neg_log_lik, init_params, method="L-BFGS-B")
    if not res.success:
        raise ValueError(f"Noise parameter optimization failed: {res.message}")

    opt_qc = math.exp(float(res.x[0]))
    opt_r = math.exp(float(res.x[1]))
    return opt_qc, opt_r, bool(res.success)


def estimate_noise_parameters(q: np.ndarray, rate_hz: float) -> NoiseParameters:
    """Estimate noise parameters (jerk_psd, measurement_var) per coordinate.

    Maximises log marginal likelihood using finite-difference initial seeds.

    Args:
        q: (N, nq) array of coordinate measurements.
        rate_hz: Sampling frequency in Hz (> 0).

    Returns:
        NoiseParameters containing estimated values and success flags.
    """
    require(isinstance(q, np.ndarray), "q must be a numpy ndarray")
    require(q.ndim == 2, "q must be a 2D array", q.shape)
    require(q.shape[0] >= 3, "q must have at least 3 frames", q.shape[0])
    require(rate_hz > 0.0, "rate_hz must be positive", rate_hz)

    dt = 1.0 / rate_hz
    F, Q_unit = white_jerk_transition(dt)
    nq = q.shape[1]

    qc_list = []
    r_list = []
    success_list = []

    for j in range(nq):
        opt_qc, opt_r, ok = _estimate_single_noise(q[:, j], rate_hz, dt, F, Q_unit)
        qc_list.append(opt_qc)
        r_list.append(opt_r)
        success_list.append(ok)

    return NoiseParameters(
        jerk_psd=np.array(qc_list, dtype=np.float64),
        measurement_var=np.array(r_list, dtype=np.float64),
        success=tuple(success_list),
    )


def smooth_kinematic(
    q: np.ndarray,
    rate_hz: float,
    *,
    jerk_psd: float | np.ndarray,
    measurement_var: float | np.ndarray,
    initial_mean: np.ndarray | None = None,
    initial_covariance: np.ndarray | None = None,
) -> KinematicSmoothingResult:
    """Perform white-jerk RTS kinematic smoothing with posterior uncertainty.

    Args:
        q: (N, nq) array of coordinate trajectories.
        rate_hz: Sampling rate in Hz (> 0).
        jerk_psd: Continuous jerk PSD (q_c > 0) as scalar or (nq,) array.
        measurement_var: Measurement noise variance (r > 0) as scalar or (nq,) array.
        initial_mean: Optional initial mean of shape (nq, 3) or (3,).
        initial_covariance: Optional initial covariance of shape (nq, 3, 3) or (3, 3).

    Returns:
        KinematicSmoothingResult with positions, velocities, accelerations and stds.
    """
    require(isinstance(q, np.ndarray), "q must be a numpy ndarray")
    require(q.ndim == 2, "q must be a 2D array", q.shape)
    require(q.shape[0] >= 3, "q must have at least 3 frames", q.shape[0])
    require(rate_hz > 0.0, "rate_hz must be positive", rate_hz)
    require(
        not bool(np.any(np.isinf(q))), "q must not contain infinities (NaN = missing)"
    )

    nq = q.shape[1]
    qc_arr = np.broadcast_to(np.asarray(jerk_psd, dtype=np.float64), (nq,)).copy()
    r_arr = np.broadcast_to(np.asarray(measurement_var, dtype=np.float64), (nq,)).copy()

    require(bool(np.all(qc_arr > 0.0)), "jerk_psd must be strictly positive", qc_arr)
    require(
        bool(np.all(r_arr > 0.0)), "measurement_var must be strictly positive", r_arr
    )

    dt = 1.0 / rate_hz
    F, Q_unit = white_jerk_transition(dt)

    require(
        (initial_mean is None) == (initial_covariance is None),
        "initial_mean and initial_covariance must be given together",
    )
    if initial_mean is not None and initial_covariance is not None:
        m0_raw = np.asarray(initial_mean, dtype=np.float64)
        m0 = (
            np.broadcast_to(m0_raw, (nq, 3)).copy()
            if m0_raw.ndim == 1
            else m0_raw.copy()
        )
        P0_raw = np.asarray(initial_covariance, dtype=np.float64)
        P0 = (
            np.broadcast_to(P0_raw, (nq, 3, 3)).copy()
            if P0_raw.ndim == 2
            else P0_raw.copy()
        )
        require(m0.shape == (nq, 3), "initial_mean must be (nq, 3) or (3,)", m0.shape)
        require(
            P0.shape == (nq, 3, 3),
            "initial_covariance must be (nq, 3, 3) or (3, 3)",
            P0.shape,
        )
    else:
        m0, P0 = _default_prior(q, dt, r_arr)

    m_pred, P_pred, m_filt, P_filt, log_lik = _forward_filter(
        q, F, Q_unit, qc_arr, r_arr, m0, P0
    )
    m_smooth, P_smooth = _backward_smoother(m_pred, P_pred, m_filt, P_filt, F)

    pos = m_smooth[:, :, 0]
    vel = m_smooth[:, :, 1]
    acc = m_smooth[:, :, 2]

    pos_std = np.sqrt(np.maximum(P_smooth[:, :, 0, 0], 0.0))
    vel_std = np.sqrt(np.maximum(P_smooth[:, :, 1, 1], 0.0))
    acc_std = np.sqrt(np.maximum(P_smooth[:, :, 2, 2], 0.0))

    return KinematicSmoothingResult(
        position=pos,
        velocity=vel,
        acceleration=acc,
        position_std=pos_std,
        velocity_std=vel_std,
        acceleration_std=acc_std,
        log_marginal_likelihood=float(np.sum(log_lik)),
        noise_parameters=NoiseParameters(jerk_psd=qc_arr, measurement_var=r_arr),
    )

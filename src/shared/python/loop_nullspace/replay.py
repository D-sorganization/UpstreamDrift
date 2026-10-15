"""Open-loop versus feedback-assisted replay of allocated torques (NSX-31, #12571).

Integrates allocated generalized torques forward under constrained equations of
motion and reports:
- In ``"OPEN_LOOP"`` mode: position and velocity drift curves and time to
  divergence (with zero feedback gain).
- In ``"FEEDBACK"`` mode: the corrective PD feedback torque series
  ``tau_fb(t) = kp * (q_ref(t) - q(t)) + kd * (v_ref(t) - v(t))`` recorded in a
  separate :class:`FeedbackBudgetReport` channel with L2 and L-infinity budget
  ratios relative to the allocated torque, and ``validation_claim_allowed=False``
  (feedback-assisted tracking is never physical or open-loop validation).
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Literal

import numpy as np

from src.engines.feedback_comparison import ReplayPolicy

ReplayMode = Literal["OPEN_LOOP", "FEEDBACK"]

_BUDGET_EPS = 1e-12


def _readonly_array(arr: Any, ndim: int, name: str) -> np.ndarray:
    out = np.array(arr, dtype=float, copy=True)
    if out.ndim != ndim or out.size == 0:
        raise ValueError(f"{name} must be a non-empty {ndim}-D array")
    if not np.all(np.isfinite(out)):
        raise ValueError(f"{name} must contain only finite values")
    out.setflags(write=False)
    return out


@dataclass(frozen=True)
class ReplayAllocation:
    """Allocated generalized torque series and reference state trajectory for replay."""

    allocated_torques: np.ndarray
    q_ref: np.ndarray
    v_ref: np.ndarray
    dt: float = 0.01

    def __post_init__(self) -> None:
        tau = _readonly_array(self.allocated_torques, 2, "allocated_torques")
        q_r = _readonly_array(self.q_ref, 2, "q_ref")
        v_r = _readonly_array(self.v_ref, 2, "v_ref")
        if tau.shape != q_r.shape or tau.shape != v_r.shape:
            raise ValueError(
                "allocated_torques, q_ref, and v_ref must share shape (T, n_dof)"
            )
        if self.dt <= 0.0 or not math.isfinite(self.dt):
            raise ValueError("dt must be a positive finite float")
        object.__setattr__(self, "allocated_torques", tau)
        object.__setattr__(self, "q_ref", q_r)
        object.__setattr__(self, "v_ref", v_r)


@dataclass(frozen=True)
class ConstrainedEOMProvider:
    """Constrained equations-of-motion provider solving KKT closed-loop dynamics.

    Computes constrained acceleration:
        ``a = M^{-1} (tau_total - b + J_c^T lambda_c)``
    where ``lambda_c`` enforces holonomic loop closure ``J_c a + jdot_v = 0``.
    """

    mass_matrix: np.ndarray
    bias_vector: np.ndarray
    closure_jacobian: np.ndarray | None = None
    closure_drift: np.ndarray | None = None

    def __post_init__(self) -> None:
        m_mat = _readonly_array(self.mass_matrix, 2, "mass_matrix")
        if m_mat.shape[0] != m_mat.shape[1]:
            raise ValueError("mass_matrix must be square")
        b_vec = _readonly_array(self.bias_vector, 1, "bias_vector")
        if b_vec.shape[0] != m_mat.shape[0]:
            raise ValueError("bias_vector length must match mass_matrix dimension")
        object.__setattr__(self, "mass_matrix", m_mat)
        object.__setattr__(self, "bias_vector", b_vec)

        if self.closure_jacobian is not None:
            j_c = _readonly_array(self.closure_jacobian, 2, "closure_jacobian")
            if j_c.shape[1] != m_mat.shape[0]:
                raise ValueError("closure_jacobian column count must match DOF count")
            object.__setattr__(self, "closure_jacobian", j_c)
            if self.closure_drift is not None:
                drift = _readonly_array(self.closure_drift, 1, "closure_drift")
                if drift.shape[0] != j_c.shape[0]:
                    raise ValueError(
                        "closure_drift length must match closure_jacobian rows"
                    )
                object.__setattr__(self, "closure_drift", drift)

    def acceleration(
        self, q: np.ndarray, v: np.ndarray, tau_total: np.ndarray
    ) -> np.ndarray:
        """Compute constrained generalized acceleration at state (q, v) under tau_total."""
        _ = q, v
        rhs_unconstrained = np.asarray(tau_total, dtype=float) - self.bias_vector
        m_inv_rhs = np.linalg.solve(self.mass_matrix, rhs_unconstrained)
        if self.closure_jacobian is None or self.closure_jacobian.shape[0] == 0:
            return m_inv_rhs

        j_c = self.closure_jacobian
        drift = (
            self.closure_drift
            if self.closure_drift is not None
            else np.zeros(j_c.shape[0], dtype=float)
        )
        m_inv_jc_t = np.linalg.solve(self.mass_matrix, j_c.T)
        delassus = j_c @ m_inv_jc_t
        lam_rhs = -drift - (j_c @ m_inv_rhs)
        lambda_c, *_ = np.linalg.lstsq(delassus, lam_rhs, rcond=1e-12)
        return np.linalg.solve(self.mass_matrix, rhs_unconstrained + j_c.T @ lambda_c)


@dataclass(frozen=True)
class FeedbackBudgetReport:
    """Separate feedback torque channel and relative norm budget ratios."""

    feedback_torque_series: np.ndarray
    allocated_torque_series: np.ndarray
    l2_budget_ratio: float
    l_inf_budget_ratio: float
    peak_feedback_nm: float

    def __post_init__(self) -> None:
        fb = _readonly_array(self.feedback_torque_series, 2, "feedback_torque_series")
        alloc = _readonly_array(
            self.allocated_torque_series, 2, "allocated_torque_series"
        )
        if fb.shape != alloc.shape:
            raise ValueError(
                "feedback_torque_series and allocated_torque_series shapes must match"
            )
        if self.l2_budget_ratio < 0.0 or self.l_inf_budget_ratio < 0.0:
            raise ValueError("Feedback budget ratios must be non-negative")
        object.__setattr__(self, "feedback_torque_series", fb)
        object.__setattr__(self, "allocated_torque_series", alloc)


@dataclass(frozen=True)
class ReplayReport:
    """Result of open-loop or feedback-assisted torque replay."""

    mode: ReplayMode
    q_trajectory: np.ndarray
    v_trajectory: np.ndarray
    position_drift_norm: np.ndarray
    velocity_drift_norm: np.ndarray
    time_to_divergence_s: float | None
    feedback_budget: FeedbackBudgetReport | None
    validation_claim_allowed: bool
    label: str

    def __post_init__(self) -> None:
        if self.mode not in ("OPEN_LOOP", "FEEDBACK"):
            raise ValueError(f"Invalid replay mode {self.mode!r}")
        if self.mode == "FEEDBACK" and self.validation_claim_allowed:
            raise ValueError(
                "Feedback-assisted replay is tracking and may never set validation_claim_allowed=True"
            )
        if self.mode == "FEEDBACK" and self.feedback_budget is None:
            raise ValueError("FEEDBACK mode requires a populated FeedbackBudgetReport")
        object.__setattr__(
            self, "q_trajectory", _readonly_array(self.q_trajectory, 2, "q_trajectory")
        )
        object.__setattr__(
            self, "v_trajectory", _readonly_array(self.v_trajectory, 2, "v_trajectory")
        )
        object.__setattr__(
            self,
            "position_drift_norm",
            _readonly_array(self.position_drift_norm, 1, "position_drift_norm"),
        )
        object.__setattr__(
            self,
            "velocity_drift_norm",
            _readonly_array(self.velocity_drift_norm, 1, "velocity_drift_norm"),
        )


def _normalize_mode(mode: ReplayMode | ReplayPolicy) -> ReplayMode:
    if isinstance(mode, ReplayPolicy):
        return "OPEN_LOOP" if mode == ReplayPolicy.INDEPENDENT_TIME_ONLY else "FEEDBACK"
    if mode in ("OPEN_LOOP", "FEEDBACK"):
        return mode
    raise ValueError(
        f"Unsupported replay mode {mode!r}; expected 'OPEN_LOOP' or 'FEEDBACK'"
    )


def _extract_allocation(allocation: Any, default_dt: float) -> ReplayAllocation:
    if isinstance(allocation, ReplayAllocation):
        return allocation
    tau = getattr(allocation, "allocated_torques", getattr(allocation, "torques", None))
    q_ref = getattr(allocation, "q_ref", getattr(allocation, "q_trajectory", None))
    v_ref = getattr(allocation, "v_ref", getattr(allocation, "v_trajectory", None))
    dt_val = float(getattr(allocation, "dt", default_dt))
    if tau is None or q_ref is None or v_ref is None:
        raise ValueError("allocation must provide allocated_torques, q_ref, and v_ref")
    return ReplayAllocation(allocated_torques=tau, q_ref=q_ref, v_ref=v_ref, dt=dt_val)


def _eval_acceleration(
    provider: Any, q: np.ndarray, v: np.ndarray, tau: np.ndarray
) -> np.ndarray:
    if hasattr(provider, "acceleration"):
        return np.asarray(provider.acceleration(q, v, tau), dtype=float)
    if callable(provider):
        return np.asarray(provider(q, v, tau), dtype=float)
    raise TypeError("provider must implement .acceleration(q, v, tau) or be callable")


def _build_feedback_budget(
    fb_series: np.ndarray, alloc_series: np.ndarray
) -> FeedbackBudgetReport:
    fb_l2 = float(np.linalg.norm(fb_series))
    alloc_l2 = max(float(np.linalg.norm(alloc_series)), _BUDGET_EPS)
    fb_linf = float(np.max(np.abs(fb_series)))
    alloc_linf = max(float(np.max(np.abs(alloc_series))), _BUDGET_EPS)
    return FeedbackBudgetReport(
        feedback_torque_series=fb_series,
        allocated_torque_series=alloc_series,
        l2_budget_ratio=fb_l2 / alloc_l2,
        l_inf_budget_ratio=fb_linf / alloc_linf,
        peak_feedback_nm=fb_linf,
    )


def replay(
    allocation: Any,
    provider: Any,
    *,
    mode: ReplayMode | ReplayPolicy = "OPEN_LOOP",
    kp: float = 0.0,
    kd: float = 0.0,
    divergence_threshold: float = 0.1,
    dt: float = 0.01,
) -> ReplayReport:
    """Integrate allocated torques forward in OPEN_LOOP or FEEDBACK mode."""
    resolved_mode = _normalize_mode(mode)
    if divergence_threshold <= 0.0 or not math.isfinite(divergence_threshold):
        raise ValueError("divergence_threshold must be a positive finite float")
    if dt <= 0.0 or not math.isfinite(dt):
        raise ValueError("dt must be a positive finite float")
    if kp < 0.0 or kd < 0.0:
        raise ValueError("Feedback gains kp and kd must be non-negative")

    alloc = _extract_allocation(allocation, dt)
    step_dt = float(dt)
    n_steps, n_dof = alloc.allocated_torques.shape

    q_traj = np.zeros((n_steps, n_dof), dtype=float)
    v_traj = np.zeros((n_steps, n_dof), dtype=float)
    fb_series = np.zeros((n_steps, n_dof), dtype=float)

    q_curr = alloc.q_ref[0].copy()
    v_curr = alloc.v_ref[0].copy()

    eff_kp = float(kp) if resolved_mode == "FEEDBACK" else 0.0
    eff_kd = float(kd) if resolved_mode == "FEEDBACK" else 0.0

    for k in range(n_steps):
        q_traj[k] = q_curr
        v_traj[k] = v_curr
        q_err = alloc.q_ref[k] - q_curr
        v_err = alloc.v_ref[k] - v_curr
        tau_fb = eff_kp * q_err + eff_kd * v_err
        fb_series[k] = tau_fb
        tau_total = alloc.allocated_torques[k] + tau_fb
        acc_k = _eval_acceleration(provider, q_curr, v_curr, tau_total)
        q_curr = q_curr + v_curr * step_dt + 0.5 * acc_k * (step_dt**2)
        v_curr = v_curr + acc_k * step_dt

    pos_drift = np.linalg.norm(q_traj - alloc.q_ref, axis=1)
    vel_drift = np.linalg.norm(v_traj - alloc.v_ref, axis=1)

    exceeded = np.flatnonzero(pos_drift > divergence_threshold)
    t_div = float(exceeded[0]) * step_dt if exceeded.size > 0 else None

    fb_report = (
        _build_feedback_budget(fb_series, alloc.allocated_torques)
        if resolved_mode == "FEEDBACK"
        else None
    )
    label = (
        "OPEN_LOOP_DRIFT_TEST"
        if resolved_mode == "OPEN_LOOP"
        else "FEEDBACK_ASSISTED_TRACKING_NOT_VALIDATION"
    )
    return ReplayReport(
        mode=resolved_mode,
        q_trajectory=q_traj,
        v_trajectory=v_traj,
        position_drift_norm=pos_drift,
        velocity_drift_norm=vel_drift,
        time_to_divergence_s=t_div,
        feedback_budget=fb_report,
        validation_claim_allowed=False,
        label=label,
    )


__all__ = [
    "ConstrainedEOMProvider",
    "FeedbackBudgetReport",
    "ReplayAllocation",
    "ReplayMode",
    "ReplayReport",
    "replay",
]

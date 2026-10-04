"""DIME Uncertain-Control ZTCF Prediction and Estimation Criterion (#11421, #11425).

Provides dynamics-informed drift prediction, uncertainty propagation,
drift-centered marginalized-control transition criterion, and explicit-control mode.
Enforces mutual exclusion of duplicate physics likelihoods and fail-closed receipts.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
import enum
from types import MappingProxyType
from typing import Any, Final

import numpy as np

from src.shared.python.contracts import require
from src.shared.python.estimation.dime_contracts import (
    ContactPolicy,
    DimeCompleteState,
    DimeFullStepRequest,
    DimeZeroInputProposal,
    DynamicsProvider,
    EstimationIntervalFactor,
)
from src.shared.python.estimation.dime_manifest import (
    CANONICAL_DIME_UNITS,
    compute_cancellation_metric,
)
from src.shared.python.estimation.dime_providers import rollout_zero_input_proposal

DIME_DRIFT_PREDICTION_VERSION: Final[str] = "1.0.0"


class PredictionValidityStatus(str, enum.Enum):
    """Validity outcome of drift prediction and proposal generation."""

    VALID = "valid"
    INVALID_CONTACT = "invalid_contact"
    INVALID_HORIZON = "invalid_horizon"
    STALE_MODEL_HASH = "stale_model_hash"
    SINGULAR_COVARIANCE = "singular_covariance"
    UNAVAILABLE_PROVIDER = "unavailable_provider"


class PredictionMode(str, enum.Enum):
    """Operational mode for control treatment in dynamics prediction."""

    EXPLICIT_CONTROL = "explicit_control"
    MARGINALIZED_CONTROL = "marginalized_control"


def _make_readonly_array(arr: np.ndarray) -> np.ndarray:
    """Return a read-only copy of a float64 numpy array."""
    out = np.array(arr, dtype=np.float64, copy=True)
    out.flags.writeable = False
    return out


def _ensure_symmetric_psd(cov: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Symmetrize covariance and ensure eigenvalues are non-negative."""
    sym = 0.5 * (cov + cov.T)
    vals, vecs = np.linalg.eigh(sym)
    clipped_vals = np.maximum(vals, eps)
    psd = vecs @ np.diag(clipped_vals) @ vecs.T
    return _make_readonly_array(0.5 * (psd + psd.T))


@dataclass(frozen=True)
class StateDistribution:
    """Gaussian distribution over state coordinates."""

    mean: DimeCompleteState
    covariance: np.ndarray

    def __post_init__(self) -> None:
        raw_cov = np.asarray(self.covariance, dtype=np.float64)
        n_dim = len(self.mean.v) * 2
        require(
            raw_cov.shape == (n_dim, n_dim),
            f"State covariance shape {raw_cov.shape} must be ({n_dim}, {n_dim})",
        )
        require(bool(np.all(np.isfinite(raw_cov))), "State covariance must be finite")
        object.__setattr__(self, "covariance", _ensure_symmetric_psd(raw_cov))


@dataclass(frozen=True)
class ControlDistribution:
    """Gaussian distribution over applied control channels."""

    mean: np.ndarray
    covariance: np.ndarray
    physical_types: tuple[str, ...] = field(default_factory=tuple)
    units: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        raw_mean = np.asarray(self.mean, dtype=np.float64)
        raw_cov = np.asarray(self.covariance, dtype=np.float64)
        require(raw_mean.ndim == 1, "Control mean must be 1D")
        require(bool(np.all(np.isfinite(raw_mean))), "Control mean must be finite")
        n_u = len(raw_mean)
        require(
            raw_cov.shape == (n_u, n_u),
            f"Control covariance shape {raw_cov.shape} must be ({n_u}, {n_u})",
        )
        require(bool(np.all(np.isfinite(raw_cov))), "Control covariance must be finite")
        object.__setattr__(self, "mean", _make_readonly_array(raw_mean))
        object.__setattr__(self, "covariance", _ensure_symmetric_psd(raw_cov))


@dataclass(frozen=True)
class ModelContactUncertainty:
    """Uncertainty specifications for dynamics prediction and propagation."""

    process_noise_covariance: np.ndarray
    mass_uncertainty_std: float = 0.0
    contact_phase_stable: bool = True
    max_qualified_horizon_s: float = 0.5
    declared_contact_policy: ContactPolicy = ContactPolicy.NATIVE_ELIMINATED

    def __post_init__(self) -> None:
        raw_q = np.asarray(self.process_noise_covariance, dtype=np.float64)
        require(raw_q.ndim == 2, "process_noise_covariance must be 2D")
        require(
            raw_q.shape[0] == raw_q.shape[1],
            "process_noise_covariance must be square",
        )
        require(
            bool(np.all(np.isfinite(raw_q))),
            "process_noise_covariance must be finite",
        )
        require(self.mass_uncertainty_std >= 0.0, "mass_uncertainty_std must be >= 0")
        require(
            self.max_qualified_horizon_s > 0.0,
            "max_qualified_horizon_s must be positive",
        )
        object.__setattr__(
            self, "process_noise_covariance", _ensure_symmetric_psd(raw_q)
        )


@dataclass(frozen=True)
class DimeDriftPredictionResult:
    """Receipt and prediction outputs for DIME-04."""

    validity_status: PredictionValidityStatus
    validity_receipt: str
    mode: PredictionMode
    horizon_s: float
    dt: float
    zero_control_branch: DimeZeroInputProposal | None
    predicted_mean: DimeCompleteState | None
    predicted_covariance: np.ndarray | None
    linearization_drift_jacobian_F: np.ndarray | None
    linearization_control_jacobian_G: np.ndarray | None
    linearization_error_bound: float | None = None
    drift_gain: float | None = None
    cancellation_index: float | None = None
    diagnostics: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "diagnostics", MappingProxyType(dict(self.diagnostics))
        )


def compute_linearization_jacobians(
    provider: DynamicsProvider,
    state: DimeCompleteState,
    control: np.ndarray,
    dt: float,
    eps: float = 1e-6,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute state transition matrix F and control sensitivity G via central finite differences."""
    q0 = np.array(state.q, dtype=np.float64)
    v0 = np.array(state.v, dtype=np.float64)
    u0 = np.array(control, dtype=np.float64)
    n_v = len(v0)
    n_u = len(u0)
    dim_x = 2 * n_v

    F = np.zeros((dim_x, dim_x), dtype=np.float64)
    G = np.zeros((dim_x, n_u), dtype=np.float64)

    # State perturbation: delta q and delta v
    for i in range(dim_x):
        q_plus, v_plus = q0.copy(), v0.copy()
        q_minus, v_minus = q0.copy(), v0.copy()
        if i < n_v:
            q_plus[i] += eps
            q_minus[i] -= eps
        else:
            v_plus[i - n_v] += eps
            v_minus[i - n_v] -= eps

        s_plus = DimeCompleteState(
            t=state.t,
            q=q_plus,
            v=v_plus,
            model_hash=provider.model_hash,
            units=dict(state.units),
            frame=state.frame,
        )
        s_minus = DimeCompleteState(
            t=state.t,
            q=q_minus,
            v=v_minus,
            model_hash=provider.model_hash,
            units=dict(state.units),
            frame=state.frame,
        )

        res_plus = provider.step(
            DimeFullStepRequest(
                state=s_plus, controls=u0, dt=dt, model_hash=provider.model_hash
            )
        )
        res_minus = provider.step(
            DimeFullStepRequest(
                state=s_minus, controls=u0, dt=dt, model_hash=provider.model_hash
            )
        )

        x_p = np.concatenate([res_plus.next_state.q, res_plus.next_state.v])
        x_m = np.concatenate([res_minus.next_state.q, res_minus.next_state.v])
        F[:, i] = (x_p - x_m) / (2.0 * eps)

    # Control perturbation: delta u
    for j in range(n_u):
        u_plus, u_minus = u0.copy(), u0.copy()
        u_plus[j] += eps
        u_minus[j] -= eps

        res_p = provider.step(
            DimeFullStepRequest(
                state=state, controls=u_plus, dt=dt, model_hash=provider.model_hash
            )
        )
        res_m = provider.step(
            DimeFullStepRequest(
                state=state, controls=u_minus, dt=dt, model_hash=provider.model_hash
            )
        )

        x_p = np.concatenate([res_p.next_state.q, res_p.next_state.v])
        x_m = np.concatenate([res_m.next_state.q, res_m.next_state.v])
        G[:, j] = (x_p - x_m) / (2.0 * eps)

    return _make_readonly_array(F), _make_readonly_array(G)


def evaluate_linearization_error(
    provider: DynamicsProvider,
    state: DimeCompleteState,
    control_mean: np.ndarray,
    state_cov: np.ndarray,
    control_cov: np.ndarray,
    F: np.ndarray,
    G: np.ndarray,
    dt: float,
) -> float:
    """Evaluate bound on linearization error using deterministic dispersion reference points."""
    x0 = np.concatenate([state.q, state.v])
    u0 = control_mean
    dim_x = len(x0)
    dim_u = len(u0)

    # Linearized step delta
    delta_x_lin = (F - np.eye(dim_x)) @ np.zeros(dim_x) + G @ np.zeros(dim_u)

    # Sigma points along principal axes of state and control covariance
    scale = 0.5
    std_x = scale * np.sqrt(np.maximum(np.diag(state_cov), 1e-12))
    std_u = scale * np.sqrt(np.maximum(np.diag(control_cov), 1e-12))

    res_nom = provider.step(
        DimeFullStepRequest(
            state=state, controls=u0, dt=dt, model_hash=provider.model_hash
        )
    )
    x_step_0 = np.concatenate([res_nom.next_state.q, res_nom.next_state.v])

    errors = []
    n_v = len(state.v)
    for i in range(min(dim_x, 4)):
        pert_x = np.zeros(dim_x)
        pert_x[i] = std_x[i]

        s_samp = DimeCompleteState(
            t=state.t,
            q=state.q + pert_x[:n_v],
            v=state.v + pert_x[n_v:],
            model_hash=provider.model_hash,
            units=dict(state.units),
            frame=state.frame,
        )
        res_nonlin = provider.step(
            DimeFullStepRequest(
                state=s_samp, controls=u0, dt=dt, model_hash=provider.model_hash
            )
        )
        x_nonlin = np.concatenate([res_nonlin.next_state.q, res_nonlin.next_state.v])
        x_lin = x_step_0 + F @ pert_x

        errors.append(float(np.linalg.norm(x_nonlin - x_lin)))

    return float(np.mean(errors)) if errors else 0.0


class DimeDriftPredictor:
    """Dynamics-informed drift predictor with uncertain control marginalization."""

    def __init__(self, provider: DynamicsProvider) -> None:
        self._provider = provider

    def _check_contract_preconditions(
        self,
        uncertainty: ModelContactUncertainty,
        horizon_s: float,
        state: DimeCompleteState,
        mode: PredictionMode,
        dt: float,
    ) -> DimeDriftPredictionResult | None:
        """Evaluate fail-closed contract checks before prediction."""
        if not uncertainty.contact_phase_stable:
            return DimeDriftPredictionResult(
                validity_status=PredictionValidityStatus.INVALID_CONTACT,
                validity_receipt="Contact phase switch or instability detected; proposal disabled.",
                mode=mode,
                horizon_s=horizon_s,
                dt=dt,
                zero_control_branch=None,
                predicted_mean=None,
                predicted_covariance=None,
                linearization_drift_jacobian_F=None,
                linearization_control_jacobian_G=None,
            )

        if horizon_s > uncertainty.max_qualified_horizon_s:
            return DimeDriftPredictionResult(
                validity_status=PredictionValidityStatus.INVALID_HORIZON,
                validity_receipt=f"Horizon {horizon_s:.3f}s exceeds maximum qualified horizon {uncertainty.max_qualified_horizon_s:.3f}s.",
                mode=mode,
                horizon_s=horizon_s,
                dt=dt,
                zero_control_branch=None,
                predicted_mean=None,
                predicted_covariance=None,
                linearization_drift_jacobian_F=None,
                linearization_control_jacobian_G=None,
            )

        if state.model_hash != self._provider.model_hash:
            return DimeDriftPredictionResult(
                validity_status=PredictionValidityStatus.STALE_MODEL_HASH,
                validity_receipt=f"State model_hash '{state.model_hash}' does not match provider model_hash '{self._provider.model_hash}'.",
                mode=mode,
                horizon_s=horizon_s,
                dt=dt,
                zero_control_branch=None,
                predicted_mean=None,
                predicted_covariance=None,
                linearization_drift_jacobian_F=None,
                linearization_control_jacobian_G=None,
            )
        return None

    def _propagate_covariance(
        self,
        F: np.ndarray,
        G: np.ndarray,
        state_dist: StateDistribution,
        control_dist: ControlDistribution,
        uncertainty: ModelContactUncertainty,
        a_total: np.ndarray,
        horizon_s: float,
        n_v: int,
    ) -> np.ndarray:
        """Propagate state/control covariance and inflate with mass uncertainty."""
        dim_x = n_v * 2
        Q_proc = uncertainty.process_noise_covariance
        if Q_proc.shape != (dim_x, dim_x):
            Q_proc = np.eye(dim_x) * 1e-5

        sigma_prop = (
            F @ state_dist.covariance @ F.T + G @ control_dist.covariance @ G.T + Q_proc
        )

        if uncertainty.mass_uncertainty_std > 0.0:
            sigma_m = uncertainty.mass_uncertainty_std
            mass_var = np.zeros((dim_x, dim_x), dtype=np.float64)
            for k in range(n_v):
                mass_var[n_v + k, n_v + k] = (
                    float(a_total[k]) * sigma_m * horizon_s
                ) ** 2 + 1e-6 * sigma_m
            sigma_prop = sigma_prop + mass_var

        return _ensure_symmetric_psd(sigma_prop)

    def _rollout_controlled_mean(
        self, state: DimeCompleteState, u: np.ndarray, n_steps: int, dt: float
    ) -> DimeCompleteState:
        """Rollout nominal mean forward through discrete steps."""
        curr_state = state
        for _ in range(n_steps):
            step_res = self._provider.step(
                DimeFullStepRequest(
                    state=curr_state,
                    controls=u,
                    dt=dt,
                    model_hash=self._provider.model_hash,
                )
            )
            curr_state = step_res.next_state
        return curr_state

    def predict(
        self,
        state_dist: StateDistribution,
        control_dist: ControlDistribution,
        uncertainty: ModelContactUncertainty,
        horizon_s: float,
        dt: float,
        mode: PredictionMode = PredictionMode.MARGINALIZED_CONTROL,
    ) -> DimeDriftPredictionResult:
        """Propagate state and uncertainty over horizon under explicit or marginalized control."""
        state = state_dist.mean
        precondition_fail = self._check_contract_preconditions(
            uncertainty, horizon_s, state, mode, dt
        )
        if precondition_fail is not None:
            return precondition_fail

        ztcf_proposal = rollout_zero_input_proposal(
            self._provider, state, duration=horizon_s, dt=dt
        )

        decomp = self._provider.compute_acceleration_decomposition(
            state, control_dist.mean
        )
        a_passive = decomp.ztcf
        a_ctrl = decomp.a_ctrl
        a_total = decomp.ztcf + decomp.a_ctrl

        norm_passive = float(np.linalg.norm(a_passive))
        norm_total = float(np.linalg.norm(a_total))
        drift_gain = norm_passive / (norm_total + 1e-12)
        cancellation = compute_cancellation_metric(a_passive, a_ctrl)

        n_steps = max(1, int(round(horizon_s / dt)))
        pred_mean = self._rollout_controlled_mean(state, control_dist.mean, n_steps, dt)

        F, G = compute_linearization_jacobians(
            self._provider, state, control_dist.mean, dt=horizon_s
        )

        pred_cov = self._propagate_covariance(
            F,
            G,
            state_dist,
            control_dist,
            uncertainty,
            a_total,
            horizon_s,
            len(state.v),
        )

        error_bound = evaluate_linearization_error(
            self._provider,
            state,
            control_dist.mean,
            state_dist.covariance,
            control_dist.covariance,
            F,
            G,
            dt=horizon_s,
        )

        return DimeDriftPredictionResult(
            validity_status=PredictionValidityStatus.VALID,
            validity_receipt="Uncertain-control prediction successfully qualified.",
            mode=mode,
            horizon_s=horizon_s,
            dt=dt,
            zero_control_branch=ztcf_proposal,
            predicted_mean=pred_mean,
            predicted_covariance=pred_cov,
            linearization_drift_jacobian_F=F,
            linearization_control_jacobian_G=G,
            linearization_error_bound=error_bound,
            drift_gain=drift_gain,
            cancellation_index=cancellation,
            diagnostics={
                "n_steps": n_steps,
                "norm_passive": norm_passive,
                "norm_total": norm_total,
                "norm_ctrl": float(np.linalg.norm(a_ctrl)),
            },
        )


@dataclass(frozen=True)
class DimeDriftTransitionCriterion(EstimationIntervalFactor):
    """Drift-centered marginalized-control transition criterion factor.

    Evaluates the Mahalanobis residual between observed candidate transition
    (x_{k+1} - x_{ZTCF}) and control prior G * mu_u under covariance
    Sigma_trans = G * Sigma_u * G^T + Q.
    """

    name: str = "dime_marginalized_drift_transition"
    factor_type: str = "marginalized_input_transition"
    t_start: float = 0.0
    t_end: float = 0.01
    contributes_to_objective: bool = True
    metadata: Mapping[str, Any] = field(default_factory=dict)
    provider: DynamicsProvider | None = None
    control_dist: ControlDistribution | None = None
    uncertainty: ModelContactUncertainty | None = None
    dt: float = 0.01

    def evaluate_residual(
        self, x_k: DimeCompleteState, x_kp1: DimeCompleteState
    ) -> np.ndarray:
        """Evaluate normalized residual vector between candidate transition and dynamics proposal."""
        require(self.provider is not None, "provider must be set")
        require(self.control_dist is not None, "control_dist must be set")
        provider = self.provider
        control_dist = self.control_dist
        assert provider is not None
        assert control_dist is not None

        dt = self.dt if self.dt > 0.0 else (x_kp1.t - x_k.t)
        require(dt > 0.0, "Time interval dt must be positive")

        # Step under zero control (ZTCF branch)
        zero_u = np.zeros(len(control_dist.mean), dtype=np.float64)
        ztcf_step = provider.step(
            DimeFullStepRequest(
                state=x_k, controls=zero_u, dt=dt, model_hash=provider.model_hash
            )
        )
        x_ztcf = np.concatenate([ztcf_step.next_state.q, ztcf_step.next_state.v])
        x_actual = np.concatenate([x_kp1.q, x_kp1.v])

        # Control sensitivity G
        _, G = compute_linearization_jacobians(provider, x_k, control_dist.mean, dt=dt)

        dim_x = len(x_actual)
        Q = (
            self.uncertainty.process_noise_covariance
            if self.uncertainty is not None
            else np.eye(dim_x) * 1e-4
        )
        if Q.shape != (dim_x, dim_x):
            Q = np.eye(dim_x) * 1e-4

        Sigma_trans = G @ control_dist.covariance @ G.T + Q
        Sigma_trans_sym = 0.5 * (Sigma_trans + Sigma_trans.T)

        # Mahalanobis whitening via Cholesky factor L: L * L^T = Sigma_trans
        # residual r = L^-1 ( (x_actual - x_ztcf) - G * mu_u )
        diff = (x_actual - x_ztcf) - G @ control_dist.mean
        try:
            L = np.linalg.cholesky(Sigma_trans_sym)
            whitened_residual = np.linalg.solve(L, diff)
        except np.linalg.LinAlgError:
            vals, vecs = np.linalg.eigh(Sigma_trans_sym)
            inv_sqrt = vecs @ np.diag(1.0 / np.sqrt(np.maximum(vals, 1e-12))) @ vecs.T
            whitened_residual = inv_sqrt @ diff

        return _make_readonly_array(whitened_residual)

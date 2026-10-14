"""Humanoid and manipulation transfer for MOSAIC (MOSAIC-12).

Demonstrates that MOSAIC estimation kernels transfer to humanoid retargeting and
manipulation:
- Known-payload anchor: an end-effector tool / grasped object provides a known-inertia
  force sensor to identify unknown payload parameters pi_obj with observability reports.
- Whole-Body Controller (WBC) export: outputs reference trajectory (q*, v*, a*),
  feedforward torques u*, identified inertias pi, and time-varying Riccati feedback
  gains K_t in structured dictionary and JSON artifact formats.
- WBC schedule validation: verifies round-trip serialization and simulated closed-loop
  replay with bounded feedback effort.
- Floating-base unactuated rows: validation for 6-DoF base dynamics.

NON-GOALS & NON-CLAIMS:
- No real-time execution guarantees: reference-plus-gains generation is an offline estimation pipeline.
- No global-optimality claims: the outer problem is non-convex and convergence is to local Gauss-Newton minima.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
import json
from typing import Any, TypeAlias

import numpy as np
import numpy.typing as npt

from src.shared.python.core.contracts import require
from src.shared.python.estimation.mosaic.inertial import SPATIAL_PARAMETERS_PER_BODY
from src.shared.python.estimation.mosaic.local_policy import (
    ReplayReport,
    replay_acceptance,
)

FloatArray: TypeAlias = npt.NDArray[np.float64]
StepFunction = Callable[[FloatArray, FloatArray], FloatArray]

NON_CLAIMS: tuple[str, ...] = (
    "No real-time execution guarantees: reference-plus-gains schedule generation is an offline pipeline.",
    "No global-optimality claims: outer problem is non-convex; convergence is to local Gauss-Newton minima.",
)

DEFAULT_GRAVITY_3D: FloatArray = np.array([0.0, 0.0, -9.81], dtype=np.float64)


# ---------------------------------------------------------------------------
# Spatial algebra helpers
# ---------------------------------------------------------------------------


def skew3(v: FloatArray) -> FloatArray:
    """Skew-symmetric 3x3 matrix [v]_x from 3-vector."""
    v_arr = np.asarray(v, dtype=np.float64)
    require(v_arr.shape == (3,), "v must be a 3-vector", v_arr.shape)
    return np.array(
        [
            [0.0, -v_arr[2], v_arr[1]],
            [v_arr[2], 0.0, -v_arr[0]],
            [-v_arr[1], v_arr[0], 0.0],
        ],
        dtype=np.float64,
    )


def spatial_payload_regressor(
    linear_acceleration: FloatArray,
    angular_velocity: FloatArray,
    angular_acceleration: FloatArray,
    gravity: FloatArray | None = None,
) -> FloatArray:
    """Batched 6x10 spatial inertial regressor A(t) such that W = A @ pi.

    The 10 spatial parameters per body follow MOSAIC conventions:
        pi = [m, h_x, h_y, h_z, I_xx, I_xy, I_xz, I_yy, I_yz, I_zz]
    Returns shape (N, 6, 10).
    """
    a_lin = np.asarray(linear_acceleration, dtype=np.float64)
    omega = np.asarray(angular_velocity, dtype=np.float64)
    alpha = np.asarray(angular_acceleration, dtype=np.float64)
    grav = (
        DEFAULT_GRAVITY_3D if gravity is None else np.asarray(gravity, dtype=np.float64)
    )

    n_samples = a_lin.shape[0]
    require(a_lin.shape == (n_samples, 3), "linear_acceleration must be (N, 3)")
    require(omega.shape == (n_samples, 3), "angular_velocity must be (N, 3)")
    require(alpha.shape == (n_samples, 3), "angular_acceleration must be (N, 3)")
    require(grav.shape == (3,), "gravity must be (3,)")

    regressor = np.zeros((n_samples, 6, SPATIAL_PARAMETERS_PER_BODY), dtype=np.float64)
    net_acc = a_lin - grav  # (N, 3)

    for t in range(n_samples):
        a_net_t = net_acc[t]
        w_t = omega[t]
        alpha_t = alpha[t]

        w_skew = skew3(w_t)
        alpha_skew = skew3(alpha_t)
        omega_mat = alpha_skew + w_skew @ w_skew

        # Force rows (0..2):
        # f = m * a_net + (alpha_skew + w_skew^2) * h
        regressor[t, :3, 0] = a_net_t
        regressor[t, :3, 1:4] = omega_mat

        # Torque rows (3..5):
        # tau = -[a_net]_x * h + I * alpha + w x (I * w)
        regressor[t, 3:6, 1:4] = -skew3(a_net_t)

        # Rotational inertia terms for flat [Ixx, Ixy, Ixz, Iyy, Iyz, Izz]
        # I * alpha:
        j_alpha = np.array(
            [
                [alpha_t[0], alpha_t[1], alpha_t[2], 0.0, 0.0, 0.0],
                [0.0, alpha_t[0], 0.0, alpha_t[1], alpha_t[2], 0.0],
                [0.0, 0.0, alpha_t[0], 0.0, alpha_t[1], alpha_t[2]],
            ],
            dtype=np.float64,
        )
        # I * omega:
        j_omega = np.array(
            [
                [w_t[0], w_t[1], w_t[2], 0.0, 0.0, 0.0],
                [0.0, w_t[0], 0.0, w_t[1], w_t[2], 0.0],
                [0.0, 0.0, w_t[0], 0.0, w_t[1], w_t[2]],
            ],
            dtype=np.float64,
        )
        regressor[t, 3:6, 4:] = j_alpha + w_skew @ j_omega

    return regressor


# ---------------------------------------------------------------------------
# Payload Identification
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PayloadObservabilityReport:
    """Observability diagnostic for spatial payload inertia identification."""

    condition_number: float
    singular_values: FloatArray
    observable_dimension: int
    observable_fraction: float
    is_fully_observable: bool


@dataclass(frozen=True)
class PayloadIdentificationResult:
    """Estimated payload inertial parameters and observability report."""

    parameters: FloatArray
    report: PayloadObservabilityReport
    residual_rms: float


def identify_payload_inertia(
    wrench: FloatArray,
    linear_acceleration: FloatArray,
    angular_velocity: FloatArray,
    angular_acceleration: FloatArray,
    known_tool_inertia: FloatArray | None = None,
    gravity: FloatArray | None = None,
    svd_threshold: float = 1e-4,
) -> PayloadIdentificationResult:
    """Identify 10-parameter spatial inertia of an unknown payload from wrench data.

    When known_tool_inertia is provided, the tool's contribution is subtracted
    from the transmitted wrench, playing the role of a known-inertia force sensor.
    """
    w_meas = np.asarray(wrench, dtype=np.float64)
    require(
        w_meas.ndim == 2 and w_meas.shape[1] == 6, "wrench must be (N, 6)", w_meas.shape
    )
    n_samples = w_meas.shape[0]

    require(
        linear_acceleration.shape == (n_samples, 3),
        "linear_acceleration shape mismatch",
    )
    require(
        angular_velocity.shape == (n_samples, 3),
        "angular_velocity shape mismatch",
    )
    require(
        angular_acceleration.shape == (n_samples, 3),
        "angular_acceleration shape mismatch",
    )

    regressor = spatial_payload_regressor(
        linear_acceleration=linear_acceleration,
        angular_velocity=angular_velocity,
        angular_acceleration=angular_acceleration,
        gravity=gravity,
    )
    # Stacked regressor: (6*N, 10)
    a_flat = regressor.reshape(-1, SPATIAL_PARAMETERS_PER_BODY)

    if known_tool_inertia is not None:
        pi_tool = np.asarray(known_tool_inertia, dtype=np.float64)
        require(
            pi_tool.shape == (SPATIAL_PARAMETERS_PER_BODY,),
            "known_tool_inertia must be 10-vector",
            pi_tool.shape,
        )
        w_tool = a_flat @ pi_tool
        w_net = w_meas.ravel() - w_tool
    else:
        w_net = w_meas.ravel()

    # SVD observability analysis
    svals = np.asarray(np.linalg.svd(a_flat, compute_uv=False), dtype=np.float64)
    sigma_max = float(svals[0])
    sigma_min = float(svals[-1])
    cond_num = sigma_max / max(1e-15, sigma_min)

    cutoff = svd_threshold * sigma_max
    obs_dim = int(np.sum(svals > cutoff))
    obs_fraction = float(obs_dim / SPATIAL_PARAMETERS_PER_BODY)
    is_fully_obs = obs_dim == SPATIAL_PARAMETERS_PER_BODY

    report = PayloadObservabilityReport(
        condition_number=cond_num,
        singular_values=svals,
        observable_dimension=obs_dim,
        observable_fraction=obs_fraction,
        is_fully_observable=is_fully_obs,
    )

    sol_raw, _, _, _ = np.linalg.lstsq(a_flat, w_net, rcond=svd_threshold)
    sol = np.asarray(sol_raw, dtype=np.float64)
    residual_rms = float(np.sqrt(np.mean((a_flat @ sol - w_net) ** 2)))

    return PayloadIdentificationResult(
        parameters=sol,
        report=report,
        residual_rms=residual_rms,
    )


# ---------------------------------------------------------------------------
# WBC Schedule Export and Validation
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class WBCSchedule:
    """Structured whole-body-controller reference trajectory and feedback gains schedule."""

    times: FloatArray
    q_ref: FloatArray
    v_ref: FloatArray
    a_ref: FloatArray
    u_feedforward: FloatArray
    gains_k: FloatArray
    inertial_parameters: FloatArray
    geometry_parameters: FloatArray
    model_name: str = "generic_humanoid"
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        t_count = self.times.size
        require(self.times.ndim == 1, "times must be 1-D")
        require(self.q_ref.shape[0] == t_count, "q_ref length mismatch")
        require(self.v_ref.shape[0] == t_count, "v_ref length mismatch")
        require(self.a_ref.shape[0] == t_count, "a_ref length mismatch")
        require(self.u_feedforward.shape[0] == t_count, "u_feedforward length mismatch")
        require(self.gains_k.shape[0] == t_count, "gains_k length mismatch")

    def to_dict(self) -> dict[str, Any]:
        """Convert schedule to JSON-serializable dictionary."""
        return {
            "model_name": self.model_name,
            "times": self.times.tolist(),
            "q_ref": self.q_ref.tolist(),
            "v_ref": self.v_ref.tolist(),
            "a_ref": self.a_ref.tolist(),
            "u_feedforward": self.u_feedforward.tolist(),
            "gains_k": self.gains_k.tolist(),
            "inertial_parameters": self.inertial_parameters.tolist(),
            "geometry_parameters": self.geometry_parameters.tolist(),
            "metadata": dict(self.metadata),
        }

    def to_json(self, indent: int = 2) -> str:
        """Serialize schedule to JSON formatted string."""
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> WBCSchedule:
        """Reconstruct WBCSchedule from dictionary."""
        return cls(
            times=np.array(data["times"], dtype=np.float64),
            q_ref=np.array(data["q_ref"], dtype=np.float64),
            v_ref=np.array(data["v_ref"], dtype=np.float64),
            a_ref=np.array(data["a_ref"], dtype=np.float64),
            u_feedforward=np.array(data["u_feedforward"], dtype=np.float64),
            gains_k=np.array(data["gains_k"], dtype=np.float64),
            inertial_parameters=np.array(data["inertial_parameters"], dtype=np.float64),
            geometry_parameters=np.array(data["geometry_parameters"], dtype=np.float64),
            model_name=data.get("model_name", "generic_humanoid"),
            metadata=data.get("metadata", {}),
        )

    @classmethod
    def from_json(cls, json_str: str) -> WBCSchedule:
        """Reconstruct WBCSchedule from JSON string."""
        return cls.from_dict(json.loads(json_str))


def export_wbc_schedule(
    times: FloatArray,
    q_ref: FloatArray,
    v_ref: FloatArray,
    a_ref: FloatArray,
    u_feedforward: FloatArray,
    gains_k: FloatArray,
    inertial_parameters: FloatArray,
    **kwargs: Any,
) -> WBCSchedule:
    """Export fitted trajectory, inertias, and feedback gains to structured WBC schedule."""
    geometry_parameters: FloatArray = kwargs.get(
        "geometry_parameters", np.array([], dtype=np.float64)
    )
    model_name: str = kwargs.get("model_name", "generic_humanoid")
    extra_metadata: dict[str, Any] | None = kwargs.get("extra_metadata")

    meta: dict[str, Any] = {
        "non_claims": list(NON_CLAIMS),
        "source": "MOSAIC_estimation",
    }
    if extra_metadata is not None:
        meta.update(extra_metadata)

    return WBCSchedule(
        times=np.asarray(times, dtype=np.float64),
        q_ref=np.asarray(q_ref, dtype=np.float64),
        v_ref=np.asarray(v_ref, dtype=np.float64),
        a_ref=np.asarray(a_ref, dtype=np.float64),
        u_feedforward=np.asarray(u_feedforward, dtype=np.float64),
        gains_k=np.asarray(gains_k, dtype=np.float64),
        inertial_parameters=np.asarray(inertial_parameters, dtype=np.float64),
        geometry_parameters=np.asarray(geometry_parameters, dtype=np.float64),
        model_name=model_name,
        metadata=meta,
    )


def validate_wbc_schedule(
    schedule: WBCSchedule,
    step_fn: StepFunction,
    initial_perturbation: FloatArray,
    position_tolerance: float,
    dt: float,
) -> ReplayReport:
    """Validate WBC schedule via JSON round-trip and simulated closed-loop tracking replay."""
    # 1. Round-trip verification
    serialized = schedule.to_json()
    reconstructed = WBCSchedule.from_json(serialized)
    np.testing.assert_allclose(schedule.q_ref, reconstructed.q_ref, atol=1e-12)
    np.testing.assert_allclose(schedule.gains_k, reconstructed.gains_k, atol=1e-12)

    # 2. Closed-loop tracking replay across the schedule horizon
    ref_states = np.concatenate([schedule.q_ref, schedule.v_ref], axis=-1)
    n_nodes = ref_states.shape[0]
    require(n_nodes >= 2, "schedule must contain at least 2 time nodes for replay")

    if schedule.u_feedforward.shape[0] == n_nodes:
        inputs = schedule.u_feedforward[:-1]
        gains = schedule.gains_k[:-1]
    else:
        inputs = schedule.u_feedforward
        gains = schedule.gains_k

    return replay_acceptance(
        step=step_fn,
        reference=ref_states,
        inputs=inputs,
        gains=gains,
        initial_perturbation=initial_perturbation,
        position_tolerance=position_tolerance,
        dt=dt,
    )


def floating_base_actuation_mask(
    n_dof: int, n_floating_dof: int = 6
) -> npt.NDArray[np.bool_]:
    """Return actuation mask indicating unactuated floating base DoFs."""
    require(
        n_dof >= n_floating_dof,
        "n_dof must be >= n_floating_dof",
        (n_dof, n_floating_dof),
    )
    require(n_floating_dof >= 0, "n_floating_dof must be non-negative")
    mask = np.ones(n_dof, dtype=bool)
    mask[:n_floating_dof] = False
    return mask

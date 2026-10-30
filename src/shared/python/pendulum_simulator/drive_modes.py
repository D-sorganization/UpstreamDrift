"""Drive-mode switch and variational STM sensitivity for canonical 3DOF (SAP-4b, #13886)."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math
from typing import Any, Sequence

from double_pendulum_model.physics import triple_pendulum as tp
import numpy as np

from src.shared.python.loop_nullspace.impedance_spec import JointImpedance

_CANONICAL_DYN = tp.TriplePendulumDynamics()
_UNDAMPED_DYN = tp.TriplePendulumDynamics(
    tp.TriplePendulumParameters(
        segments=_CANONICAL_DYN.parameters.segments, damping=(0.0, 0.0, 0.0)
    )
)


class DriveMode(str, Enum):
    TORQUE_UNDAMPED = "torque_undamped"
    TORQUE_CANONICAL = "torque_canonical"
    JOINT_IMPEDANCE = "joint_impedance"


@dataclass(frozen=True)
class NominalTrajectory:
    times: np.ndarray
    q: np.ndarray
    v: np.ndarray
    torques: np.ndarray
    stage_states: np.ndarray
    stage_undamped_torques: np.ndarray

    def at(self, t: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        t0, t1 = float(self.times[0]), float(self.times[-1])
        if not math.isfinite(t) or t < t0 - 1e-12 or t > t1 + 1e-12:
            raise ValueError(f"time {t} outside nominal grid [{t0}, {t1}]")
        q_t = np.array([np.interp(t, self.times, self.q[:, i]) for i in range(3)])
        v_t = np.array([np.interp(t, self.times, self.v[:, i]) for i in range(3)])
        idx = int(np.searchsorted(self.times, t, side="right")) - 1
        tau_c = self.torques[min(max(idx, 0), len(self.torques) - 1)]
        st = tp.TriplePendulumState(*q_t, *v_t)
        acc = _CANONICAL_DYN.forward_dynamics(st, tuple(tau_c))
        return q_t, v_t, tau_c, np.asarray(_UNDAMPED_DYN.inverse_dynamics(st, acc))


@dataclass(frozen=True)
class DriveModeSensitivityReceipt:
    drive_mode: str
    impedance_hypotheses: tuple[str, ...]
    sigma_max_window: float
    singular_values_window: tuple[float, ...]
    ftle_window: float
    det_phi_window: float
    directional_gains: dict[str, float]
    phi_window_norm: np.ndarray
    validation_status: str = "UNVALIDATED"

    def __post_init__(self) -> None:
        sv = self.singular_values_window
        if any(not math.isfinite(s) or s < 0.0 for s in sv):
            raise ValueError("singular_values_window must be finite and non-negative")
        if list(sv) != sorted(sv, reverse=True):
            raise ValueError("singular_values_window must be sorted descending")
        if not math.isfinite(self.det_phi_window) or self.det_phi_window <= 0.0:
            raise ValueError("det_phi_window must be positive")

    def gain_along(self, direction: str | Sequence[float] | np.ndarray) -> float:
        if isinstance(direction, str):
            if direction not in self.directional_gains:
                raise ValueError(f"unknown direction key: {direction!r}")
            return float(self.directional_gains[direction])
        vec = np.asarray(direction, dtype=np.float64)
        norm = float(np.linalg.norm(vec))
        if vec.shape != (6,) or not np.all(np.isfinite(vec)) or norm <= 0.0:
            raise ValueError("direction must be a finite non-zero 6-vector")
        return float(np.linalg.norm(self.phi_window_norm @ (vec / norm)))

    def to_dict(self) -> dict[str, Any]:
        sv = [round(s, 6) for s in self.singular_values_window]
        dg = {k: round(v, 6) for k, v in self.directional_gains.items()}
        return {
            "drive_mode": self.drive_mode,
            "impedance_hypotheses": list(self.impedance_hypotheses),
            "sigma_max_window": round(self.sigma_max_window, 6),
            "singular_values_window": sv,
            "ftle_window": round(self.ftle_window, 6),
            "det_phi_window": round(self.det_phi_window, 6),
            "directional_gains": dg,
            "validation_status": self.validation_status,
        }


def _coerce_mode(mode: DriveMode | str) -> DriveMode:
    try:
        return mode if isinstance(mode, DriveMode) else DriveMode(mode)
    except ValueError as exc:
        raise ValueError(f"unknown drive mode: {mode!r}") from exc


def drive_torque(
    mode: DriveMode | str,
    t: float,
    q: Any,
    v: Any,
    nominal: NominalTrajectory,
    impedance: JointImpedance | None = None,
    ref_stage: Any = None,
) -> tuple[float, float, float]:
    m = _coerce_mode(mode)
    st_r, tau_c, tau_u = ref_stage or (None, *nominal.at(t)[2:])
    q_ref, v_ref = (st_r[:3], st_r[3:]) if st_r is not None else nominal.at(t)[:2]
    if m is DriveMode.TORQUE_CANONICAL:
        return (float(tau_c[0]), float(tau_c[1]), float(tau_c[2]))
    if m is DriveMode.TORQUE_UNDAMPED:
        return (float(tau_u[0]), float(tau_u[1]), float(tau_u[2]))
    if impedance is None:
        raise ValueError("impedance is required for JOINT_IMPEDANCE mode")
    if impedance.is_zero:
        return (float(tau_c[0]), float(tau_c[1]), float(tau_c[2]))
    dq, dv = q_ref - np.asarray(q, dtype=float), v_ref - np.asarray(v, dtype=float)
    tau = tau_c + impedance.torque(dq, dv, t)
    return (float(tau[0]), float(tau[1]), float(tau[2]))

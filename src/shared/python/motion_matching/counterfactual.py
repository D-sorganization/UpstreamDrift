"""Versioned native counterfactual intervention and spatial wrench contracts.

Implements CF-1, CF-2, and CF-3 for epic #10286:
- Explicit intervention contracts (actual, ZTCF, ZVCF)
- 3D spatial wrench accounting and moment transport
- Native constrained counterfactual provider with acceleration & reaction closures
- 3D spatial power, work, and impulse integration
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
import logging
from types import MappingProxyType
from typing import Any, Literal, Protocol

import numpy as np
from numpy.typing import NDArray

from src.shared.python.biomechanics.interaction_evidence import (
    InterfaceDescriptor,
    SpatialWrenchTrajectory,
)

logger = logging.getLogger(__name__)

SCHEMA_VERSION = "upstreamdrift/native-counterfactual/1"
_CLOSURE_RTOL = 1e-9
_CLOSURE_ATOL = 1e-10

SplitName = Literal[
    "actual", "ztcf", "zvcf", "control", "control_preserved_zero_velocity"
]


def _finite_3d_vector(name: str, val: Any) -> NDArray[np.float64]:
    arr = np.asarray(val, dtype=np.float64).reshape(-1)
    if arr.size != 3:
        raise ValueError(f"{name} must have shape (3,), got {arr.shape}")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must contain only finite numbers")
    arr.flags.writeable = False
    return arr


@dataclass(frozen=True, slots=True)
class SpatialWrench:
    """3D spatial wrench with explicit frame, application point, and action direction.

    Layout of the 6D wrench vector is [Fx, Fy, Fz, Mx, My, Mz].
    Units: Force in Newtons (N), Moment/Torque in Newton-meters (N*m).
    """

    force_N: NDArray[np.float64]
    torque_Nm: NDArray[np.float64]
    point_of_application_m: NDArray[np.float64]
    frame: str = "world"
    action_direction: str = "proximal_on_distal"
    units: str = "SI"

    def __post_init__(self) -> None:
        force = _finite_3d_vector("force_N", self.force_N)
        torque = _finite_3d_vector("torque_Nm", self.torque_Nm)
        point = _finite_3d_vector("point_of_application_m", self.point_of_application_m)
        if not self.frame.strip():
            raise ValueError("frame must be a non-empty string")
        if not self.action_direction.strip():
            raise ValueError("action_direction must be a non-empty string")
        if not self.units.strip():
            raise ValueError("units must be a non-empty string")
        object.__setattr__(self, "force_N", force)
        object.__setattr__(self, "torque_Nm", torque)
        object.__setattr__(self, "point_of_application_m", point)

    @property
    def vector(self) -> NDArray[np.float64]:
        """Return the combined 6D vector [Fx, Fy, Fz, Mx, My, Mz]."""
        vec = np.concatenate([self.force_N, self.torque_Nm])
        vec.flags.writeable = False
        return vec

    def change_point_of_application(self, new_point_m: Any) -> SpatialWrench:
        """Transport the moment origin according to Varignon's theorem.

        M_P = M_O + r_{P/O} x F = M_O + (O - P) x F.
        """
        new_point = _finite_3d_vector("new_point_m", new_point_m)
        lever_from_new_to_old = self.point_of_application_m - new_point
        transported_moment = self.torque_Nm + np.cross(
            lever_from_new_to_old, self.force_N
        )
        return SpatialWrench(
            force_N=self.force_N,
            torque_Nm=transported_moment,
            point_of_application_m=new_point,
            frame=self.frame,
            action_direction=self.action_direction,
            units=self.units,
        )

    def negate(self) -> SpatialWrench:
        """Return the equal and opposite reaction wrench (Newton's 3rd law)."""
        new_direction = (
            "distal_on_proximal"
            if self.action_direction == "proximal_on_distal"
            else "proximal_on_distal"
        )
        return SpatialWrench(
            force_N=-self.force_N,
            torque_Nm=-self.torque_Nm,
            point_of_application_m=self.point_of_application_m,
            frame=self.frame,
            action_direction=new_direction,
            units=self.units,
        )


@dataclass(frozen=True)
class PointwiseCounterfactualSample:
    """Pointwise evaluation of actual, ZTCF, and ZVCF dynamics and constraint reactions."""

    time_s: float
    coordinates: Mapping[str, float]
    rates: Mapping[str, float]
    applied_efforts: Mapping[str, float]
    actual_acceleration: Mapping[str, float]
    ztcf_acceleration: Mapping[str, float]
    zvcf_acceleration: Mapping[str, float]
    control_increment_acceleration: Mapping[str, float]
    actual_reaction_wrench: SpatialWrench
    ztcf_reaction_wrench: SpatialWrench
    zvcf_reaction_wrench: SpatialWrench
    control_increment_reaction_wrench: SpatialWrench
    is_affine: bool = True
    parent_run_id: str | None = None
    schema_version: str = SCHEMA_VERSION


class ConstrainedDynamicsModel(Protocol):
    """Protocol for models capable of evaluating constrained forward dynamics and reactions."""

    def evaluate_dynamics(
        self,
        q: Mapping[str, float],
        v: Mapping[str, float],
        u: Mapping[str, float],
    ) -> tuple[dict[str, float], SpatialWrench]:
        """Evaluate (accelerations, reaction_wrench) for state (q, v) and applied efforts u."""
        ...


class NativeConstrainedCounterfactualProvider:
    """Evaluates pointwise counterfactuals on constrained multibody models without mutating caller state."""

    def __init__(self, model: Any) -> None:
        self._model = model

    def _eval_state(
        self, q: Mapping[str, float], v: Mapping[str, float], u: Mapping[str, float]
    ) -> tuple[dict[str, float], SpatialWrench]:
        if hasattr(self._model, "evaluate_dynamics"):
            return self._model.evaluate_dynamics(q, v, u)
        if hasattr(self._model, "accelerations") and hasattr(
            self._model, "closure_reaction_wrench"
        ):
            accel = self._model.accelerations(q, v, u)
            force, torque, point = self._model.closure_reaction_wrench("world")
            wrench = SpatialWrench(
                force_N=force, torque_Nm=torque, point_of_application_m=point
            )
            return accel, wrench
        raise NotImplementedError(
            "Model does not implement constrained dynamics evaluation"
        )

    def evaluate_pointwise(
        self,
        coordinates: Mapping[str, float],
        rates: Mapping[str, float],
        applied_efforts: Mapping[str, float],
        *,
        time_s: float = 0.0,
        parent_run_id: str | None = None,
    ) -> PointwiseCounterfactualSample:
        """Compute actual, ZTCF, and ZVCF accelerations and reaction wrenches at state (q, v, u)."""
        q = MappingProxyType(dict(coordinates))
        v = MappingProxyType(dict(rates))
        u = MappingProxyType(dict(applied_efforts))
        zero_u = MappingProxyType(dict.fromkeys(u, 0.0))
        zero_v = MappingProxyType(dict.fromkeys(v, 0.0))

        # 1. Actual evaluation: (q, v, u)
        act_accel, act_wrench = self._eval_state(q, v, u)

        # 2. Pointwise ZTCF: (q, v, u=0)
        ztcf_accel, ztcf_wrench = self._eval_state(q, v, zero_u)

        # 3. Pointwise ZVCF: (q, v=0, u=0)
        zvcf_accel, zvcf_wrench = self._eval_state(q, zero_v, zero_u)

        # 4. Control increments: actual - ztcf
        ctrl_accel = {k: act_accel[k] - ztcf_accel[k] for k in act_accel}
        ctrl_wrench = SpatialWrench(
            force_N=act_wrench.force_N - ztcf_wrench.force_N,
            torque_Nm=act_wrench.torque_Nm - ztcf_wrench.torque_Nm,
            point_of_application_m=act_wrench.point_of_application_m,
            frame=act_wrench.frame,
            action_direction=act_wrench.action_direction,
            units=act_wrench.units,
        )

        return PointwiseCounterfactualSample(
            time_s=time_s,
            coordinates=q,
            rates=v,
            applied_efforts=u,
            actual_acceleration=MappingProxyType(act_accel),
            ztcf_acceleration=MappingProxyType(ztcf_accel),
            zvcf_acceleration=MappingProxyType(zvcf_accel),
            control_increment_acceleration=MappingProxyType(ctrl_accel),
            actual_reaction_wrench=act_wrench,
            ztcf_reaction_wrench=ztcf_wrench,
            zvcf_reaction_wrench=zvcf_wrench,
            control_increment_reaction_wrench=ctrl_wrench,
            parent_run_id=parent_run_id,
        )


@dataclass(frozen=True)
class CounterfactualTrajectory:
    """Collection of time-series counterfactual evaluations with 3D power and work accounting."""

    time_s: NDArray[np.float64]
    coordinate_names: tuple[str, ...]
    actual_accelerations: NDArray[np.float64]
    ztcf_accelerations: NDArray[np.float64]
    control_accelerations: NDArray[np.float64]
    zvcf_accelerations: NDArray[np.float64]
    actual_wrenches: NDArray[np.float64]
    ztcf_wrenches: NDArray[np.float64]
    control_wrenches: NDArray[np.float64]
    zvcf_wrenches: NDArray[np.float64]
    twist: NDArray[np.float64]
    parent_run_id: str | None = None
    model_tier: str = "native_constrained"

    def __post_init__(self) -> None:
        t = np.asarray(self.time_s, dtype=np.float64).reshape(-1)
        if t.size < 2 or not np.all(np.isfinite(t)):
            raise ValueError("time_s must contain at least two finite samples")
        if np.any(np.diff(t) <= 0.0):
            raise ValueError("time_s must be strictly increasing")
        n = t.size
        d = len(self.coordinate_names)
        for name, arr, expected in (
            ("actual_accelerations", self.actual_accelerations, (n, d)),
            ("ztcf_accelerations", self.ztcf_accelerations, (n, d)),
            ("control_accelerations", self.control_accelerations, (n, d)),
            ("zvcf_accelerations", self.zvcf_accelerations, (n, d)),
            ("actual_wrenches", self.actual_wrenches, (n, 6)),
            ("ztcf_wrenches", self.ztcf_wrenches, (n, 6)),
            ("control_wrenches", self.control_wrenches, (n, 6)),
            ("zvcf_wrenches", self.zvcf_wrenches, (n, 6)),
            ("twist", self.twist, (n, 6)),
        ):
            a = np.asarray(arr, dtype=np.float64)
            if a.shape != expected or not np.all(np.isfinite(a)):
                raise ValueError(f"{name} must have finite shape {expected}")

    def _wrench(self, split: str) -> NDArray[np.float64]:
        mapping = {
            "actual": self.actual_wrenches,
            "total": self.actual_wrenches,
            "ztcf": self.ztcf_wrenches,
            "drift": self.ztcf_wrenches,
            "control": self.control_wrenches,
            "zvcf": self.zvcf_wrenches,
        }
        if split not in mapping:
            raise ValueError(f"Unknown split: {split}")
        return mapping[split]

    def power(self, split: str) -> NDArray[np.float64]:
        """Compute 3D instantaneous spatial power P = F . v + M . omega."""
        wrench = self._wrench(split)
        # wrench is [Fx, Fy, Fz, Mx, My, Mz], twist is [vx, vy, vz, wx, wy, wz]
        # P = F . v + M . omega = sum(wrench * twist, axis=1)
        p = np.sum(wrench * self.twist, axis=1)
        p.flags.writeable = False
        return p

    def work(self, split: str) -> NDArray[np.float64]:
        """Compute cumulative mechanical work W(t) = integral_0^t P(tau) dtau."""
        p = self.power(split)
        w = np.zeros_like(p)
        dt = np.diff(self.time_s)
        avg_p = 0.5 * (p[:-1] + p[1:])
        w[1:] = np.cumsum(avg_p * dt)
        w.flags.writeable = False
        return w

    def linear_impulse(self, split: str) -> NDArray[np.float64]:
        """Compute cumulative linear impulse I(t) = integral_0^t F(tau) dtau."""
        force = self._wrench(split)[:, :3]
        imp = np.zeros_like(force)
        dt = np.diff(self.time_s)[:, None]
        avg_f = 0.5 * (force[:-1] + force[1:])
        imp[1:] = np.cumsum(avg_f * dt, axis=0)
        imp.flags.writeable = False
        return imp

    def angular_impulse(self, split: str) -> NDArray[np.float64]:
        """Compute cumulative angular impulse L(t) = integral_0^t M(tau) dtau."""
        moment = self._wrench(split)[:, 3:]
        imp = np.zeros_like(moment)
        dt = np.diff(self.time_s)[:, None]
        avg_m = 0.5 * (moment[:-1] + moment[1:])
        imp[1:] = np.cumsum(avg_m * dt, axis=0)
        imp.flags.writeable = False
        return imp

    def to_interaction_evidence(
        self,
        interface_name: str = "lead_hand_weld",
        proximal_body: str = "hub",
        distal_body: str = "grip",
        frame: str = "world",
    ) -> SpatialWrenchTrajectory:
        """Export as an engine-neutral SpatialWrenchTrajectory."""
        desc = InterfaceDescriptor(
            name=interface_name,
            proximal_body=proximal_body,
            distal_body=distal_body,
            frame=frame,
            reference_point="joint_origin",
            action_direction="proximal_on_distal",
        )
        samples = self.time_s.size
        ref_pos = np.zeros((samples, 1, 3))
        w_total = self.actual_wrenches[:, None, :]
        w_drift = self.ztcf_wrenches[:, None, :]
        w_ctrl = self.control_wrenches[:, None, :]
        w_zvcf = self.zvcf_wrenches[:, None, :]
        twist_expanded = self.twist[:, None, :]

        return SpatialWrenchTrajectory(
            time=self.time_s,
            interfaces=(desc,),
            reference_position_m=ref_pos,
            wrench_total=w_total,
            wrench_drift=w_drift,
            wrench_control=w_ctrl,
            twist=twist_expanded,
            model_tier=self.model_tier,
            wrench_zvcf=w_zvcf,
        )

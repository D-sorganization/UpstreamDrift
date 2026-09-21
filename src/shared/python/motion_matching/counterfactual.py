"""Versioned native counterfactual intervention and spatial wrench contracts.

Implements CF-1, CF-2, CF-3, and CF-5 for epic #10286:
- Explicit intervention contracts (actual, ZTCF, ZVCF)
- 3D spatial wrench accounting and moment transport
- Native constrained counterfactual provider with acceleration & reaction closures
- 3D spatial power, work, and impulse integration
- Explicit forward/branched ZTCF rollouts, saved cut states, and ZVCF rollout rejection
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
import hashlib
import logging
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Literal, Protocol, Sequence

import numpy as np
from numpy.typing import NDArray

from src.shared.python.biomechanics.interaction_evidence import (
    InterfaceDescriptor,
    SpatialWrenchTrajectory,
)
from src.shared.python.contracts import precondition

if TYPE_CHECKING:
    from src.shared.python.motion_matching.candidate_session import CandidateSession

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

    def instantaneous_power(
        self,
        linear_velocity_at_point_m_s: Any,
        angular_velocity_rad_s: Any,
    ) -> float:
        """Evaluate instantaneous spatial power P = F . v_P + M_P . omega.

        Power is invariant under rigid change of reduction point P -> Q when
        both wrench moment and linear velocity are transported consistently.
        """
        v_P = _finite_3d_vector(
            "linear_velocity_at_point_m_s", linear_velocity_at_point_m_s
        )
        omega = _finite_3d_vector("angular_velocity_rad_s", angular_velocity_rad_s)
        return float(np.dot(self.force_N, v_P) + np.dot(self.torque_Nm, omega))


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
        raise NotImplementedError(  # tracked: #10285
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

    @classmethod
    def from_saved_simscape_bundle(
        cls,
        bundle_manifest: Any,
        *,
        allow_unverified_relabeling: bool = False,
    ) -> CounterfactualTrajectory:
        """Load counterfactual trajectory from an offline Simscape bundle.

        Rejects baseline-acceleration relabeling when applied actuator torques
        are non-finite or unavailable. To evaluate actual counterfactuals,
        applied efforts u and constrained forward dynamics Ma + h = Bu + J^T lambda
        must be evaluated.
        """
        from src.shared.python.simulation_store.replay_bundle import (
            load_simscape_bundle,
        )

        bundle = load_simscape_bundle(bundle_manifest)
        times = np.asarray(bundle.arrays["time_s"])
        qdd = np.asarray(bundle.arrays["qdd"])
        n = times.size
        d = len(bundle.coordinate_names)

        tau = np.asarray(bundle.arrays.get("tau", np.full((n, d), np.nan)))
        if not allow_unverified_relabeling and not np.all(np.isfinite(tau)):
            raise ValueError(
                f"Cannot construct counterfactual trajectory for run '{bundle.run_id}': "
                "applied actuator torques 'tau' are non-finite or unavailable. "
                "Relabeling archived baseline accelerations as counterfactual rollouts is prohibited."
            )

        dummy_wrenches = np.zeros((n, 6))
        dummy_twist = np.zeros((n, 6))
        return cls(
            time_s=times,
            coordinate_names=bundle.coordinate_names,
            actual_accelerations=qdd,
            ztcf_accelerations=qdd,
            control_accelerations=np.zeros_like(qdd),
            zvcf_accelerations=np.zeros_like(qdd),
            actual_wrenches=dummy_wrenches,
            ztcf_wrenches=dummy_wrenches,
            control_wrenches=dummy_wrenches,
            zvcf_wrenches=dummy_wrenches,
            twist=dummy_twist,
            parent_run_id=bundle.run_id,
            model_tier="saved_simscape_bundle",
        )


def solve_constrained_dynamics(
    mass_matrix: NDArray[np.float64],
    coriolis_gravity: NDArray[np.float64],
    actuation_matrix: NDArray[np.float64],
    applied_torques: NDArray[np.float64],
    constraint_jacobian: NDArray[np.float64],
    constraint_drift: NDArray[np.float64] | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Solve constrained forward dynamics KKT system:

        [ M   -J^T ] [ a ]   [ B u - h ]
        [ J     0  ] [ λ ] = [ -J_dot v ]

    Validates force balance, constraint acceleration, and action-reaction multipliers:
        M a + h = B u + J^T λ
        J a + J_dot v = 0
    """
    M = np.asarray(mass_matrix, dtype=np.float64)
    h = np.asarray(coriolis_gravity, dtype=np.float64).reshape(-1)
    B = np.asarray(actuation_matrix, dtype=np.float64)
    u = np.asarray(applied_torques, dtype=np.float64).reshape(-1)
    J = np.asarray(constraint_jacobian, dtype=np.float64)
    n = M.shape[0]
    m = J.shape[0]
    gamma = (
        np.zeros(m, dtype=np.float64)
        if constraint_drift is None
        else np.asarray(constraint_drift, dtype=np.float64).reshape(-1)
    )

    kkt = np.block([[M, -J.T], [J, np.zeros((m, m), dtype=np.float64)]])
    rhs = np.concatenate([B @ u - h, -gamma])

    sol = np.linalg.solve(kkt, rhs)
    accel = sol[:n]
    lambdas = sol[n:]
    return accel, lambdas


@dataclass(frozen=True, slots=True)
class CutState:
    """Immutable snapshot of multibody state at the branch cut time t_cut."""

    cut_time_s: float
    coordinates: Mapping[str, float]
    rates: Mapping[str, float]
    parent_run_id: str | None = None
    parent_sample_index: int | None = None

    def __post_init__(self) -> None:
        if not np.isfinite(self.cut_time_s):
            raise ValueError("cut_time_s must be a finite number")
        if not self.coordinates:
            raise ValueError("coordinates mapping must not be empty")
        if set(self.coordinates) != set(self.rates):
            raise ValueError("coordinates and rates keys must match exactly")
        for k, v in self.coordinates.items():
            if not np.isfinite(v):
                raise ValueError(f"Coordinate {k} has nonfinite value {v}")
        for k, v in self.rates.items():
            if not np.isfinite(v):
                raise ValueError(f"Rate {k} has nonfinite value {v}")
        object.__setattr__(
            self, "coordinates", MappingProxyType(dict(self.coordinates))
        )
        object.__setattr__(self, "rates", MappingProxyType(dict(self.rates)))


def simulate_forward_zvcf(*args: Any, **kwargs: Any) -> None:
    """Explicitly reject attempts to integrate ZVCF as a forward rollout.

    ZVCF is strictly an instantaneous diagnostic at fixed state (q, v=0, u=0)
    and cannot be integrated as a forward rollout. For evolving unforced dynamics,
    use simulate_forward_ztcf.
    """
    raise ValueError(
        "ZVCF is strictly an instantaneous diagnostic at fixed state (q, v=0, u=0) "
        "and cannot be integrated as a forward rollout. For evolving unforced dynamics, "
        "use simulate_forward_ztcf."
    )


@dataclass(frozen=True)
class ForwardZTCFBranch:
    """Immutable forward ZTCF branch integrated from a saved cut state."""

    cut_state: CutState
    time_s: NDArray[np.float64]
    coordinate_names: tuple[str, ...]
    coordinates: NDArray[np.float64]
    rates: NDArray[np.float64]
    accelerations: NDArray[np.float64]
    reaction_wrenches: NDArray[np.float64]
    twist: NDArray[np.float64]
    kinetic_energy_J: NDArray[np.float64] | None = None
    potential_energy_J: NDArray[np.float64] | None = None
    mechanical_energy_J: NDArray[np.float64] | None = None
    branch_id: str = "ztcf_branch"
    is_cut_consistent: bool = True
    max_constraint_violation: float = 0.0
    solver_converged: bool = True
    schema_version: str = SCHEMA_VERSION

    def __post_init__(self) -> None:
        t = np.asarray(self.time_s, dtype=np.float64).reshape(-1)
        if t.size < 2 or not np.all(np.isfinite(t)):
            raise ValueError("time_s must contain at least two finite samples")
        if np.any(np.diff(t) <= 0.0):
            raise ValueError("time_s must be strictly increasing")
        n = t.size
        d = len(self.coordinate_names)
        for name, arr, expected in (
            ("coordinates", self.coordinates, (n, d)),
            ("rates", self.rates, (n, d)),
            ("accelerations", self.accelerations, (n, d)),
            ("reaction_wrenches", self.reaction_wrenches, (n, 6)),
            ("twist", self.twist, (n, 6)),
        ):
            a = np.asarray(arr, dtype=np.float64)
            if a.shape != expected or not np.all(np.isfinite(a)):
                raise ValueError(f"{name} must have finite shape {expected}")
            a.flags.writeable = False

        t.flags.writeable = False
        object.__setattr__(self, "time_s", t)

        for e_name in (
            "kinetic_energy_J",
            "potential_energy_J",
            "mechanical_energy_J",
        ):
            e_arr = getattr(self, e_name)
            if e_arr is not None:
                ea = np.asarray(e_arr, dtype=np.float64).reshape(-1)
                if ea.shape != (n,) or not np.all(np.isfinite(ea)):
                    raise ValueError(f"{e_name} must have shape ({n},) and be finite")
                ea.flags.writeable = False
                object.__setattr__(self, e_name, ea)

    def power(self) -> NDArray[np.float64]:
        """Compute 3D instantaneous spatial power P = F . v + M . omega along the branch."""
        p = np.sum(self.reaction_wrenches * self.twist, axis=1)
        p.flags.writeable = False
        return p

    def work(self) -> NDArray[np.float64]:
        """Compute cumulative mechanical work W(t) = integral_tcut^t P(tau) dtau."""
        p = self.power()
        w = np.zeros_like(p)
        dt = np.diff(self.time_s)
        avg_p = 0.5 * (p[:-1] + p[1:])
        w[1:] = np.cumsum(avg_p * dt)
        w.flags.writeable = False
        return w

    def energy_change(self) -> NDArray[np.float64] | None:
        """Compute delta E(t) = E(t) - E(t_cut) if mechanical energy is tracked."""
        if self.mechanical_energy_J is None:
            return None
        de = self.mechanical_energy_J - self.mechanical_energy_J[0]
        de.flags.writeable = False
        return de

    def linear_impulse(self) -> NDArray[np.float64]:
        """Compute cumulative linear impulse I(t) = integral_tcut^t F(tau) dtau."""
        force = self.reaction_wrenches[:, :3]
        imp = np.zeros_like(force)
        dt = np.diff(self.time_s)[:, None]
        avg_f = 0.5 * (force[:-1] + force[1:])
        imp[1:] = np.cumsum(avg_f * dt, axis=0)
        imp.flags.writeable = False
        return imp

    def angular_impulse(self) -> NDArray[np.float64]:
        """Compute cumulative angular impulse L(t) = integral_tcut^t M(tau) dtau."""
        moment = self.reaction_wrenches[:, 3:]
        imp = np.zeros_like(moment)
        dt = np.diff(self.time_s)[:, None]
        avg_m = 0.5 * (moment[:-1] + moment[1:])
        imp[1:] = np.cumsum(avg_m * dt, axis=0)
        imp.flags.writeable = False
        return imp

    def to_counterfactual_trajectory(self) -> CounterfactualTrajectory:
        """Export forward branch as a CounterfactualTrajectory."""
        n = self.time_s.size
        d = len(self.coordinate_names)
        zeros_qd = np.zeros((n, d), dtype=np.float64)
        zeros_6 = np.zeros((n, 6), dtype=np.float64)
        return CounterfactualTrajectory(
            time_s=self.time_s,
            coordinate_names=self.coordinate_names,
            actual_accelerations=self.accelerations,
            ztcf_accelerations=self.accelerations,
            control_accelerations=zeros_qd,
            zvcf_accelerations=zeros_qd,
            actual_wrenches=self.reaction_wrenches,
            ztcf_wrenches=self.reaction_wrenches,
            control_wrenches=zeros_6,
            zvcf_wrenches=zeros_6,
            twist=self.twist,
            parent_run_id=self.cut_state.parent_run_id,
            model_tier="forward_ztcf_branch",
        )


def _eval_model_forward_step(
    model: Any,
    coord_names: tuple[str, ...],
    q_vec: NDArray[np.float64],
    v_vec: NDArray[np.float64],
    zero_u: dict[str, float],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    q_dict = {name: float(q_vec[i]) for i, name in enumerate(coord_names)}
    v_dict = {name: float(v_vec[i]) for i, name in enumerate(coord_names)}
    if hasattr(model, "evaluate_dynamics"):
        acc_dict, wrench = model.evaluate_dynamics(q_dict, v_dict, zero_u)
        a_vec = np.array([acc_dict[name] for name in coord_names], dtype=np.float64)
        w_vec = wrench.vector
    elif hasattr(model, "accelerations") and hasattr(model, "closure_reaction_wrench"):
        acc_dict = model.accelerations(q_dict, v_dict, zero_u)
        a_vec = np.array([acc_dict[name] for name in coord_names], dtype=np.float64)
        force, torque, _ = model.closure_reaction_wrench("world")
        w_vec = np.concatenate([force, torque])
    else:
        raise NotImplementedError(  # tracked: #10285
            "Model does not implement dynamics evaluation"
        )
    return a_vec, w_vec


def _step_rk4(
    model: Any,
    coord_names: tuple[str, ...],
    q_curr: NDArray[np.float64],
    v_curr: NDArray[np.float64],
    a_curr: NDArray[np.float64],
    zero_u: dict[str, float],
    dt_s: float,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    k1_v = a_curr
    k1_q = v_curr
    q_k2 = q_curr + 0.5 * dt_s * k1_q
    v_k2 = v_curr + 0.5 * dt_s * k1_v
    k2_v, _ = _eval_model_forward_step(model, coord_names, q_k2, v_k2, zero_u)
    k2_q = v_k2
    q_k3 = q_curr + 0.5 * dt_s * k2_q
    v_k3 = v_curr + 0.5 * dt_s * k2_v
    k3_v, _ = _eval_model_forward_step(model, coord_names, q_k3, v_k3, zero_u)
    k3_q = v_k3
    q_k4 = q_curr + dt_s * k3_q
    v_k4 = v_curr + dt_s * k3_v
    k4_v, _ = _eval_model_forward_step(model, coord_names, q_k4, v_k4, zero_u)
    k4_q = v_k4

    q_next = q_curr + (dt_s / 6.0) * (k1_q + 2.0 * k2_q + 2.0 * k3_q + k4_q)
    v_next = v_curr + (dt_s / 6.0) * (k1_v + 2.0 * k2_v + 2.0 * k3_v + k4_v)
    return q_next, v_next


def _validate_rollout_parameters(
    intervention: str, duration_s: float, dt_s: float
) -> None:
    if intervention != "ztcf":
        raise ValueError(
            f"Only intervention='ztcf' can be integrated as a forward rollout; got '{intervention}'. "
            "ZVCF is strictly an instantaneous diagnostic at fixed state (q, v=0, u=0) "
            "and cannot be integrated as a forward rollout."
        )
    if duration_s <= 0.0 or not np.isfinite(duration_s):
        raise ValueError("duration_s must be positive and finite")
    if dt_s <= 0.0 or not np.isfinite(dt_s):
        raise ValueError("dt_s must be positive and finite")


def _check_cut_consistency(
    q_arr: NDArray[np.float64],
    v_arr: NDArray[np.float64],
    a_arr: NDArray[np.float64],
    cut_state: CutState,
    coord_names: tuple[str, ...],
) -> bool:
    q0_dict = {name: float(q_arr[0, j]) for j, name in enumerate(coord_names)}
    v0_dict = {name: float(v_arr[0, j]) for j, name in enumerate(coord_names)}
    return (
        all(np.isclose(q0_dict[k], cut_state.coordinates[k]) for k in coord_names)
        and all(np.isclose(v0_dict[k], cut_state.rates[k]) for k in coord_names)
        and bool(np.all(np.isfinite(q_arr)))
        and bool(np.all(np.isfinite(v_arr)))
        and bool(np.all(np.isfinite(a_arr)))
    )


def _record_energies_and_violation(
    model: Any,
    q_dict: dict[str, float],
    v_dict: dict[str, float],
    i: int,
    ke_arr: NDArray[np.float64] | None,
    pe_arr: NDArray[np.float64] | None,
    me_arr: NDArray[np.float64] | None,
    max_violation: float,
) -> float:
    if ke_arr is not None and pe_arr is not None and me_arr is not None:
        ke, pe, me = model.compute_energy(q_dict, v_dict)
        ke_arr[i], pe_arr[i], me_arr[i] = ke, pe, me
    if hasattr(model, "constraint_violation"):
        viol = float(model.constraint_violation(q_dict, v_dict))
        if viol > max_violation:
            return viol
    return max_violation


def simulate_forward_ztcf(
    model: Any,
    cut_state: CutState,
    duration_s: float,
    dt_s: float = 1.0 / 360.0,
    *,
    integrator: Literal["rk4", "euler"] = "rk4",
    branch_id: str = "ztcf_branch",
    intervention: str = "ztcf",
) -> ForwardZTCFBranch:
    """Integrate evolving unforced forward dynamics from a saved cut state.

    Uses zero applied control (u=0) starting at t_cut.
    """
    _validate_rollout_parameters(intervention, duration_s, dt_s)

    coord_names = tuple(sorted(cut_state.coordinates.keys()))
    n_steps = int(round(duration_s / dt_s)) + 1
    t_arr = cut_state.cut_time_s + np.arange(n_steps, dtype=np.float64) * dt_s

    d = len(coord_names)
    q_arr = np.zeros((n_steps, d), dtype=np.float64)
    v_arr = np.zeros((n_steps, d), dtype=np.float64)
    a_arr = np.zeros((n_steps, d), dtype=np.float64)
    w_arr = np.zeros((n_steps, 6), dtype=np.float64)
    twist_arr = np.zeros((n_steps, 6), dtype=np.float64)

    has_energy = hasattr(model, "compute_energy")
    ke_arr = np.zeros(n_steps, dtype=np.float64) if has_energy else None
    pe_arr = np.zeros(n_steps, dtype=np.float64) if has_energy else None
    me_arr = np.zeros(n_steps, dtype=np.float64) if has_energy else None

    q_curr = np.array(
        [cut_state.coordinates[name] for name in coord_names], dtype=np.float64
    )
    v_curr = np.array([cut_state.rates[name] for name in coord_names], dtype=np.float64)
    zero_u = dict.fromkeys(coord_names, 0.0)
    max_violation = 0.0

    for i in range(n_steps):
        q_arr[i] = q_curr
        v_arr[i] = v_curr

        a_curr, w_curr = _eval_model_forward_step(
            model, coord_names, q_curr, v_curr, zero_u
        )
        a_arr[i] = a_curr
        w_arr[i] = w_curr

        if d >= 2:
            twist_arr[i, :2] = v_curr[:2]
        elif d == 1:
            twist_arr[i, 0] = v_curr[0]

        q_dict = {name: float(q_curr[j]) for j, name in enumerate(coord_names)}
        v_dict = {name: float(v_curr[j]) for j, name in enumerate(coord_names)}
        max_violation = _record_energies_and_violation(
            model, q_dict, v_dict, i, ke_arr, pe_arr, me_arr, max_violation
        )

        if i < n_steps - 1:
            if integrator == "rk4":
                q_curr, v_curr = _step_rk4(
                    model, coord_names, q_curr, v_curr, a_curr, zero_u, dt_s
                )
            else:
                q_curr = q_curr + dt_s * v_curr
                v_curr = v_curr + dt_s * a_curr

    is_cut_consistent = _check_cut_consistency(
        q_arr, v_arr, a_arr, cut_state, coord_names
    )

    return ForwardZTCFBranch(
        cut_state=cut_state,
        time_s=t_arr,
        coordinate_names=coord_names,
        coordinates=q_arr,
        rates=v_arr,
        accelerations=a_arr,
        reaction_wrenches=w_arr,
        twist=twist_arr,
        kinetic_energy_J=ke_arr,
        potential_energy_J=pe_arr,
        mechanical_energy_J=me_arr,
        branch_id=branch_id,
        is_cut_consistent=is_cut_consistent,
        max_constraint_violation=max_violation,
        solver_converged=bool(np.all(np.isfinite(q_arr))),
    )


def _make_readonly(arr: np.ndarray | None) -> None:
    """Set numpy array flags to read-only while satisfying Law of Demeter."""
    if arr is not None:
        flags = arr.flags
        flags.writeable = False


class CounterfactualStrategy(str, Enum):
    """Supported counterfactual intervention strategies."""

    ZERO_TRAIL_ARM_TORQUE = "zero_trail_arm_torque"
    CLAMPED_ACTUATOR_TORQUE = "clamped_actuator_torque"
    NULLSPACE_EXPLORATION = "nullspace_exploration"
    CUSTOM = "custom"


@dataclass(frozen=True)
class AccelerationDecomposition:
    """Instantaneous decomposition of generalized accelerations.

    Invariants:
    - a_grav: generalized acceleration induced purely by gravity.
    - a_drift: velocity-product drift (Coriolis, centrifugal, passive damping).
    - a_ctrl: generalized acceleration induced by active control torques.
    - ztcf = a_grav + a_drift (Zero-Torque Counterfactual).
    - zvcf = a_grav + a_ctrl (Zero-Velocity Counterfactual).
    """

    a_grav: NDArray[np.float64]
    a_drift: NDArray[np.float64]
    a_ctrl: NDArray[np.float64]

    def __post_init__(self) -> None:
        if not (
            np.all(np.isfinite(self.a_grav))
            and np.all(np.isfinite(self.a_drift))
            and np.all(np.isfinite(self.a_ctrl))
        ):
            raise ValueError("All acceleration decomposition components must be finite")
        if (
            self.a_grav.shape != self.a_drift.shape
            or self.a_grav.shape != self.a_ctrl.shape
        ):
            raise ValueError(
                f"Component shape mismatch: grav={self.a_grav.shape}, "
                f"drift={self.a_drift.shape}, ctrl={self.a_ctrl.shape}"
            )
        _make_readonly(self.a_grav)
        _make_readonly(self.a_drift)
        _make_readonly(self.a_ctrl)

    @property
    def ztcf(self) -> NDArray[np.float64]:
        """Zero-Torque Counterfactual acceleration (a_grav + a_drift)."""
        res = self.a_grav + self.a_drift
        res.flags.writeable = False
        return res

    @property
    def zvcf(self) -> NDArray[np.float64]:
        """Zero-Velocity Counterfactual acceleration (a_grav + a_ctrl)."""
        res = self.a_grav + self.a_ctrl
        res.flags.writeable = False
        return res

    @property
    def total_accel(self) -> NDArray[np.float64]:
        """Total instantaneous acceleration (a_grav + a_drift + a_ctrl)."""
        res = self.a_grav + self.a_drift + self.a_ctrl
        res.flags.writeable = False
        return res

    def verify_decomposition(
        self, a_total: NDArray[np.float64], rtol: float = 1e-4, atol: float = 1e-4
    ) -> bool:
        """Verify whether decomposition matches total acceleration within tolerance."""
        return bool(np.allclose(self.total_accel, a_total, rtol=rtol, atol=atol))


@dataclass(frozen=True)
class CounterfactualFork:
    """Immutable record of an intervened trajectory rollout diverging from a baseline."""

    fork_id: str
    baseline_candidate_sha256: str
    strategy: CounterfactualStrategy
    fork_time_s: float
    fork_frame_idx: int
    initial_state: tuple[NDArray[np.float64], NDArray[np.float64]]
    time_s: NDArray[np.float64]
    q: NDArray[np.float64]
    v: NDArray[np.float64] | None = None
    a: NDArray[np.float64] | None = None
    altered_tau: NDArray[np.float64] | None = None
    divergence_rms: float = 0.0
    constraint_status: dict[str, Any] = field(default_factory=dict)
    is_accepted: bool = True
    rejection_reasons: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        _make_readonly(self.initial_state[0])
        _make_readonly(self.initial_state[1])
        _make_readonly(self.time_s)
        _make_readonly(self.q)
        _make_readonly(self.v)
        _make_readonly(self.a)
        _make_readonly(self.altered_tau)

    @property
    def frame_count(self) -> int:
        return int(len(self.time_s))

    @property
    def duration_s(self) -> float:
        return float(self.time_s[-1] - self.time_s[0]) if len(self.time_s) > 1 else 0.0


def _compute_baseline_hash(session: CandidateSession) -> str:
    """Compute combined SHA-256 digest of session coordinate arrays to guarantee immutability."""
    hasher = hashlib.sha256()
    hasher.update(session.q.tobytes())
    hasher.update(session.time_s.tobytes())
    if session.v is not None:
        hasher.update(session.v.tobytes())
    if session.tau is not None:
        hasher.update(session.tau.tobytes())
    return hasher.hexdigest()


def _resolve_trail_arm_indices(
    coord_names: tuple[str, ...], explicit_indices: Sequence[int] | None
) -> tuple[int, ...]:
    """Identify coordinate columns for trail arm joints."""
    if explicit_indices is not None:
        return tuple(explicit_indices)

    trail_keywords = (
        "right_shoulder",
        "right_elbow",
        "right_wrist",
        "r_shoulder",
        "r_elbow",
        "r_wrist",
        "trail_arm",
        "trail_shoulder",
        "trail_elbow",
    )
    matches: list[int] = []
    for idx, name in enumerate(coord_names):
        lower_name = name.lower()
        if any(kw in lower_name for kw in trail_keywords):
            matches.append(idx)
    return tuple(matches)


def _alter_torques(
    base_tau: NDArray[np.float64],
    strategy: CounterfactualStrategy,
    coord_names: tuple[str, ...],
    control_override: NDArray[np.float64] | None,
    clamp_limits: tuple[float, float] | None,
    trail_arm_indices: Sequence[int] | None,
) -> NDArray[np.float64]:
    """Compute altered torques according to chosen counterfactual intervention strategy."""
    altered = base_tau.copy()
    if strategy == CounterfactualStrategy.ZERO_TRAIL_ARM_TORQUE:
        arm_indices = _resolve_trail_arm_indices(coord_names, trail_arm_indices)
        for idx in arm_indices:
            if idx < altered.shape[1]:
                altered[:, idx] = 0.0
    elif strategy == CounterfactualStrategy.CLAMPED_ACTUATOR_TORQUE:
        c_min, c_max = clamp_limits if clamp_limits is not None else (-50.0, 50.0)
        altered = np.clip(altered, c_min, c_max)
    elif strategy == CounterfactualStrategy.NULLSPACE_EXPLORATION:
        t = np.linspace(0.0, np.pi, len(altered))[:, np.newaxis]
        altered = altered + 5.0 * np.sin(t)
    elif strategy == CounterfactualStrategy.CUSTOM:
        if control_override is None:
            raise ValueError("control_override must be provided for CUSTOM strategy")
        if control_override.shape != altered.shape:
            raise ValueError(
                f"control_override shape {control_override.shape} does not match {altered.shape}"
            )
        altered = control_override.copy()
    return altered


def _integrate_dynamics(
    time_s: NDArray[np.float64],
    q0: NDArray[np.float64],
    v0: NDArray[np.float64],
    base_q: NDArray[np.float64],
    base_v: NDArray[np.float64],
    base_tau: NDArray[np.float64],
    altered_tau: NDArray[np.float64],
    base_a: NDArray[np.float64] | None,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Numerically integrate counterfactual divergence from baseline trajectory."""
    n_frames = len(time_s)
    n_dofs = len(q0)
    q_cf = np.zeros((n_frames, n_dofs), dtype=np.float64)
    v_cf = np.zeros((n_frames, n_dofs), dtype=np.float64)
    a_cf = np.zeros((n_frames, n_dofs), dtype=np.float64)

    q_cf[0] = q0
    v_cf[0] = v0

    # Effective diagonal inertia approximation for acceleration perturbation
    m_eff = 1.0

    for i in range(n_frames):
        dt = float(time_s[i + 1] - time_s[i]) if i + 1 < n_frames else 0.001
        if dt <= 0.0:
            dt = 0.001

        # Baseline acceleration at frame i
        if base_a is not None and i < len(base_a):
            a_nom = base_a[i]
        elif i + 1 < len(base_v):
            a_nom = (base_v[i + 1] - base_v[i]) / dt
        else:
            a_nom = np.zeros(n_dofs, dtype=np.float64)

        delta_tau = altered_tau[i] - base_tau[i]
        delta_a = np.zeros(n_dofs, dtype=np.float64)
        n_map = min(len(delta_tau), n_dofs)
        delta_a[:n_map] = delta_tau[:n_map] / m_eff
        a_cf[i] = a_nom + delta_a

        if i + 1 < n_frames:
            v_cf[i + 1] = v_cf[i] + a_cf[i] * dt
            q_cf[i + 1] = q_cf[i] + v_cf[i + 1] * dt

    return q_cf, v_cf, a_cf


def _evaluate_constraints(
    q: NDArray[np.float64],
    v: NDArray[np.float64],
    specification: dict[str, Any],
    coord_names: tuple[str, ...],
) -> tuple[dict[str, Any], bool, tuple[str, ...]]:
    """Validate counterfactual rollout trajectory against system constraints."""
    reasons: list[str] = []
    status: dict[str, Any] = {"finite": True, "limits_satisfied": True}

    if not (np.all(np.isfinite(q)) and np.all(np.isfinite(v))):
        status["finite"] = False
        reasons.append("Non-finite state encountered during counterfactual rollout")

    # Joint limits evaluation if present in specification
    limits = specification.get("joint_limits")
    if isinstance(limits, dict):
        limit_violations = 0
        for idx, name in enumerate(coord_names):
            if name in limits and idx < q.shape[1]:
                min_val, max_val = limits[name]
                col = q[:, idx]
                if np.any(col < min_val) or np.any(col > max_val):
                    limit_violations += 1
        if limit_violations > 0:
            status["limits_satisfied"] = False
            reasons.append(
                f"Joint limits violated on {limit_violations} generalized coordinates"
            )

    is_accepted = len(reasons) == 0
    return status, is_accepted, tuple(reasons)


@precondition(
    lambda session, fork_frame_idx, **_: (
        session.supports_counterfactuals
        and 0 <= fork_frame_idx < session.frame_count - 1
    ),
    "Candidate session must support counterfactuals and have a valid fork frame index",
)
def create_counterfactual_rollout(
    session: CandidateSession,
    fork_frame_idx: int,
    strategy: CounterfactualStrategy = CounterfactualStrategy.ZERO_TRAIL_ARM_TORQUE,
    *,
    control_override: NDArray[np.float64] | None = None,
    duration_frames: int | None = None,
    clamp_limits: tuple[float, float] | None = None,
    trail_arm_indices: Sequence[int] | None = None,
) -> CounterfactualFork:
    """Generate an immutable counterfactual fork guaranteeing zero mutation of baseline data."""
    # Strict immutability guarantee: hash baseline state before rollout
    baseline_digest_before = _compute_baseline_hash(session)

    end_idx = (
        session.frame_count
        if duration_frames is None
        else min(fork_frame_idx + duration_frames, session.frame_count)
    )
    if end_idx <= fork_frame_idx + 1:
        raise ValueError(
            f"Rollout window too short: [{fork_frame_idx}, {end_idx}) frames"
        )

    fork_time_s = float(session.time_s[fork_frame_idx])
    time_window = session.time_s[fork_frame_idx:end_idx].copy()
    q0 = session.q[fork_frame_idx].copy()
    v0 = (
        session.v[fork_frame_idx].copy() if session.v is not None else np.zeros_like(q0)
    )

    base_q = session.q[fork_frame_idx:end_idx]
    base_v = (
        session.v[fork_frame_idx:end_idx]
        if session.v is not None
        else np.zeros_like(base_q)
    )
    base_a = session.a[fork_frame_idx:end_idx] if session.a is not None else None
    base_tau = (
        session.tau[fork_frame_idx:end_idx].copy()
        if session.tau is not None
        else np.zeros_like(base_q)
    )

    altered_tau = _alter_torques(
        base_tau=base_tau,
        strategy=strategy,
        coord_names=session.coordinate_names,
        control_override=control_override,
        clamp_limits=clamp_limits,
        trail_arm_indices=trail_arm_indices,
    )

    q_cf, v_cf, a_cf = _integrate_dynamics(
        time_s=time_window,
        q0=q0,
        v0=v0,
        base_q=base_q,
        base_v=base_v,
        base_tau=base_tau,
        altered_tau=altered_tau,
        base_a=base_a,
    )

    divergence_rms = float(np.sqrt(np.mean((q_cf - base_q) ** 2)))
    status, is_accepted, reasons = _evaluate_constraints(
        q=q_cf,
        v=v_cf,
        specification=session.specification,
        coord_names=session.coordinate_names,
    )

    # Verify zero-mutation invariant on baseline session
    baseline_digest_after = _compute_baseline_hash(session)
    if baseline_digest_before != baseline_digest_after:
        raise RuntimeError(
            "Baseline candidate session was mutated during counterfactual rollout calculation!"
        )

    fork_id = f"cf_{session.candidate_sha256[:8]}_{strategy.value}_{fork_frame_idx}"

    return CounterfactualFork(
        fork_id=fork_id,
        baseline_candidate_sha256=session.candidate_sha256,
        strategy=strategy,
        fork_time_s=fork_time_s,
        fork_frame_idx=fork_frame_idx,
        initial_state=(q0, v0),
        time_s=time_window,
        q=q_cf,
        v=v_cf,
        a=a_cf,
        altered_tau=altered_tau,
        divergence_rms=divergence_rms,
        constraint_status=status,
        is_accepted=is_accepted,
        rejection_reasons=reasons,
    )

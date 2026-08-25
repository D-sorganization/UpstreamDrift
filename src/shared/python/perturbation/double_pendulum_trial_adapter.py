"""Execute canonical Tools variation rows with the analytical pendulum.

Tools owns variation-plan construction and sampling. This module owns the
UpstreamDrift execution boundary: mapping supported sampled variables into the
two-degree-of-freedom model, applying localized torque commands, retaining a
complete trace, and classifying a sampled clubhead-to-target observation.

Contact classification is deliberately modest. It uses the minimum sampled
clubhead-centre distance and does not claim a continuous collision solution or
ball-flight outcome. The retained trace lets reviewers replace or falsify that
event rule without rerunning the variation sampler.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from src.shared.python.simulation_backends.model_params import GolfModelParams
from src.shared.python.simulation_backends.ode_backend import ODEBackend
from src.shared.python.simulation_backends.protocol import SimState, Trace

from .trial_evidence import (
    CanonicalTrialEvidence,
    ClosestApproach,
    ImpactObservation,
    SampledInput,
    TrialOutcome,
    TrialTrace,
)
from .trial_adapter_contracts import (
    TrialObservation,
    collect_trial_evidence,
    collect_trial_failure,
    make_trial_evidence_identity,
    require_fixed_step_horizon,
    require_localized_time_window,
    require_trial_result_geometry,
)

SHOULDER_DAMPING_KEY = "swing_sim.swing.damping_shoulder"
WRIST_DAMPING_KEY = "swing_sim.swing.damping_wrist"
SHOULDER_TORQUE_KEY = "swing_sim.swing.shoulder_commanded_torque_offset_nm"
WRIST_TORQUE_KEY = "swing_sim.swing.wrist_commanded_torque_offset_nm"

SHOULDER_JOINT_ID = "joint.shoulder"
WRIST_JOINT_ID = "joint.wrist"

_DAMPING_KEYS = {
    SHOULDER_DAMPING_KEY: "damping_shoulder",
    WRIST_DAMPING_KEY: "damping_wrist",
}
_TORQUE_KEYS = {
    SHOULDER_TORQUE_KEY: (0, SHOULDER_JOINT_ID),
    WRIST_TORQUE_KEY: (1, WRIST_JOINT_ID),
}
_UNITS = {
    SHOULDER_DAMPING_KEY: "N·m·s",
    WRIST_DAMPING_KEY: "N·m·s",
    SHOULDER_TORQUE_KEY: "N·m",
    WRIST_TORQUE_KEY: "N·m",
}
_COORDINATE_IDS = (
    "joint.shoulder.angle",
    "joint.wrist.relative_angle",
)
_MARKER_IDS = (WRIST_JOINT_ID, "clubhead.center")
_TARGET_ID = "ball.center"


class NoiseSpecification(Protocol):
    """Narrow canonical plan column required by this adapter."""

    variable_key: str
    time_window_s: tuple[float, float] | None
    point_ids: tuple[str, ...]


@dataclass(frozen=True)
class DoublePendulumTrialConfig:
    """Immutable execution and observation configuration for one plan.

    The target is a point in ``frame_id`` and contact is observed when a
    retained clubhead-centre sample lies within ``contact_radius_m``. The
    duration must contain an integer number of fixed integration steps.
    """

    model_params: GolfModelParams
    initial_state: SimState
    duration_s: float
    dt_s: float
    base_torques_nm: tuple[float, float]
    target_position_m: tuple[float, float, float]
    contact_radius_m: float
    frame_id: str
    alignment_id: str

    def __post_init__(self) -> None:
        if not isinstance(self.model_params, GolfModelParams):
            raise TypeError("model_params must be GolfModelParams")
        if not isinstance(self.initial_state, SimState):
            raise TypeError("initial_state must be SimState")
        q = np.array(self.initial_state.q, dtype=float, copy=True).reshape(-1)
        v = np.array(self.initial_state.v, dtype=float, copy=True).reshape(-1)
        if (
            q.shape != (2,)
            or v.shape != (2,)
            or not np.isfinite(q).all()
            or not np.isfinite(v).all()
        ):
            raise ValueError("initial_state must contain two finite q and v values")
        if self.initial_state.time != 0.0:
            raise ValueError("initial_state time must be zero")
        q.flags.writeable = False
        v.flags.writeable = False
        object.__setattr__(self, "initial_state", SimState(q=q, v=v, time=0.0))

        require_fixed_step_horizon(self.duration_s, self.dt_s)

        torques = np.asarray(self.base_torques_nm, dtype=float)
        if torques.shape != (2,) or not np.isfinite(torques).all():
            raise ValueError("base_torques_nm must contain two finite values")
        target = np.asarray(self.target_position_m, dtype=float)
        if target.shape != (3,) or not np.isfinite(target).all():
            raise ValueError("target_position_m must contain three finite values")
        radius = float(self.contact_radius_m)
        if not math.isfinite(radius) or radius <= 0.0:
            raise ValueError("contact_radius_m must be positive and finite")
        for value, name in (
            (self.frame_id, "frame_id"),
            (self.alignment_id, "alignment_id"),
        ):
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be non-empty")

    @property
    def horizon(self) -> int:
        """Return the exact number of fixed integration steps."""
        return require_fixed_step_horizon(self.duration_s, self.dt_s)


@dataclass(frozen=True)
class DoublePendulumTrialResult:
    """Raw analytical result before conversion to canonical trial evidence."""

    trace: Trace
    closest_sample_index: int
    closest_distance_m: float
    contact_observed: bool

    def __post_init__(self) -> None:
        if not isinstance(self.trace, Trace):
            raise TypeError("trace must be a simulation_backends Trace")
        require_trial_result_geometry(
            self.trace, self.closest_sample_index, self.closest_distance_m
        )
        if type(self.contact_observed) is not bool:
            raise TypeError("contact_observed must be bool")


@dataclass(frozen=True)
class _Column:
    key: str
    unit: str
    time_window_s: tuple[float, float] | None
    point_ids: tuple[str, ...]


class DoublePendulumTrialAdapter:
    """Run and collect canonical rows for the analytical ODE backend."""

    engine_id = "simulation_backends.ode"
    model_id = "planar-relative-angle-double-pendulum/v1"

    def __init__(
        self,
        *,
        plan: object,
        config: DoublePendulumTrialConfig,
        plan_sha256: str,
        scenario_sha256: str,
        tools_revision: str,
        engine_revision: str,
    ) -> None:
        if not isinstance(config, DoublePendulumTrialConfig):
            raise TypeError("config must be DoublePendulumTrialConfig")
        if getattr(plan, "mode", None) != "swing":
            raise ValueError("double-pendulum execution requires a swing plan")
        self._config = config
        self._columns = self._validate_columns(plan)
        self._identity = make_trial_evidence_identity(
            plan_sha256,
            scenario_sha256,
            self._execution_config_digest(),
            tools_revision,
            self.engine_id,
            engine_revision,
            self.model_id,
        )

    def _validate_columns(self, plan: object) -> tuple[_Column, ...]:
        raw_noise = getattr(plan, "noise", None)
        if not isinstance(raw_noise, tuple) or not raw_noise:
            raise ValueError("plan noise must contain canonical specifications")
        columns: list[_Column] = []
        for raw_spec in raw_noise:
            key = getattr(raw_spec, "variable_key", None)
            window = getattr(raw_spec, "time_window_s", None)
            points = getattr(raw_spec, "point_ids", None)
            if key not in _UNITS:
                raise ValueError(f"unsupported double-pendulum variable {key!r}")
            if not isinstance(points, tuple):
                raise ValueError("plan point_ids must be a tuple")
            if key in _DAMPING_KEYS:
                if window is not None or points:
                    raise ValueError(
                        "global damping variables must not declare a locus"
                    )
            else:
                self._validate_torque_locus(key, window, points)
            columns.append(_Column(key, _UNITS[key], window, points))
        return tuple(columns)

    def _execution_config_digest(self) -> str:
        model_params = self._config.model_params
        payload = {
            "schema_version": "double-pendulum-trial-config/v1",
            "model_params": model_params.model_dump(mode="json"),
            "initial_state": {
                "q": self._config.initial_state.q.tolist(),
                "v": self._config.initial_state.v.tolist(),
            },
            "duration_s": self._config.duration_s,
            "dt_s": self._config.dt_s,
            "base_torques_nm": list(self._config.base_torques_nm),
            "target_position_m": list(self._config.target_position_m),
            "contact_radius_m": self._config.contact_radius_m,
            "contact_rule": "minimum-sampled-clubhead-centre-distance/v1",
            "frame_id": self._config.frame_id,
            "alignment_id": self._config.alignment_id,
            "columns": [
                {
                    "key": column.key,
                    "unit": column.unit,
                    "time_window_s": column.time_window_s,
                    "point_ids": column.point_ids,
                }
                for column in self._columns
            ],
        }
        canonical = json.dumps(
            payload,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        return hashlib.sha256(canonical).hexdigest()

    def _validate_torque_locus(
        self,
        key: str,
        window: object,
        points: tuple[str, ...],
    ) -> None:
        require_localized_time_window(window, self._config.duration_s)
        expected_point = _TORQUE_KEYS[key][1]
        if points != (expected_point,):
            raise ValueError(
                f"localized torque point must be exactly {expected_point!r}"
            )

    def run(self, sampled_row: np.ndarray) -> DoublePendulumTrialResult:
        """Execute one finite row through the real analytical RK4 backend."""
        row = np.asarray(sampled_row, dtype=float).reshape(-1)
        if row.shape != (len(self._columns),) or not np.isfinite(row).all():
            raise ValueError(
                "sampled row must contain one finite value per plan column"
            )
        params = self._parameters_for(row)
        controls = self._controls_for(row)
        backend = ODEBackend(params, dt=self._config.dt_s)
        backend.reset(self._config.initial_state.copy())
        trace = backend.rollout(controls, self._config.horizon, self._config.dt_s)
        markers = self._markers(trace.q, params)
        if not all(
            np.isfinite(values).all()
            for values in (trace.t, trace.q, trace.v, controls, markers)
        ):
            raise FloatingPointError("analytical trial produced non-finite values")
        trace.markers = markers
        target = np.asarray(self._config.target_position_m, dtype=float)
        offsets = markers[:, 1, :] - target
        distances = np.sqrt(np.einsum("ij,ij->i", offsets, offsets))
        closest_index = int(np.argmin(distances))
        closest_distance = float(distances[closest_index])
        return DoublePendulumTrialResult(
            trace=trace,
            closest_sample_index=closest_index,
            closest_distance_m=closest_distance,
            contact_observed=closest_distance <= self._config.contact_radius_m,
        )

    def _parameters_for(self, row: np.ndarray) -> GolfModelParams:
        model_params = self._config.model_params
        values = model_params.model_dump()
        for column, sampled_value in zip(self._columns, row, strict=True):
            field = _DAMPING_KEYS.get(column.key)
            if field is not None:
                values[field] = float(sampled_value)
        return GolfModelParams.model_validate(values)

    def _controls_for(self, row: np.ndarray) -> np.ndarray:
        horizon = self._config.horizon
        controls = np.tile(np.asarray(self._config.base_torques_nm), (horizon, 1))
        times = np.arange(horizon, dtype=float) * self._config.dt_s
        for column, sampled_value in zip(self._columns, row, strict=True):
            torque_mapping = _TORQUE_KEYS.get(column.key)
            if torque_mapping is None:
                continue
            if column.time_window_s is None:
                raise ValueError("joint torque offsets require a time window")
            start_s, end_s = column.time_window_s
            active = (times >= start_s) & (times < end_s)
            controls[active, torque_mapping[0]] += float(sampled_value)
        return controls

    @staticmethod
    def _markers(q: np.ndarray, params: GolfModelParams) -> np.ndarray:
        theta1 = q[:, 0]
        theta12 = theta1 + q[:, 1]
        upper_length = params.upper.length_m
        lower_length = params.lower.length_m
        wrist = np.column_stack(
            (
                upper_length * np.sin(theta1),
                np.zeros(theta1.size),
                -upper_length * np.cos(theta1),
            )
        )
        clubhead = wrist + np.column_stack(
            (
                lower_length * np.sin(theta12),
                np.zeros(theta1.size),
                -lower_length * np.cos(theta12),
            )
        )
        return np.stack((wrist, clubhead), axis=1)

    def collect_success(
        self,
        trial_index: int,
        plan_seed: int,
        sampled_row: np.ndarray,
        result: object,
    ) -> CanonicalTrialEvidence:
        """Convert one completed hit or miss into immutable evidence."""
        if not isinstance(result, DoublePendulumTrialResult):
            raise TypeError("result must be DoublePendulumTrialResult")
        trace = result.trace
        if trace.markers is None:
            raise ValueError("double-pendulum trial trace must retain markers")
        trial_trace = TrialTrace(
            times_s=trace.t,
            q=trace.q,
            v=trace.v,
            coordinate_ids=_COORDINATE_IDS,
            coordinate_units=("rad", "rad"),
            velocity_units=("rad/s", "rad/s"),
            markers_m=trace.markers,
            marker_ids=_MARKER_IDS,
            frame_id=self._config.frame_id,
            alignment_id=self._config.alignment_id,
            complete=True,
        )
        closest_index = result.closest_sample_index
        closest = ClosestApproach(
            time_s=float(trace.t[closest_index]),
            distance_m=result.closest_distance_m,
            source_marker_id=_MARKER_IDS[1],
            target_id=_TARGET_ID,
            contact_observed=result.contact_observed,
        )
        impact = None
        outcome: TrialOutcome = "no_impact"
        if result.contact_observed:
            impact = ImpactObservation(
                time_s=float(trace.t[closest_index]),
                state=self._impact_state(
                    trace, closest_index, self._config.model_params
                ),
            )
            outcome = "hit"
        return collect_trial_evidence(
            self._identity,
            trial_index,
            plan_seed,
            sampled_row,
            self._columns,
            TrialObservation(outcome, trial_trace, impact, closest),
        )

    def collect_failure(
        self,
        trial_index: int,
        plan_seed: int,
        sampled_row: np.ndarray,
        error: Exception,
    ) -> CanonicalTrialEvidence:
        """Retain a declared domain/numerical failure without outputs."""
        return collect_trial_failure(
            self._identity,
            trial_index,
            plan_seed,
            sampled_row,
            self._columns,
            error,
        )

    @staticmethod
    def _impact_state(
        trace: Trace,
        index: int,
        params: GolfModelParams,
    ) -> tuple[SampledInput, ...]:
        q = trace.q[index]
        v = trace.v[index]
        theta1, theta2 = float(q[0]), float(q[1])
        omega1, omega2 = float(v[0]), float(v[1])
        absolute_rate = omega1 + omega2
        absolute_angle = theta1 + theta2
        vx = (
            params.upper.length_m * math.cos(theta1) * omega1
            + params.lower.length_m * math.cos(absolute_angle) * absolute_rate
        )
        vz = (
            params.upper.length_m * math.sin(theta1) * omega1
            + params.lower.length_m * math.sin(absolute_angle) * absolute_rate
        )
        return (
            SampledInput("joint.shoulder.angle", theta1, "rad"),
            SampledInput("joint.wrist.relative_angle", theta2, "rad"),
            SampledInput("joint.shoulder.angular_velocity", omega1, "rad/s"),
            SampledInput("joint.wrist.relative_angular_velocity", omega2, "rad/s"),
            SampledInput("clubhead.center.speed", math.hypot(vx, vz), "m/s"),
        )


__all__ = [
    "DoublePendulumTrialAdapter",
    "DoublePendulumTrialConfig",
    "DoublePendulumTrialResult",
    "SHOULDER_DAMPING_KEY",
    "SHOULDER_TORQUE_KEY",
    "WRIST_DAMPING_KEY",
    "WRIST_TORQUE_KEY",
]

"""Execute canonical Tools variation rows on an articulated MuJoCo model.

The adapter preserves the scientific boundary between a sampled Tools plan and
an UpstreamDrift execution. A binding must state how each canonical scalar is
allocated to named MuJoCo joints; no anatomical allocation is inferred. The
retained result uses actual MuJoCo contact for hit classification and reports
body-centre closest approach separately so reviewers can falsify the event
rule without reconstructing a trajectory.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import math
from dataclasses import dataclass
from typing import Any, Literal, Protocol, cast

import numpy as np

from src.shared.python.simulation_backends.protocol import Trace

from .double_pendulum_trial_adapter import (
    SHOULDER_DAMPING_KEY,
    SHOULDER_TORQUE_KEY,
    WRIST_DAMPING_KEY,
    WRIST_TORQUE_KEY,
)
from .trial_evidence import (
    CanonicalTrialEvidence,
    ClosestApproach,
    ImpactObservation,
    SampledInput,
    TrialOutcome,
    TrialTrace,
)

BindingKind = Literal["joint_torque_offset", "joint_damping"]

_EXPECTED_UNITS = {
    SHOULDER_DAMPING_KEY: "N·m·s",
    WRIST_DAMPING_KEY: "N·m·s",
    SHOULDER_TORQUE_KEY: "N·m",
    WRIST_TORQUE_KEY: "N·m",
}
_KIND_BY_KEY: dict[str, BindingKind] = {
    SHOULDER_DAMPING_KEY: "joint_damping",
    WRIST_DAMPING_KEY: "joint_damping",
    SHOULDER_TORQUE_KEY: "joint_torque_offset",
    WRIST_TORQUE_KEY: "joint_torque_offset",
}


class NoiseSpecification(Protocol):
    """Narrow canonical plan column consumed by the adapter."""

    variable_key: str
    time_window_s: tuple[float, float] | None
    point_ids: tuple[str, ...]


@dataclass(frozen=True)
class NamedJointTorque:
    """One constant generalized torque applied to a named scalar joint."""

    joint_name: str
    value_nm: float

    def __post_init__(self) -> None:
        _require_name(self.joint_name, "joint_name")
        if not math.isfinite(self.value_nm):
            raise ValueError("value_nm must be finite")


@dataclass(frozen=True)
class MujocoVariationBinding:
    """Explicit allocation of one canonical variable to MuJoCo joints."""

    variable_key: str
    unit: str
    kind: BindingKind
    target_joint_names: tuple[str, ...]
    allocation_weights: tuple[float, ...]
    plan_point_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.variable_key not in _EXPECTED_UNITS:
            raise ValueError(f"unsupported canonical variable {self.variable_key!r}")
        if self.unit != _EXPECTED_UNITS[self.variable_key]:
            raise ValueError("binding unit does not match the canonical variable")
        if self.kind != _KIND_BY_KEY[self.variable_key]:
            raise ValueError("binding kind does not match the canonical variable")
        _require_unique_names(self.target_joint_names, "target_joint_names")
        if len(self.allocation_weights) != len(self.target_joint_names):
            raise ValueError("allocation_weights must identify every target joint")
        weights = np.asarray(self.allocation_weights, dtype=float)
        if not np.isfinite(weights).all() or not np.any(weights != 0.0):
            raise ValueError("allocation_weights must be finite and not all zero")
        if self.kind == "joint_damping" and np.any(weights < 0.0):
            raise ValueError("joint damping allocation weights must be non-negative")
        if not isinstance(self.plan_point_ids, tuple):
            raise TypeError("plan_point_ids must be a tuple")
        if self.plan_point_ids:
            _require_unique_names(self.plan_point_ids, "plan_point_ids")


@dataclass(frozen=True)
class ArticulatedMujocoTrialConfig:
    """Immutable model, topology, control, and observation configuration."""

    model_xml: str
    model_id: str
    duration_s: float
    dt_s: float
    coordinate_joint_names: tuple[str, ...]
    marker_body_names: tuple[str, ...]
    source_body_name: str
    target_body_name: str
    base_joint_torques: tuple[NamedJointTorque, ...]
    bindings: tuple[MujocoVariationBinding, ...]
    frame_id: str
    alignment_id: str
    initial_qpos: tuple[float, ...] | None = None
    initial_qvel: tuple[float, ...] | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.model_xml, str) or not self.model_xml.strip():
            raise ValueError("model_xml must be non-empty")
        for value, name in (
            (self.model_id, "model_id"),
            (self.source_body_name, "source_body_name"),
            (self.target_body_name, "target_body_name"),
            (self.frame_id, "frame_id"),
            (self.alignment_id, "alignment_id"),
        ):
            _require_name(value, name)
        duration = float(self.duration_s)
        dt = float(self.dt_s)
        if not math.isfinite(duration) or duration <= 0.0:
            raise ValueError("duration_s must be positive and finite")
        if not math.isfinite(dt) or dt <= 0.0:
            raise ValueError("dt_s must be positive and finite")
        steps = duration / dt
        if not math.isclose(steps, round(steps), rel_tol=0.0, abs_tol=1e-12):
            raise ValueError("duration_s must contain an integer number of steps")
        _require_unique_names(self.coordinate_joint_names, "coordinate_joint_names")
        _require_unique_names(self.marker_body_names, "marker_body_names")
        if self.source_body_name not in self.marker_body_names:
            raise ValueError("source_body_name must be retained as a marker")
        if self.target_body_name not in self.marker_body_names:
            raise ValueError("target_body_name must be retained as a marker")
        if any(
            not isinstance(value, NamedJointTorque) for value in self.base_joint_torques
        ):
            raise TypeError("base_joint_torques must contain NamedJointTorque records")
        if not self.bindings or any(
            not isinstance(value, MujocoVariationBinding) for value in self.bindings
        ):
            raise TypeError("bindings must contain MujocoVariationBinding records")
        binding_keys = tuple(binding.variable_key for binding in self.bindings)
        if len(set(binding_keys)) != len(binding_keys):
            raise ValueError("bindings must identify unique canonical variables")
        if (self.initial_qpos is None) != (self.initial_qvel is None):
            raise ValueError("initial_qpos and initial_qvel must be provided together")
        for values, name in (
            (self.initial_qpos, "initial_qpos"),
            (self.initial_qvel, "initial_qvel"),
        ):
            if (
                values is not None
                and not np.isfinite(np.asarray(values, dtype=float)).all()
            ):
                raise ValueError(f"{name} must be finite")

    @property
    def horizon(self) -> int:
        """Return the exact fixed-step horizon."""
        return round(self.duration_s / self.dt_s)


@dataclass(frozen=True)
class ArticulatedMujocoTrialResult:
    """Raw articulated result before conversion to canonical evidence."""

    trace: Trace
    closest_sample_index: int
    closest_distance_m: float
    first_contact_index: int | None

    def __post_init__(self) -> None:
        if not isinstance(self.trace, Trace):
            raise TypeError("trace must be a simulation_backends Trace")
        if type(self.closest_sample_index) is not int or not (
            0 <= self.closest_sample_index < self.trace.t.size
        ):
            raise ValueError("closest_sample_index must identify a trace sample")
        if not math.isfinite(self.closest_distance_m) or self.closest_distance_m < 0.0:
            raise ValueError("closest_distance_m must be finite and non-negative")
        if self.first_contact_index is not None and (
            type(self.first_contact_index) is not int
            or not 0 <= self.first_contact_index < self.trace.t.size
        ):
            raise ValueError("first_contact_index must identify a trace sample")

    @property
    def contact_observed(self) -> bool:
        """Return whether actual source-target geom contact was retained."""
        return self.first_contact_index is not None


@dataclass(frozen=True)
class _Column:
    key: str
    unit: str
    kind: BindingKind
    target_joint_names: tuple[str, ...]
    allocation_weights: tuple[float, ...]
    time_window_s: tuple[float, float] | None
    point_ids: tuple[str, ...]


@dataclass(frozen=True)
class _ModelTopology:
    qpos_addresses: tuple[int, ...]
    dof_addresses: tuple[int, ...]
    joint_index_by_name: dict[str, int]
    marker_body_ids: tuple[int, ...]
    source_body_id: int
    target_body_id: int


class ArticulatedMujocoTrialAdapter:
    """Run canonical variation rows on a named articulated MuJoCo topology."""

    engine_id = "mujoco-articulated"

    def __init__(
        self,
        *,
        plan: object,
        config: ArticulatedMujocoTrialConfig,
        plan_sha256: str,
        scenario_sha256: str,
        tools_revision: str,
        engine_revision: str,
    ) -> None:
        if not isinstance(config, ArticulatedMujocoTrialConfig):
            raise TypeError("config must be ArticulatedMujocoTrialConfig")
        if getattr(plan, "mode", None) != "swing":
            raise ValueError("articulated MuJoCo execution requires a swing plan")
        self._mujoco = importlib.import_module("mujoco")
        self._config = config
        model = self._compile_model()
        self._topology = self._resolve_topology(model)
        self._columns = self._validate_columns(plan)
        self._validate_control_topology()
        self._plan_sha256 = plan_sha256
        self._scenario_sha256 = scenario_sha256
        self._execution_config_sha256 = self._execution_config_digest()
        self._tools_revision = tools_revision
        self._engine_revision = engine_revision

    @property
    def model_id(self) -> str:
        """Return the configured stable articulated-model identifier."""
        return self._config.model_id

    def _compile_model(self) -> Any:
        try:
            model = self._mujoco.MjModel.from_xml_string(self._config.model_xml)
        except Exception as error:  # noqa: BLE001 - normalize optional C-extension error
            raise ValueError("model_xml is not a valid MuJoCo model") from error
        model.opt.timestep = self._config.dt_s
        if self._config.initial_qpos is not None:
            if len(self._config.initial_qpos) != model.nq:
                raise ValueError("initial_qpos must identify every model qpos")
            if len(cast(tuple[float, ...], self._config.initial_qvel)) != model.nv:
                raise ValueError("initial_qvel must identify every model dof")
        return model

    def _resolve_topology(self, model: Any) -> _ModelTopology:
        joint_ids = tuple(
            self._name_to_id(model, self._mujoco.mjtObj.mjOBJ_JOINT, name, "joint")
            for name in self._config.coordinate_joint_names
        )
        hinge_type = int(self._mujoco.mjtJoint.mjJNT_HINGE)
        if any(int(model.jnt_type[joint_id]) != hinge_type for joint_id in joint_ids):
            raise ValueError("coordinate joints must be scalar hinge joints")
        qpos_addresses = tuple(
            int(model.jnt_qposadr[joint_id]) for joint_id in joint_ids
        )
        dof_addresses = tuple(int(model.jnt_dofadr[joint_id]) for joint_id in joint_ids)
        marker_ids = tuple(
            self._name_to_id(model, self._mujoco.mjtObj.mjOBJ_BODY, name, "body")
            for name in self._config.marker_body_names
        )
        joint_index = {
            name: index
            for index, name in enumerate(self._config.coordinate_joint_names)
        }
        return _ModelTopology(
            qpos_addresses=qpos_addresses,
            dof_addresses=dof_addresses,
            joint_index_by_name=joint_index,
            marker_body_ids=marker_ids,
            source_body_id=marker_ids[
                self._config.marker_body_names.index(self._config.source_body_name)
            ],
            target_body_id=marker_ids[
                self._config.marker_body_names.index(self._config.target_body_name)
            ],
        )

    def _name_to_id(self, model: Any, object_type: Any, name: str, label: str) -> int:
        identifier = int(self._mujoco.mj_name2id(model, object_type, name))
        if identifier < 0:
            raise ValueError(f"{label} {name!r} was not found in model topology")
        return identifier

    def _validate_columns(self, plan: object) -> tuple[_Column, ...]:
        raw_noise = getattr(plan, "noise", None)
        if not isinstance(raw_noise, tuple) or not raw_noise:
            raise ValueError("plan noise must contain canonical specifications")
        bindings = {binding.variable_key: binding for binding in self._config.bindings}
        raw_plan_keys = tuple(getattr(spec, "variable_key", None) for spec in raw_noise)
        if any(not isinstance(key, str) for key in raw_plan_keys):
            raise ValueError("plan variable keys must be strings")
        plan_keys = cast(tuple[str, ...], raw_plan_keys)
        if len(set(plan_keys)) != len(plan_keys):
            raise ValueError("plan variable keys must be unique")
        if set(plan_keys) != set(bindings):
            raise ValueError(
                "plan variables and articulated bindings must match exactly"
            )
        columns: list[_Column] = []
        for raw_spec in raw_noise:
            key = getattr(raw_spec, "variable_key", None)
            assert isinstance(key, str)
            binding = bindings[key]
            window = getattr(raw_spec, "time_window_s", None)
            points = getattr(raw_spec, "point_ids", None)
            if not isinstance(points, tuple):
                raise ValueError("plan point_ids must be a tuple")
            if binding.kind == "joint_damping":
                if window is not None or points or binding.plan_point_ids:
                    raise ValueError(
                        "global damping variables must not declare a locus"
                    )
            else:
                self._validate_torque_locus(window, points, binding.plan_point_ids)
            columns.append(
                _Column(
                    key=key,
                    unit=binding.unit,
                    kind=binding.kind,
                    target_joint_names=binding.target_joint_names,
                    allocation_weights=binding.allocation_weights,
                    time_window_s=window,
                    point_ids=points,
                )
            )
        return tuple(columns)

    def _validate_torque_locus(
        self,
        window: object,
        points: tuple[str, ...],
        declared_points: tuple[str, ...],
    ) -> None:
        if not isinstance(window, tuple) or len(window) != 2:
            raise ValueError("localized torque requires one half-open time window")
        if not all(isinstance(value, (int, float)) for value in window):
            raise ValueError("localized torque time window must be finite")
        start_s, end_s = (float(value) for value in window)
        if not math.isfinite(start_s) or not math.isfinite(end_s):
            raise ValueError("localized torque time window must be finite")
        if start_s < 0.0 or start_s >= end_s:
            raise ValueError(
                "localized torque time window must satisfy 0 <= start < end"
            )
        if end_s > self._config.duration_s:
            raise ValueError("localized torque time window exceeds trial duration")
        if not declared_points or points != declared_points:
            raise ValueError("localized torque point does not match its binding")

    def _validate_control_topology(self) -> None:
        available = self._topology.joint_index_by_name
        names = tuple(torque.joint_name for torque in self._config.base_joint_torques)
        names += tuple(
            name for column in self._columns for name in column.target_joint_names
        )
        for name in names:
            if name not in available:
                raise ValueError(f"joint {name!r} was not found in coordinate topology")

    def _execution_config_digest(self) -> str:
        payload = {
            "schema_version": "articulated-mujoco-trial-config/v1",
            "model_xml_sha256": hashlib.sha256(
                self._config.model_xml.encode("utf-8")
            ).hexdigest(),
            "model_id": self._config.model_id,
            "duration_s": self._config.duration_s,
            "dt_s": self._config.dt_s,
            "coordinate_joint_names": self._config.coordinate_joint_names,
            "marker_body_names": self._config.marker_body_names,
            "source_body_name": self._config.source_body_name,
            "target_body_name": self._config.target_body_name,
            "base_joint_torques": [
                {"joint_name": value.joint_name, "value_nm": value.value_nm}
                for value in self._config.base_joint_torques
            ],
            "columns": [
                {
                    "key": value.key,
                    "unit": value.unit,
                    "kind": value.kind,
                    "target_joint_names": value.target_joint_names,
                    "allocation_weights": value.allocation_weights,
                    "time_window_s": value.time_window_s,
                    "point_ids": value.point_ids,
                }
                for value in self._columns
            ],
            "initial_qpos": self._config.initial_qpos,
            "initial_qvel": self._config.initial_qvel,
            "frame_id": self._config.frame_id,
            "alignment_id": self._config.alignment_id,
            "contact_rule": "actual-source-target-geom-contact/v1",
            "closest_approach_rule": "sampled-body-centre-distance/v1",
        }
        canonical = json.dumps(
            payload, ensure_ascii=False, separators=(",", ":"), sort_keys=True
        ).encode("utf-8")
        return hashlib.sha256(canonical).hexdigest()

    def run(self, sampled_row: np.ndarray) -> ArticulatedMujocoTrialResult:
        """Execute one finite row and retain every configured state sample."""
        row = np.asarray(sampled_row, dtype=float).reshape(-1)
        if row.shape != (len(self._columns),) or not np.isfinite(row).all():
            raise ValueError(
                "sampled row must contain one finite value per plan column"
            )
        model = self._compile_model()
        data = self._mujoco.MjData(model)
        self._mujoco.mj_resetData(model, data)
        if self._config.initial_qpos is not None:
            data.qpos[:] = self._config.initial_qpos
            data.qvel[:] = cast(tuple[float, ...], self._config.initial_qvel)
        self._apply_damping(model, row)
        self._mujoco.mj_forward(model, data)

        sample_count = self._config.horizon + 1
        times = np.empty(sample_count, dtype=float)
        q = np.empty((sample_count, len(self._config.coordinate_joint_names)))
        v = np.empty_like(q)
        applied = np.zeros_like(q)
        markers = np.empty((sample_count, len(self._config.marker_body_names), 3))
        contacts = np.zeros(sample_count, dtype=bool)
        self._record_sample(data, times, q, v, markers, contacts, 0, model)
        for step in range(self._config.horizon):
            applied[step] = self._apply_torque(data, row, step)
            self._mujoco.mj_step(model, data)
            self._record_sample(data, times, q, v, markers, contacts, step + 1, model)
        arrays = (times, q, v, applied, markers)
        if not all(np.isfinite(values).all() for values in arrays):
            raise FloatingPointError(
                "articulated MuJoCo trial produced non-finite values"
            )
        trace = Trace(
            t=times,
            q=q,
            v=v,
            u=applied,
            dt=self._config.dt_s,
            backend=self.engine_id,
            meta={
                "model_id": self._config.model_id,
                "contact_rule": "actual-source-target-geom-contact/v1",
            },
            markers=markers,
        )
        source_index = self._config.marker_body_names.index(
            self._config.source_body_name
        )
        target_index = self._config.marker_body_names.index(
            self._config.target_body_name
        )
        offsets = markers[:, source_index] - markers[:, target_index]
        distances = np.sqrt(np.einsum("ij,ij->i", offsets, offsets))
        closest_index = int(np.argmin(distances))
        contact_indices = np.flatnonzero(contacts)
        return ArticulatedMujocoTrialResult(
            trace=trace,
            closest_sample_index=closest_index,
            closest_distance_m=float(distances[closest_index]),
            first_contact_index=(
                int(contact_indices[0]) if contact_indices.size else None
            ),
        )

    def _apply_damping(self, model: Any, row: np.ndarray) -> None:
        for column, sampled_value in zip(self._columns, row, strict=True):
            if column.kind != "joint_damping":
                continue
            for name, weight in zip(
                column.target_joint_names, column.allocation_weights, strict=True
            ):
                damping = float(sampled_value) * weight
                if damping < 0.0:
                    raise ValueError("sampled joint damping must be non-negative")
                index = self._topology.joint_index_by_name[name]
                model.dof_damping[self._topology.dof_addresses[index]] = damping

    def _apply_torque(
        self,
        data: Any,
        row: np.ndarray,
        step: int,
    ) -> np.ndarray:
        data.qfrc_applied[:] = 0.0
        for torque in self._config.base_joint_torques:
            index = self._topology.joint_index_by_name[torque.joint_name]
            data.qfrc_applied[self._topology.dof_addresses[index]] += torque.value_nm
        time_s = step * self._config.dt_s
        for column, sampled_value in zip(self._columns, row, strict=True):
            if column.kind != "joint_torque_offset":
                continue
            assert column.time_window_s is not None
            start_s, end_s = column.time_window_s
            if not start_s <= time_s < end_s:
                continue
            for name, weight in zip(
                column.target_joint_names, column.allocation_weights, strict=True
            ):
                index = self._topology.joint_index_by_name[name]
                data.qfrc_applied[self._topology.dof_addresses[index]] += (
                    float(sampled_value) * weight
                )
        return np.asarray(data.qfrc_applied[list(self._topology.dof_addresses)]).copy()

    def _record_sample(
        self,
        data: Any,
        times: np.ndarray,
        q: np.ndarray,
        v: np.ndarray,
        markers: np.ndarray,
        contacts: np.ndarray,
        index: int,
        model: Any,
    ) -> None:
        times[index] = float(data.time)
        q[index] = data.qpos[list(self._topology.qpos_addresses)]
        v[index] = data.qvel[list(self._topology.dof_addresses)]
        markers[index] = data.xpos[list(self._topology.marker_body_ids)]
        contacts[index] = self._source_target_contact(model, data)

    def _source_target_contact(self, model: Any, data: Any) -> bool:
        expected = {self._topology.source_body_id, self._topology.target_body_id}
        for index in range(int(data.ncon)):
            contact = data.contact[index]
            bodies = {
                int(model.geom_bodyid[int(contact.geom1)]),
                int(model.geom_bodyid[int(contact.geom2)]),
            }
            if bodies == expected:
                return True
        return False

    def collect_success(
        self,
        trial_index: int,
        plan_seed: int,
        sampled_row: np.ndarray,
        result: object,
    ) -> CanonicalTrialEvidence:
        """Convert one completed articulated trajectory into typed evidence."""
        if not isinstance(result, ArticulatedMujocoTrialResult):
            raise TypeError("result must be ArticulatedMujocoTrialResult")
        trace = result.trace
        assert trace.markers is not None
        trial_trace = TrialTrace(
            times_s=trace.t,
            q=trace.q,
            v=trace.v,
            coordinate_ids=tuple(
                f"joint.{name}" for name in self._config.coordinate_joint_names
            ),
            coordinate_units=("rad",) * len(self._config.coordinate_joint_names),
            velocity_units=("rad/s",) * len(self._config.coordinate_joint_names),
            markers_m=trace.markers,
            marker_ids=tuple(f"body.{name}" for name in self._config.marker_body_names),
            frame_id=self._config.frame_id,
            alignment_id=self._config.alignment_id,
            complete=True,
        )
        source_id = f"body.{self._config.source_body_name}"
        target_id = f"body.{self._config.target_body_name}"
        closest = ClosestApproach(
            time_s=float(trace.t[result.closest_sample_index]),
            distance_m=result.closest_distance_m,
            source_marker_id=source_id,
            target_id=target_id,
            contact_observed=result.contact_observed,
        )
        outcome: TrialOutcome = "no_impact"
        impact = None
        if result.first_contact_index is not None:
            outcome = "hit"
            impact = ImpactObservation(
                time_s=float(trace.t[result.first_contact_index]),
                state=self._impact_state(trace, result.first_contact_index),
            )
        return CanonicalTrialEvidence(
            trial_index=trial_index,
            seed=plan_seed,
            plan_sha256=self._plan_sha256,
            scenario_sha256=self._scenario_sha256,
            execution_config_sha256=self._execution_config_sha256,
            tools_revision=self._tools_revision,
            engine_id=self.engine_id,
            engine_revision=self._engine_revision,
            model_id=self.model_id,
            sampled_inputs=self._sampled_inputs(sampled_row),
            outcome=outcome,
            trace=trial_trace,
            impact=impact,
            closest_approach=closest,
        )

    def collect_failure(
        self,
        trial_index: int,
        plan_seed: int,
        sampled_row: np.ndarray,
        error: Exception,
    ) -> CanonicalTrialEvidence:
        """Retain a declared articulated domain or numerical failure."""
        return CanonicalTrialEvidence(
            trial_index=trial_index,
            seed=plan_seed,
            plan_sha256=self._plan_sha256,
            scenario_sha256=self._scenario_sha256,
            execution_config_sha256=self._execution_config_sha256,
            tools_revision=self._tools_revision,
            engine_id=self.engine_id,
            engine_revision=self._engine_revision,
            model_id=self.model_id,
            sampled_inputs=self._sampled_inputs(sampled_row),
            outcome="numerical_failure",
            trace=None,
            failure_reason=f"{type(error).__name__}: {error}",
        )

    def _sampled_inputs(self, sampled_row: np.ndarray) -> tuple[SampledInput, ...]:
        row = np.asarray(sampled_row, dtype=float).reshape(-1)
        if row.shape != (len(self._columns),):
            raise ValueError("sampled row does not match plan columns")
        return tuple(
            SampledInput(column.key, float(value), column.unit)
            for column, value in zip(self._columns, row, strict=True)
        )

    def _impact_state(self, trace: Trace, index: int) -> tuple[SampledInput, ...]:
        samples: list[SampledInput] = []
        for coordinate, angle, rate in zip(
            self._config.coordinate_joint_names,
            trace.q[index],
            trace.v[index],
            strict=True,
        ):
            samples.append(
                SampledInput(f"joint.{coordinate}.angle", float(angle), "rad")
            )
            samples.append(
                SampledInput(
                    f"joint.{coordinate}.angular_velocity", float(rate), "rad/s"
                )
            )
        assert trace.markers is not None
        marker_index = self._config.marker_body_names.index(
            self._config.source_body_name
        )
        if index == 0:
            velocity = (
                trace.markers[1, marker_index] - trace.markers[0, marker_index]
            ) / self._config.dt_s
        elif index == trace.t.size - 1:
            velocity = (
                trace.markers[-1, marker_index] - trace.markers[-2, marker_index]
            ) / self._config.dt_s
        else:
            velocity = (
                trace.markers[index + 1, marker_index]
                - trace.markers[index - 1, marker_index]
            ) / (2.0 * self._config.dt_s)
        samples.append(
            SampledInput(
                f"body.{self._config.source_body_name}.speed",
                float(np.linalg.norm(velocity)),
                "m/s",
            )
        )
        return tuple(samples)


def _require_name(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be non-empty")
    return value


def _require_unique_names(values: tuple[str, ...], name: str) -> None:
    if not isinstance(values, tuple) or not values:
        raise ValueError(f"{name} must contain names")
    if any(not isinstance(value, str) or not value.strip() for value in values):
        raise ValueError(f"{name} must contain non-empty names")
    if len(set(values)) != len(values):
        raise ValueError(f"{name} must be unique")


__all__ = [
    "ArticulatedMujocoTrialAdapter",
    "ArticulatedMujocoTrialConfig",
    "ArticulatedMujocoTrialResult",
    "MujocoVariationBinding",
    "NamedJointTorque",
]

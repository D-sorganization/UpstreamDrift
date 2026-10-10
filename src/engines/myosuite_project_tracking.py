"""Frozen dimensionless native tracking costs, separate from physical admission.

Decode complete states on owned scratch data without integrating or forwarding.
The same objective scores provisional search and fully guarded SDK histories.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, fields
import hashlib
import json
from pathlib import Path
from threading import Lock
from typing import Iterator, TypeAlias

import numpy as np
from numpy.typing import NDArray

from src.engines.myosuite_project_feedback import (
    ProjectTaskObservation,
    _immutable,
    snapshot_project_task_state,
)
from src.engines.myosuite_project_forecast import (
    ProjectTaskForecaster,
    _admit_observation_native_state,
    _require_current_observation,
)
from src.engines.myosuite_project_native_search import ProvisionalCommandPrediction
from src.engines.myosuite_project_task_producer import (
    ProjectTaskCommandHistory,
    native_action_bounds,
)
from src.engines.native_replay_contracts import (
    native_clock_interval_matches,
    require_native_step_clock,
)
from src.engines.physics_engines.myosuite.python.native_direct_model_replay import (
    _require_normalized_quaternions,
)

Array: TypeAlias = NDArray[np.float64]


def _source_sha256() -> str:
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


@dataclass(frozen=True)
class NativeTrackingReference:
    """Native targets at absolute clocks, with caller-declared source provenance.

    A provenance digest binds a declaration; it does not establish anatomical
    correspondence, measured subject identity or independent validation.
    """

    time_seconds: Array
    qpos: Array
    qvel: Array
    activation: Array
    provenance_sha256: str

    def __post_init__(self) -> None:
        digest = self.provenance_sha256
        if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
            raise ValueError("reference provenance requires lowercase SHA-256")
        for name in ("time_seconds", "qpos", "qvel", "activation"):
            value = _immutable(getattr(self, name))
            if not np.isfinite(value).all():
                raise ValueError("reference targets and clocks must be finite")
            object.__setattr__(self, name, value)


@dataclass(frozen=True)
class NativeTrackingScales:
    """Positive physical magnitudes, not inverse weights or learned tolerances.

    Position scales use each native tangent DOF's m/rad units; velocity scales
    use m/s or rad/s. Activation is dimensionless; post-mapping command slew
    scales use each native control channel's units. Empty activation vectors
    are allowed for motor models.
    """

    position: Array
    velocity: Array
    activation: Array
    command_slew: Array

    def __post_init__(self) -> None:
        for entry in fields(self):
            value = _immutable(getattr(self, entry.name))
            if value.ndim != 1 or not np.isfinite(value).all() or np.any(value <= 0):
                raise ValueError(
                    "physical tracking scales must be finite positive vectors"
                )
            object.__setattr__(self, entry.name, value)


@dataclass(frozen=True)
class NativeTrackingCost:
    """Dimensionless sums on one frozen grid, not timestep-invariant integrals.

    Velocity residuals compare native qvel coordinates without twist transport.
    """

    position: float
    velocity: float
    activation: float
    command_slew: float

    @property
    def total(self) -> float:
        return self.position + self.velocity + self.activation + self.command_slew


class ProjectTaskTrackingObjective:
    """Current-state-bound scoring; numerical cost cannot override hard gates.

    Caller owns the SDK forecaster and serializes live mutation. Scratch data
    alone receives saved states. No dynamics derivative or contact-smoothness
    assumption is made. Initial state, absolute clock and objective parameters
    stay frozen for this solve; a new receding-horizon solve needs a new anchor.
    """

    def __init__(
        self,
        forecaster: ProjectTaskForecaster,
        observation: ProjectTaskObservation,
        reference: NativeTrackingReference,
        scales: NativeTrackingScales,
    ) -> None:
        if not isinstance(forecaster, ProjectTaskForecaster):
            raise TypeError("tracking requires an admitted owned SDK forecaster")
        if not isinstance(reference, NativeTrackingReference) or not isinstance(
            scales, NativeTrackingScales
        ):
            raise TypeError("tracking requires explicit frozen reference and scales")
        self._forecaster = forecaster
        self._mj = forecaster._mj
        self._lock = Lock()
        self._source = _source_sha256()
        self._reference, self._scales = reference, scales
        self._parameters: str | None = None
        if forecaster._closed:
            raise RuntimeError("tracking SDK forecaster is closed")
        with forecaster._live_guard() as initial:
            live = forecaster._live
            self._model = live.model
            current = snapshot_project_task_state(live.model, live.data, sample_index=0)
            _require_current_observation(observation, current)
            _admit_observation_native_state(live.model, live.data, current)
            forecaster._verify_live(initial)
            self._initial = _immutable(initial)
            self._previous = _immutable(current.control)
            self._ordered_ids = current.ordered_actuator_ids
            self._origin = current.native_time_seconds
            self._scratch = self._mj.MjData(self._model)
            self._validate_reference()
            self._units = self._position_units()
            self._parameters = self._parameter_digest()
            self._verify_objective()

    @property
    def reference(self) -> NativeTrackingReference:
        return self._reference

    @property
    def scales(self) -> NativeTrackingScales:
        return self._scales

    @property
    def position_units(self) -> tuple[str, ...]:
        return self._units

    @property
    def source_sha256(self) -> str:
        return self._source

    @property
    def parameters_sha256(self) -> str:
        assert self._parameters is not None
        return self._parameters

    def _verify_objective(self) -> None:
        if _source_sha256() != self._source:
            raise ValueError("tracking objective source changed")
        if (
            self._parameters is not None
            and self._parameter_digest() != self._parameters
        ):
            raise ValueError("tracking objective parameters changed")

    @contextmanager
    def _guard(self) -> Iterator[None]:
        if not self._lock.acquire(blocking=False):
            raise RuntimeError("tracking objective is busy; reentrant use rejected")
        try:
            if self._forecaster._closed:
                raise RuntimeError("tracking SDK forecaster is closed")
            with self._forecaster._live_guard() as initial:
                self._verify_objective()
                if not np.array_equal(initial, self._initial):
                    raise ValueError(
                        "tracking live anchor differs from frozen initial state"
                    )
                try:
                    yield
                finally:
                    self._verify_objective()
        finally:
            self._lock.release()

    def _validate_reference(self) -> None:
        reference, scales, model = self.reference, self.scales, self._model
        count = len(reference.time_seconds)
        if reference.time_seconds.ndim != 1 or count < 2:
            raise ValueError("reference clock requires a complete positive horizon")
        if reference.time_seconds[0] != self._origin:
            raise ValueError("reference clock differs from current absolute epoch")
        for name, width in (
            ("qpos", model.nq),
            ("qvel", model.nv),
            ("activation", model.na),
        ):
            if getattr(reference, name).shape != (count, width):
                raise ValueError(
                    "reference requires complete native-order target dimensions"
                )
        for name, width in (
            ("position", model.nv),
            ("velocity", model.nv),
            ("activation", model.na),
            ("command_slew", model.nu),
        ):
            if getattr(scales, name).shape != (width,):
                raise ValueError("tracking scale dimensions differ from native model")
        options = model.opt
        for before, after in zip(
            reference.time_seconds[:-1], reference.time_seconds[1:], strict=True
        ):
            require_native_step_clock(
                float(before), float(after), float(options.timestep)
            )
        for qpos in reference.qpos:
            _require_normalized_quaternions(model, qpos)

    def _position_units(self) -> tuple[str, ...]:
        joint = self._mj.mjtJoint
        units: list[str] = []
        for kind in self._model.jnt_type:
            if kind == joint.mjJNT_FREE:
                units.extend(("m", "m", "m", "rad", "rad", "rad"))
            elif kind == joint.mjJNT_BALL:
                units.extend(("rad",) * 3)
            else:
                units.append("m" if kind == joint.mjJNT_SLIDE else "rad")
        if len(units) != self._model.nv:
            raise ValueError("native tangent unit layout is incomplete")
        return tuple(units)

    def _parameter_digest(self) -> str:
        live = self._forecaster._live
        source = live.source
        payload = {
            "model": live.model_sha256,
            "source": source.source_model_sha256,
            "closure": source.resource_closure_sha256,
            "forecast_source": self._forecaster._source,
            "initial": self._initial.tolist(),
            "previous_controls": self._previous.tolist(),
            "ordered_actuators": self._ordered_ids,
            "position_units": self.position_units,
            "reference": {
                entry.name: getattr(self.reference, entry.name).tolist()
                if entry.name != "provenance_sha256"
                else self.reference.provenance_sha256
                for entry in fields(self.reference)
            },
            "scales": {
                entry.name: getattr(self.scales, entry.name).tolist()
                for entry in fields(self.scales)
            },
            "objective": self.source_sha256,
        }
        encoded = json.dumps(
            payload, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
        return hashlib.sha256(encoded).hexdigest()

    def prediction_cost(
        self, prediction: ProvisionalCommandPrediction
    ) -> NativeTrackingCost:
        """Score copied full provisional states; retain provisional classification."""
        with self._guard():
            if not isinstance(prediction, ProvisionalCommandPrediction):
                raise TypeError("tracking requires an explicit provisional prediction")
            live = self._forecaster._live
            source = live.source
            model = prediction.planned_bundle.model
            if (
                prediction.ordered_actuator_ids != self._ordered_ids
                or model.loaded_native_model_sha256 != live.model_sha256
                or model.source_model_sha256 != source.source_model_sha256
            ):
                raise ValueError("prediction actuator order or compiled model differs")
            return self._score(
                prediction.integration_states,
                prediction.time_seconds + self._origin,
                prediction.applied_actuator_commands,
            )

    def history_cost(self, history: ProjectTaskCommandHistory) -> NativeTrackingCost:
        """Recompute cost from guarded SDK states, not provisional arrays/scores."""
        with self._guard():
            if not isinstance(history, ProjectTaskCommandHistory):
                raise TypeError("tracking requires an explicit guarded SDK history")
            live = self._forecaster._live
            source = live.source
            task = live.task
            if (
                history.loaded_native_model_sha256 != live.model_sha256
                or history.source_model_sha256 != source.source_model_sha256
                or history.resource_closure_sha256 != source.resource_closure_sha256
                or history.sdk_binding != live.sdk
                or history.project_task_source_sha256 != task.project_task_source_sha256
            ):
                raise ValueError("tracking history model/source/SDK binding differs")
            return self._score(
                history.integration_states,
                history.time_seconds,
                history.applied_actuator_commands,
            )

    def _score(
        self, supplied_states: Array, supplied_times: Array, supplied_commands: Array
    ) -> NativeTrackingCost:
        states = np.asarray(supplied_states, dtype=np.float64).copy()
        times = np.asarray(supplied_times, dtype=np.float64).copy()
        commands = np.asarray(supplied_commands, dtype=np.float64).copy()
        self._forecaster._verify_live(self._initial)
        self._verify_objective()
        reference, scales = self.reference, self.scales
        count = len(reference.time_seconds)
        if (
            states.shape != (count, len(self._initial))
            or times.shape != (count,)
            or commands.shape != (count - 1, self._model.nu)
        ):
            raise ValueError("tracking requires the exact complete declared horizon")
        if any(
            not np.isfinite(value).all() for value in (states, times, commands)
        ) or not np.array_equal(states[0], self._initial):
            raise ValueError("tracking requires finite history and exact initial state")
        if times[0] != self._origin or any(
            not native_clock_interval_matches(
                float(t - self._origin), float(r - self._origin)
            )
            for t, r in zip(times[1:], reference.time_seconds[1:], strict=True)
        ):
            raise ValueError("tracking clock differs from declared absolute grid")
        low, high = native_action_bounds(self._model)
        options = self._model.opt
        for before, after in zip(times[:-1], times[1:], strict=True):
            require_native_step_clock(
                float(before), float(after), float(options.timestep)
            )
        if np.any(commands < low) or np.any(commands > high):
            raise ValueError("tracking commands violate native control bounds")
        return self._residual_sums(states, times, commands)

    def _residual_sums(
        self, states: Array, times: Array, commands: Array
    ) -> NativeTrackingCost:
        position = velocity = activation = 0.0
        tangent = np.empty(self._model.nv, dtype=np.float64)
        reference, scales = self.reference, self.scales
        state_kind = self._mj.mjtState
        for index, state in enumerate(states):
            self._mj.mj_setState(
                self._model, self._scratch, state, state_kind.mjSTATE_INTEGRATION
            )
            if self._scratch.time != times[index] or (
                index and not np.array_equal(self._scratch.ctrl, commands[index - 1])
            ):
                raise ValueError(
                    "tracking complete state clock/control correspondence differs"
                )
            _require_normalized_quaternions(self._model, self._scratch.qpos)
            self._mj.mj_differentiatePos(
                self._model, tangent, 1.0, reference.qpos[index], self._scratch.qpos
            )
            position += float(np.sum((tangent / scales.position) ** 2))
            velocity += float(
                np.sum(
                    ((self._scratch.qvel - reference.qvel[index]) / scales.velocity)
                    ** 2
                )
            )
            activation += float(
                np.sum(
                    (
                        (self._scratch.act - reference.activation[index])
                        / scales.activation
                    )
                    ** 2
                )
            )
        previous = np.vstack((self._previous, commands[:-1]))
        slew = float(np.sum(((commands - previous) / scales.command_slew) ** 2))
        result = NativeTrackingCost(position, velocity, activation, slew)
        if not np.isfinite(result.total):
            raise ValueError(
                "tracking objective overflowed its declared physical scales"
            )
        return result

"""Provisional owned-native search; selected plans retain guarded SDK authority.

Invocation-boundary model fingerprints are deliberately distinct from the SDK
producer's per-step checks. Search arrays and rankings never qualify execution.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field, replace
import hashlib
from pathlib import Path
from threading import Lock
from time import perf_counter
from typing import Any

import numpy as np
from numpy.typing import NDArray

from src.engines.myosuite_project_feedback import ProjectTaskObservation, _immutable
from src.engines.myosuite_project_forecast import (
    ProjectTaskForecaster,
    _admit_observation_native_state,
    _require_current_observation,
)
from src.engines.myosuite_project_feedback import snapshot_project_task_state
from src.engines.myosuite_project_task_producer import (
    ProjectTaskCommandHistory,
    _model_sha256,
)
from src.engines.native_direct_model_provider import create_native_direct_model
from src.engines.native_replay_contracts import (
    native_replay_contract_types,
    require_no_global_mujoco_callbacks,
    validate_native_replay_bundle,
)
from src.engines.physics_engines.myosuite.python import (
    native_direct_model_replay as direct,
)
from src.engines.project_task_replay_artifacts import (
    FrozenProjectTaskReplay,
    _initial_state,
    freeze_project_task_history,
)


@dataclass(frozen=True)
class ProvisionalCommandPrediction:
    """Copied search arrays; no producer history or execution qualification."""

    time_seconds: NDArray[np.float64]
    qpos: NDArray[np.float64]
    qvel: NDArray[np.float64]
    activation: NDArray[np.float64]
    integration_states: NDArray[np.float64]
    applied_actuator_commands: NDArray[np.float64]
    ordered_actuator_ids: tuple[str, ...]
    evaluator_source_sha256: str
    planned_input_sha256: str
    planned_bundle: Any
    classification: str = field(default="provisional-native-search", init=False)


@dataclass(frozen=True)
class GuardedCommandPlan:
    """Guarded full-plan recording/replay and recomputed caller numerical cost.

    A validated future plan is distinct from an executed live prefix. Caller
    numerical criteria do not grant anatomical or physiological acceptance.
    """

    history: ProjectTaskCommandHistory
    replay_artifact: FrozenProjectTaskReplay
    objective: float
    validation_seconds: float


def _source_sha256() -> str:
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


class ProjectTaskNativeSearch:
    """Reuse one owned native context and the existing direct execution kernel.

    Commands are ordered post-mapping controls, never implicitly clamped raw
    SDK actions. The caller owns the forecaster and serializes live mutations.
    Search fingerprints the model at invocation boundaries; transient mutation
    absence is not proven. Promotion always recomputes guarded history and cost.
    """

    def __init__(
        self,
        forecaster: ProjectTaskForecaster,
        seed: ProjectTaskCommandHistory,
        *,
        model_id: str,
        variant_id: str,
        experiment_id: str,
        max_steps: int,
    ) -> None:
        if not isinstance(forecaster, ProjectTaskForecaster):
            raise TypeError("native search requires an owned SDK forecaster")
        if type(max_steps) is not int or max_steps <= 0:
            raise ValueError("native search requires a positive maximum step budget")
        started = perf_counter()
        self._forecaster = forecaster
        self._mj = forecaster._mj
        self._lock = Lock()
        self._closed = False
        self._source = _source_sha256()
        self._max_steps = max_steps
        self._model_id, self._variant_id = model_id, variant_id
        self._experiment_id = experiment_id
        self.last_attempt_seconds: float | None = None
        self._kernel = direct._execute_commands
        self._kernel_code = self._kernel.__code__
        self._environment: Any = None
        try:
            self._require_open()
            with forecaster._live_guard():
                self._seed = self._freeze(seed, "seed")
                environment, model, data, closure = (
                    direct._load_registered_direct_model(
                        self._seed.registration, create_native_direct_model, self._mj
                    )
                )
                self._environment = environment
                self._model, self._data, self._closure = model, data, closure
                self._verify_native()
        except BaseException:
            if self._environment is not None:
                self._environment.close()
            raise
        self.construction_seconds = perf_counter() - started

    def _require_open(self) -> None:
        if self._closed or self._forecaster._closed:
            raise RuntimeError("native search or its SDK forecaster is closed")
        if _source_sha256() != self._source:
            raise ValueError("native search adapter source changed")
        if (
            direct._execute_commands is not self._kernel
            or self._kernel.__code__ is not self._kernel_code
        ):
            raise ValueError("native search execution kernel changed")

    def _verify_native(self) -> None:
        self._require_open()
        require_no_global_mujoco_callbacks(self._mj)
        environment = self._environment
        if environment.model is not self._model or environment.data is not self._data:
            raise ValueError("native search model/data handles changed")
        registration = self._seed.registration
        if (
            _model_sha256(self._mj, self._model)
            != registration.loaded_native_model_sha256
        ):
            raise ValueError("native search model changed")
        direct._validate_environment_identity(
            registration, create_native_direct_model, environment, self._closure
        )

    @contextmanager
    def _attempt(self) -> Iterator[NDArray[np.float64]]:
        if not self._lock.acquire(blocking=False):
            raise RuntimeError(
                "native search is busy; reentrant/concurrent use rejected"
            )
        started = perf_counter()
        try:
            self._require_open()
            with self._forecaster._live_guard() as initial:
                self._verify_native()
                try:
                    yield initial
                finally:
                    self._verify_native()
        finally:
            self.last_attempt_seconds = perf_counter() - started
            self._lock.release()

    def _freeze(
        self, history: ProjectTaskCommandHistory, suffix: str
    ) -> FrozenProjectTaskReplay:
        live = self._forecaster._live
        source = live.source
        return freeze_project_task_history(
            live.task,
            history,
            resource_root=source.resource_root,
            resources=source.resources,
            model_id=self._model_id,
            variant_id=self._variant_id,
            experiment_id=f"{self._experiment_id}:{suffix}",
        )

    def _prepare(
        self,
        observation: ProjectTaskObservation,
        supplied: NDArray[np.float64],
        initial: NDArray[np.float64],
    ) -> tuple[Any, NDArray[np.float64], NDArray[np.float64], str]:
        live = self._forecaster._live
        current = snapshot_project_task_state(live.model, live.data, sample_index=0)
        _require_current_observation(observation, current)
        _admit_observation_native_state(live.model, live.data, current)
        contracts = native_replay_contract_types()
        schema, state = _initial_state(live.data, initial, contracts)
        commands = np.asarray(supplied, dtype=np.float64).copy()
        if (
            commands.ndim != 2
            or not 0 < len(commands) <= self._max_steps
            or commands.shape[1] != self._model.nu
            or not np.isfinite(commands).all()
        ):
            raise ValueError("search commands require a finite bounded native horizon")
        bundle = self._state_bundle(state, schema, commands)
        bundle = validate_native_replay_bundle(bundle, contracts)
        direct._validate_bundle_identity(bundle, self._seed.registration, contracts)
        # Preparation can invoke supplied conversions or bundle construction.
        # Verify handles, source, callbacks and live state after all such work,
        # immediately before native admission; reuse that full provider check.
        self._forecaster._verify_live(initial)
        self._verify_native()
        times, commands = direct._validate_native_execution_identity(
            bundle,
            self._model,
            self._seed.registration,
        )
        profile = direct.compiled_actuator_profile_bytes(
            bundle, self._seed.registration, self._model, self._closure
        )
        return bundle, times, commands, hashlib.sha256(profile).hexdigest()

    def _state_bundle(
        self, state: Any, schema: Any, commands: NDArray[np.float64]
    ) -> Any:
        seed = self._seed.bundle
        if schema != seed.model.state_schema:
            raise ValueError("search native state schema changed")
        contracts = native_replay_contract_types()
        held = np.vstack((commands, commands[-1]))
        options = self._model.opt
        times = np.arange(len(held), dtype=np.float64) * float(options.timestep)
        return contracts.build_experiment_replay_bundle(
            f"{self._experiment_id}:proposal",
            seed.model,
            seed.capabilities,
            state,
            seed.input_history.channels,
            seed.input_history.input_kind,
            seed.input_history.interpolation,
            tuple(times),
            tuple(tuple(row) for row in held),
            seed.policy,
        )

    def _restore(self, bundle: Any) -> NDArray[np.float64]:
        mj = self._mj
        states = direct._bundle_state(bundle, self._model)
        mj.mj_resetData(self._model, self._data)
        mj.mj_setState(
            self._model,
            self._data,
            states["integration"],
            mj.mjtState.mjSTATE_INTEGRATION,
        )
        direct._verify_restored_state(self._model, self._data, states)
        mj.mj_forward(self._model, self._data)
        direct._verify_restored_state(self._model, self._data, states)
        return states["integration"]

    def predict(
        self, observation: ProjectTaskObservation, applied_commands: NDArray[np.float64]
    ) -> ProvisionalCommandPrediction:
        """Evaluate ordered post-mapping commands; return only provisional copies."""
        with self._attempt() as initial:
            bundle, times, commands, profile_sha = self._prepare(
                observation, applied_commands, initial
            )
            integration = self._restore(bundle)
            registration = self._seed.registration
            schedule = direct._DirectCommandSchedule(
                times,
                commands,
                integration,
                bundle,
                self._closure,
                registration.actuator_law_manifest_sha256,
                profile_sha,
            )
            result = self._kernel(self._model, self._data, schedule)
            return ProvisionalCommandPrediction(
                _immutable(result.time_seconds),
                _immutable(result.qpos),
                _immutable(result.qvel),
                _immutable(result.actuator_activation),
                _immutable(result.integration_states),
                _immutable(result.applied_actuator_commands),
                observation.ordered_actuator_ids,
                self._source,
                bundle.applied_input_sha256,
                bundle,
            )

    def promote(
        self,
        observation: ProjectTaskObservation,
        applied_commands: NDArray[np.float64],
        *,
        objective: Callable[[ProjectTaskCommandHistory], float],
        admit: Callable[[ProjectTaskCommandHistory], bool],
    ) -> GuardedCommandPlan:
        """Recompute the full guarded plan, numerical criteria and fresh replay.

        No provisional states or scores are accepted. This does not apply a live
        command; callers must retain current-state admission for any prefix.
        """
        started = perf_counter()
        with self._attempt() as initial:
            if not callable(objective) or not callable(admit):
                raise TypeError(
                    "promotion requires objective and hard-criterion callbacks"
                )
            _, _, held, _ = self._prepare(observation, applied_commands, initial)
            commands = held[:-1]
            recorded = self._forecaster.predict(observation, commands).history
            history = replace(
                recorded,
                time_seconds=_immutable(recorded.time_seconds),
                integration_states=_immutable(recorded.integration_states),
                applied_actuator_commands=_immutable(
                    recorded.applied_actuator_commands
                ),
            )
            if not np.array_equal(history.applied_actuator_commands, commands):
                raise ValueError(
                    "guarded SDK mapping differs from post-mapping search commands"
                )
            admitted = admit(history)
            if type(admitted) is not bool or not admitted:
                raise ValueError("guarded command plan failed declared hard criteria")
            cost = float(objective(history))
            if not np.isfinite(cost):
                raise ValueError("guarded command objective must be finite")
            artifact = self._freeze(history, "guarded-plan")
        return GuardedCommandPlan(history, artifact, cost, perf_counter() - started)

    def close(self) -> None:
        """Close only this owned native context; caller retains the SDK forecaster."""
        if not self._lock.acquire(blocking=False):
            raise RuntimeError("native search is busy; closure rejected")
        try:
            if not self._closed:
                self._closed = True
                self._environment.close()
        finally:
            self._lock.release()

    def __enter__(self) -> ProjectTaskNativeSearch:
        self._require_open()
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

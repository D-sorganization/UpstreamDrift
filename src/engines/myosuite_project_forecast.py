"""Owned SDK forecasts from the live plant's complete current native state.

These predictions supply a shooting primitive, not an optimizer or a tracking
qualification. They reuse the SDK producer; independent replay stays separate.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
from pathlib import Path
from threading import Lock
from time import perf_counter
from typing import Any, Iterator

import numpy as np
from numpy.typing import NDArray

from src.engines.myosuite_project_feedback import (
    ProjectTaskObservation,
    snapshot_project_task_state,
)
from src.engines.myosuite_project_task_producer import (
    ProjectTaskCommandHistory,
    _admit_task,
    _model_sha256,
    _read_native_state,
    _verify_source_files,
    create_project_golf_task,
    record_project_task_commands,
)
from src.engines.native_replay_contracts import require_no_global_mujoco_callbacks
from src.engines.physics_engines.myosuite.python.native_direct_model_replay import (
    _verify_restored_state,
)


@dataclass(frozen=True)
class ProjectTaskForecast:
    """An SDK producer history and the guarded forecasting adapter identity."""

    history: ProjectTaskCommandHistory
    forecast_adapter_source_sha256: str


@dataclass(frozen=True)
class _TaskIdentity:
    task: Any
    model: Any
    data: Any
    source: Any
    sdk: Any
    model_sha256: str

    @classmethod
    def capture(cls, task: Any, mj: Any) -> _TaskIdentity:
        _admit_task(task, mj)
        _verify_source_files(task.project_model_source)
        return cls(
            task,
            task.model,
            task.data,
            task.project_model_source,
            task.project_sdk_binding,
            _model_sha256(mj, task.model),
        )

    def verify(self, mj: Any) -> None:
        if self.task.model is not self.model or self.task.data is not self.data:
            raise ValueError("forecast native model/data handles changed")
        if (
            self.task.project_model_source != self.source
            or self.task.project_sdk_binding != self.sdk
        ):
            raise ValueError("forecast model source or SDK lineage changed")
        _admit_task(self.task, mj)
        _verify_source_files(self.source)
        if _model_sha256(mj, self.model) != self.model_sha256:
            raise ValueError("forecast native model changed")


def _source_sha256() -> str:
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def _require_current_observation(
    supplied: ProjectTaskObservation, current: ProjectTaskObservation
) -> None:
    if not isinstance(supplied, ProjectTaskObservation):
        raise TypeError("forecast requires a current project task observation")
    if (
        type(supplied.sample_index) is not int
        or supplied.sample_index < 0
        or supplied.native_time_seconds != current.native_time_seconds
        or supplied.ordered_actuator_ids != current.ordered_actuator_ids
        or any(
            not np.array_equal(getattr(supplied, field), getattr(current, field))
            for field in ("qpos", "qvel", "activation", "control", "integration_state")
        )
    ):
        raise ValueError("forecast observation differs from current live plant")


def _admit_observation_native_state(
    model: Any, data: Any, observation: ProjectTaskObservation
) -> None:
    """Apply the existing replay restoration/scope check to a native snapshot."""
    _verify_restored_state(
        model,
        data,
        {
            "qpos": observation.qpos,
            "qvel": observation.qvel,
            "actuator_internal": observation.control,
            "actuator_activation": observation.activation,
            "integration": observation.integration_state,
        },
    )


class ProjectTaskForecaster:
    """Reuse one owned native plant, restoring every integration field per call.

    The caller owns the live task and serializes its mutations. Concurrent or
    reentrant forecasts are rejected. Failures return no partial history; live
    mutation is detected, not rolled back. Timings include admission overhead.
    """

    def __init__(self, live_task: Any) -> None:
        import mujoco as mj

        started = perf_counter()
        self._mj = mj
        self._lock = Lock()
        self._closed = False
        self.last_attempt_seconds: float | None = None
        require_no_global_mujoco_callbacks(mj)
        self._source = _source_sha256()
        self._live = _TaskIdentity.capture(live_task, mj)
        candidate = None
        try:
            with self._live_guard():
                source = self._live.source
                candidate = create_project_golf_task(
                    Path(live_task.model_path),
                    self._live.sdk,
                    resource_root=source.resource_root,
                )
                self._candidate = candidate
                self._owned = _TaskIdentity.capture(candidate, mj)
                if (
                    candidate.model is self._live.model
                    or candidate.data is self._live.data
                    or self._owned.model_sha256 != self._live.model_sha256
                    or self._owned.source != self._live.source
                ):
                    raise ValueError(
                        "forecast requires an owned source-identical model"
                    )
        except BaseException:
            if candidate is not None and candidate is not live_task:
                candidate.close()
            raise
        self.construction_seconds = perf_counter() - started

    def _verify_live(self, state: NDArray[np.float64]) -> None:
        require_no_global_mujoco_callbacks(self._mj)
        self._live.verify(self._mj)
        if _source_sha256() != self._source:
            raise ValueError("forecast adapter source changed")
        if not np.array_equal(
            state, _read_native_state(self._mj, self._live.model, self._live.data)
        ):
            raise ValueError("forecast changed the live plant state")

    @contextmanager
    def _live_guard(self) -> Iterator[NDArray[np.float64]]:
        require_no_global_mujoco_callbacks(self._mj)
        self._live.verify(self._mj)
        state = _read_native_state(self._mj, self._live.model, self._live.data)
        self._verify_live(state)
        try:
            yield state
        finally:
            self._verify_live(state)

    def _restore(self, state: NDArray[np.float64]) -> None:
        mj = self._mj
        self._owned.verify(self._mj)
        self._candidate.reset(seed=0)
        self._owned.verify(self._mj)
        self._mj.mj_setState(
            self._owned.model,
            self._owned.data,
            state,
            mj.mjtState.mjSTATE_INTEGRATION,
        )
        self._mj.mj_forward(self._owned.model, self._owned.data)
        if not np.array_equal(
            state, _read_native_state(self._mj, self._owned.model, self._owned.data)
        ):
            raise ValueError("forecast did not restore complete native state")

    def predict(
        self, observation: ProjectTaskObservation, commands: NDArray[np.float64]
    ) -> ProjectTaskForecast:
        """Forecast SDK-mapped commands without stepping or resetting the live task."""
        if not self._lock.acquire(blocking=False):
            raise RuntimeError("forecast is busy; concurrent or reentrant use rejected")
        started = perf_counter()
        try:
            if self._closed:
                raise RuntimeError("forecast is closed")
            with self._live_guard() as state:
                current = snapshot_project_task_state(
                    self._live.model, self._live.data, sample_index=0
                )
                _admit_observation_native_state(
                    self._live.model, self._live.data, current
                )
                _require_current_observation(observation, current)
                actions = np.asarray(commands, dtype=np.float64).copy()
                # Conversion can execute caller code. Validate before owned stepping.
                self._verify_live(state)
                self._restore(state)
                history = record_project_task_commands(self._candidate, actions)
                self._owned.verify(self._mj)
            return ProjectTaskForecast(history, self._source)
        finally:
            self.last_attempt_seconds = perf_counter() - started
            self._lock.release()

    def close(self) -> None:
        """Release only the owned task; repeated closure is harmless."""
        if not self._lock.acquire(blocking=False):
            raise RuntimeError("forecast is busy; closure rejected")
        try:
            if not self._closed:
                self._closed = True
                self._candidate.close()
        finally:
            self._lock.release()

    def __enter__(self) -> ProjectTaskForecaster:
        if self._closed:
            raise RuntimeError("forecast is closed")
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()
